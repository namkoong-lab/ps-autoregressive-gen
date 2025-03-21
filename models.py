import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.distributions.beta import Beta

class ModelWithPrior(nn.Module):
    def __init__(self, modelclass, prior_scale, *args, **kwargs):
        super(ModelWithPrior, self).__init__()
        self.model = modelclass(*args, **kwargs)
        self.prior = modelclass(*args, **kwargs)
        self.prior_scale = prior_scale
    def forward(self, *args, **kwargs):
        model_output = self.model(*args, **kwargs)
        with torch.no_grad():
            prior_output = self.prior(*args, **kwargs)
        return model_output + self.prior_scale * prior_output

# Randomized prior functions:
# Ian Osband, John Aslanides, and Albin Cassirer. Randomized prior functions for deep
# reinforcement learning.
class LinearRandPrior(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None) -> None:
        super(LinearRandPrior, self).__init__(in_features, out_features, bias)
        #self.prior_weights = torch.empty((out_features, in_features))
        self.prior_weights = torch.nn.parameter.Parameter(torch.empty((out_features, in_features), requires_grad=False))
        
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight+self.prior_weights, self.bias)


class VariableMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_layers=3, width=50, rand_prior=False, last_fn='sigmoid', init_bias=None):
        super(VariableMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.width = width
        self.rand_prior = rand_prior
        self.last_fn = last_fn
        self.init_bias = init_bias

        hidden_sizes = [self.width]*self.num_layers
        self.hidden_sizes = hidden_sizes

        layers = []

        if rand_prior:
            # Add input layer
            layers.append(LinearRandPrior(input_dim, hidden_sizes[0]))
            layers.append(nn.ReLU())
    
            # Add hidden layers
            for i in range(len(hidden_sizes) - 1):
                layers.append(LinearRandPrior(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
    
            # Add output layer
            layers.append(LinearRandPrior(hidden_sizes[-1], output_dim))
        else:
            # Add input layer
            layers.append(nn.Linear(input_dim, hidden_sizes[0]))
            layers.append(nn.ReLU())
    
            # Add hidden layers
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
    
            # Add output layer
            layers.append(nn.Linear(hidden_sizes[-1], output_dim))

        # Initialize weights with Xavier uniform
        self.apply(self.init_weights)

        if self.init_bias is not None:
            m = layers[-1]
            nn.init.constant_(m.bias, self.init_bias)
        
        if self.last_fn is not None and self.last_fn.lower() != 'none':
            if self.last_fn.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.last_fn.lower() == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError('last fn argument value not supported') 
        self.model = nn.Sequential(*layers)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        if isinstance(m, LinearRandPrior):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
            nn.init.xavier_uniform_(m.prior_weights)

    def forward(self, x):
        return self.model(x)



class BertEncoder(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
    
    def forward(self, kwargs):
        embed = self.bert_model(**kwargs).last_hidden_state[:,0,:]
        return embed

    def output_dim(self):
        return self.bert_model.config.hidden_size

        

class MarginalPredictor(nn.Module):
    
    def __init__(self, bert_encoder=None, category_args=None, Z_dim=None, MLP_layer=6, MLP_width=50, rand_prior=False, prior_scale=0):
        super().__init__()
        self.use_bert = False
        self.use_category = False
        self.MLP_width = MLP_width
        self.MLP_layer = MLP_layer
        self.rand_prior = rand_prior
        self.prior_scale = prior_scale # osband type
        print('model prior scale', prior_scale)
        if bert_encoder is not None:
            self.z_encoder = BertEncoder(bert_encoder)
            self.z_encoder_output_dim = self.z_encoder.output_dim()
            self.use_bert = True
        elif category_args is not None:
            self.z_encoder = nn.Embedding(num_embeddings=category_args['num_embeddings'],
                                          embedding_dim=category_args['embedding_dim'])
            nn.init.xavier_uniform_(self.z_encoder.weight)
            self.z_encoder_output_dim = category_args['embedding_dim']
            self.use_category = True
        elif Z_dim is not None:
            self.z_encoder = lambda x: x
            self.z_encoder_output_dim = Z_dim
        else:
            raise ValueError("Need a Z feature")

        if self.prior_scale == 0:
            self.top_layer = VariableMLP(input_dim=self.z_encoder_output_dim,
                             num_layers=MLP_layer, width=MLP_width, rand_prior=rand_prior)
        else:
            print('make model with prior')
            self.top_layer = ModelWithPrior(VariableMLP, self.prior_scale, 
                             input_dim=self.z_encoder_output_dim,
                             num_layers=MLP_layer, width=MLP_width, rand_prior=False, last_fn='none')

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, kwargs):
        embed = self.z_encoder(kwargs)
        return self.sigmoid(self.top_layer(embed))


    def eval_seq(self, Zkwargs, Y, N=None, return_preds=False, trainlen=None, exact=False):
        if N is None:
            N = Y.shape[1]
        N = min(N, Y.shape[1])
        
        # not a sequence model (marginal loss)
        p_hats = self.forward(Zkwargs)
        # true success_p and p_hat_pred have dimensions (rows, N)
        p_hat_pred = p_hats.repeat((1, N))
        loss_matrix = nn.functional.binary_cross_entropy(p_hat_pred, Y[:,:N], 
                                                  reduction='none')
        if return_preds:
            return loss_matrix, p_hat_pred
        return loss_matrix

class SequentialBetaBernoulli(nn.Module):
    def __init__(self, bert_encoder=None, MLP_layer=3, MLP_width=50, Z_dim=None):
        # skip category for now
        # one shared bert / feature, then two heads on top for (alpha / (alpha+beta)) and (alpha + beta)
        super().__init__()
        self.use_bert = False
        self.use_category = False # not supported
        self.MLP_layer = MLP_layer
        self.MLP_width = MLP_width
        if bert_encoder is not None:
            self.z_encoder = BertEncoder(bert_encoder)
            self.z_encoder_output_dim = self.z_encoder.output_dim()
            self.use_bert = True
        elif Z_dim is not None:
            self.z_encoder = lambda x: x
            self.z_encoder_output_dim = Z_dim
        else:
            # There are no Z features
            self.z_encoder = lambda x: None
            self.z_encoder_output_dim = 0

        # one shared bert / feature, then two heads on top for (alpha / (alpha+beta)) and (alpha + beta)
        self.top_layer_mean = VariableMLP(input_dim=self.z_encoder_output_dim,
                             num_layers=MLP_layer, width=MLP_width)
        self.top_layer_aplusb = VariableMLP(input_dim=self.z_encoder_output_dim,
                             num_layers=MLP_layer, width=MLP_width, last_fn='relu', init_bias=1)

    def get_device(self):
        return next(self.top_layer_mean.parameters()).device

    def init_model_states(self, batch_size):
        init_state = torch.zeros((batch_size, 2))
        return init_state.to(self.get_device())


    def next_model_states(self, prev_Ys):
        if len(prev_Ys.shape)==2:
            ones_count = prev_Ys.sum(1)
            zeros_count = (1-prev_Ys).sum(1)
        else:
            ones_count = prev_Ys
            zeros_count = 1-prev_Ys
        return torch.cat([zeros_count.unsqueeze(-1), ones_count.unsqueeze(-1)],1).to(self.get_device())
    
    def update_state(self, curr_state, prev_Ys):
        new_stuff = self.next_model_states(prev_Ys)
        return curr_state + new_stuff

    def get_p_pred(self, a, b, prev_Ys):
        curr_state = self.next_model_states(prev_Ys)
        ones = curr_state[:,1]
        zeros = curr_state[:,0]
        return (a+ones)/(a+b+ones+zeros)

    def get_alpha_beta(self, encoded_Z):
        mean = self.top_layer_mean(encoded_Z)
        aplusb = self.top_layer_aplusb(encoded_Z)
        a = mean * aplusb
        b = aplusb - a
        return a,b

    # no mask as input, use mask later. 
    # no trainlen stuff here
    def eval_seq(self, Zkwargs, Y, N=None, return_preds=False, trainlen=None, exact=False):
        if N is None:
            N = Y.shape[1]
        N = min(N, Y.shape[1])
        encoded_Z = self.z_encoder(Zkwargs)
        Ys = Y[:,:N]
        all_ones = torch.cat([torch.zeros((Ys.shape[0],1)).to(Ys.device), Ys.cumsum(1)],1)
        all_zeros = torch.cat([torch.zeros((Ys.shape[0],1)).to(Ys.device), (1-Ys).cumsum(1)],1)
        a,b = self.get_alpha_beta(encoded_Z)
        p_hat_pred = (a+all_ones)/(a+b+all_ones+all_zeros)
        loss_matrix = nn.functional.binary_cross_entropy(p_hat_pred[:,:-1], Ys, 
                                                  reduction='none')
        if return_preds:
            return loss_matrix, p_hat_pred
        return loss_matrix

    # get closed form posterior draws. num_imagined is not used at all. 
    def get_posterior_draws(self, Z_input, curr_state, num_imagined, num_repetitions, repeat_batch_size=None):
        a, b = self.get_alpha_beta(Z_input)
        a = a.flatten()
        b = b.flatten()
        ones = curr_state[:,1]
        zeros = curr_state[:,0]
        beta = Beta(a + ones, b + zeros)
        samples = beta.sample((num_repetitions,)).T
        return samples


class SequentialBetaBernoulliAlphaBeta(SequentialBetaBernoulli):
    def __init__(self, bert_encoder=None, MLP_layer=3, MLP_width=50, Z_dim=None):
        # skip category for now
        # one shared bert / feature, then two heads on top for (alpha / (alpha+beta)) and (alpha + beta)
        super().__init__()
        self.use_bert = False
        self.use_category = False # not supported
        self.MLP_layer = MLP_layer
        self.MLP_width = MLP_width
        if bert_encoder is not None:
            self.z_encoder = BertEncoder(bert_encoder)
            self.z_encoder_output_dim = self.z_encoder.output_dim()
            self.use_bert = True
        elif Z_dim is not None:
            self.z_encoder = lambda x: x
            self.z_encoder_output_dim = Z_dim
        else:
            # There are no Z features
            self.z_encoder = lambda x: None
            self.z_encoder_output_dim = 0

        # one shared bert / feature, then two heads on top for (alpha / (alpha+beta)) and (alpha + beta)
        self.top_layer_alpha = VariableMLP(input_dim=self.z_encoder_output_dim,
                             num_layers=MLP_layer, width=MLP_width, last_fn='relu', init_bias=1)
        self.top_layer_beta = VariableMLP(input_dim=self.z_encoder_output_dim,
                             num_layers=MLP_layer, width=MLP_width, last_fn='relu', init_bias=1)

    def get_alpha_beta(self, encoded_Z):
        alpha = self.top_layer_alpha(encoded_Z)
        beta = self.top_layer_beta(encoded_Z)
        return alpha, beta


class SequentialPredictor(nn.Module):
    def __init__(self, bert_encoder=None, category_args=None, Z_dim=None, MLP_layer=6, 
                 MLP_width=50, init_mean=0.5, repeat_suffstat=100, MLP_last_fn='sigmoid'):
        super(SequentialPredictor, self).__init__()
        self.use_bert = False
        self.use_category = False
        self.init_mean = init_mean
        self.MLP_layer = MLP_layer
        self.MLP_width = MLP_width
        self.repeat_suffstat = repeat_suffstat

        if bert_encoder is not None:
            self.z_encoder = BertEncoder(bert_encoder)
            self.z_encoder_output_dim = self.z_encoder.output_dim()
            self.use_bert = True

        elif category_args is not None:
            self.z_encoder = nn.Embedding(num_embeddings=category_args['num_embeddings'],
                                          embedding_dim=category_args['embedding_dim'])
            nn.init.xavier_uniform_(self.z_encoder.weight)
            self.z_encoder_output_dim = category_args['embedding_dim']
            self.use_category = True

        elif Z_dim is not None:
            self.z_encoder = lambda x: x
            self.z_encoder_output_dim = Z_dim

        else:
            # There are no Z features
            self.z_encoder = lambda x: None
            self.z_encoder_output_dim = 0

        self.top_layer = VariableMLP(input_dim=self.z_encoder_output_dim+2*repeat_suffstat,
                             num_layers=MLP_layer, width=MLP_width, last_fn=MLP_last_fn)


    def get_device(self):
        return next(self.top_layer.parameters()).device

    def init_model_states(self, batch_size):
        init_state = torch.tensor([self.init_mean, 1])  # mean, 1/n+1
        return init_state.repeat((batch_size,1)).repeat_interleave(self.repeat_suffstat, dim=1).to(self.get_device())


    def next_model_states(self, prev_Ys, click_mask=None):
        # prev_Ys must have dimensions (row, observation)
        assert prev_Ys.dim() == 2
        if prev_Ys.shape[1] == 0:
            return self.init_model_states(prev_Ys.shape[0])
        if click_mask is None:
            click_mask = torch.ones_like(prev_Ys).to(prev_Ys.device)
        counts = torch.sum(click_mask, axis=1).unsqueeze(1)
        counts_denom = counts.clone().to(prev_Ys.device)
        counts_denom[counts == 0]=1
        means = torch.sum(prev_Ys*click_mask, axis=1).unsqueeze(1) / counts_denom
        inv_N = torch.ones(means.shape).to(prev_Ys.device)/(counts+1)
        
        device = self.get_device()
        inv_N = inv_N.to(device)
        return torch.cat([means, inv_N], 1).repeat_interleave(self.repeat_suffstat, dim=1).to(device)


    def get_p_pred(self, encoded_Z, prev_Ys, click_mask=None):
        curr_state = self.next_model_states(prev_Ys, click_mask)
        
        if encoded_Z is not None:
            state = torch.cat([encoded_Z, curr_state], 1) 
        else:
            state = curr_state
        
        return self.top_layer(state)


    # no mask as input, use mask later. 
    def eval_seq(self, Zkwargs, Y, true_p=None, N=None, return_preds=False, trainlen=None, exact=False):
        if N is None:
            N = Y.shape[1]
        N = min(N, Y.shape[1])
        if trainlen is None: trainlen = N
        encoded_Z = self.z_encoder(Zkwargs)
        all_p_hats = []
        for j in range(N):
            if exact:
                start_idx = max(0, j-trainlen)
            else:
                start_idx = (j // trainlen) * trainlen
            prev_Ys = Y[:,start_idx:j]
            
            p_hat = self.get_p_pred(encoded_Z, prev_Ys=prev_Ys)
            all_p_hats.append(p_hat)
            
        p_hat_pred = torch.cat(all_p_hats,1)
        loss_matrix = nn.functional.binary_cross_entropy(p_hat_pred, Y[:,:N], 
                                                  reduction='none')
        if return_preds:
            return loss_matrix, p_hat_pred

        return loss_matrix

    
    def update_state(self, curr_state, Y_sample):
        next_state = torch.clone(curr_state)
        if len(Y_sample.shape) == 1:
            Y_sample = Y_sample.unsqueeze(-1)
    
        if len(Y_sample.shape) == 2:
            num_arms, num_Ys = Y_sample.shape
            prev_means = next_state[:,:self.repeat_suffstat]
            prev_cnt = 1 / next_state[:,self.repeat_suffstat:2*self.repeat_suffstat] - 1 

            new_cnt = prev_cnt + num_Ys
            Y_sum = Y_sample.sum(axis=1).repeat(new_cnt.shape[1],1).T
            new_means = ( prev_means * prev_cnt + Y_sum) / new_cnt

            # update means, and then 1/(n+1)'s
            next_state[:,:self.repeat_suffstat] = new_means
            next_state[:,self.repeat_suffstat:2*self.repeat_suffstat] = 1 / ( new_cnt + 1 ) 
        else:
            raise ValueError('Y_sample must have 1 or 2 dimensions')
        return next_state

    
    def autoregressive_sampling(self, curr_state, encoded_Z, num_imagined):
        device = self.get_device()

        generated_Ys = []
        encoded_Z = encoded_Z.to(device)

        Y_prev_tmp = curr_state[:,:0].to(device)
        curr_state_tmp = torch.clone(curr_state).to(device)
        for k in range(num_imagined):
            if encoded_Z is not None:
                state = torch.cat([encoded_Z, curr_state_tmp], 1) 
            else:
                state = curr_state_tmp
            p_pred = self.top_layer(state)
            Y_sample = torch.bernoulli(p_pred).to(device)
            Y_prev_tmp = torch.cat([Y_prev_tmp, Y_sample], 1)
            generated_Ys.append(Y_sample)

            # update state
            curr_state_tmp = self.update_state(curr_state_tmp, Y_sample)
            
        all_generated_Ys = torch.cat(generated_Ys, axis=1)
        return all_generated_Ys

    def dpt_sampling(self, curr_state, encoded_Z):
        device = self.get_device()
        encoded_Z = encoded_Z.to(device)

        Y_prev_tmp = curr_state[:,:0].to(device)
        curr_state_tmp = torch.clone(curr_state).to(device)
        if encoded_Z is not None:
            state = torch.cat([encoded_Z, curr_state_tmp], 1) 
        else:
            state = curr_state_tmp
        p_pred = self.top_layer(state)
        return p_pred 


    def get_posterior_draws(self, Z_input, curr_state, num_imagined, num_repetitions, 
                            repeat_batch_size=None):
        device = self.get_device()
        if repeat_batch_size is None:
            repeat_batch_size = num_repetitions
            
        with torch.no_grad():

            # Get encoded Z
            if self.use_category:
                encoded_Z = self.z_encoder(Z_input).to(device)
            elif self.use_bert:
                encoded_Z = Z_input.to(device)
            elif Z_input is not None:
                encoded_Z = Z_input.to(device)
            else:
                encoded_Z = None

            phats_list = []  

            for i in range(int(np.ceil(num_repetitions/repeat_batch_size))):
                this_batch_size = repeat_batch_size
                if repeat_batch_size * (i+1) > num_repetitions:
                    this_batch_size = num_repetitions - repeat_batch_size * i

                # repeat for number of batch repetitions
                curr_state_repeated = curr_state.repeat((this_batch_size, 1)) 
                if encoded_Z is None:
                    encoded_Z_repeated = None
                else:
                    encoded_Z_repeated = encoded_Z.repeat((this_batch_size, 1))  

                # Autoregressive forward resampling
                repeat_generated_Ys = self.autoregressive_sampling(curr_state_repeated, encoded_Z_repeated, num_imagined)
                num_rows = curr_state.shape[0]
                splits = torch.split(repeat_generated_Ys, num_rows)
                repeat_batch_phats = torch.cat([x.mean(axis=1).unsqueeze(-1) for x in splits], 1)

                assert repeat_batch_phats.shape[0] == num_rows and repeat_batch_phats.shape[1] == this_batch_size
                phats_list.append(repeat_batch_phats)

            res = torch.cat(phats_list,1)
            assert res.shape[0] == num_rows and res.shape[1] == num_repetitions
            return res

            
    def get_posterior_draws_horizonDependent(self, Z_input, curr_state, T, num_repetitions, past_obs,
                            repeat_batch_size=None, return_Ys=False):
        device = self.get_device()
        if repeat_batch_size is None:
            repeat_batch_size = num_repetitions
            
        with torch.no_grad():

            # Get encoded Z
            if self.use_category:
                encoded_Z = self.z_encoder(Z_input).to(device)
            elif self.use_bert:
                encoded_Z = Z_input.to(device)
            elif Z_input is not None:
                encoded_Z = Z_input.to(device)
            else:
                encoded_Z = None

            phats_list = []
            Ys_list = []

            for i in range(int(np.ceil(num_repetitions/repeat_batch_size))):
                this_batch_size = repeat_batch_size
                if repeat_batch_size * (i+1) > num_repetitions:
                    this_batch_size = num_repetitions - repeat_batch_size * i

                # repeat for number of batch repetitions
                curr_state_repeated = curr_state.repeat((this_batch_size, 1)) 
                if encoded_Z is None:
                    encoded_Z_repeated = None
                else:
                    encoded_Z_repeated = encoded_Z.repeat((this_batch_size, 1))  

                # See what maximum generation length to use
                #maxseen = max([ len(x) for x in past_obs ]); maxgen = T - maxseen

                # Autoregressive forward resampling
                repeat_generated_Ys = self.autoregressive_sampling(curr_state_repeated, encoded_Z_repeated, T)
                num_rows = curr_state.shape[0]
                splits = torch.split(repeat_generated_Ys, num_rows)

                # Form posterior draws
                repeat_batch_phats = []
                for new_obs in splits:
                    gen_obs = [ new[:T-len(old)] for old, new in zip(past_obs, new_obs) ]
                    all_obs = [ torch.cat([old, new]) for old, new in zip(past_obs, gen_obs) ]
                    all_obs_matrix = torch.vstack(all_obs)
                    phats = all_obs_matrix.mean(axis=1)
                    repeat_batch_phats.append(phats)

                repeat_batch_phats = torch.vstack(repeat_batch_phats).T

                assert repeat_batch_phats.shape[0] == num_rows and repeat_batch_phats.shape[1] == this_batch_size
                phats_list.append(repeat_batch_phats)

                if return_Ys:
                    Ys_list.append( repeat_generated_Ys )

            res = torch.cat(phats_list,1)
            assert res.shape[0] == num_rows and res.shape[1] == num_repetitions
            if return_Ys:
                import ipdb; ipdb.set_trace()
                return Ys_list, res
            return res
        

