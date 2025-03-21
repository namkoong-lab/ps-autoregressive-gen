import torch
from torch import nn
from scipy import special


# This function is not used in any notebooks to my knowledge
#def make_gg_model_data_split(marginal_preds):
#    prior_dict = get_gg_params_data_split(marginal_preds)
#    gp_model_eval_marginal = GaussianGaussian(prior_dict)
#    return gp_model_eval_marginal
    

class GaussianGaussian(nn.Module):
   
    def __init__(self, prior_dict = None):
        super(GaussianGaussian, self).__init__()
        self.prior_dict = prior_dict
        self.noise_var = prior_dict['noise_var']
        self.prior_var = prior_dict['prior_var']
        self.prior_mean = prior_dict['prior_mean']

    
    def compute_posterior(self, obs_seq):
        cnt = obs_seq.shape[1]
        if cnt == 0:
            return self.prior_mean, self.prior_var
            
        sum_obs = torch.sum(obs_seq, axis=1)
    
        posterior_var = 1/( cnt/self.noise_var + 1/self.prior_var )
        posterior_mean = posterior_var * ( self.prior_mean / self.prior_var + sum_obs / self.noise_var )

        return posterior_mean, posterior_var

    
    def compute_posterior_predictive(self, obs_seq):
        posterior_mean, posterior_var = self.compute_posterior(obs_seq)
        return posterior_mean, posterior_var + self.noise_var

    
    def compute_posterior_seq(self, obs_seq):
        cumsum_obs_seq = torch.cumsum(obs_seq, axis=1)
        cnt_seq = torch.arange(obs_seq.shape[1])+1
        
        posterior_var_seq = 1/( cnt_seq/self.noise_var + 1/self.prior_var )
        posterior_mean_seq = posterior_var_seq * ( self.prior_mean.reshape(-1,1) / self.prior_var + cumsum_obs_seq / self.noise_var )

        posterior_mean_seq = torch.cat( [self.prior_mean.reshape(-1,1), posterior_mean_seq[:,:-1]], axis=1 )
        posterior_var_seq = torch.cat( [ torch.tensor([self.prior_var]), posterior_var_seq[:-1] ] )
        
        return posterior_mean_seq, posterior_var_seq

    
    def posterior_samples(self, prev_obs, num_repetitions):
        if len(prev_obs) == 0:
            post_samples = torch.normal(self.prior_mean.repeat(num_repetitions,1).T, torch.sqrt(self.prior_var))
        else:
            post_mean_seq, post_var_seq = self.compute_posterior_seq(prev_obs)
            post_mean = post_mean_seq[:,-1]
            post_var = post_var_seq[-1]
            post_samples = torch.normal(post_mean.repeat(num_repetitions,1).T, torch.sqrt(post_var))
            
        return post_samples

    
    def posterior_samples_autoregressive(self, prev_obs, num_repetitions, num_imagined):
        all_post_samples = []
        for rep in range(num_repetitions):
            curr_prev_obs = prev_obs
            
            for k in range(num_imagined):
                post_pred_mean, post_pred_var = self.compute_posterior_predictive(curr_prev_obs)
                post_pred_samples = torch.normal(post_pred_mean, torch.sqrt(post_pred_var))
                curr_prev_obs = torch.cat([curr_prev_obs, post_pred_samples.reshape(-1,1)], axis=1)
        
            post_sample = torch.mean( curr_prev_obs[:,-num_imagined:], axis=1 )
            all_post_samples.append( post_sample )
        
        all_post_samples = torch.vstack(all_post_samples).T
        return all_post_samples



class BetaBernoulli(nn.Module):
   
    def __init__(self, prior_dict = None):
        super(BetaBernoulli, self).__init__()
        self.prior_dict = prior_dict
        self.alpha = prior_dict['alpha'].squeeze()
        self.beta = prior_dict['beta'].squeeze()

    
    def compute_posterior(self, obs_seq):
        
        cnt = obs_seq.shape[1]
        if cnt == 0:
            return self.alpha, self.beta

        post_alpha = self.alpha + torch.sum(obs_seq, axis=1)
        post_beta = self.beta + torch.sum(1-obs_seq, axis=1)

        return post_alpha.squeeze(), post_beta.squeeze()

    
    def compute_posterior_seq(self, obs_seq):

        post_alpha = self.alpha.unsqueeze(1) + torch.cumsum(obs_seq, axis=1)
        post_beta = self.beta.unsqueeze(1) + torch.cumsum(1-obs_seq, axis=1)

        post_alpha = torch.cat( [self.alpha.reshape(-1,1), post_alpha[:,:-1]], axis=1 )
        post_beta = torch.cat( [self.beta.reshape(-1,1), post_beta[:,:-1]], axis=1 )
        
        return post_alpha, post_beta

    
    def compute_post_mean_var(self, post_alpha, post_beta):
        post_mean = post_alpha / (post_alpha + post_beta)
        post_var = post_alpha*post_beta / ( (1 + post_alpha + post_beta) * torch.square(post_alpha + post_beta) )
        return post_mean, post_var

    
    def posterior_samples(self, prev_obs, num_repetitions):
        
        if len(prev_obs) == 0:
            beta = torch.distributions.beta.Beta(self.alpha, self.beta)
            samples = beta.sample((1, num_repetitions)).squeeze().T
        else:
            post_alpha, post_beta = self.compute_posterior(prev_obs)
            beta = torch.distributions.beta.Beta(post_alpha, post_beta)
            post_samples = beta.sample((1, num_repetitions)).squeeze().T
            
        return post_samples


class BetaBernoulliMixture(nn.Module):
   
    def __init__(self, prior_dict = None):
        # Assumes prior is a mixture of two betas
        super(BetaBernoulliMixture, self).__init__()
        
        self.prior_dict = prior_dict
        self.mixweight = prior_dict['mixweight'] # how much to weight the first beta dist
        self.alpha1 = prior_dict['alpha1'].squeeze()
        self.beta1 = prior_dict['beta1'].squeeze()
        self.alpha2 = prior_dict['alpha2'].squeeze()
        self.beta2 = prior_dict['beta2'].squeeze()

    
    def compute_posterior(self, obs_seq):
        # nice reference on posterior distribution for mixture priors
            # https://www.mas.ncl.ac.uk/~nmf16/teaching/mas3301/week11.pdf
        
        cnt = obs_seq.shape[1]
        if cnt == 0:
            post_dict = {
                "alpha1": self.alpha1,
                "beta1": self.beta1,
                "alpha2": self.alpha2,
                "beta2": self.beta2,
                "mixweight": self.mixweight,
                }
            return post_dict

        post_alpha1 = self.alpha1 + torch.sum(obs_seq, axis=1)
        post_beta1 = self.beta1 + torch.sum(1-obs_seq, axis=1)
        
        post_alpha2 = self.alpha2 + torch.sum(obs_seq, axis=1)
        post_beta2 = self.beta2 + torch.sum(1-obs_seq, axis=1)
        
        # Formula for marginal likelihood is here on page 24 of
            # https://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-1/01-intro-to-Bayes.pdf
        clog1 = special.betaln(post_alpha1, post_beta1) - special.betaln(self.alpha1, self.beta1)
        clog2 = special.betaln(post_alpha2, post_beta2) - special.betaln(self.alpha2, self.beta2)

        bigconst = -clog1 # to make numerically stable
            # eg. see discussion in https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        c1 = self.mixweight * torch.exp( clog1 + bigconst )
        c2 = (1-self.mixweight) * torch.exp( clog2 + bigconst )

        post_mixweight = c1 / (c1+c2)
        assert torch.isnan(post_mixweight).sum() == 0

        post_dict = {
                "alpha1": post_alpha1,
                "beta1": post_beta1,
                "alpha2": post_alpha2,
                "beta2": post_beta2,
                "mixweight": post_mixweight,
                }

        return post_dict

    
    def compute_posterior_seq(self, obs_seq):

        post_alpha1_raw = self.alpha1.unsqueeze(1) + torch.cumsum(obs_seq, axis=1)
        post_beta1_raw = self.beta1.unsqueeze(1) + torch.cumsum(1-obs_seq, axis=1)

        post_alpha2_raw = self.alpha2.unsqueeze(1) + torch.cumsum(obs_seq, axis=1)
        post_beta2_raw = self.beta2.unsqueeze(1) + torch.cumsum(1-obs_seq, axis=1)

        post_alpha1 = torch.cat( [self.alpha1.reshape(-1,1), post_alpha1_raw[:,:-1]], axis=1 )
        post_beta1 = torch.cat( [self.beta1.reshape(-1,1), post_beta1_raw[:,:-1]], axis=1 )

        post_alpha2 = torch.cat( [self.alpha2.reshape(-1,1), post_alpha2_raw[:,:-1]], axis=1 )
        post_beta2 = torch.cat( [self.beta2.reshape(-1,1), post_beta2_raw[:,:-1]], axis=1 )

        # Formula for marginal likelihood is here on page 24 of
            # https://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-1/01-intro-to-Bayes.pdf
        clog1 = special.betaln(post_alpha1, post_beta1) - special.betaln(self.alpha1, self.beta1).unsqueeze(1)
        clog2 = special.betaln(post_alpha2, post_beta2) - special.betaln(self.alpha2, self.beta2).unsqueeze(1)

        bigconst = -clog1 # to make numerically stable
            # eg. see discussion in https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        c1 = self.mixweight * torch.exp( clog1 + bigconst )
        c2 = (1-self.mixweight) * torch.exp( clog2 + bigconst )
        
        post_mixweight = c1 / (c1+c2)
        assert torch.isnan(post_mixweight).sum() == 0
        
        post_dict = {
                "alpha1": post_alpha1,
                "beta1": post_beta1,
                "alpha2": post_alpha2,
                "beta2": post_beta2,
                "mixweight": post_mixweight,
                }

        return post_dict
        
    
    def compute_post_mean(self, post_dict):
        post_means1 = post_dict['alpha1'] / (post_dict['alpha1'] + post_dict['beta1'])
        post_means2 = post_dict['alpha2'] / (post_dict['alpha2'] + post_dict['beta2'])
        post_mean = post_dict['mixweight'] * post_means1 + (1-post_dict['mixweight']) * post_means2
        return post_mean

    
    def posterior_samples(self, prev_obs, num_repetitions):
        
        if len(prev_obs) == 0:
            beta1 = torch.distributions.beta.Beta(self.alpha1, self.beta1)
            beta2 = torch.distributions.beta.Beta(self.alpha2, self.beta2)
            samples1 = beta1.sample((1, num_repetitions)).squeeze().T
            samples2 = beta2.sample((1, num_repetitions)).squeeze().T
            ind_matrix = torch.bernoulli(self.mixweight*torch.ones(samples1.shape))
            post_samples = samples1 * ind_matrix + samples2 * (1-ind_matrix)
            
        else:
            post_dict = self.compute_posterior(prev_obs)
            
            beta1 = torch.distributions.beta.Beta(post_dict['alpha1'], post_dict['beta1'])
            beta2 = torch.distributions.beta.Beta(post_dict['alpha2'], post_dict['beta2'])
            samples1 = beta1.sample((1, num_repetitions)).squeeze().T
            samples2 = beta2.sample((1, num_repetitions)).squeeze().T
            if type(post_dict['mixweight']) == type(0.5):
                ind_matrix = torch.bernoulli(post_dict['mixweight']*torch.ones(samples1.shape))
            else:
                ind_matrix = torch.bernoulli(post_dict['mixweight'].unsqueeze(1)*torch.ones(samples1.shape))
            post_samples = samples1 * ind_matrix + samples2 * (1-ind_matrix)
            
        return post_samples
