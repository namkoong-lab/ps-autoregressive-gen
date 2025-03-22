import copy
import sys
import torch
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from transformers import DistilBertModel
from torch import nn
from dataset_MIND import get_token_dict
from models import MarginalPredictor, SequentialPredictor, SequentialBetaBernoulli, SequentialBetaBernoulliAlphaBeta

##############################################################################
# Functions to define optimizers 
##############################################################################

def get_optimizer(model, config):
    param_list = list(model.named_parameters())

    optimizer_parameters = [ 
        { 
            "params": [ p for n, p in param_list if 'prior_weights' not in n ],
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate,
        },
    ]
    
    opt = torch.optim.AdamW(optimizer_parameters, lr=config.learning_rate,
                            weight_decay=config.weight_decay, betas=(0.9, 0.95))
    return opt 


def get_optimizer_bert(model, config):
    if config.bert_weight_decay is None:
        config.bert_weight_decay = config.weight_decay
    if config.bert_learning_rate is None:
        config.bert_learning_rate = config.learning_rate
    if not hasattr(config, 'aplusb_learning_rate') or config.aplusb_learning_rate is None:
        config.aplusb_learning_rate = config.learning_rate

    param_list = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "aplusb"]

    optimizer_parameters = [ 
        { # Top layer with weight decay
            "params": [ p for n, p in param_list if not any(nd in n for nd in no_decay) and 'top_layer' in n and 'prior_weights' not in n and 'aplusb' not in n ],
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate,
        },
        { # Top layer with weight decay
            "params": [ p for n, p in param_list if not any(nd in n for nd in no_decay) and 'top_layer' in n and 'prior_weights' not in n and 'aplusb' in n ],
            "weight_decay": config.weight_decay,
            "lr": config.aplusb_learning_rate,
        },

        { # Not top layer with weight decay
            "params": [ p for n, p in param_list if not any(nd in n for nd in no_decay) and 'top_layer' not in n and 'prior_weights' not in n ],
            "weight_decay": config.bert_weight_decay,
            "lr": config.bert_learning_rate,
        },
        { # Not top layer without weight decay
            "params": [ p for n, p in param_list if any(nd in n for nd in no_decay) and 'top_layer' not in n and 'prior_weights' not in n ],
            "weight_decay": 0.0,
            "lr": config.bert_learning_rate,
        },
        { # Top layer without weight decay
            "params": [ p for n, p in param_list if any(nd in n for nd in no_decay) and 'top_layer' in n and 'prior_weights' not in n and 'aplusb' not in n ],
            "weight_decay": 0.0,
            "lr": config.learning_rate,
        },
        { # Top layer without weight decay
            "params": [ p for n, p in param_list if any(nd in n for nd in no_decay) and 'top_layer' in n and 'prior_weights' not in n and 'aplusb' in n ],
            "weight_decay": 0.0,
            "lr": config.aplusb_learning_rate,
        },


    ]
    opt = torch.optim.AdamW(optimizer_parameters, betas=(0.9, 0.95))
    
    return opt 



##############################################################################
# Functions to load models
##############################################################################

def compareModelWeights(model_a, model_b):
    module_a = model_a._modules
    module_b = model_b._modules
    if len(list(module_a.keys())) != len(list(module_b.keys())):
        return False
    a_modules_names = list(module_a.keys())
    b_modules_names = list(module_b.keys())
    for i in range(len(a_modules_names)):
        layer_name_a = a_modules_names[i]
        layer_name_b = b_modules_names[i]
        if layer_name_a != layer_name_b:
            return False
        layer_a = module_a[layer_name_a]
        layer_b = module_b[layer_name_b]
        if (
            (type(layer_a) == nn.Module) or (type(layer_b) == nn.Module) or
            (type(layer_a) == nn.Sequential) or (type(layer_b) == nn.Sequential)
            ):
            if not compareModelWeights(layer_a, layer_b):
                return False
        if hasattr(layer_a, 'weight') and hasattr(layer_b, 'weight'):
            if not torch.equal(layer_a.weight.data, layer_b.weight.data):
                return False
    return True


def get_model_and_optimizer_MIND(config, category2id_dict=None):
    is_sequential = config.marginal_vs_sequential == 'sequential'
    logging.info(f"IS SEQUENTIAL {is_sequential}")
    if not is_sequential and config.marginal_vs_sequential != 'marginal':
        raise ValueError('config.marginal_vs_sequential must be either marginal or sequential')

    if not hasattr(config, 'rand_prior'):
        config.rand_prior = 0
    if not hasattr(config, 'prior_scale'):
        config.prior_scale = 0
    # dataset consists of dumped embeddings; we only train the head

    if not hasattr(config, 'embed_data_dir'):
        setattr(config, 'embed_data_dir', False)

    if hasattr(config, "sequential_beta_bernoulli") and config.sequential_beta_bernoulli:
        assert is_sequential

    if config.use_text:
        bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(config.device)
        # Model uses text and BERT features --------------------------
        if is_sequential:
            if hasattr(config, "sequential_beta_bernoulli") and config.sequential_beta_bernoulli:
                if hasattr(config, "sequential_beta_bernoulli_alpha_beta") and config.sequential_beta_bernoulli_alpha_beta:
                    model = SequentialBetaBernoulliAlphaBeta(bert, 
                            MLP_width=config.MLP_width, 
                            MLP_layer=config.MLP_layer).to(config.device)
                else:
                    model = SequentialBetaBernoulli(bert, 
                            MLP_width=config.MLP_width, 
                            MLP_layer=config.MLP_layer).to(config.device)
            else:
                if hasattr(config, "sequential_init_mean"):
                    model = SequentialPredictor(bert, 
                            MLP_width=config.MLP_width, 
                            MLP_layer=config.MLP_layer,
                            init_mean=config.sequential_init_mean, 
                            repeat_suffstat=config.repeat_suffstat).to(config.device)
                else:
                    model = SequentialPredictor(bert, 
                            MLP_width=config.MLP_width, 
                            MLP_layer=config.MLP_layer,
                            repeat_suffstat=config.repeat_suffstat).to(config.device)

        else:
            print('train fns prior scale')
            model = MarginalPredictor(bert, MLP_width=config.MLP_width, MLP_layer=config.MLP_layer,
                                        rand_prior=config.rand_prior, prior_scale=config.prior_scale).to(config.device)

        if config.embed_data_dir:
            model.z_encoder = nn.Identity()

        if config.freeze_bert:
            for param in model.z_encoder.parameters():
                param.requires_grad = False
            print('Freezing bert parameters')

        if hasattr(config, 'end2end') and config.end2end:
            optimizer_encoder = get_optimizer_bert(model.z_encoder, config)
            optimizer_MLP = get_optimizer_bert(model.top_layer, config)
            optimizer_dict = { 
                'encoder': optimizer_encoder,
                'MLP': optimizer_MLP,
            }
        else:
            optimizer_dict = { 'all': get_optimizer_bert(model, config) }
    
    else:
        # Model uses doesn't use text features --------------------------
        if config.use_category:

            category_args = { "num_embeddings": len(category2id_dict), 
                              "embedding_dim": 100 }
        else:
            category_args = None

        if is_sequential:
            # Not doing interpolating model right now
            model = SequentialPredictor(category_args=category_args,
                                        MLP_width=config.MLP_width, MLP_layer=config.MLP_layer,
                                        init_mean=config.sequential_init_mean,
                                        repeat_suffstat=config.repeat_suffstat
                                        ).to(config.device)
        else:
            # Marginal prediction model with no Z features is trivial
            assert category_args is not None
            model = MarginalPredictor(category_args=category_args,
                                        MLP_width=config.MLP_width, MLP_layer=config.MLP_layer,
                                        rand_prior=config.rand_prior, prior_scale=config.prior_scale).to(config.device)
        if config.embed_data_dir:
            model.z_encoder = nn.Identity()

        if hasattr(config, 'end2end') and config.end2end:
            optimizer_encoder = get_optimizer(model.z_encoder, config)
            optimizer_MLP = get_optimizer(model.top_layer, config)
            optimizer_dict = { 
                'encoder': optimizer_encoder,
                'MLP': optimizer_MLP,
            }
        else:
            optimizer_dict = { 'all': get_optimizer(model, config) }
    return model, optimizer_dict


def get_model_and_optimizer_synthetic(config):
    is_sequential = config.marginal_vs_sequential == 'sequential'
    logging.info(f"IS SEQUENTIAL {is_sequential}")
    if not is_sequential and config.marginal_vs_sequential != 'marginal':
        raise ValueError('config.marginal_vs_sequential must be either marginal or sequential')

    if not hasattr(config, 'rand_prior'):
        config.rand_prior = 0
    if not hasattr(config, 'prior_scale'):
        config.prior_scale = 0
    if not hasattr(config, 'MLP_last_fn'):
        config.MLP_last_fn = 'sigmoid'
    # Model uses doesn't use text features --------------------------
    if is_sequential:
        if hasattr(config, "sequential_beta_bernoulli") and config.sequential_beta_bernoulli:
            if hasattr(config, "sequential_beta_bernoulli_alpha_beta") and config.sequential_beta_bernoulli_alpha_beta:
                model = SequentialBetaBernoulliAlphaBeta(Z_dim=config.Z_dim, 
                        MLP_width=config.MLP_width, 
                        MLP_layer=config.MLP_layer).to(config.device)
            else:
                model = SequentialBetaBernoulli(Z_dim=config.Z_dim, 
                        MLP_width=config.MLP_width, 
                        MLP_layer=config.MLP_layer).to(config.device)
        else:
            model = SequentialPredictor(Z_dim=config.Z_dim, 
                        init_mean=config.sequential_init_mean,
                        MLP_width=config.MLP_width, 
                        MLP_layer=config.MLP_layer,
                        repeat_suffstat=config.repeat_suffstat,
                        MLP_last_fn=config.MLP_last_fn,
                        ).to(config.device)
    else:
        # Marginal prediction model with no Z features is trivial
        assert config.Z_dim > 0
        model = MarginalPredictor(Z_dim=config.Z_dim,
                                    MLP_width=config.MLP_width, MLP_layer=config.MLP_layer,
                                    rand_prior=config.rand_prior, prior_scale=config.prior_scale).to(config.device)

    optimizer_dict = { 'all': get_optimizer(model, config) }
    return model, optimizer_dict



##############################################################################
# Functions to compute losses
##############################################################################

def loss_from_loss_matrix(loss_matrix, orig_click_mask, how='sum_avg_per_row', weight_factor=1):
    click_mask = orig_click_mask * weight_factor**torch.arange(loss_matrix.shape[1]).to(loss_matrix.device)
    masked_losses = loss_matrix * click_mask # click mask is always 1 in our current setup

    if how == 'avg_per_row':
        loss = masked_losses.sum(1) / click_mask.sum(1)
        return loss.mean()
    
    elif how == 'avg_per_obs':
        loss = masked_losses.sum() / click_mask.sum()
        return loss.mean()

    elif how == 'sum_avg_per_row':
        loss = masked_losses.sum(1) / click_mask.sum(1)
        return loss.sum()
    
    elif how == 'sum_per_obs':
        loss = masked_losses.sum()
        return loss.sum()

    else:
        raise ValueError('Argument "how" not accepted')



def get_val_loss(model, val_loader, device, category2id=None, loss_agg='sum_avg_per_row', 
                 sequential_one_length=None, weight_factor=1, trainlen=None, exact=False,
                 embed_data=False, verbose=False):
    
    total_loss_per_t = None
    total_loss = 0
    total_rows = 0
    model.eval()
    theta_hats = []
    click_obs_means = []
    click_rates = []
    click_obs_counts = []
    click_obs = []
    click_obs_masks = []
    cat_info = []
    all_model_input = []
    encoder = hasattr(model, 'z_encoder_output_dim') and model.z_encoder_output_dim is not None
    use_category = hasattr(model, 'use_category') and model.use_category
    i=0
    with torch.no_grad():
        for batch in val_loader:
            i+=1
            if verbose:
                print(f'{i} out of {len(val_loader)}')
                sys.stdout.flush()
            for k,v in batch.items():
                batch[k] = v.to(device)

            click_mask = batch['click_length_mask']
            if hasattr(model, 'use_bert') and model.use_bert and not embed_data:
                model_input = get_token_dict(batch)
            elif use_category:
                model_input = batch['category_ids']
                all_model_input.append(model_input)
            elif hasattr(model, 'z_encoder_output_dim') and (model.z_encoder_output_dim is not None or embed_data):
                model_input = batch['Z']
                all_model_input.append(model_input)
            else:
                model_input = None
            
            loss_matrix, row_theta_hats = model.eval_seq(model_input, 
                                                         batch['click_obs'],
                                                         N=None, return_preds=True, trainlen=trainlen, exact=exact)
                # no trainlen or exact if using context  
            if sequential_one_length is not None:
                loss_matrix = loss_matrix[:,[sequential_one_length]]
                click_mask_loss = copy.deepcopy(click_mask[:,[sequential_one_length]])
                loss = loss_from_loss_matrix(loss_matrix, click_mask_loss, how=loss_agg, weight_factor=weight_factor).detach().cpu()
            else:
                loss = loss_from_loss_matrix(loss_matrix, click_mask, how=loss_agg, weight_factor=weight_factor).detach().cpu()
            
            theta_hats.append(row_theta_hats.detach().cpu())
            click_obs_means.append((batch['click_obs']*click_mask).sum(dim=1).cpu())
            click_obs.append(batch['click_obs'].cpu())
            click_obs_masks.append(batch['click_length_mask'].cpu())
            click_rates.append(batch['click_rates'].cpu())
            click_obs_counts.append(click_mask.sum(dim=1).cpu())
            
            if use_category:
                cat_info.append(batch['category_ids'])
            total_loss += loss.detach().cpu().item()
            total_rows += len(batch['click_obs'])
            
            if total_loss_per_t is None:
                total_loss_per_t = loss_matrix.detach().sum(dim=0).cpu()
            else:
                total_loss_per_t += loss_matrix.detach().sum(dim=0).cpu()
                
    
    return_dict =  {
            'loss':             total_loss / total_rows, 
            'loss_per_t':       total_loss_per_t / total_rows,
            'theta_hats':       torch.concatenate(theta_hats).cpu(), 
            'click_obs_means':  torch.concatenate(click_obs_means).cpu(), 
            'click_rates':      torch.concatenate(click_rates).cpu(),
            'click_obs_counts': torch.concatenate(click_obs_counts).cpu(),
            'click_obs':        torch.concatenate(click_obs).cpu(),
            'click_obs_masks':  torch.concatenate(click_obs_masks).cpu(),
    }
    if not (hasattr(model, 'use_bert') and model.use_bert):
        if encoder or use_category:
            return_dict['Z'] = torch.concatenate(all_model_input).cpu()
        
    if hasattr(model, 'use_category') and model.use_category:
        return_dict['category_ids'] = torch.concatenate(cat_info).cpu()
        return_dict['category2id'] = category2id
        
    return return_dict

