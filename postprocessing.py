import torch
import copy
import os
import argparse
from argparse import Namespace
from util import parse_bool, parse_int_list, set_seed

from train_fns import get_val_loss, get_model_and_optimizer_MIND, get_model_and_optimizer_synthetic
from dataset_MIND import get_token_dict, get_loaders_MIND
from dataset_synthetic import get_loaders_synthetic

def get_row_embeds(config, model, loader, device):
    with torch.no_grad():
        row_embeds = []
        for batch in loader:
            for k,v in batch.items():
                batch[k] = v.to(device)

            if config.dataset_type == 'MIND' and not config.embed_data_dir:
                if model.use_bert:
                    row_embeds.append(model.z_encoder(get_token_dict(batch)).detach().cpu())
                elif model.use_category:
                    row_embeds.append(batch['category_ids'].detach().cpu())
            elif config.dataset_type == 'synthetic' or config.embed_data_dir:
                row_embeds.append(batch['Z'])
            
    return torch.concatenate(row_embeds)


def get_row_embeds_fname(out_dir, prefix='best_loss_'):
    return out_dir + f'/{prefix}row_embeds.pt'

def get_predictions_fname(out_dir, prefix='best_loss_'):
    return out_dir + f'/{prefix}predictions.pt'

def get_posterior_samples_fname(out_dir, prefix='best_loss_'):
    return out_dir + f'/{prefix}posterior_samples.pt'


def save_row_embeds(config, model, loader_dict, out_dir, device, loader_names, prefix='best_loss_', recalc=False):
    save_fname = get_row_embeds_fname(out_dir, prefix)
    if not recalc and os.path.exists(save_fname):
        res = torch.load(save_fname, map_location='cpu')
    else:
        res = {}
    with torch.no_grad():
        for loader_name in loader_names:
            if loader_name in res.keys(): continue
            print(f'Saving row embeds for {loader_name}')
            res[loader_name] = get_row_embeds(config, model, loader_dict[loader_name+'_loader'], device)
    torch.save(res, save_fname)
    return res


def save_model_predictions(model, loader_dict, out_dir, device, loader_names, prefix='best_loss_', 
                           category2id=None, recalc=False, embed_data=False, 
                           use_X=False, use_X_model=False):
    save_fname = get_predictions_fname(out_dir, prefix)
    if not recalc and os.path.exists(save_fname):
        res = torch.load(save_fname, map_location='cpu')
    else:
        res = {}
    for loader_name in loader_names:
        if loader_name in res.keys(): continue
        print(f'Saving model loss for {loader_name}')
        loader = loader_dict[loader_name+'_loader']
        res[loader_name] = get_val_loss(model, loader, device, category2id,embed_data=embed_data,
                                       use_X=use_X, use_X_model=use_X_model)
    torch.save(res, save_fname)
    return res


def save_posterior_samples(config, model, out_dir, device, loader_names, 
        predictions=None,
        row_embeds=None, 
        all_num_prev_obs=[0,1,5,10,25], 
        num_imagined=500,
        prefix='best_loss_', 
        recalc=False, 
        num_repetitions=100,
        max_post_rows=None):
    assert predictions is not None
    assert row_embeds is not None
    
    save_fname = get_posterior_samples_fname(out_dir, prefix)
    if not recalc and os.path.exists(save_fname):
        res = torch.load(save_fname, map_location='cpu')
    else:
        res = {}

    for loader_name in loader_names:
        posterior_dict = {}
        for num_prev_obs in all_num_prev_obs:
            param_keys = (num_prev_obs, num_imagined, num_repetitions)
            if loader_name in res.keys() and param_keys in res[loader_name].keys(): continue
            print("num_prev_obs", num_prev_obs)

            if config.dataset_type == 'MIND':
                if model.use_bert:
                    Z_input = row_embeds[loader_name].to(device)
                elif model.use_category:
                    Z_input = row_embeds[loader_name].to(device)
                    #raise ValueError('Not implemented') 
                    # though maybe we can just use the same thing as for use_bert
                    # since we dump row embeddings
                else:
                    Z_input = None
            elif config.dataset_type == 'synthetic':
                Z_input = row_embeds[loader_name].to(device)
            else:
                raise ValueError('Invalid dataset type')

            # Get current state
            data_dict = predictions[loader_name]
            R_obs = data_dict['click_obs'].to(device)
            
            if Z_input is not None and max_post_rows is not None:
                Z_input = Z_input[:max_post_rows]
                R_obs = R_obs[:max_post_rows]
            
            prev_obs_og = R_obs[:, :num_prev_obs]
            curr_state = model.next_model_states(prev_obs_og)

            val_post_samples = model.get_posterior_draws(Z_input, curr_state,
                                                        num_imagined, num_repetitions).cpu()

            posterior_dict[param_keys] = {
                'true_p': data_dict['click_rates'],
                'post_samples' : val_post_samples,
            }
        res[loader_name] = posterior_dict
    torch.save(res, save_fname)
    return res
    

def get_device(gpu):
    if gpu is not None and int(gpu) >= 0:
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_old_model(config, sd, check=None):
    
    if config.use_category:
        assert check is not None
        category2id_dict = check['val_loss_dict']['category2id']
    else:
        category2id_dict = None

    if config.dataset_type == 'MIND':
        model, optimizer = get_model_and_optimizer_MIND(config, category2id_dict=category2id_dict)
    elif config.dataset_type == 'synthetic':
        model, optimizer = get_model_and_optimizer_synthetic(config)
    else:
        raise ValueError("invalid dataset type")
    model.to('cpu')
    model.load_state_dict(sd)

    return model


def do_postprocessing(args):
    print(f'ARGS: {args}')

    device = args.device
    check = torch.load(args.run_dir + '/best_loss.pt', map_location='cpu')
    
    config = copy.deepcopy(check['config'])
    if not hasattr(config, 'embed_data_dir'):
        setattr(config, 'embed_data_dir', False)
    setattr(config, 'batch_size', args.batch_size)
    setattr(config, 'device', args.device)
    model = load_old_model(config, check['state_dict'], check)

    set_seed(config.seed)

    category2id = None
    if config.dataset_type == 'MIND' and not config.embed_data_dir:
        loaders = get_loaders_MIND(config, train_deterministic_row_order=True, extras=True)
        if config.use_category:
            category2id = loaders['category2id']
    elif config.dataset_type == 'synthetic' or config.embed_data_dir:
        loaders = get_loaders_synthetic(config, train_deterministic_row_order=True, extras=True)
    else:
        raise ValueError("Invalid dataset_type")

    model.to(device)
    model.eval()

    recalc = args.postproc_force_recalc

    #if model.use_category:
    #    raise ValueError('Category models not supported presently')
        # examples: feeding in Z for posterior sampling

    all_loader_names = [x.split('_loader')[0] for x in loaders.keys() if x.endswith('_loader')]

    if not hasattr(config, 'embed_data_dir'):
        config.embed_data_dir=False

    # this will probably do some unnecessary computation, e.g. on val set,
    # where those outputs are already in e.g. best_loss.pt (but I think we don't care)
    predictions = save_model_predictions(model, loaders, 
            args.run_dir, device, all_loader_names, recalc=recalc, category2id=category2id,
            embed_data = config.embed_data_dir,
            use_X=config.use_X, use_X_model=config.use_X_model) 
    print(predictions['train']['theta_hats'].shape)

    # we have only tested this with BERT embeddings but not category
    row_embeds = save_row_embeds(config, model, loaders, 
                args.run_dir, device, all_loader_names, recalc=recalc)
    
    set_seed(config.seed)
    if config.marginal_vs_sequential == 'sequential' and not config.use_X:
        max_post_rows = None
        loader_names = ['val']
        if config.extra_eval_data is not None:
            loader_names = ['val','extra_eval']
            max_post_rows=1000
            
        post_samples = save_posterior_samples(config, model, args.run_dir, device, 
                loader_names=loader_names, 
                predictions = predictions,
                row_embeds = row_embeds,
                all_num_prev_obs = args.post_sample_all_num_prev_obs, 
                num_repetitions = args.post_sample_num_repetitions,
                num_imagined = args.post_sample_num_imagined,
                recalc=recalc,
                max_post_rows=max_post_rows)
    

def add_default_postproc_params(parser):
    parser.add_argument('--postproc_force_recalc', type=parse_bool, default=True) # true for actual usage
    parser.add_argument('--post_sample_all_num_prev_obs', type=parse_int_list, 
            help='an integer or a list of integers separated by commas', 
            default=[0,1,2,5,10,25])
    parser.add_argument('--post_sample_num_repetitions', type=int, default=250)
    parser.add_argument('--post_sample_num_imagined', type=int, default=500)
                            #                    default=[0, 1])
    #parser.add_argument('--post_sample_num_repetitions', type=int, default=5)
    #parser.add_argument('--post_sample_num_imagined', type=int, default=25)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, help='directory with model outputs')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--wandb_entity', default='ps-autoregressive')
    add_default_postproc_params(parser)
    args = parser.parse_args()

    import wandb
    wandb.login()
    wandb.init(project='postprocessing', entity=args.wandb_entity)
    
    device = get_device(args.gpu)
    args.device = device
    print(f'Arg gpu: {args.gpu}, Device: {device}')
    
    do_postprocessing(args)

if __name__ == "__main__":
    main()
