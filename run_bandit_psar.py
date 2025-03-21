import numpy as np
import torch
import argparse
from util import parse_bool, set_seed, make_parent_dir
from postprocessing import load_old_model
from rl import run_bandit, get_bandit_envs, PosteriorHallucinationAlg, GreedyPosteriorMeanAlg, BinaryRewardEnv_horizonDependent, PosteriorHallucinationAlg_horizonDependent, SquareCB
from rl import SampledGreedyPosteriorMeanAlg
from rl import DPTSequenceAlg
import time
import os

def get_article_ordering(seed, N):
    rng_tmp = np.random.default_rng(seed)
    article_ordering = np.arange(N)
    rng_tmp.shuffle(article_ordering)
    return article_ordering

def load_bandit_rewards(bandit_dir, all_bandit_envs, success_p_all):
    env_rewards_dict = {}
    action_arms_dict = {}
    for f in os.listdir(bandit_dir):
        idx = int(f.split('.')[0].split('=')[1])
        if idx >= len(all_bandit_envs): continue
        
        # verify environments are the same between loaded runs
        c = torch.load(bandit_dir + '/' + f)
        chosen_arms = all_bandit_envs[idx][1]

        for k,v in c.items():
            if isinstance(v, torch.Tensor):
                c[k] = v.detach().numpy()

        assert np.abs(all_bandit_envs[idx][1] - c['env_chosen_arms']).mean() == 0
        assert np.abs(success_p_all[chosen_arms] - c['env_click_rates']).mean() == 0
        if 'reward_dict' not in c.keys():
            print(c.keys())
        env_rewards_dict[idx] = c['reward_dict']['expected_rewards']
        action_arms_dict[idx] = c['reward_dict']['action_arms']
    missing = [idx for idx in range(len(all_bandit_envs)) if idx not in env_rewards_dict or idx not in action_arms_dict]
    if len(missing) > 0:
        raise ValueError(f'Missing idxs: {" ".join(missing)}')
    
    all_rewards = [ env_rewards_dict[idx] for idx in range(len(all_bandit_envs)) ]
    all_action_arms = [ action_arms_dict[idx] for idx in range(len(all_bandit_envs)) ]
    return {'rewards':np.array(all_rewards),'action_arms':all_action_arms}


def get_file_savename(args):
    dataset_str = f'dataset={args.dataset}'
    if args.bandit_alg in ['sequential_horizon', 'greedy']:
        name = f'num_arms={args.num_arms},T={args.T},seed={args.seed},{dataset_str},alg={args.bandit_alg}'
    else:
        name = f'num_arms={args.num_arms},T={args.T},num_imagined={args.num_imagined},seed={args.seed},{dataset_str},alg={args.bandit_alg}'
    
    if args.bandit_alg == 'sequential':
        if args.randomly_break_ties:
            name += ',rand_break_ties'
    if args.horizonDependent:
        name += ",horizonDep"
    if args.finite_horizon_alg and args.bandit_alg not in ['linearTS11','linearfeatureTS11','greedy']:
        name += ",finite_horizon_alg"
    if args.no_shuffle_boot:
        name += ',no_shuffle_boot'
    if args.use_bandit_split:
        name += ',bandit_split'
    name += f"/env_idx={args.env_idx}.pt"
    return name


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    # where to save outputs: save_dir is not given but a model_dir is given, we use that
    parser.add_argument('--save_dir', type=str, help='directory to save in', default=None)

    # load model
    parser.add_argument('--model_dir', type=str, help='directory for model to load', default=None)
    parser.add_argument('--num_imagined', type=int, default=100)
    parser.add_argument('--bandit_alg', type=str, default='sequential', 
                        choices=['sequential','greedy','squarecb','sequential_horizon', 'linearTS11','linearfeatureTS11','dpt','sampled_greedy']) 
    parser.add_argument('--randomly_break_ties', type=parse_bool, default=False) # this is mostly for sequential PSAR 
    # bandit env params
    parser.add_argument('--env_idx', type=int)
    parser.add_argument('--T', type=int, help='number of timesteps')
    parser.add_argument('--num_arms', type=int, help='number of bandit arms')
    parser.add_argument('--seed', type=int, default=23485223) # where would we use this. idk
    parser.add_argument('--dataset', type=str, default='val')
    parser.add_argument('--horizonDependent', type=int, default=0) # boolean
    
    parser.add_argument('--finite_horizon_alg', type=parse_bool, default=False, help='finite horizon algorithm')

    parser.add_argument('--bandit_dir', type=str, default='bandit')
    parser.add_argument('--no_shuffle_boot', type=parse_bool, default=False) # for debugging bootstrap
    
    parser.add_argument('--use_bandit_split', type=parse_bool, default=False) # use bandit data split


    args = parser.parse_args()
    print(args)
        
    assert args.model_dir is not None 
    # save outputs in model_dir if save_dir is not provided
    # assert args.save_dir is not None or args.model_dir is not None

    if args.save_dir is None and args.model_dir is not None:
        args.save_dir = args.model_dir + '/' + args.bandit_dir + '/'

    # load model, and also click rates and embeddings
    assert args.model_dir is not None
    model_path = args.model_dir + "/best_loss.pt"
    check = torch.load(model_path, map_location=torch.device('cpu'))
    config_path = args.model_dir + "/config.pt"
    config = torch.load(config_path, map_location=torch.device('cpu'))
    config.device = 'cpu'
    model = load_old_model(config, check['state_dict'], check)
    model.eval()

    if args.bandit_alg=='dpt':
        if config.dataset_type=='synthetic':
            # load up val dataset
            data_path = config.data_dir 
            if args.use_bandit_split:
                val_data = torch.load(data_path + '/bandit_data.pt')
            else:
                val_data = torch.load(data_path + '/eval_data.pt')
            val_batch_size = len(val_data['click_rate'])
            article_ordering = get_article_ordering(args.seed, val_batch_size)
            orig_click_rates = val_data['click_rate'].flatten().detach().numpy()
            click_rates = orig_click_rates[article_ordering]
            all_Zs = val_data['Z']
            Z_representation = model.z_encoder(all_Zs[article_ordering])
        elif config.dataset_type=='MIND':
            orig_click_rates = check['val_loss_dict']['click_rates']
            val_batch_size = len(orig_click_rates)
            article_ordering = get_article_ordering(args.seed, val_batch_size)
            Z_representation = check['val_loss_dict']['Z_representation'][article_ordering]
            click_rates = orig_click_rates[article_ordering]
    else:
        # click rates for env
        click_rates = check[f'{args.dataset}_loss_dict']['click_rates']
    
        embed_path = args.model_dir + "/best_loss_row_embeds.pt"
        embeds = torch.load(embed_path, map_location=torch.device('cpu'))

        # shuffle bandit environment click rates
        orig_click_rates = click_rates.numpy()
        article_ordering = get_article_ordering(args.seed, len(orig_click_rates))
        click_rates = orig_click_rates[article_ordering]

        if model.use_category:
            Z_representation = check[f'{args.dataset}_loss_dict']['category_ids'][article_ordering]
        elif model.z_encoder is not None:
            Z_representation = embeds[args.dataset][article_ordering]
        else:
            Z_representation = None
    

    # make bandit envs, then select the correct one
    all_bandit_envs = get_bandit_envs(args.num_arms, args.T, args.env_idx+1, click_rates, seed=args.seed,
                                     horizonDependent=args.horizonDependent)

    bandit_env, chosen_arms = all_bandit_envs[args.env_idx]
    Z_representation = Z_representation[chosen_arms]

    file_savename = args.save_dir + '/' + get_file_savename(args)
    make_parent_dir(file_savename)
    loss_matrix = None
    if args.bandit_alg != 'dpt':
        try:
            # compute prediction loss matrix
            loss_matrix = model.eval_seq(Z_representation, bandit_env.potential_outcomes)
        except:
            print('no loss matrix')
    print('Make alg')
    if args.bandit_alg=='greedy':
        bandit_alg = GreedyPosteriorMeanAlg(seq_model=model, 
                                    Z_representation=Z_representation,
                                    num_arms=args.num_arms) 
    if args.bandit_alg=='sampled_greedy':
        bandit_alg = SampledGreedyPosteriorMeanAlg(seq_model=model, 
                                    Z_representation=Z_representation,
                                    num_arms=args.num_arms,num_samples=args.num_imagined) 
    elif args.bandit_alg=='squarecb':
        bandit_alg = SquareCB(seq_model=model,
                Z_representation=Z_representation,
                num_arms=args.num_arms,T=args.T)
    elif args.bandit_alg=='sequential':
        bandit_alg = PosteriorHallucinationAlg(seq_model=model, 
                                       Z_representation=Z_representation, 
                                       num_arms=args.num_arms,
                                       num_imagined=args.num_imagined,
                                       randomly_break_ties=args.randomly_break_ties)
    elif args.bandit_alg=='sequential_horizon':
        bandit_alg = PosteriorHallucinationAlg_horizonDependent(seq_model=model, 
                                       Z_representation=Z_representation, 
                                       num_arms=args.num_arms,
                                       T=args.T)
    elif args.bandit_alg == 'dpt':
        bandit_alg = DPTSequenceAlg(model, Z_representation, args.num_arms)
    else:
        raise ValueError(f'unrecognized bandit alg: {args.bandit_alg}')
    print('run bandits')
    set_seed(args.seed)
    reward_dict = run_bandit(bandit_env, bandit_alg, args.T, return_extra=True)
    res = {
            'reward_dict': reward_dict,
            'loss_matrix': loss_matrix,
    }
    res['env_chosen_arms'] = chosen_arms
    res['env_article_ordering'] = article_ordering
    res['orig_click_rates'] = orig_click_rates,
    res['env_click_rates'] = click_rates[chosen_arms]
    torch.save(res, file_savename)
    end = time.time()
    print(f'Saved to {file_savename}')
    print(f'Total time: {(end-start)} seconds = {(end-start)/60:0.2f} minutes')

if __name__ == "__main__":
    main()
