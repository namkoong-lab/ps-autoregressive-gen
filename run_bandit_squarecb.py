import numpy as np
import torch
import argparse
from util import set_seed, make_parent_dir
from postprocessing import load_old_model
from rl import run_bandit, get_bandit_envs, SquareCB
import time

def get_article_ordering(seed, N):
    rng_tmp = np.random.default_rng(seed)
    article_ordering = np.arange(N)
    rng_tmp.shuffle(article_ordering)
    return article_ordering

def get_file_savename(args):
    name = f'num_arms={args.num_arms},T={args.T},seed={args.seed},dataset={args.dataset},alg={args.bandit_alg},gamma0={args.gamma0},rho={args.rho}'
    
    name += f"/env_idx={args.env_idx}.pt"
    return name


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    # where to save outputs: save_dir is not given but a model_dir is given, we use that
    parser.add_argument('--save_dir', type=str, help='directory to save in', default=None)

    # load model
    parser.add_argument('--model_dir', type=str, help='directory for model whose bert embeddings to use', default=None)
    parser.add_argument('--bandit_alg', type=str, default='squarecb') 

    
    # bandit env params
    parser.add_argument('--env_idx', type=int)
    parser.add_argument('--T', type=int, help='number of timesteps')
    parser.add_argument('--num_arms', type=int, help='number of bandit arms')
    parser.add_argument('--seed', type=int, default=23485223) # where would we use this. idk
    parser.add_argument('--dataset', type=str, default='val')
    
    # RL args
    parser.add_argument('--gamma0', type=float)
    parser.add_argument('--rho', type=float)

# unused:
    parser.add_argument('--horizonDependent', type=int, default=0) # boolean

    args = parser.parse_args()
    print(args)

    assert args.bandit_alg == 'squarecb'
    assert args.model_dir is not None
    # save outputs in model_dir if save_dir is not provided
    # assert args.save_dir is not None or args.model_dir is not None

    if args.save_dir is None and args.model_dir is not None:
        args.save_dir = args.model_dir + '/bandit/'

    # load model, and also click rates and embeddings
    if args.model_dir is not None:
        model_path = args.model_dir + "/best_loss.pt"
        check = torch.load(model_path, map_location=torch.device('cpu'))
        config_path = args.model_dir + "/config.pt"
        config = torch.load(config_path, map_location=torch.device('cpu'))
        config.device = 'cpu'
        model = load_old_model(config, check['state_dict'], check)
        model.eval()


        # click rates for env
        click_rates = check[f'{args.dataset}_loss_dict']['click_rates']
    
        embed_path = args.model_dir + "/best_loss_row_embeds.pt"
        embeds = torch.load(embed_path, map_location=torch.device('cpu'))

    # shuffle bandit environment click rates
    orig_click_rates = click_rates.numpy()
    article_ordering = get_article_ordering(args.seed, len(orig_click_rates))
    click_rates = orig_click_rates[article_ordering]

    # make bandit envs, then select the correct one
    all_bandit_envs = get_bandit_envs(args.num_arms, args.T, args.env_idx+1, click_rates, seed=args.seed,
                                     horizonDependent=args.horizonDependent)
    bandit_env, chosen_arms = all_bandit_envs[args.env_idx]

    file_savename = args.save_dir + '/' + get_file_savename(args)
    make_parent_dir(file_savename)


    if model.use_category:
        Z_representation = check[f'{args.dataset}_loss_dict']['category_ids'][article_ordering][chosen_arms]
    elif model.z_encoder is not None: 
        Z_representation = embeds[args.dataset][article_ordering][chosen_arms]
    else:
        Z_representation = None


    bandit_alg = SquareCB(seq_model=model, Z_representation=Z_representation, 
            num_arms=len(chosen_arms), 
            T=args.T, 
            hyparam_dict = {'gamma0': args.gamma0, 'rho': args.rho})

    print('run bandits')
    set_seed(args.seed)
    reward_dict = run_bandit(bandit_env, bandit_alg, args.T)
    res = {
            'reward_dict': reward_dict,
            'orig_click_rates': orig_click_rates,
            'env_click_rates': click_rates[chosen_arms],
            'env_chosen_arms': chosen_arms,
            'env_article_ordering': article_ordering
    }
    torch.save(res, file_savename)
    end = time.time()
    print(f'Total time: {(end-start)} seconds = {(end-start)/60:0.2f} minutes')

if __name__ == "__main__":
    main()
