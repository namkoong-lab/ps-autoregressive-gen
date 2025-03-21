import torch
import numpy as np

def generate_data_beta(D, N, cnts):
    Z = torch.rand( (D,1) )    # one dimensional Z
    
    # mode 1
    alpha = Z*cnts + 1
    beta = (1-Z)*cnts + 1
    
    # sample a click rate
    success_p = torch.distributions.beta.Beta(alpha, beta).sample()
    
    res = {}
    res['Z'] = Z
    res['alpha'] = alpha; res['beta'] = beta
    res['cnts'] = cnts
    res['click_rate'] = success_p # ensure dimension
    res['Y'] = torch.bernoulli(res['click_rate'].repeat(1,N))

    return res


def fit_beta_dist(success_ps):
    mean = torch.mean(success_ps)
    var = torch.var(success_ps)
    
    alpha = ( (1-mean)/var - 1/mean ) * mean**2
    beta = alpha * (1/mean - 1)
        
    return {'alpha': alpha, 'beta': beta}


def get_beta_postpred(val_dict, fit_prior=False, prior_dict={'alpha': 1, 'beta':1}):
    # Fit / specify prior ====================================
    if fit_prior:
        prior = fit_beta_dist( val_dict['click_rates'] )
    else:
        prior = prior_dict

    # Compute posterior predictives (Bernoullis)
    click_obs = val_dict['click_obs']

    # Start with 0 observations (no look-ahead)
    click_obs = torch.concatenate([torch.zeros(click_obs.shape[0],1), click_obs],1)[:,:-1]
    cum_rewards = torch.cumsum( click_obs, axis=1 )

    # Row length increases with more observations, starting with 0
    row_length = torch.ones_like(cum_rewards).cumsum(axis=1) - 1
    p_hats = ( prior['alpha'] + cum_rewards ) / ( prior['alpha'] + prior['beta'] + row_length )
    
    return p_hats


def main():
    import argparse

    import os
    import sys
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 
    
    from util import set_seed, make_parent_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=100,
                        help="number of domains (training)")
    parser.add_argument("--D_eval", type=int, default=1000,
                        help="number of domains (evaluation)")
    parser.add_argument("--cnts", type=int, default=100)
    parser.add_argument("--N", type=int, default=50,
                        help="number of obervations per domain")
    parser.add_argument("--datadir", type=str, 
                        help="save dataset to this path")
    args = parser.parse_args()
    print("")
    print(vars(args))
    set_seed(382374298)
    
    
    # TRAINING SET ------------------
    data_dict = generate_data_beta(args.D, args.N, args.cnts) 
    data_dict['N'] = args.N
    data_dict['D'] = args.D
    
    fname = f"{args.datadir}/N={args.N},D={args.D},D_eval={args.D_eval},cnts={args.cnts}/train_data.pt" 
    make_parent_dir(fname)
    
    with open(fname, 'wb') as f:
        torch.save(data_dict, f)
        
        
    # EVAL SET ------------------
    data_dict = generate_data_beta(args.D_eval, args.N, args.cnts) 
    data_dict['N'] = args.N
    data_dict['D'] = args.D
    
    fname = f"{args.datadir}/N={args.N},D={args.D},D_eval={args.D_eval},cnts={args.cnts}/eval_data.pt" 
    make_parent_dir(fname)
    
    with open(fname, 'wb') as f:
        torch.save(data_dict, f)


if __name__ == "__main__":
    main()
