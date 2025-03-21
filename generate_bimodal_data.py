import torch
from pathlib import Path

def generate_data_bimodal(D, N, cnts1, cnts2):
    Z = torch.rand( (D, 2) )    # two dimensional Z
    mean1 = Z[:,0]*0.25; mean2 = 0.75+Z[:,1]*0.25
    
    # mode 1
    alpha1 = mean1*cnts1 + 1
    beta1 = 2 + cnts1 - alpha1
    
    # mode 2
    alpha2 = mean2*cnts2 + 1
    beta2 = 2 + cnts2 - alpha2
    
    # sample a click rate
    mode1_ind = torch.bernoulli(torch.ones(D)*0.5)
    mode1sample = torch.distributions.beta.Beta(alpha1, beta1).sample()
    mode2sample = torch.distributions.beta.Beta(alpha2, beta2).sample()
      
    success_p = mode1_ind * mode1sample + (1-mode1_ind) * mode2sample
    
    res = {}
    res['Z'] = Z
    res['alpha1'] = alpha1; res['beta1'] = beta1
    res['alpha2'] = alpha2; res['beta2'] = beta2
    res['cnts1'] = cnts1; res['cnts2'] = cnts2
    res['mode'] = mode1_ind
    res['click_rate'] = success_p.unsqueeze(1)
    res['Y'] = torch.bernoulli(res['click_rate'].repeat(1,N))

    return res

def main():
    import argparse
    from util import set_seed, make_parent_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=100,
                        help="number of domains (training)")
    parser.add_argument("--D_eval", type=int, default=1000,
                        help="number of domains (evaluation)")
    parser.add_argument("--D_bandit", type=int, default=None,
                        help="number of domains (evaluation)")
    parser.add_argument("--cnts1", type=int, default=100)
    parser.add_argument("--cnts2", type=int, default=100)
    parser.add_argument("--N", type=int, default=50,
                        help="number of obervations per domain")
    parser.add_argument("--datadir", type=str, 
                        help="save dataset to this path")
    args = parser.parse_args()
    print("")
    print(vars(args))
    set_seed(382374298)
    Path(args.datadir).mkdir(parents=True, exist_ok=True)    
    
    # TRAINING SET ------------------
    data_dict = generate_data_bimodal(args.D, args.N, args.cnts1, args.cnts2) 
    data_dict['N'] = args.N
    data_dict['D'] = args.D
    
    if args.D_bandit is not None and args.D_bandit > 0:
        path = f"{args.datadir}/N={args.N},D={args.D},D_eval={args.D_eval},D_bandit={args.D_bandit},cnts1={args.cnts1},cnts2={args.cnts2}" 
    else:
        path = f"{args.datadir}/N={args.N},D={args.D},D_eval={args.D_eval},cnts1={args.cnts1},cnts2={args.cnts2}" 
    fname = path + "/train_data.pt"
    make_parent_dir(fname)
    
    with open(fname, 'wb') as f:
        torch.save(data_dict, f)
        
        
    # EVAL SET ------------------
    data_dict = generate_data_bimodal(args.D_eval, args.N, args.cnts1, args.cnts2) 
    data_dict['N'] = args.N
    data_dict['D'] = args.D
    
    fname = path + "/eval_data.pt"
    make_parent_dir(fname)
    
    with open(fname, 'wb') as f:
        torch.save(data_dict, f)
   
    # Bandit set
    if args.D_bandit is not None and args.D_bandit > 0:
        data_dict = generate_data_bimodal(args.D_bandit, args.N, args.cnts1, args.cnts2) 
        data_dict['N'] = args.N
        data_dict['D'] = args.D_bandit # what is this even used for. 
        
        fname = path + "/bandit_data.pt"
        make_parent_dir(fname)
        
        with open(fname, 'wb') as f:
            torch.save(data_dict, f)



if __name__ == "__main__":
    main()
