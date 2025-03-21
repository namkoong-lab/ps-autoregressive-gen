import os
from pathlib import Path
import numpy as np
import torch
import sys
import argparse

def concat_tensor_dicts(dicts, dim=0):
    """
    Recursively concatenates dictionaries of tensors, preserving nested structure.
    
    Args:
        dicts: list of dictionaries with matching structure.
               Values can be tensors or nested dictionaries of tensors.
        dim: Dimension along which to concatenate tensors (default: 0)
    
    Returns:
        dict: A new dictionary with the same structure where tensors are concatenated.
    
    Example:
        dict1 = {
            'a': tensor([[1, 2]]),  # shape: (1, 2)
            'b': {'x': tensor([3]), 'y': tensor([[4]])}
        }
        dict2 = {
            'a': tensor([[5, 6]]),  # shape: (1, 2)
            'b': {'x': tensor([7]), 'y': tensor([[8]])}
        }
        # Concatenate along dim=0 (default)
        result = concat_tensor_dicts(dict1, dict2)
        # result['a'] shape: (2, 2)
        
        # Concatenate along dim=1
        result = concat_tensor_dicts(dict1, dict2, dim=1)
        # result['a'] shape: (1, 4)
    """
    if not dicts:
        return {}
    
    # Use first dict as reference for structure
    first_dict = dicts[0]
    result = {}
    
    for key in first_dict:
        # Get all values for this key across input dicts
        values = [d[key] for d in dicts]
        first_val = values[0] 
        # If the value is a dictionary, recurse with same dim
        if isinstance(first_val, dict):
            result[key] = concat_tensor_dicts(values, dim=dim)
        # if lists, add lists
        elif isinstance(first_val, list):
            result[key] = sum(values, [])
        elif hasattr(first_val, 'shape') and hasattr(first_val, 'dtype'):
            result[key] = torch.cat(values, dim=dim)
        # Otherwise, concatenate tensors along specified dimension
        else:
            print(values)
            result[key] = values
    
    return result

def argmax_random_tiebreak(tensor):
    # Find the maximum values and their indices
    max_value = torch.max(tensor)

    # Create a mask for the maximum values
    max_mask = tensor == max_value

    # Generate random numbers for tie-breaking
    random_values = torch.rand(tensor.size())

    # Multiply the random numbers by the mask to only affect ties
    masked_random_values = random_values * max_mask

    # Add the random values to the original tensor, then use argmax
    return torch.argmax(tensor + masked_random_values)


def get_best_run(d):
    sds = os.listdir(d)
    best_loss = np.inf
    best_sd = None
    for sd in sds:
        try:
            c = torch.load(d + '/' + sd + '/best_loss.pt', map_location='cpu')
            loss = c['val_loss_dict']['loss']
            epoch = c['epoch']
            if loss < best_loss:
                best_loss = loss
                best_sd = sd
        except:
            pass
    print(best_loss)
    return best_sd


def make_plots_good():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
    matplotlib.rcParams.update({'font.size': 14})

# from https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch
def pcustom_roll(arr, r_tup):
    m = r_tup 
    arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].clone() #need `copy`
    strd_0, strd_1 = arr_roll.stride()
    n = arr.shape[1]
    result = torch.as_strided(arr_roll, (*arr.shape, n), (strd_0 ,strd_1, strd_1))
    return result[torch.arange(arr.shape[0]), (n-m)%n]

def get_device(gpu):
    if gpu is not None and int(gpu) >= 0:
        return torch.device(f'cuda:{gpu}')
    else: 
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_bool(v): 
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_int_list(s):
    try:
        return [int(item) for item in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list of integers: {}".format(s))

def make_parent_dir(fname):
    os.makedirs(os.path.dirname(fname), 0o775, exist_ok=True)


def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed+100)
    import numpy as np
    np.random.seed(seed+1000)


def parse_fname(fname):
    fname = fname.replace(':',',')
    res = {}
    for s in fname.split(','):
        splitvals = s.split('=')
        if len(splitvals) == 2:
            k,v = tuple(splitvals)

            try:
                v = int(v)
            except ValueError:
                try: 
                    v = float(v)
                except ValueError:
                    pass
            res[k]=v
    return res

def satisfies_dict(d, target):
    for k,v in target.items():
        if k not in d.keys() or d[k] != v:
            return False
    return True

def get_best_subdir(base_dir, required_dict={}, prefix=None):
    ok_dirs = [x for x in os.listdir(base_dir) if x.startswith(prefix)]
    best_loss = np.inf
    best_subdir = None
    for subdir in ok_dirs:
        subdir_conf = parse_fname(subdir)
        if satisfies_dict(subdir_conf, required_dict): 
            check = torch.load(base_dir + '/' + subdir + '/best_loss.pt')
            this_loss = check['val_loss_dict']['loss']
            if this_loss < best_loss:
                best_loss = this_loss
                best_subdir = subdir
    return best_subdir

