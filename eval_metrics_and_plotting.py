# Standard packages
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# Importing from files
from util import get_best_subdir, make_parent_dir
#from models import BetaProcess, SequentialPredictor


#################### Prediction Evaluation and Plotting #####################

def get_prediction_metric_values(pred, click_rate, metric):
    assert len(click_rate.shape)==1
    reshaped_click_rate = click_rate.unsqueeze(-1).repeat(1,pred.shape[1])
    if metric == 'mse':
        return F.mse_loss(pred, reshaped_click_rate, reduction='none')
    elif metric == 'joint_loss':
        return F.binary_cross_entropy(pred, reshaped_click_rate, reduction='none')
    else:
        raise ValueError('metric must be mse, or joint_loss')


def make_plot_from_predictions(pred_dict, metric, click_rate, timesteps=20, ax=None, 
                               linestyle_dict=None, dict_key=None, ylim=None, title="", skip_val=1,
                                ylogscale=False, plotkeys=None):
    '''
    timesteps: max number of t (num obs) for plotting
    pred_dict: dict of predictions
    click_rate: true click rates
    '''
    assert metric in ['mse','joint_loss']
    if linestyle_dict is None:
        linestyle_dict = defaultdict(lambda: '-')
    K = timesteps
    
    if ax is None:
        fig, ax = plt.subplots(1)
        
    for k,v in pred_dict.items():
        if plotkeys is not None and k not in plotkeys:
            continue
        if dict_key is None:
            metric_values = get_prediction_metric_values(v, click_rate, metric)
        else:
            metric_values = get_prediction_metric_values(v[dict_key], click_rate, metric)
            
        if timesteps is not None:
            metric_values = metric_values[:,:K][:,::skip_val]
        metric_means = metric_values.mean(dim=0)
        metric_sd = metric_values.std(dim=0) / metric_values.shape[0]**0.5

        ax.errorbar(x=np.arange(K)[::skip_val],
                    y=metric_means,
                    yerr=metric_sd, 
                    label=k, 
                    linestyle=linestyle_dict[k])
        ax.set_ylabel(metric.upper())
    ax.set_xlabel('Number of observations')
    ax.legend();
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if ylogscale:
        ax.set_yscale('log')

    ax.set_title(title)


# FUNCTION TO FORM CREDIBLE INTERVALS ========================================

def credible_interval_eval(sig_val, post_samples, true_p):

    # form credible interval
    interval = np.quantile(post_samples, [sig_val/2, 1-sig_val/2], axis=1)

    # evaluate coverage
    above_lower = np.greater(true_p, interval[0])
    below_upper = np.less(true_p, interval[1])
    in_interval = np.logical_and(above_lower, below_upper)
    emp_coverage = torch.mean(in_interval*1.0)
    emp_coverage_sterr = torch.std(in_interval*1.0) / np.sqrt(len(in_interval))

    # interval width
    widths = interval[1] - interval[0]
    averge_width = np.mean(widths)
    averge_width_sterr = np.std(widths) / np.sqrt(len(widths))

    # corrected interval width
    #interval = np.quantile(post_samples, [sig_val/2, 1-sig_val/2], axis=1)
    
    return {
        "credible_intervals": interval,
        "empirical_coverage": (emp_coverage.item(), emp_coverage_sterr.item()),
        "interval_width": (averge_width, averge_width_sterr),
    }


def credible_interval_results(posterior_dict, all_num_prev_obs, all_sig_val = None):

    if all_sig_val is None:
        all_sig_val = np.arange(0.05,1,0.05)
        
    credible_dict = {
        'sigval': [],
        'model': [],
        'coverage': [],
        'coverage_stderr': [],
        'width': [],
        'width_stderr': [],
        'num_prev_obs': []
    }
    
    for num_prev_obs in all_num_prev_obs:
        for model_name, model_post_dict in posterior_dict.items():
            for sig_val in all_sig_val:
                eval_dict_tmp = credible_interval_eval(sig_val, 
                                           post_samples=model_post_dict[num_prev_obs]['val_post_samples'], 
                                           true_p=model_post_dict[num_prev_obs]['val_true_p'])
                coverage, coverage_stderr = eval_dict_tmp['empirical_coverage']
                width, width_stderr = eval_dict_tmp['interval_width']
                
                credible_dict['model'].append(model_name); credible_dict['sigval'].append(sig_val); 
                credible_dict['num_prev_obs'].append(num_prev_obs)
                credible_dict['coverage'].append(coverage); credible_dict['coverage_stderr'].append(coverage_stderr)
                credible_dict['width'].append(width); credible_dict['width_stderr'].append(width_stderr)
    
    return credible_dict

##################### General plotting ##########################

def plot_scatter(x_vals, y_vals, ax=None, alpha=1, buffer=None, s=5, title=None, xlabel=None, ylabel=None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    ax.scatter(x_vals, y_vals, alpha=alpha, s=s)
    all_vals = np.concatenate([x_vals, y_vals])
    if buffer is None:
        buffer = (np.max(all_vals) - np.min(all_vals) ) / 80
    valrange = (np.min(all_vals)-buffer, np.max(all_vals)+buffer)
    ax.set_xlim(valrange)
    ax.set_ylim(valrange)
    vals = np.arange(valrange[0], valrange[1]+1, 1)
    ax.plot(vals, vals, color='r', alpha=0.3)
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)



# todo: refactor with next fn
def draw_posterior_samples_MIND(all_model_dicts, all_num_prev_obs, num_imagined, num_repetitions, use_savefile=False, save_file=None, savefile_prefix=None, verbose=True):
    '''
    If use_savefile is True, then save results in default location. 
    If savefile_prefix is not None, save results in a file beginning with savefile_prefix. 
    If save_file is not None, save results in save_file.
    '''
    assert savefile_prefix is None or save_file is None
    if use_savefile and save_file is None: 
        save_str = "model={}_img={}_rep={}_Yprev={}".format(",".join(sorted(all_model_dicts.keys())), num_imagined, num_repetitions, ",".join([str(x) for x in all_num_prev_obs]))
        if savefile_prefix is not None:
            save_str = f'{savefile_prefix}:{save_str}'
        save_file = f'posterior_samples/post_samples_{save_str}.pt'

    if use_savefile and os.path.isfile(save_file):
        if verbose:
            print(f'Loading from {save_file}')
        posterior_dict = torch.load(save_file)

    else:   
        # Draw posterior samples
        posterior_dict = {}

        for model_name, model_dict in all_model_dicts.items():
            if "marginal" in model_name:
                continue
            print("\n", model_name)

            posterior_dict[model_name] = {}

            for num_prev_obs in all_num_prev_obs:
                print("num_prev_obs", num_prev_obs)

                joint_model = model_dict['joint_model']
                val_data_dict = model_dict['val_loss_dict']
                val_true_p = val_data_dict['click_rates']
                train_data_dict = model_dict['train_subset_loss_dict']
                train_true_p = train_data_dict['click_rates']
                if joint_model.use_bert:
                    Z_input = val_data_dict['article_bert_emb']
                elif joint_model.use_category:
                    Z_input = val_data_dict['category_ids']
                else:
                    Z_input = None

                # Get current state
                R_obs = val_data_dict['click_obs']
                prev_obs_og = R_obs[:, :num_prev_obs]
                curr_state = joint_model.next_model_states(prev_obs_og)

                val_post_samples = joint_model.get_posterior_draws(Z_input, curr_state,
                                                            num_imagined, num_repetitions)

                train_post_samples = joint_model.get_posterior_draws(Z_input, curr_state,
                                                              num_imagined, num_repetitions)
                posterior_dict[model_name][num_prev_obs] = {
                    'val_true_p': val_true_p,
                    'val_post_samples' : val_post_samples,
                    'train_true_p': train_true_p,
                    'train_post_samples' : train_post_samples,
                }
        if save_file is not None:
            make_parent_dir(save_file)
            if verbose: 
                print(f'Saving to {save_file}')
            torch.save(posterior_dict, save_file)

    return posterior_dict



def draw_posterior_samples_synthetic(cnts=5, rows_and_cols=[(5000,500),(500,500)],
                                     all_num_prev_obs=[0,1,5,25],
                                     all_num_imagined=[500], # sequence length
                                     num_repetitions=250, #number of posterior samples
                                     num_rows=1000,
                                     use_savefile=True,
                                     savefile_prefix='synthetic',
                                     save_file=None,
                                     verbose=True):

    def list_to_string(things):
        return ",".join([str(x) for x in things])
    assert savefile_prefix is None or save_file is None
    if use_savefile and save_file is None: 
        rowandcol_str = list_to_string([f'({row},{col})' for (row,col) in rows_and_cols])
        save_str = f"cnts={cnts}_rowsandcols={rowandcol_str}_img={list_to_string(all_num_imagined)}_rep={num_repetitions}_Yprev={list_to_string(all_num_prev_obs)}"
        if savefile_prefix is not None:
            save_str = f'{savefile_prefix}:{save_str}'
        save_file = f'posterior_samples/post_samples_{save_str}.pt'

    if use_savefile and os.path.isfile(save_file):
        if verbose:
            print(f'Loading from {save_file}')
        return torch.load(save_file)

    all_oracle_samples = {}
    all_postpred_oracle_samples = {}
    all_sequential_post_samples = defaultdict(dict)
    true_click_rates = None
    for num_obs in all_num_prev_obs:
        for num_imagined in all_num_imagined:
            print(num_imagined, num_obs)
            for (D, cols) in rows_and_cols:
                model_path, extra_eval_data = get_synthetic_model_path_and_data(cnts, D, cols, num_imagined, num_repetitions, num_obs=num_obs)
                sequential_post_samples = get_sequential_posts(model_path, extra_eval_data, num_imagined, num_repetitions, num_obs, num_rows)
                all_sequential_post_samples[(num_obs,num_imagined)][(D,cols)] = sequential_post_samples
                these_click_rates = extra_eval_data['click_rate'].flatten()

                # make sure the true click rates are the same across settings
                if true_click_rates is not None:
                    assert torch.allclose(these_click_rates, true_click_rates)
                else:
                    true_click_rates = these_click_rates

            # posterior predictive, using DGP
            postpred_oracle = get_oracle_postpred_posts(extra_eval_data, num_imagined, num_repetitions, num_obs, num_rows)
            all_postpred_oracle_samples[(num_obs,num_imagined)] = postpred_oracle

        oracle_samples = get_oracle_posts(extra_eval_data, num_repetitions, num_obs, num_rows)
        all_oracle_samples[num_obs] = oracle_samples
    res = {
        'oracle_samples':all_oracle_samples,
        'postpred_oracle_samples':all_postpred_oracle_samples,
        'sequential_samples':all_sequential_post_samples,
        'true_click_rates': true_click_rates[:num_rows]}

    if save_file is not None:
        make_parent_dir(save_file)
        if verbose: 
            print(f'Saving to {save_file}')
        torch.save(res, save_file)
    return res

#################### Custom Prediction Plotting ########################


####################### Scatter plots for posterior samples ###########################

      
def mean_abs_diff_from_mean(samples):
    sample_mean = samples.mean(1).unsqueeze(1).repeat(1,samples.shape[1])
    return (samples - sample_mean).abs()

def compare_posteriors(sample_draws, oracle_draws, title='Posterior Sampling', savefile=None, s=10, xlabel=None, figsize=(8,3)):
    fig,ax = plt.subplots(1,3,figsize=figsize)
    
    metrics = {
        'Mean': lambda x: x.mean(1),
        'Mean |sample - mean|': lambda x: mean_abs_diff_from_mean(x).mean(1),
        'Standard Deviation': lambda x: x.std(1),
    }
    sd_metrics = {
        'Mean': lambda x: x.std(1) / x.shape[1] ** 0.5,
        'Mean |sample - mean|': lambda x: mean_abs_diff_from_mean(x).std(1) / x.shape[1] ** 0.5,
    }
    for i, (metric, fn) in enumerate(metrics.items()):
        x = fn(sample_draws)
        y = fn(oracle_draws)
        plot_scatter(x, y, ax=ax[i], title=metric, s=s)    
        if metric in sd_metrics.keys():
            xerr = sd_metrics[metric](sample_draws)
            yerr = sd_metrics[metric](oracle_draws)
            ax[i].errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", c='r', alpha=0.5)
    
    if xlabel is None: 
        xlabel = 'Approximate Posterior'
    for a in ax:
        a.set_xlabel(xlabel)
        a.set_ylabel('Oracle Posterior')
    plt.suptitle(title)
    fig.tight_layout()
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')
        
def do_sampling_and_posterior_comparison_plots(model_path, extra_eval_data, num_imagined=100, num_post_samples=500,
                         num_obs=0, num_rows=100, verbose=True):
    ###### Get posterior samples ######
    # sequential samples
    sequential_post_samples = get_sequential_posts(model_path, extra_eval_data, num_imagined, num_post_samples, num_obs, num_rows)

    # oracle samples
    oracle_samples = get_oracle_posts(extra_eval_data, num_imagined, num_post_samples, num_obs, num_rows)

    # posterior predictive, using DGP
    postpred_oracle = get_oracle_postpred_posts(extra_eval_data, num_imagined, num_post_samples, num_obs, num_rows)

    compare_posteriors(postpred_oracle, oracle_samples, title='Posterior predictive oracle')
    compare_posteriors(sequential_post_samples, oracle_samples, title='Sequential sampling from NN')

