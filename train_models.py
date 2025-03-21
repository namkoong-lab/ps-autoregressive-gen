import time
import sys
import numpy as np
import argparse
import torch
import wandb
from transformers import get_scheduler
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Imports from other files
from util import make_parent_dir, set_seed, parse_bool
from dataset_MIND import get_token_dict, get_loaders_MIND
from dataset_synthetic import get_loaders_synthetic
from train_fns import get_val_loss, get_model_and_optimizer_MIND, get_model_and_optimizer_synthetic, loss_from_loss_matrix
from postprocessing import do_postprocessing, add_default_postproc_params

##############################################################################
# Functions for parsing arguments and training
##############################################################################

def get_argparser():
    parser = argparse.ArgumentParser()
    
    # File Saving -------------------------------------
    parser.add_argument('--save_name', type=str, help='save name')
    parser.add_argument('--data_dir', type=str, help='directory with data for training')
    parser.add_argument('--embed_data_dir', type=parse_bool, help='data has embeddings dump', default=False)
    parser.add_argument('--extra_eval_data', type=str, help='location of extra eval dataset (optional)', default=None)
    parser.add_argument('--wandb_user', type=str)
   
    # General Training -------------------------------------
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32) # note that this means rows OR obs
    parser.add_argument('--eval_batch_size', type=int, default=32) # note that this means rows OR obs
    parser.add_argument('--marginal_vs_sequential', type=str, choices=['sequential','marginal'])
    parser.add_argument('--sequential_beta_bernoulli', type=parse_bool, default=False)
    parser.add_argument('--sequential_beta_bernoulli_alpha_beta', type=parse_bool, default=False)
    parser.add_argument('--seed', type=int, default=2340923)
    parser.add_argument('--onelayer', type=parse_bool, default=False)
    parser.add_argument('--MLP_width', type=int, default=50)
    parser.add_argument('--MLP_layer', type=int, default=3)
    parser.add_argument('--MLP_last_fn', type=str, default='sigmoid')
    parser.add_argument('--repeat_suffstat', type=int, default=1,
                       help="number of times to repeat sufficient statitic input")
    parser.add_argument('--rand_prior', type=int, default=0,
                       help="Bool for whether the MLP uses randomized prior functions")
    # uses new randomized prior code in ModelWithPrior
    parser.add_argument('--prior_scale', type=float, default=0)
    parser.add_argument('--postprocess_often', type=int, default=0,
                       help="Bool for whether to postprocess everytime a new model is saved")
    
    # Train on only one timestep / number of observations to condition on
    parser.add_argument('--sequential_one_length', type=int, default=None)
    parser.add_argument('--weight_factor', type=float, default=1)
    parser.add_argument('--scheduler_type', type=str, default='constant')

    # Dataset Processing -------------------------------------
    parser.add_argument('--dataset_type', type=str, choices=['MIND','synthetic'], default='MIND')
    parser.add_argument('--sample_frac', type=float, default=1.0, 
            help='for testing purposes, subsample train and eval datasets for faster testing')
    parser.add_argument('--num_loader_obs', type=int, default=500)
    parser.add_argument('--num_loader_obs_train', type=int, default=500)
    parser.add_argument('--datasplit', type=str, default=None, 
                       help="Write first_60 to train on first 60% of the train dataset. \
                            Write last_30 to train on last 30% of the train dataset")
    parser.add_argument('--savelens', type=str, default=None, 
                       help="Save best models for listed timesteps, by val loss on that timestep")

    # Synthetic Arguments --------------------------------------
    parser.add_argument('--Z_dim', type=int, default=1)
    parser.add_argument('--X_dim', type=int, default=1)

    # MIND Arguments --------------------------------------
    parser.add_argument('--use_text', type=int, default=1, help="whether to use text")
    parser.add_argument('--use_category', type=int, default=0, help="whether to use category")
    parser.add_argument('--transform_success_p_alpha', type=str, default=1, 
            help='we have several datasets where we transformed the success probability to be centered around 0.5, for better (effective) sample size. This transform is parameterized by parameter alpha that goes from 0 to 1, where 0 is the original success probabilities and 1 centers success probabilities around 0.5')
    parser.add_argument('--click_data_suffix', type=str, default=None, help="any additional appendix to click data name") 
    parser.add_argument('--bootstrap_seed', type=int, default=None)
    
    # BERT/GPU Arguments (only for MIND)
    parser.add_argument('--bert_learning_rate', type=float, default=None)
    parser.add_argument('--bert_weight_decay', type=float, default=None)
    parser.add_argument('--freeze_bert', type=parse_bool, default=False)
    parser.add_argument('--load_bert_file', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=None)
   
    # arguments just for beta bernoulli model
    parser.add_argument('--aplusb_learning_rate', type=float, default=None)
    # sequential models have state [running mean, 1/n]. When n=0, set running mean to the following value:
    parser.add_argument('--sequential_init_mean', type=float, default=0.5)

    return parser

def main():
    parser = get_argparser()
    add_default_postproc_params(parser)

    # Parse Arguments, Initialize save files and logging ============================================
    config = parser.parse_args()

    # Currently data splitting only implemented for MIND
    #if config.dataset_type == "synthetic":
    #    assert config.datasplit in [None, "first_100", "last_100"]
    if config.onelayer:
        assert config.marginal_vs_sequential == 'marginal'
    if config.sequential_beta_bernoulli_alpha_beta:
        config.sequential_beta_bernoulli = True
    if config.sequential_beta_bernoulli:
        assert config.marginal_vs_sequential == 'sequential'
        assert config.use_category != 1
    descriptive_name = f"{config.marginal_vs_sequential}:epochs={config.epochs},bs={config.batch_size},lr={config.learning_rate},wd={config.weight_decay},MLP_layers={config.MLP_layer},MLP_width={config.MLP_width},weight_factor={config.weight_factor},max_obs={config.num_loader_obs},"
    if config.marginal_vs_sequential == 'sequential':
        descriptive_name += f"repeat_suffstat={config.repeat_suffstat},"

    if config.rand_prior:
        assert config.marginal_vs_sequential == 'marginal'
        descriptive_name += "rand_prior=True,"
    # this uses the new randomized prior code
    if config.prior_scale != 0:
        assert config.marginal_vs_sequential == 'marginal'
        descriptive_name += f"prior_scale={config.prior_scale},"

    if config.num_loader_obs != config.num_loader_obs_train:
        descriptive_name += f'max_obs_train={config.num_loader_obs_train},'
        
    if config.sequential_one_length is not None:
        assert config.marginal_vs_sequential == 'sequential'
        descriptive_name += f'sequential_one_length={config.sequential_one_length},'

    if config.datasplit is not None:
        descriptive_name += f'datasplit={config.datasplit},'

    if config.aplusb_learning_rate is not None:
        descriptive_name += f'aplusb_lr={config.aplusb_learning_rate},'

    if config.savelens is not None:
        assert config.sequential_one_length is None
        descriptive_name += f'savelens={config.savelens},'
        savelens = [int(x) for x in config.savelens.split(",")]
    else:
        savelens = []

    if config.dataset_type == "MIND":
        # MIND DATASET -----------------------------------------------------------------------------

        # Different file saving for text and no text features
        if config.use_text:
            descriptive_name += f'bert_lr={config.bert_learning_rate},'
            if config.freeze_bert:
                descriptive_name += f'freezebert={config.freeze_bert},'
        else:
            descriptive_name += f'category={config.use_category},'
        
        descriptive_name += f'sample_frac={config.sample_frac},p_alpha={config.transform_success_p_alpha},'
        if config.click_data_suffix is not None:
            descriptive_name += f'click_suff={config.click_data_suffix},'
        
    
    elif config.dataset_type == "synthetic":
         # SYNTHETIC DATASET -----------------------------------------------------------------------------
        descriptive_name += f'Zdim={config.Z_dim},'

    else:
        raise ValueError("Invalid dataset type")
    if config.bootstrap_seed is not None:
        descriptive_name += f'boot_seed:{config.bootstrap_seed},'
    if config.embed_data_dir is True:
        descriptive_name += 'embed_data,'
    if config.scheduler_type != 'linear':
        descriptive_name += f'sched={config.scheduler_type},'
    if config.sequential_beta_bernoulli:
        descriptive_name += 'BB=True,'
    if config.sequential_beta_bernoulli_alpha_beta:
        descriptive_name += 'alphabeta=True,'
    # previous descriptive_name ends with a comma (,)
    descriptive_name = descriptive_name + f'seed={config.seed}'

    save_dir = config.data_dir + '/models/' + config.save_name + '/' + descriptive_name + '/'
    logging.info(descriptive_name)
    logging.info(f"Saving in {save_dir}")
    make_parent_dir(save_dir)
    torch.save(config, save_dir + '/config.pt')

    save_dir_savelens = {}
    for t in savelens:
        save_dir_savelens[t] = save_dir + f'/savelen={t}/'
        make_parent_dir(save_dir_savelens[t])

    wandb.login()
    wandb.init(project=config.save_name, entity=config.wandb_user,
            dir=save_dir,
            config=config,
            name=descriptive_name+config.data_dir.split("/")[-1])
    
    logging.info(config)
    set_seed(config.seed)
        
    if config.gpu is not None and int(config.gpu) >= 0:
        config.device = torch.device(f'cuda:{config.gpu}')
    else:
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.dataset_type == "MIND" and config.use_text:
        if config.bert_learning_rate is None:
            config.bert_learning_rate = config.learning_rate
        if config.bert_weight_decay is None:
            config.bert_weight_decay = config.weight_decay
        
    
    # Loading Datasets and Making Dataset Objects =====================================================================
    category2id = None
    if config.dataset_type == "MIND":
        if config.embed_data_dir:
            # load bert embeddings, instead of evaluating them
            loader_dict = get_loaders_synthetic(config)

        else:
            loader_dict = get_loaders_MIND(config)
            tokenizer = loader_dict['tokenizer']
            if config.use_category:
                category2id = loader_dict['category2id']

    elif config.dataset_type == "synthetic":
        loader_dict = get_loaders_synthetic(config)
        
    train_loader = loader_dict['train_loader']
    train_dataset = loader_dict['train_dataset']
    train_fixed_subset_loader = loader_dict['train_fixed_subset_loader']
    val_loader = loader_dict['val_loader']
    val_dataset = loader_dict['val_dataset']

    if config.extra_eval_data is not None:
        extra_eval_loader = loader_dict['extra_eval_loader']
        extra_eval_dataset = loader_dict['extra_eval_dataset']
     
    # Initialize Prediction Models ===================================================================================
    model_seed = config.seed
    if config.bootstrap_seed is not None:
        model_seed += config.bootstrap_seed
    set_seed(model_seed)
    if config.dataset_type == "MIND":
        if config.use_category:
            category2id_dict = train_dataset.category2id
        else:
            category2id_dict = None
        model, optimizer_dict = get_model_and_optimizer_MIND(config, 
                                                             category2id_dict = category2id_dict)
        
    elif config.dataset_type == "synthetic":
        model, optimizer_dict = get_model_and_optimizer_synthetic(config)

    if hasattr(config, 'end2end') and config.end2end:
        optimizer = optimizer_dict['encoder']
    else:
        optimizer = optimizer_dict['all']
        
    logging.info(model)
    total_batches = len(train_loader) * config.epochs
    if not hasattr(config, 'scheduler_type'):
        setattr(config, 'scheduler_type', 'constant')
    scheduler = get_scheduler(config.scheduler_type, optimizer,
            num_training_steps=total_batches, num_warmup_steps=0)

    # Training Loop =================================================================
    logging.info("Begin training")
    best_loss = np.inf
    best_loss_savelens = {}
    for t in savelens:
        best_loss_savelens[t] = np.inf

    for epoch in range(config.epochs):
        start = time.time()
        logging.info(f"=== Epoch {epoch} ===")
        epoch_loss_unweighted = 0
        epoch_obs = 0
        epoch_loss_train = 0
        for i, batch in enumerate(train_loader):

            for k,v in batch.items():
                batch[k] = v.to(config.device)

            model.train()
            optimizer.zero_grad()
            
            click_mask = batch['click_length_mask']
            if config.dataset_type == "synthetic" or (config.dataset_type=='MIND' and config.embed_data_dir):
                model_input = batch['Z']
            elif config.dataset_type == "MIND":
                if config.use_category:
                    model_input = batch['category_ids']
                else:
                    model_input = get_token_dict(batch)

            loss_matrix = model.eval_seq(model_input,
                                         batch['click_obs'], N=None)
            if config.sequential_one_length is not None:
                loss_matrix = loss_matrix[:,[config.sequential_one_length]]
                click_mask = click_mask[:,[config.sequential_one_length]]

            loss_train = loss_from_loss_matrix(loss_matrix, click_mask, 
                                    how='sum_avg_per_row', weight_factor=config.weight_factor)
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            if i % 50 == 0:
                if config.dataset_type == "MIND" and not config.embed_data_dir:
                    logging.info(f"  iter [{i}]: training loss (weight={config.weight_factor}) {loss_train.item():5.4f} {tokenizer.decode(batch['text_token_ids'][0])[6:80]} {batch['click_obs'].mean()}")
                else:
                    logging.info(f"  iter [{i}]: training loss (weight={config.weight_factor}) {loss_train.item():5.4f} {batch['click_obs'].mean()}")
                        
            epoch_loss_train += loss_train.detach().cpu().item()

            if config.weight_factor != 1:
                # less computation for regular loss by not tracking gradients
                with torch.no_grad():
                    loss_unweighted = loss_from_loss_matrix(loss_matrix, click_mask, 
                                                 how='sum_avg_per_row')

                 # log (unweighted) loss, regardless of training objective
                if i % 50 == 0:
                    if config.dataset_type == "MIND" and not config.embed_data_dir:
                        logging.info(f"  iter [{i}]: loss {loss_unweighted.item():5.4f} {tokenizer.decode(batch['text_token_ids'][0])[6:80]} {batch['click_obs'].mean()}")
                    else:
                        logging.info(f"  iter [{i}]: loss {loss_unweighted.item():5.4f} {batch['click_obs'].mean()}")

                epoch_loss_unweighted += loss_unweighted.detach().cpu().item()
            
            epoch_obs += len(batch['click_obs'])
                
        logging.info(f'Finished epoch {epoch}; epoch loss {epoch_loss_train/epoch_obs}')
        wandb.log({'train_loss': epoch_loss_train/epoch_obs}, step=epoch)
        if config.weight_factor != 1:
            wandb.log({'weighted_train_loss': epoch_loss_unweighted/epoch_obs}, step=epoch)


        val_start = time.time()
        val_loss_dict = get_val_loss(model, val_loader, config.device, category2id=category2id, weight_factor=1, embed_data=config.embed_data_dir)
        val_end = time.time()
        wandb.log({'epoch_val_secs': (val_end - val_start)}, step=epoch)
        train_subset_loss_dict = get_val_loss(model, train_fixed_subset_loader, config.device, train_dataset, embed_data=config.embed_data_dir)

        val_loss = val_loss_dict['loss']

        train_subset_loss = train_subset_loss_dict['loss']

        wandb.log({'val_loss': val_loss}, step=epoch)
        wandb.log({'train_subset_loss': train_subset_loss}, step=epoch)
        
        
        val_loss_savelens = {}
        for t in savelens:
            one_timestep_loss = get_val_loss(model, val_loader, config.device, val_dataset, trainlen=t, exact=True, embed_data=config.embed_data_dir)
            val_loss_savelens[t] = one_timestep_loss['loss']

        if config.weight_factor != 1:
            val_loss_dict_weighted = get_val_loss(model, val_loader, config.device, val_dataset, weight_factor=config.weight_factor, embed_data=config.embed_data_dir)
            train_subset_loss_dict_weighted = get_val_loss(model, train_fixed_subset_loader, config.device, 
                                                               train_dataset, weight_factor=config.weight_factor, embed_data=config.embed_data_dir)
                
            wandb.log({'val_loss_weighted': val_loss_dict_weighted['loss']}, step=epoch)
            wandb.log({'train_subset_loss_weighted': train_subset_loss_dict_weighted['loss']}, step=epoch)
            logging.info(f'val_loss_weighted: {val_loss_dict_weighted["loss"]}')


        # log MSE for val and train_subset
        for name, loss_dict in [('val', val_loss_dict), ('train_subset', train_subset_loss_dict)]:
            predicted_probs = loss_dict['theta_hats']
            ground_truth = loss_dict['click_rates'].unsqueeze(1).repeat(1, predicted_probs.shape[1])
            squared_error = (predicted_probs - ground_truth)**2
            
            # MSE on average, over all observed lengths
            mse_any_obs = squared_error.mean()
            wandb.log({f'{name} mse, all lengths': mse_any_obs}, step=epoch)
            loss_dict['mse_any_obs'] = mse_any_obs

            # MSE on average, after 0 observations
            mse_0_obs = squared_error[:,0].mean()
            wandb.log({f'{name} mse, after 0 obs': mse_0_obs}, step=epoch)
            loss_dict['mse_0_obs'] = mse_0_obs

            # MSE on average, after max number of observations
            mse_max_obs = squared_error[:,-1].mean()
            wandb.log({f'{name} mse, after max obs': mse_max_obs}, step=epoch)
            loss_dict['mse_max_obs'] = mse_max_obs

            loss_per_t = loss_dict['loss_per_t']
            mse_per_t = squared_error.mean(axis=0)
            for t in [0, 1, 2, 3, 4, 5, 10, 25, 100]:
                if t < config.num_loader_obs and t < len(loss_per_t):
                    wandb.log({f'{name}_loss, after {str(t)} obs': loss_per_t[t]}, step=epoch)
                    wandb.log({f'{name}_mse, after {str(t)} obs': mse_per_t[t]}, step=epoch)

        logging.info(f'val_loss: {val_loss}')
        
        save_dict = {
            'state_dict':model.state_dict(),
            'optimizer': optimizer_dict,
            #'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'val_loss_dict': val_loss_dict,
            'train_subset_loss_dict': train_subset_loss_dict,
            'config': config,
        }
        if val_loss < best_loss:
            best_loss = val_loss
            
            if config.extra_eval_data is not None:
                extra_eval_loss_dict = get_val_loss(model, extra_eval_loader, config.device, extra_eval_dataset, embed_data=config.embed_data_dir)
                save_dict['extra_eval_loss_dict'] = extra_eval_loss_dict

            torch.save(save_dict, save_dir + '/best_loss.pt')
            if config.postprocess_often:
                setattr(config, 'run_dir', save_dir)
                do_postprocessing(config)

        for t,loss in best_loss_savelens.items():
            if val_loss_savelens[t] < loss:
                best_loss_savelens[t] = val_loss_savelens[t]
                torch.save(save_dict, save_dir_savelens[t] + '/best_loss.pt')
            print(f'Saving best savelen {t} at epoch {epoch}')        
        # save latest
        torch.save(save_dict, save_dir + '/latest.pt') 
        end = time.time()
        wandb.log({'epoch_train_secs': (end - start)}, step=epoch)
    

    # do postprocessing
    setattr(config, 'run_dir', save_dir)
    do_postprocessing(config)

if __name__ == '__main__':
    main()

