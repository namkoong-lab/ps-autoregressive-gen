import time
import sys
import numpy as np
import argparse
import torch
import wandb
from transformers import get_scheduler
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from train_fns_dpt import get_val_loss_dpt, get_dpt_history_loader, eval_all_Zreps
# Imports from other files
from util import make_parent_dir, set_seed, parse_bool
from dataset_MIND import get_token_dict, get_loaders_MIND
from dataset_synthetic import get_loaders_synthetic
from train_fns import get_model_and_optimizer_MIND, get_model_and_optimizer_synthetic
from postprocessing import add_default_postproc_params

##############################################################################
# Functions for parsing arguments and training
##############################################################################

# do this dirichlet thing
def get_probs(num_arms):
    cov = np.random.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    alpha = np.ones(num_arms)
    probs = np.random.dirichlet(alpha)
    probs2 = np.zeros(num_arms)
    rand_index = np.random.choice(np.arange(num_arms))
    probs2[rand_index] = 1.0
    probs = (1 - cov) * probs + cov * probs2
    return probs

def get_argparser():
    parser = argparse.ArgumentParser()
    
    # File Saving -------------------------------------
    parser.add_argument('--save_name', type=str, help='save name')
    parser.add_argument('--data_dir', type=str, help='directory with data for training')
    parser.add_argument('--val_dpt_history_data', type=str, help='val dpt history data')
    parser.add_argument('--embed_data_dir', type=parse_bool, help='data has embeddings dump', default=False)
    parser.add_argument('--wandb_user', type=str)
   
    # General Training -------------------------------------
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_arms', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32) # note that this means rows OR obs
    parser.add_argument('--eval_batch_size', type=int, default=32) # note that this means rows OR obs
    parser.add_argument('--hist_batch_size', type=int, default=32) # for evaluating histories, in val only
    parser.add_argument('--sequential_beta_bernoulli', type=parse_bool, default=False)
    parser.add_argument('--sequential_beta_bernoulli_alpha_beta', type=parse_bool, default=False)
    parser.add_argument('--seed', type=int, default=2340923)
    parser.add_argument('--onelayer', type=parse_bool, default=False)
    parser.add_argument('--MLP_width', type=int, default=50)
    parser.add_argument('--MLP_layer', type=int, default=3)
    parser.add_argument('--MLP_last_fn', type=str, default='none') # it is none for DPT models
    parser.add_argument('--repeat_suffstat', type=int, default=1,
                       help="number of times to repeat sufficient statitic input")
    parser.add_argument('--rand_prior', type=int, default=0,
                       help="Booling for whether the MLP uses randomized prior functions")
    # uses new randomized prior code in ModelWithPrior
    parser.add_argument('--prior_scale', type=float, default=0)
    parser.add_argument('--postprocess_often', type=int, default=0,
                       help="Booling for whether to postprocess everytime a new model is saved")
    
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
    config.marginal_vs_sequential='sequential'
    config.extra_eval_data=None

    if config.onelayer:
        assert config.marginal_vs_sequential == 'marginal'
    if config.sequential_beta_bernoulli_alpha_beta:
        config.sequential_beta_bernoulli = True
    if config.sequential_beta_bernoulli:
        assert config.marginal_vs_sequential == 'sequential'
        assert config.use_category != 1
    descriptive_name = f"DPT:num_arms={config.num_arms}:epochs={config.epochs},bs={config.batch_size},lr={config.learning_rate},wd={config.weight_decay},MLP_layers={config.MLP_layer},MLP_width={config.MLP_width},weight_factor={config.weight_factor},max_obs={config.num_loader_obs},"
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

    hist_dataset, hist_loader = get_dpt_history_loader(config.val_dpt_history_data, 
        config.hist_batch_size, is_train=False)
     
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
    num_arms = config.num_arms

    for epoch in range(config.epochs):
        start = time.time()
        logging.info(f"=== Epoch {epoch} ===")
        epoch_loss_unweighted = 0
        epoch_obs = 0
        epoch_loss_train = 0
        for i, batch in enumerate(train_loader):

            prev_Ys = batch['click_obs']
            batch_size, T = prev_Ys.shape
            num_envs = batch_size // num_arms
            #assert num_envs == batch_size / num_arms
            batch_size = num_envs * num_arms
            
            for k,v in batch.items():
                batch[k] = v[:batch_size].to(config.device)

            model.train()
            optimizer.zero_grad()

            # only for synthetic for now

            prev_Ys = batch['click_obs']

            # generate / simulate arm counts, using dirichlet thing from DPT
            num_arms = config.num_arms
            arm_counts = torch.zeros(batch_size)
            for i in range(num_envs):
                # behavior policy from DPT
                probs = get_probs(num_arms)
                H = torch.randint(low=0,high=T,size=(1,)).item()

                for h in range(H):
                    chosen_arm = np.random.choice(np.arange(num_arms), p=probs)    
                    arm_counts[num_arms * i + chosen_arm] += 1

            # get preds

            if config.dataset_type == "synthetic" or (config.dataset_type=='MIND' and config.embed_data_dir):
                model_input = batch['Z']
            elif config.dataset_type == "MIND":
                if config.use_category:
                    model_input = batch['category_ids']
                else:
                    model_input = get_token_dict(batch)

            encoded_Z = model.z_encoder(model_input)
            cols = torch.arange(T).unsqueeze(0)  
            click_mask = (cols < arm_counts.unsqueeze(1)).to(config.device)
            pred = model.get_p_pred(encoded_Z, prev_Ys, click_mask)

            # shape into num_envs x num_arms 
            preds_reshaped = pred.view(num_envs, num_arms)
            clickrates_reshaped = batch['click_rates'].view(num_envs, num_arms)
            max_idx = clickrates_reshaped.argmax(1)
            targets = torch.zeros_like(clickrates_reshaped).to(config.device)
            targets.scatter_(1, max_idx.unsqueeze(1), 1)

            loss_train = torch.nn.functional.cross_entropy(preds_reshaped,targets,reduction='sum')           
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            if i % 50 == 0:
                if config.dataset_type == "MIND" and not config.embed_data_dir:
                    logging.info(f"  iter [{i}]: training loss (weight={config.weight_factor}) {loss_train.item():5.4f} {tokenizer.decode(batch['text_token_ids'][0])[6:80]} {batch['click_obs'].mean()}")
                else:
                    logging.info(f"  iter [{i}]: training loss (weight={config.weight_factor}) {loss_train.item():5.4f} {batch['click_obs'].mean()}")
                        
            epoch_loss_train += loss_train.detach().cpu().item()
            epoch_obs += num_envs 
        logging.info(f'Finished epoch {epoch}; epoch loss {epoch_loss_train/epoch_obs}')
        wandb.log({'train_loss': epoch_loss_train/epoch_obs}, step=epoch)
        if config.weight_factor != 1:
            wandb.log({'weighted_train_loss': epoch_loss_unweighted/epoch_obs}, step=epoch)

        val_start = time.time()
        all_Zreps = None
        if config.dataset_type=='MIND':
            all_Zreps, val_click_rates = eval_all_Zreps(model, val_loader, config.device, click_rates=True)
        val_loss_dict = get_val_loss_dpt(model, hist_loader, config.device, config.num_arms, all_Zreps) 
        if config.dataset_type=='MIND':
            val_loss_dict['Z_representation'] = all_Zreps
            val_loss_dict['click_rates'] = val_click_rates
        
        val_end = time.time()
        wandb.log({'epoch_val_secs': (val_end - val_start)}, step=epoch)

        val_loss = val_loss_dict['loss']

        wandb.log({'val_loss': val_loss}, step=epoch)
        
        
        logging.info(f'val_loss: {val_loss}')
        
        save_dict = {
            'state_dict':model.state_dict(),
            'optimizer': optimizer_dict,
            #'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'val_loss_dict': val_loss_dict,
            #'train_subset_loss_dict': train_subset_loss_dict,
            'config': config,
        }
        if val_loss < best_loss:
            best_loss = val_loss
            
            torch.save(save_dict, save_dir + '/best_loss.pt')

        # save latest
        torch.save(save_dict, save_dir + '/latest.pt') 
        end = time.time()
        wandb.log({'epoch_train_secs': (end - start)}, step=epoch)
    

if __name__ == '__main__':
    main()

