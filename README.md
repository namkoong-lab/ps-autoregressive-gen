
This is the code for the experiments in the paper "Active Exploration via Autoregressive Generation of Missing Data" (https://arxiv.org/abs/2405.19466). 

# Requirements
**Python environment**

    requirements.txt

**Hardware?**

Everything here we have run on CPU, except where specified. There, each used a single NVIDIA A40 GPU. 

# Create datasets
See instructions in the data_prep folder!

# Train autoregressive sequence models $p_\theta$
For all of these, the shell scripts and/or config files must be modified to use your own data directories and weights and biases (wandb) username. 
The trained models are saved in subdirectories inside the data directories. 

**Synthetic (bimodal) dataset**
To train flexible neural net sequence models on synthetic dataset: run 

    ./shell_scripts/train_synthetic_bimodal_flex_nn.sh

To train beta-bernoulli neural net sequence models on synthetic dataset: run 

    ./shell_scripts/train_synthetic_bimodal_bb.sh

**MIND (news) dataset**
To train flexible neural net sequence models on MIND dataset, using category features: run 

    ./shell_scripts/train_MIND_category.sh

To train flexible neural net sequence models on MIND datset, using text features: run the following on GPU, using the weights and biases config in

    wandb_conf/train_MIND_text_flex_nn.yaml

To train beta-bernoulli neural net sequence models on MIND dataset, using text features: run the following on GPU, using the weights and biases config in

    wandb_conf/train_MIND_text_bb.yaml


# Comparison with ensembling
The shell scripts must be modified to use the directories containing the sequential models trained in the previous step. 

First, train BERT features, which we'll keep fixed; the MLP heads on top will be trained on bootstrapped data. 

    wandb_confs/train_MIND_text_marginal.yaml

Then we train a 50 randomly initialized MLP heads on bootstrapped subsets of the data, fixing the BERT features from before. Please parallelize appropriately according to your resources. 

    ./shell_scripts/train_MIND_ensembles.sh

# Run bandit algorithms
We run and cache some bandit algorithms, with commands listed here. Others (that are faster) we run directly alongside plotting in a jupyter notebook (see next section). 
* For all of these, the shell scripts must be modified to use the directories containing the sequential models trained in the previous step. 
* In these shell scripts, we have a for-loop over many environments. Please parallelize appropriately according to your resources. 
* Adjust number of timesteps T as appropriate for the setting

    ./shell_scripts/run_bandits_psar.sh


# Notebooks
**Synthetic (bimodal) dataset**

    synthetic_posteriors.ipynb
    synthetic_bandit.ipynb

**MIND (news) dataset**

    MIND_posterior.ipynb
    MIND_bandit.ipynb

# Extras
Additional code for experiments in the appendix are in the extras folder

# Details for using weights and biases
We run weights and biases configs using

    wandb sweep SOMETHING.yaml

which then outputs a sweep agent name, which we use as input to
    
    wandb_agent_gpus.py

which is run on a single machine. 
