
This is the code for the experiments in the paper. 

# Requirements
**Python environment**

    requirements.txt

**Hardware?**

Everything here we have run on CPU, except where specified. There, each used a single NVIDIA A40 GPU. 

# Create datasets
For all of these, the shell scripts must be modified to use your own data directories. 

**Synthetic (bimodal) dataset**
This data is generated. 

    ./shell_scripts/generate_data_bimodal.sh

**MIND (news) dataset**
Some data is downloaded in the script. Then, we post-process it. 

    ./shell_scripts/generate_data_MIND.sh

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

**PS-AR**

    ./shell_scripts/run_bandits_psar.sh

**SquareCB**

    ./shell_scripts/run_bandits_squarecb.sh


# Notebooks
**Synthetic (bimodal) dataset**

    synthetic_posteriors.ipynb
    synthetic_bandit.ipynb

**MIND (news) dataset**

    MIND_posterior.ipynb
    MIND_bandit.ipynb

# Empirical Bayes example
**Generate data**

    ./shell_scripts/generate_data_beta.sh

**Train**

    ./shell_scripts/train_synthetic_beta_bb.sh

Then, to generate posterior samples (posterior samples with no observations are prior samples)
run 

    postprocessing.py

on model outputs, to get samples from the prior. Then, for plotting, see

    empirical_bayes.ipynb

# Comparison of PS-AR generation horizons
In the Appendix, we discuss variations on PS-AR: one in which we impute any unobserved outcomes, and another in which we generate a fixed number of imagined outcomes, both in order to construct $\hat\mu^{(a)}$. We compare these in the Appendix. To compute regret, 

    ./shell_scripts/run_bandits_psar_compare_horizon.sh

For plotting, see

    MIND_bandit_psar_variant_comparison.ipynb

# TS-Action: modeling best action, rather than reward
Our code here is very similar to before, but models the best action rather than the reward.

**Generate histories** 

    generate_DPT_histories_synthetic.ipynb

**Train models**

    ./shell_scripts/train_dpt_bimodal.sh

**bandit**
    
    ./shell_scripts/run_bandit_dpt.sh

# Comparing TS-PSAR to TS-Action on smaller datasets
**Generate smaller datasets**

    ./shell_scripts/generate_data_bimodal_small.sh

**Train models**

    ./shell_scripts/train_dpt_bimodal_small.sh
    ./shell_scripts/train_synthetic_bimodal_small.sh

**Bandit**

    No specific scripts, but you run the bandit scripts on the trained models. 

# Details for using weights and biases
I run weights and biases configs using

    wandb sweep SOMETHING.yaml

which then outputs a sweep agent name, which I use as input to
    
    wandb_agent_gpus.py

which is run on a single machine, which can have multiple GPUs. 
