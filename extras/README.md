
This is the code for the appendix experiments in the paper. To use this code, please move the python files in this folder into the main directory. We separated out these secondary experiments to make reproducing the main paper experiments clearer.

# Empirical Bayes example
**Generate data**

    ./shell_scripts/extras/generate_data_beta.sh

**Train**

    ./shell_scripts/extras/train_synthetic_beta_bb.sh

Then, to generate posterior samples (posterior samples with no observations are prior samples)
run 

    postprocessing.py

on model outputs, to get samples from the prior. Then, for plotting, see

    extras/empirical_bayes.ipynb

# Comparison of PS-AR generation horizons
In the Appendix, we discuss variations on PS-AR: one in which we impute any unobserved outcomes, and another in which we generate a fixed number of imagined outcomes, both in order to construct $\hat\mu^{(a)}$. We compare these in the Appendix. To compute regret, 

    ./shell_scripts/extras/run_bandits_psar_compare_horizon.sh

For plotting, see

    extras/MIND_bandit_psar_variant_comparison.ipynb

# TS-Action: modeling best action, rather than reward
Our code here is very similar to before, but models the best action rather than the reward.

**Generate histories** 

    extras/generate_ts_action_histories_synthetic.ipynb

**Train models**

    ./shell_scripts/extras/train_dpt_bimodal.sh

**bandit**
    
    ./shell_scripts/extras/run_bandit_dpt.sh

# Comparing TS-PSAR to TS-Action on smaller datasets
**Generate smaller datasets**

    ./shell_scripts/extras/generate_data_bimodal_small.sh

**Train models**

    ./shell_scripts/extras/train_dpt_bimodal_small.sh
    ./shell_scripts/extras/train_synthetic_bimodal_small.sh

**Bandit**

    No specific scripts, but you run the bandit scripts on the trained models. 


