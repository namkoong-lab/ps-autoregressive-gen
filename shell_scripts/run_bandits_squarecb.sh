# replace with yours
model_dir='/shared/share_mala/implicitbayes/dataset_files/MIND_data/filter100/models/seq_bert_rerun/sequential:epochs=500,bs=100,lr=1e-05,wd=0.01,MLP_layers=3,MLP_width=100,weight_factor=1,max_obs=500,repeat_suffstat=100,bert_lr=1e-08,sample_frac=1.0,p_alpha=1,seed=2340923'

bandit_alg='squarecb'
T=2000
num_arms=10
for gamma0 in 10 50 100
do
for rho in 0.25 0.5
do
for env_idx in {0..500} 
do
    echo $env_idx
    python run_bandit_squarecb.py --model_dir $model_dir --env_idx $env_idx --T $T --num_arms $num_arms --rho $rho --gamma0 $gamma0
done
done
done
