# REPLACE WITH YUORS
user=YOUR_WANDB_USERNAME
base_data_dir=/shared/share_mala/implicitbayes/dataset_files/synthetic_data/beta/

# =====

save_name=beta_sequential_bb

# training params
epochs=2000
bs=500
cnts=5
num_loader_obs=500
ms='sequential'
repeat_suffstat=10
sequential_beta_bernoulli=True
sequential_beta_bernoulli_alpha_beta=True
lr=0.001
data_subdir="N=500,D=25000,D_eval=10000,cnts=${cnts}"
python train_models.py --data_dir $base_data_dir/$data_subdir \
        --epochs $epochs --num_loader_obs $num_loader_obs \
        --save_name $save_name --wandb_user $user --marginal_vs_sequential $ms \
        --MLP_width 100 --batch_size $bs --eval_batch_size $bs \
        --dataset_type synthetic --learning_rate $lr --repeat_suffstat $repeat_suffstat \
        --sequential_beta_bernoulli $sequential_beta_bernoulli --sequential_beta_bernoulli_alpha_beta $sequential_beta_bernoulli_alpha_beta


