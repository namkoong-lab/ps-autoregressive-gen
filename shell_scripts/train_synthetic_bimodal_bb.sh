# replace with yours
user=YOUR_WANDB_USERNAME
base_data_dir=YOUR_DATA_DIR/dataset_files/synthetic_data/bimodal

# ======

save_name=bimodal_sequential_bb
data_subdir="N=500,D=2500,D_eval=1000,cnts1=25,cnts2=25"

# training params
bs=500
num_loader_obs=500
ms='sequential'
repeat_suffstat=10
Z_dim=2
sequential_beta_bernoulli=True
sequential_beta_bernoulli_alpha_beta=True
lr=0.001
epochs=1000


python ../train_models.py --data_dir $base_data_dir/$data_subdir \
    --epochs $epochs --num_loader_obs $num_loader_obs --Z_dim $Z_dim \
    --save_name $save_name --wandb_user $user --marginal_vs_sequential $ms \
    --MLP_width 100 --batch_size $bs --eval_batch_size $bs \
    --dataset_type synthetic --learning_rate $lr --sequential_beta_bernoulli $sequential_beta_bernoulli \
    --sequential_beta_bernoulli_alpha_beta $sequential_beta_bernoulli_alpha_beta


