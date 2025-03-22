# REPLACE WITH YOURS
user=REPLACE_WITH_YOURS
base_data_dir=/shared/share_mala/implicitbayes/dataset_files/synthetic_data/bimodal
save_name=0930_dpt_bimodal_test

epochs=10000
bs=5000
eval_bs=5000
hist_bs=5000
cnts1=25
cnts2=25
num_loader_obs=500
repeat_suffstat=10
Z_dim=2
lr=0.001
num_arms=10
data_subdir="N=500,D=2500,D_eval=1000,cnts1=${cnts1},cnts2=${cnts2}"
name="${save_name}"
val_history='/shared/share_mala/implicitbayes/dataset_files/synthetic_data/bimodal/N=500,D=2500,D_eval=1000,cnts1=25,cnts2=25,forced/dpt_histories/eval_hist_num_arms=10,E=1000,H=500,S=1,seed=234234234'
for lr in 0.001 0.0001 0.01 0.1 
do
    python ./../train_models_ts_action.py --data_dir $base_data_dir/$data_subdir \
        --use_X False \
        --MLP_last_fn none \
        --val_dpt_history_data $val_history \
        --num_arms $num_arms \
        --epochs $epochs --num_loader_obs $num_loader_obs --Z_dim $Z_dim \
        --save_name $name --wandb_user $user \
        --MLP_width 100 --batch_size $bs --eval_batch_size $bs \
        --hist_batch_size $hist_bs \
        --dataset_type synthetic --learning_rate $lr --repeat_suffstat $repeat_suffstat
done
