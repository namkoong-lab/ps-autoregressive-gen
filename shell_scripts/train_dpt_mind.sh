# REPLACE WITH YOURS
user=YOUR_WANDB_USER
data_dir=/shared/share_mala/implicitbayes/dataset_files/MIND_data/filter100/
save_name=0930_dpt_mind_test

# ===

epochs=1000
bs=500
eval_bs=500
hist_bs=5000
num_loader_obs=500
repeat_suffstat=10
num_arms=10
name="${save_name}"
freezebert=False
val_history='/shared/share_mala/implicitbayes/dataset_files/MIND_data/filter100//dpt_histories/eval_hist_num_arms=10,E=1000,H=500,S=1,seed=234234234'

for lr in 0.001 0.0001 0.01 0.1 
do
    python ./../train_models_dpt.py --data_dir $data_dir \
        --freeze_bert $freezebert \
        --use_X False \
        --MLP_last_fn none \
        --val_dpt_history_data $val_history \
        --num_arms $num_arms \
        --epochs $epochs --num_loader_obs $num_loader_obs \
        --save_name $name --wandb_user $user \
        --MLP_width 100 --batch_size $bs --eval_batch_size $eval_bs --hist_batch_size $hist_bs \
        --dataset_type MIND --learning_rate $lr --repeat_suffstat $repeat_suffstat
done

