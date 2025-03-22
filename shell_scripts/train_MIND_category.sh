user=YOUR_WANDB_USERNAME
save_name=train_mind_category
data_dir='YOUR_DATA_PATH/MIND_data/filter100'

# training params
ms='sequential'
repeat_suffstat=10
num_loader_obs=500
bs=100
MLP_width=50
lr=0.001
epochs=1000

python ../train_models.py --data_dir $data_dir \
        --epochs $epochs --num_loader_obs $num_loader_obs \
        --save_name $save_name --wandb_user $user --marginal_vs_sequential $ms \
        --MLP_width $MLP_width --batch_size $bs --eval_batch_size $bs --use_text 0 --use_category 1\
        --dataset_type 'MIND' --learning_rate $lr --repeat_suffstat $repeat_suffstat

