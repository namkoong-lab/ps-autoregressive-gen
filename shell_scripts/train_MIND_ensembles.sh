# REPLACE WITH YOURS
user=YOUR_WANDB_USERNAME
# directory of an existing trained marginal model
data_dir=YOUR_DATA_DIR
# use embeddings from trained marginal model
embed_data_dir=True

marginal_vs_sequential='marginal'
save_name='ensembles_randprior_withscale'
MLP_width=100
MLP_layer=3
batch_size=100
repeat_suffstat=1
eval_batch_size=100
transform_success_p_alpha=1
lr=1e-5
bert_lr=1e-8
wd=1e-2
sample_frac=1
num_loader_obs=500
num_loader_obs_train=500
epochs=50
rand_prior=0
for prior_scale in 1 10 100 
do
    for boot_seed in {0..50} 
do
    sleep 0.2
python ../train_models.py --save_name $save_name --wandb_user $user --data_dir $data_dir --marginal_vs_sequential $marginal_vs_sequential --MLP_width $MLP_width --MLP_layer $MLP_layer --batch_size $batch_size --repeat_suffstat $repeat_suffstat --eval_batch_size $eval_batch_size --transform_success_p_alpha 1 --learning_rate $lr --bert_learning_rate $bert_lr --weight_decay $wd --sample_frac $sample_frac --num_loader_obs $num_loader_obs --num_loader_obs_train $num_loader_obs_train --epochs $epochs --embed_data_dir $embed_data_dir --bootstrap_seed $boot_seed --seed $boot_seed --rand_prior $rand_prior --prior_scale $prior_scale
done
done
