# replace with yours
user=YOUR_USERNAME
base_data_dir=/shared/share_mala/implicitbayes/dataset_files/synthetic_data/bimodal_smalltest

# =======

save_name=1001_psar_bimodal_test_small

# training params
epochs=1000
bs=500
num_loader_obs=500
ms='sequential'
repeat_suffstat=10
Z_dim=2
for D in 25 50 100 250 500 1000
do
data_subdir="N=500,D=${D},D_eval=1000,cnts1=25,cnts2=25"
for lr in 0.001 0.01 0.0001 
do
python train_models.py --data_dir $base_data_dir/$data_subdir \
    --epochs $epochs --num_loader_obs $num_loader_obs --Z_dim $Z_dim \
    --save_name $save_name --wandb_user $user --marginal_vs_sequential $ms \
    --MLP_width 100 --batch_size $bs --eval_batch_size $bs \
    --dataset_type synthetic --learning_rate $lr --repeat_suffstat $repeat_suffstat
sleep 0.2
done
done
