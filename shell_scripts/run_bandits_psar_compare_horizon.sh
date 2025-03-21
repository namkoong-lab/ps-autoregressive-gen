# replace with yours
model_dir=YOUR_TRAINED_MODEL_DIR

T=1000
num_arms=10

for bandit_alg in 'sequential_horizon' 'sequential'
do
for env_idx in {0..500} 
do
    echo $env_idx
    python run_bandit_psar.py --model_dir $model_dir --env_idx $env_idx --T $T --num_arms $num_arms --bandit_alg $bandit_alg --horizonDependent 1
done
done
