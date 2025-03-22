# REPLACE WITH YOURS 
model_dir=YOUR_MODEL_DIR

# ===== 

T=1000
num_arms=10
context=False
bandit_dir='bandit'
alg='ts-action'
for env_idx in {0..500} 
do
    echo $env_idx
    sleep .2
    python ./../run_bandit_psar.py --model_dir $model_dir --env_idx $env_idx --T $T --num_arms $num_arms --dataset $dataset --bandit_alg $alg --context $context --bandit_dir $bandit_dir  
done



