# replace with yours
model_dir=PSAR_TRAINED_MODEL_DIRECTORY

# We can truncate T in this setting without any change in results
T=2000
num_arms=10
num_imagined=500

for env_idx in {0..500}
do
    echo $env_idx
    python run_bandit_psar.py --model_dir $model_dir --env_idx $env_idx --T $T --num_arms $num_arms --num_imagined $num_imagined
done
