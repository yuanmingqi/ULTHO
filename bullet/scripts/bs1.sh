export CUDA_VISIBLE_DEVICES=0
for seed in 1 2 3; do
    python main.py --seed ${seed} --env-name HalfCheetahBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/bs/halfcheetah/ --save-dir logs/bs/halfcheetah/ \
        --device cuda:0 > logs/bs_halfcheetah_${seed}.log 2>&1 &
done
# wait

for seed in 1 2 3; do
    python main.py --seed ${seed} --env-name HopperBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/bs/hopper/ --save-dir logs/bs/hopper/ \
        --device cuda:1 > logs/bs_hopper_${seed}.log 2>&1 &
done
wait

