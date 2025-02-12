export CUDA_VISIBLE_DEVICES=0
for seed in 1 2 3; do
    python main_rr.py --seed ${seed} --env-name HalfCheetahBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/rr/halfcheetah/ --save-dir logs/rr/halfcheetah/ \
        --device cuda:0 > logs/rr_halfcheetah_${seed}.log 2>&1 &
done
# wait

for seed in 1 2 3; do
    python main_rr.py --seed ${seed} --env-name HopperBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/rr/hopper/ --save-dir logs/rr/hopper/ \
        --device cuda:0 > logs/rr_hopper_${seed}.log 2>&1 &
done
wait

