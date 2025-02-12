export CUDA_VISIBLE_DEVICES=0
for seed in 1 2 3; do
    python main_ts.py --seed ${seed} --eta 1.0 --window_length 50 --env-name HopperBulletEnv-v0 --num-env-steps 1000000 \
        --log-dir logs/ts/hopper/ --save-dir logs/ts/hopper/ \
        --device cuda:0 > logs/ts_hopper_${seed}.log 2>&1 &
done
wait