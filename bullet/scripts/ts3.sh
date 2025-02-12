export CUDA_VISIBLE_DEVICES=1
for seed in 1 2 3; do
    python main_ts.py --seed ${seed} --eta 1.0 --window_length 50 --env-name Walker2DBulletEnv-v0 --num-env-steps 1000000 \
        --log-dir logs/ts/walker/ --save-dir logs/ts/walker/ \
        --device cuda:1 > logs/ts_walker_${seed}.log 2>&1 &
done
wait