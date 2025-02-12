export CUDA_VISIBLE_DEVICES=1
for seed in 1 2 3; do
    python main_ts.py --seed ${seed} --eta 1.0 --window_length 50 --env-name AntBulletEnv-v0 --num-env-steps 1000000 \
        --log-dir logs/ts/ant/ --save-dir logs/ts/ant/ \
        --device cuda:1 > logs/ts_ant_${seed}.log 2>&1 &
done
wait