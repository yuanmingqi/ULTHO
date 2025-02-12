export CUDA_VISIBLE_DEVICES=1
for seed in 1 2 3; do
    python main_rr.py --seed ${seed} --env-name AntBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/rr/ant/ --save-dir logs/rr/ant/ \
        --device cuda:1 > logs/rr_ant_${seed}.log 2>&1 &
done
# wait

for seed in 1 2 3; do
    python main_rr.py --seed ${seed} --env-name Walker2DBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/rr/walker/ --save-dir logs/rr/walker/ \
        --device cuda:1 > logs/rr_walker_${seed}.log 2>&1 &
done
wait

