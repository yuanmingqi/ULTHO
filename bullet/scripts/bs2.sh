export CUDA_VISIBLE_DEVICES=1
for seed in 1 2 3; do
    python main.py --seed ${seed} --env-name AntBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/bs/ant/ --save-dir logs/bs/ant/ \
        --device cuda:0 > logs/bs_ant_${seed}.log 2>&1 &
done
# wait

for seed in 1 2 3; do
    python main.py --seed ${seed} --env-name Walker2DBulletEnv-v0 --num-env-steps 2000000 \
        --log-dir logs/bs/walker/ --save-dir logs/bs/walker/ \
        --device cuda:1 > logs/bs_walker_${seed}.log 2>&1 &
done
wait

