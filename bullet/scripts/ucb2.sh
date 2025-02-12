export CUDA_VISIBLE_DEVICES=1

ec=5.0
for wl in 10 50 100; do
    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name AntBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/ant/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/ant/ \
            --device cuda:1 > logs/ucb_ant_${seed}.log 2>&1 &
    done
    # wait

    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name Walker2DBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/walker/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/walker/ \
            --device cuda:1 > logs/ucb_walker_${seed}.log 2>&1 &
    done
    wait
done

ec=1.0
for wl in 10 50 100; do
    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name AntBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/ant/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/ant/ \
            --device cuda:1 > logs/ucb_ant_${seed}.log 2>&1 &
    done
    # wait

    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name Walker2DBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/walker/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/walker/ \
            --device cuda:1 > logs/ucb_walker_${seed}.log 2>&1 &
    done
    wait
done