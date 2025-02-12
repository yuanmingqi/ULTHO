export CUDA_VISIBLE_DEVICES=0

ec=5.0
for wl in 10 50 100; do
    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name HalfCheetahBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/halfcheetah/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/halfcheetah/ \
            --device cuda:0 > logs/ucb_halfcheetah_${seed}.log 2>&1 &
    done
    # wait

    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name HopperBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/hopper/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/hopper/ \
            --device cuda:0 > logs/ucb_hopper_${seed}.log 2>&1 &
    done
    wait
done

ec=1.0
for wl in 10 50 100; do
    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name HalfCheetahBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/halfcheetah/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/halfcheetah/ \
            --device cuda:0 > logs/ucb_halfcheetah_${seed}.log 2>&1 &
    done
    # wait

    for seed in 1 2 3; do
        python main_ucb.py --seed ${seed} --window_length ${wl} --expl_coef ${ec} \
            --env-name HopperBulletEnv-v0 --num-env-steps 2000000 \
            --log-dir logs/ppo_ucb_w=${wl}_c=${ec}/hopper/ --save-dir logs/ppo_ucb_w=${wl}_c=${ec}/hopper/ \
            --device cuda:0 > logs/ucb_hopper_${seed}.log 2>&1 &
    done
    wait
done
