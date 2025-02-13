for cls in lr vfc ent lr_vfc lr_ent vfc_ent lr_vfc_ent; do
    for seed in 1 2 3; do
        # h4090
        CUDA_VISIBLE_DEVICES=0 python a2c_ucb.py --env-name PhoenixNoFrameskip-v4 --seed $seed \
            --cls ${cls} --expl_coef 1.0 --window_length 10 \
            --log-dir logs/a2c_ucb_${cls}_Phoenix_s$seed >logs/a2c_ucb_${cls}_cuda0_s$seed.log 2>&1 &

        CUDA_VISIBLE_DEVICES=1 python a2c_ucb.py --env-name QbertNoFrameskip-v4 --seed $seed \
            --cls ${cls} --expl_coef 1.0 --window_length 10 \
            --log-dir logs/a2c_ucb_${cls}_Qbert_s$seed >logs/a2c_ucb_${cls}_cuda1_s$seed.log 2>&1 &

        # h3090
        CUDA_VISIBLE_DEVICES=0 python a2c_ucb.py --env-name NameThisGameNoFrameskip-v4 --seed $seed \
            --cls ${cls} --expl_coef 1.0 --window_length 10 \
            --log-dir logs/a2c_ucb_${cls}_NameThisGame_s$seed >logs/a2c_ucb_${cls}_cuda0_s$seed.log 2>&1 &

        # e4090
        CUDA_VISIBLE_DEVICES=1 python a2c_ucb.py --env-name DoubleDunkNoFrameskip-v4 --seed $seed \
            --cls ${cls} --expl_coef 1.0 --window_length 10 \
            --log-dir logs/a2c_ucb_${cls}_DoubleDunk_s$seed >logs/a2c_ucb_${cls}_cuda1_s$seed.log 2>&1 &

        CUDA_VISIBLE_DEVICES=0 python a2c_ucb.py --env-name BattleZoneNoFrameskip-v4 --seed $seed \
            --cls ${cls} --expl_coef 1.0 --window_length 10 \
            --log-dir logs/a2c_ucb_${cls}_BattleZone_s$seed >logs/a2c_ucb_${cls}_cuda0_s$seed.log 2>&1 &
        wait
    done
done
