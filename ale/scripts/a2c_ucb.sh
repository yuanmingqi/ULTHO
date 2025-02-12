for seed in 1 2 3; do
    # h4090
    CUDA_VISIBLE_DEVICES=0 python a2c_ucb.py --env-name PhoenixNoFrameskip-v4 --seed $seed \
        --log-dir logs/a2c_ucb_Phoenix_s$seed >logs/a2c_ucb_cuda0_s$seed.log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 python a2c_ucb.py --env-name QbertNoFrameskip-v4 --seed $seed \
        --log-dir logs/a2c_ucb_Qbert_s$seed >logs/a2c_ucb_cuda1_s$seed.log 2>&1 &
    # h3090
    CUDA_VISIBLE_DEVICES=0 python a2c_ucb.py --env-name NameThisGameNoFrameskip-v4 --seed $seed \
        --log-dir logs/a2c_ucb_NameThisGame_s$seed >logs/a2c_ucb_cuda0_s$seed.log 2>&1 &
    # e4090
    CUDA_VISIBLE_DEVICES=1 python a2c_ucb.py --env-name DoubleDunkNoFrameskip-v4 --seed $seed \
        --log-dir logs/a2c_ucb_DoubleDunk_s$seed >logs/a2c_ucb_cuda1_s$seed.log 2>&1 &

    CUDA_VISIBLE_DEVICES=0 python a2c_ucb.py --env-name BattleZoneNoFrameskip-v4 --seed $seed \
        --log-dir logs/a2c_ucb_BattleZone_s$seed >logs/a2c_ucb_cuda0_s$seed.log 2>&1 &
    wait
done
