ec=1.0;
wl=10;

for cls in vfc_bs_ent_ue vfc bs ent ue; do
    for seed in 1 2 3; do
        # # e4090
        # CUDA_VISIBLE_DEVICES=0 python main.py --env-name DoubleDunkNoFrameskip-v4 --seed $seed --algo ppo --use-gae \
        #     --lr 2.5e-4 --clip-param 0.1 \
        #     --value-loss-coef 0.5 --num-processes 8 \
        #     --num-steps 128 --num-mini-batch 4 \
        #     --log-interval 1 --use-linear-lr-decay \
        #     --entropy-coef 0.01 \
        #     --cls ${cls} --expl_coef $ec --window_length $wl \
        #     --log-dir logs/ppo_ucb_DoubleDunk_s$seed >logs/ppo_ucb_cuda0_s$seed.log 2>&1 &
        # CUDA_VISIBLE_DEVICES=1 python main.py --env-name BattleZoneNoFrameskip-v4 --seed $seed --algo ppo --use-gae \
        #     --lr 2.5e-4 --clip-param 0.1 \
        #     --value-loss-coef 0.5 --num-processes 8 \
        #     --num-steps 128 --num-mini-batch 4 \
        #     --log-interval 1 --use-linear-lr-decay \
        #     --entropy-coef 0.01 \
        #     --cls ${cls} --expl_coef $ec --window_length $wl \
        #     --log-dir logs/ppo_ucb_BattleZone_s$seed >logs/ppo_ucb_cuda1_s$seed.log 2>&1 &
        # # h3090
        # CUDA_VISIBLE_DEVICES=0 python main.py --env-name NameThisGameNoFrameskip-v4 --seed $seed --algo ppo --use-gae \
        #     --lr 2.5e-4 --clip-param 0.1 \
        #     --value-loss-coef 0.5 --num-processes 8 \
        #     --num-steps 128 --num-mini-batch 4 \
        #     --log-interval 1 --use-linear-lr-decay \
        #     --entropy-coef 0.01 \
        #     --cls ${cls} --expl_coef $ec --window_length $wl \
        #     --log-dir logs/ppo_ucb_NameThisGame_s$seed >logs/ppo_ucb_cuda0_s$seed.log 2>&1 &
        # h4090
        CUDA_VISIBLE_DEVICES=0 python main.py --env-name PhoenixNoFrameskip-v4 --seed $seed --algo ppo --use-gae \
            --lr 2.5e-4 --clip-param 0.1 \
            --value-loss-coef 0.5 --num-processes 8 \
            --num-steps 128 --num-mini-batch 4 \
            --log-interval 1 --use-linear-lr-decay \
            --entropy-coef 0.01 \
            --cls ${cls} --expl_coef $ec --window_length $wl \
            --log-dir logs/ppo_ucb_Phoenix_s$seed >logs/ppo_ucb_cuda0_s$seed.log 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python main.py --env-name QbertNoFrameskip-v4 --seed $seed --algo ppo --use-gae \
            --lr 2.5e-4 --clip-param 0.1 \
            --value-loss-coef 0.5 --num-processes 8 \
            --num-steps 128 --num-mini-batch 4 \
            --log-interval 1 --use-linear-lr-decay \
            --entropy-coef 0.01 \
            --cls ${cls} --expl_coef $ec --window_length $wl \
            --log-dir logs/ppo_ucb_Qbert_s$seed >logs/ppo_ucb_cuda1_s$seed.log 2>&1 &
        wait
    done
done