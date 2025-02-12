for seed in 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 python main.py --env-name PhoenixNoFrameskip-v4 --seed $seed --algo ppo --use-gae \
        --lr 2.5e-4 --clip-param 0.1 \
        --value-loss-coef 0.5 --num-processes 8 \
        --num-steps 128 --num-mini-batch 4 \
        --log-interval 1 --use-linear-lr-decay \
        --entropy-coef 0.01 \
        --log-dir logs/ppo_bs_Phoenix_s$seed >logs/ppo_bs_cuda0_s$seed.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python main.py --env-name QbertNoFrameskip-v4 --seed $seed --algo ppo --use-gae \
        --lr 2.5e-4 --clip-param 0.1 \
        --value-loss-coef 0.5 --num-processes 8 \
        --num-steps 128 --num-mini-batch 4 \
        --log-interval 1 --use-linear-lr-decay \
        --entropy-coef 0.01 \
        --log-dir logs/ppo_bs_Qbert_s$seed >logs/ppo_bs_cuda1_s$seed.log 2>&1 &
    wait
done
