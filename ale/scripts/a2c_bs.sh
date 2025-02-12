for seed in 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 python main.py --env-name PhoenixNoFrameskip-v4 --seed $seed \
	    --log-dir logs/a2c_bs_Phoenix_s$seed > logs/a2c_bs_cuda0_s$seed.log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 python main.py --env-name QbertNoFrameskip-v4   --seed $seed \
	    --log-dir logs/a2c_bs_Qbert_s$seed   > logs/a2c_bs_cuda1_s$seed.log 2>&1 &
    wait
done
