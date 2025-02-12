for seed in 1 2 3; do
    # python a2c_atari_envpool.py --env_id BattleZone-v5   --seed $seed --device cuda:0 > logs/ae0_$seed.log 2>&1 &
    # python a2c_atari_envpool.py --env_id DoubleDunk-v5   --seed $seed --device cuda:1 > logs/ae1_$seed.log 2>&1 &
    # python a2c_atari_envpool.py --env_id NameThisGame-v5 --seed $seed --device cuda:0 > logs/ae2_$seed.log 2>&1 &
    python a2c_atari_envpool.py --env_id Phoenix-v5      --seed $seed --device cuda:0 > logs/ae3_$seed.log 2>&1 &
    python a2c_atari_envpool.py --env_id Qbert-v5        --seed $seed --device cuda:1 > logs/ae4_$seed.log 2>&1 &
    wait
done
