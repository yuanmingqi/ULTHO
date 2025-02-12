for seed in 1 2 3;
do
    python ppo_procgen_ucb.py --env_id leaper --seed ${seed} --device cuda:0 > logs/cuda0_${seed}.log 2>&1 &
    python ppo_procgen_ucb.py --env_id maze   --seed ${seed} --device cuda:1 > logs/cuda1_${seed}.log 2>&1 &
    python ppo_procgen_ucb.py --env_id miner  --seed ${seed} --device cuda:0 > logs/cuda2_${seed}.log 2>&1 &
    python ppo_procgen_ucb.py --env_id ninja  --seed ${seed} --device cuda:1 > logs/cuda4_${seed}.log 2>&1 &
    wait
    python ppo_procgen_ucb.py --env_id caveflyer  --seed ${seed} --device cuda:0 > logs/cuda5_${seed}.log 2>&1 &
    python ppo_procgen_ucb.py --env_id chaser  --seed ${seed} --device cuda:1 > logs/cuda6_${seed}.log 2>&1 &
    python ppo_procgen_ucb.py --env_id coinrun   --seed ${seed} --device cuda:0 > logs/cuda7_${seed}.log 2>&1 &
    wait 
done