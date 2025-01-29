import numpy as np
import random
import torch
import gym, os

from procgen import ProcgenEnv
from ppo_procgen import Agent

envs = [
    'bigfish', 'bossfight', 'caveflyer', 'chaser', 
        # 'climber', 'coinrun', 'dodgeball', 'fruitbot',
        # 'heist', 'jumper', 'leaper', 'maze', 
        # 'miner', 'ninja', 'plunder', 'starpilot'
        ]
score_dict = {env: [] for env in envs}
best_ucb = {'bigfish': 'ppo_ucb_runs_c=5.0_w=10', 'bossfight': 'ppo_ucb_runs_c=1.0_w=10', 'caveflyer': 'ppo_ucb_runs_c=5.0_w=10', 'chaser': 'ppo_ucb_runs_c=5.0_w=10', 
            'climber': 'ppo_ucb_runs_c=5.0_w=10', 'coinrun': 'ppo_ucb_runs_c=5.0_w=100', 'dodgeball': 'ppo_ucb_runs_c=5.0_w=100', 'fruitbot': 'ppo_ucb_runs_c=5.0_w=100', 
            'heist': 'ppo_ucb_runs_c=5.0_w=100', 'jumper': 'ppo_ucb_runs_c=5.0_w=100', 'leaper': 'ppo_ucb_runs_c=5.0_w=100', 'maze': 'ppo_ucb_runs_c=5.0_w=10', 
            'miner': 'ppo_ucb_runs_c=5.0_w=10', 'ninja': 'ppo_ucb_runs_c=5.0_w=100', 'plunder': 'ppo_ucb_runs_c=5.0_w=10', 'starpilot': 'ppo_ucb_runs_c=5.0_w=10'}

best_ts = {'bigfish': 'ppo_ts_runs_w50', 'bossfight': 'ppo_ts_runs_w5', 'caveflyer': 'ppo_ts_runs_w50', 'chaser': 'ppo_ts_runs_w5', 
           'climber': 'ppo_ts_runs_w100', 'coinrun': 'ppo_ts_runs_w50', 'dodgeball': 'ppo_ts_runs_w100', 'fruitbot': 'ppo_ts_runs_w50', 
           'heist': 'ppo_ts_runs_w50', 'jumper': 'ppo_ts_runs_w100', 'leaper': 'ppo_ts_runs_w100', 'maze': 'ppo_ts_runs_w100', 
           'miner': 'ppo_ts_runs_w50', 'ninja': 'ppo_ts_runs_w100', 'plunder': 'ppo_ts_runs_w50', 'starpilot': 'ppo_ts_runs_w10'}

for env_name in envs:
    # model_dir = 'ppo_rr_runs'
    model_dir = best_ucb[env_name]
    for seed in range(1, 4):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        device = torch.device('cuda:0')
        envs = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=0, start_level=0, distribution_mode="easy")
        envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space["rgb"]
        envs.is_vector_env = True
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        envs = gym.wrappers.NormalizeReward(envs, gamma=0.999)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

        agent = Agent(envs).to(device)
        
        for file in os.listdir(model_dir):
            if env_name in file and f'__{seed}__' in file and '.npz' not in file:
                model_file = f'{model_dir}/{file}/model.pth'
                print(env_name, seed, model_file)
                break

        model = torch.load(model_file, map_location=device)
        agent.load_state_dict(model)

        next_obs = torch.Tensor(envs.reset()).to(device)

        eps_return = []

        while len(eps_return) < 100:
            action = agent.act(next_obs)
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)

            for item in info:
                if "episode" in item.keys():
                    r = item["episode"]["r"]
                    eps_return.append(r)
                    break

        print(env_name, np.mean(eps_return))
        score_dict[env_name].append(np.mean(eps_return))

print(score_dict)