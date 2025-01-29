import numpy as np
import torch, random

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from ucb_rl2_meta.envs import VecPyTorchProcgen
from ucb_rl2_meta.model import Policy

def evaluate(env_name, model, device):
    num_processes = 1
    
    # Sample Levels From the Full Distribution 
    venv = ProcgenEnv(num_envs=num_processes, env_name=env_name, \
        num_levels=0, start_level=0, \
        distribution_mode='easy')
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    eval_envs = VecPyTorchProcgen(venv, device)

    obs_shape = eval_envs.observation_space.shape
    actor_critic = Policy(
        obs_shape,
        15,
        base_kwargs={'recurrent': False, 'hidden_size': 256})        
    actor_critic.load_state_dict(model['model_state_dict'])
    actor_critic.to(device)
    
    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.ones(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)

        obs, _, done, infos = eval_envs.step(action)
         
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return np.mean(eval_episode_rewards)

best_ucb = {'bigfish': 'drac_ucb_c=5.0_w=50', 'bossfight': 'drac_ucb_c=5.0_w=100', 'caveflyer': 'drac_ucb_c=5.0_w=50', 
            'chaser': 'drac_ucb', 'climber': 'drac_ucb_c=5.0_w=100', 'coinrun': 'drac_ucb', 'dodgeball': 'drac_ucb_c=5.0_w=100', 
            'fruitbot': 'drac_ucb_c=1.0_w=10', 'heist': 'drac_ucb', 'jumper': 'drac_ucb_c=5.0_w=100', 'leaper': 'drac_ucb_c=0.1_w=10', 
            'maze': 'drac_ucb', 'miner': 'drac_ucb', 'ninja': 'drac_ucb_c=5.0_w=50', 'plunder': 'drac_ucb_c=1.0_w=10', 'starpilot': 'drac_ucb'}
best_ts = {'bigfish': 'drac_ts_w=50_eta=1.0', 'bossfight': 'drac_ts', 'caveflyer': 'drac_ts_w=50_eta=1.0', 'chaser': 'drac_ts_w=50_eta=0.5',
            'climber': 'drac_ts_w=50_eta=0.1', 'coinrun': 'drac_ts_w=50_eta=1.0', 'dodgeball': 'drac_ts_w=50_eta=0.5', 'fruitbot': 'drac_ts_w=50_eta=0.1', 
            'heist': 'drac_ts_w=50_eta=0.1', 'jumper': 'drac_ts_w=50_eta=0.1', 'leaper': 'drac_ts_w=50_eta=0.1', 'maze': 'drac_ts_w=50_eta=0.1', 
            'miner': 'drac_ts_w=50_eta=0.1', 'ninja': 'drac_ts_w=100_eta=1.0', 'plunder': 'drac_ts_w=50_eta=0.5', 'starpilot': 'drac_ts'}


if __name__ == '__main__':
    envs = [
            # 'bigfish', 'bossfight', 'caveflyer', 'chaser', 
            # 'climber', 'coinrun', 'dodgeball', 'fruitbot', 
            'heist', 'jumper', 'leaper', 'maze', 
            # 'miner', 'ninja', 'plunder', 'starpilot'
        ]
    tag = 'ts'
    device = 'cuda:2'
    score_dict = {env: [] for env in envs}
    for env_name in envs:
         # model_dir = 'logs_drac_bs'
        # model_dir =  'logs_'+best_ucb[env_name]
        model_dir = 'logs_'+best_ts[env_name]
        for seed in range(1, 4):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            device = torch.device(device)
            model_file = f'{model_dir}/agent-drac_{tag}-{env_name}-reproduce-s{seed}.pt'
            model = torch.load(model_file, map_location=device)

            eps_return = evaluate(env_name, model, device)
            score_dict[env_name].append(np.mean(eps_return))

            print(env_name, model_file, eps_return)
    
    print(score_dict)