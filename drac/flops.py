import copy
import os
import time
from collections import deque

import numpy as np
import torch

from ucb_rl2_meta import algo, utils
from ucb_rl2_meta.model import Policy, AugCNN
from ucb_rl2_meta.storage import RolloutStorage
from test import evaluate

from baselines import logger

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from ucb_rl2_meta.envs import VecPyTorchProcgen, TransposeImageProcgen
from ucb_rl2_meta.arguments import parser
import data_augs
from thop import profile, clever_format


aug_to_func = {    
        'crop': data_augs.Crop,
        'random-conv': data_augs.RandomConv,
        'grayscale': data_augs.Grayscale,
        'flip': data_augs.Flip,
        'rotate': data_augs.Rotate,
        'cutout': data_augs.Cutout,
        'cutout-color': data_augs.CutoutColor,
        'color-jitter': data_augs.ColorJitter,
}

env_to_aug = {
    'bigfish': 'crop',
    'starpilot': 'crop',
    'fruitbot': 'crop',
    'bossfight': 'flip',
    'ninja': 'color-jitter',
    'plunder': 'crop',
    'caveflyer': 'rotate',
    'coinrun': 'random-conv',
    'jumper': 'random-conv',
    'chaser': 'crop',
    'climber': 'color-jitter',
    'dodgeball': 'crop',
    'heist': 'crop',
    'leaper': 'crop',
    'maze': 'crop',
    'miner': 'color-jitter'
}

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)
    device = torch.device(args.device)

    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name, \
        num_levels=args.num_levels, start_level=args.start_level, \
        distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)
    
    obs_shape = envs.observation_space.shape
    actor_critic = Policy(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size})        
    actor_critic.to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size,
                                aug_type=args.aug_type, split_ratio=args.split_ratio)
        
    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    aug_id = data_augs.Identity
    _aug_type = env_to_aug[args.env_name]
    print(args.env_name, 'using', _aug_type)
    if _aug_type == 'color-jitter':
        aug_func = aug_to_func[_aug_type](batch_size=batch_size, device=device)
    else:
        aug_func = aug_to_func[_aug_type](batch_size=batch_size)

    agent = algo.DrAC(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        aug_id=aug_id,
        aug_func=aug_func,
        aug_coef=args.aug_coef,
        env_name=args.env_name)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(1):
        actor_critic.train()
        for step in range(1):
            # Sample actions
            with torch.no_grad():
                obs_id = aug_id(rollouts.obs[step])
                # obs_id = torch.rand(size=(2048,3,64,64)).to(args.device)
                # value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                #     obs_id, rollouts.recurrent_hidden_states[step],
                #     rollouts.masks[step])
                macs, params = profile(actor_critic, inputs=(obs_id[0].unsqueeze(0), rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]))
                # macs, params = clever_format([macs, params], "%.3f")
                print(obs_id.shape, macs, params)

                # macs, params = profile(actor_critic, inputs=(obs_id, rollouts.recurrent_hidden_states[step],
                #     rollouts.masks[step]))
                # print(obs_id.shape, macs, params)

        #     # Obser reward and next obs
        #     obs, reward, done, infos = envs.step(action)
            
        #     for info in infos:
        #         if 'episode' in info.keys():
        #             episode_rewards.append(info['episode']['r'])

        #     # If done then clean the history of observations.
        #     masks = torch.FloatTensor(
        #         [[0.0] if done_ else [1.0] for done_ in done])
        #     bad_masks = torch.FloatTensor(
        #         [[0.0] if 'bad_transition' in info.keys() else [1.0]
        #          for info in infos])

        #     rollouts.insert(obs, recurrent_hidden_states, action,
        #                     action_log_prob, value, reward, masks, bad_masks)

        # with torch.no_grad():
        #     obs_id = aug_id(rollouts.obs[-1])
        #     next_value = actor_critic.get_value(
        #         obs_id, rollouts.recurrent_hidden_states[-1],
        #         rollouts.masks[-1]).detach()
            
        # rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # if args.use_ucb and j > 0:
        #     agent.update_ucb_values(rollouts)
        # value_loss, action_loss, dist_entropy = agent.update(rollouts)    
        # rollouts.after_update()

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
