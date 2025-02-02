import os, difflib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_and_save_event_data(event_file, output_file):
    # 加载 TensorBoard 事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    # 定义要提取的标签
    tags = [
        "losses/value_loss",
        "losses/policy_loss",
        "losses/entropy",
        "losses/old_approx_kl",
        "losses/approx_kl",
        "losses/clipfrac",
        "losses/explained_variance",
        "charts/learning_rate",
        "charts/SPS",
        "charts/episodic_return",
        "charts/episodic_length",
        "h2o/cluster_idx",
        "h2o/sub_option_idx",
        "h2o/sub_option"
    ]
    
    # 初始化存储数据的字典
    data = {tag: [] for tag in tags}
    steps = {tag: [] for tag in tags}
    
    # 遍历所有标签并提取数据
    for tag in tags:
        if tag in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(tag)
            steps[tag] = [event.step for event in events]
            data[tag] = [event.value for event in events]
        else:
            print(f"Warning: Tag {tag} not found in the event file.")
    
    # 将数据保存为 npz 文件
    np.savez(output_file, steps=steps, data=data)
    print(f"Data extracted and saved to {output_file}")
root = 'ucb_5321'

envs = ['bigfish', 'bossfight', 'caveflyer', 'chaser', 'climber', 'coinrun', 'dodgeball', 'fruitbot',
        'heist', 'jumper', 'leaper', 'maze', 'miner', 'ninja', 'plunder', 'starpilot'
        ]
for env in envs:
    for dir in os.listdir(root):
        if env in dir:
            file = os.path.join(root, dir)
            event_file = difflib.get_close_matches("events.out.tfevents", os.listdir(file), 1, 0.1)[0]
            event_file = os.path.join(file, event_file)
            extract_and_save_event_data(event_file, f"{root}/{dir}.npz")