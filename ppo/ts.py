import numpy as np
from collections import deque

class GaussianThompsonSampling:
    def __init__(self, 
                 param_values, 
                 init_mean=0., 
                 init_std=1.,
                 window_size=10,
                 lr=1.0
                 ):
        """
        初始化：候选超参数及其奖励分布（高斯分布）
        param_values: 超参数的候选值列表
        init_mean: 奖励均值的初始值
        init_std: 奖励标准差的初始值
        """
        self.param_values = param_values
        self.means = np.full(len(param_values), init_mean, dtype=float)  # 初始化均值
        self.stds = np.full(len(param_values), init_std, dtype=float)    # 初始化标准差
        self.counts = np.zeros(len(param_values), dtype=int)           # 每个参数被选择的次数
        self.avg_returns = deque(maxlen=window_size)        # 记录最近 window_size 个回合的平均奖励

        self.current_param = None
        self.lr = lr

    def select_param(self):
        """通过汤普森采样选择一个超参数"""
        epsilon = 1e-6  # 避免标准差为 0
        samples = [np.random.normal(mean, max(std, epsilon)) for mean, std in zip(self.means, self.stds)]
        self.action_idx = np.argmax(samples)
        self.current_param = self.param_values[self.action_idx]
        return self.action_idx, self.current_param

    def update_distribution(self, returns):
        """
        根据训练结果更新奖励分布
        param: 被选择的超参数
        reward: 训练的连续奖励值
        """
        self.avg_returns.append(returns.mean().item())
        reward = np.mean(self.avg_returns)
        index = self.param_values.index(self.current_param)
        self.counts[index] += 1

        # 使用增量平均更新均值
        prev_mean = self.means[index].copy()
        self.means[index] = prev_mean + self.lr * (reward - self.means[index]) / self.counts[index]
        
        
        # 更新标准差
        if self.counts[index] > 1:
            self.stds[index] = np.sqrt(((self.stds[index] ** 2) * (self.counts[index] - 1) +
                                        (reward - prev_mean) * (reward - self.means[index])) /
                                       self.counts[index])


class GTSforClusters:
    def __init__(self,
                 cluster_dict={},
                 init_mean=0., 
                 init_std=1.,
                 window_size=10,
                 lr=1.0
                 ):
        self.cluster_dict = cluster_dict
        self.cluster_names = list(cluster_dict.keys())
        self.num_clusters = len(self.cluster_names)

        self.cluster_gts = GaussianThompsonSampling(param_values=self.cluster_names,
                               init_mean=init_mean,
                               init_std=init_std,
                               window_size=window_size,
                               lr=lr)
        self.sub_gts = {key: GaussianThompsonSampling(param_values=cluster_dict[key],
                                                      init_mean=init_mean, 
                                                      init_std=init_std, 
                                                      window_size=window_size, 
                                                      lr=lr) for key in cluster_dict.keys()}
        
        self.current_cluster = None
    
    def select_param(self):
        cluster_idx, cluster_name = self.cluster_gts.select_param()
        self.current_cluster = cluster_name
        sub_idx, sub_name = self.sub_gts[cluster_name].select_param()
        return cluster_idx, cluster_name, sub_idx, sub_name
    
    def update_distribution(self, returns):
        self.cluster_gts.update_distribution(returns)
        self.sub_gts[self.current_cluster].update_distribution(returns)