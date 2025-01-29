from collections import deque
import numpy as np

class UCB:
    def __init__(self,
                 hp_list=None,
                 ucb_exploration_coef=0.5,
                 ucb_window_length=10):
        self.hp_list = hp_list
        self.num_hp_types = len(hp_list)
        self.ucb_exploration_coef = ucb_exploration_coef
        self.ucb_window_length = ucb_window_length

        self.total_num = 1
        self.num_action = [1.] * self.num_hp_types 
        self.qval_action = [0.] * self.num_hp_types 

        self.expl_action = [0.] * self.num_hp_types 
        self.ucb_action = [0.] * self.num_hp_types 

        self.return_action = []
        for i in range(self.num_hp_types):
            self.return_action.append(deque(maxlen=ucb_window_length))
    
    def select_ucb_hp(self):
        for i in range(self.num_hp_types):
            self.expl_action[i] = self.ucb_exploration_coef * \
                np.sqrt(np.log(self.total_num) / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]
        ucb_hp_id = np.argmax(self.ucb_action)
        self.current_hp_id = ucb_hp_id
        return ucb_hp_id, self.hp_list[ucb_hp_id]

    def update_ucb_values(self, returns):
        self.total_num += 1
        self.num_action[self.current_hp_id] += 1
        self.return_action[self.current_hp_id].append(returns.mean().item())
        self.qval_action[self.current_hp_id] = np.mean(self.return_action[self.current_hp_id])


class UCBforClusters:
    def __init__(self,
                 cluster_dict={},
                 ucb_exploration_coef=0.5,
                 ucb_window_length=10):
        self.cluster_dict = cluster_dict
        self.cluster_names = list(cluster_dict.keys())
        self.num_clusters = len(self.cluster_names)

        self.cluster_ucb = UCB(hp_list=self.cluster_names,
                               ucb_exploration_coef=ucb_exploration_coef,
                               ucb_window_length=ucb_window_length)
        self.sub_ucbs = {key: UCB(hp_list=cluster_dict[key],
                                  ucb_exploration_coef=ucb_exploration_coef,
                                  ucb_window_length=ucb_window_length) for key in cluster_dict.keys()}
        
        self.current_cluster = None

    def select_ucb_option(self):
        cluster_id, cluster_name = self.cluster_ucb.select_ucb_hp()
        self.current_cluster = cluster_name
        sub_id, sub_name = self.sub_ucbs[cluster_name].select_ucb_hp()
        return cluster_id, cluster_name, sub_id, sub_name
    
    def update_ucb_values(self, returns):
        self.cluster_ucb.update_ucb_values(returns)
        self.sub_ucbs[self.current_cluster].update_ucb_values(returns)