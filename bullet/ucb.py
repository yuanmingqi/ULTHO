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