"""
Modified environment supporting diagnosis actions in agent action space.
Action space: [0..symptom_size-1] = ask symptom; [symptom_size..symptom_size+diag_size-1] = diagnose disease
"""

import numpy as np
import copy
from scipy.stats import entropy


class environment_diag(object):
    def __init__(self, data_path, args):
        self.args = args
        from env import environment
        # reuse loading from original env
        base = environment(data_path, args)
        # copy relevant attributes
        self.patients = base.patients
        self.sample_size = base.sample_size
        self.idx = 0
        self.indexes = base.indexes
        self.d2idx = base.d2idx
        self.s2idx = base.s2idx
        self.disease_dict = base.disease_dict
        self.symptom_size = base.symptom_size
        self.ss_size = base.ss_size
        self.ts_size = base.ts_size
        self.diag_size = len(self.d2idx)
        # cost/earn same dimension as original + diag_size (diagnosis actions have no earn/cost by default)
        self.cost = np.ones(self.symptom_size + self.diag_size)
        self.earn = np.ones(self.symptom_size + self.diag_size)
        self.all_state = base.all_state if hasattr(base, 'all_state') else None

    def idx_to_str(self, idx, idx_type="D"):
        if idx_type == "D":
            return list(self.d2idx.keys())[list(self.d2idx.values()).index(idx)]
        if idx_type == "S":
            return list(self.s2idx.keys())[list(self.s2idx.values()).index(idx)]

    def initialize_state(self, batch_size):
        from env import environment
        return environment.initialize_state(self, batch_size)

    def step(self, s, a_p, done, right_diagnosis, agent, ent_init=None, threshold=None, ent=None):
        # s: (batch, symptom_size)
        # a_p: (batch,) actions chosen by agent in [0, symptom_size+diag_size)
        # Returns: s_, reward, done, right_diagnosis, diag_triggered(索引布尔数组), ent_, a_d_
        s_ = copy.deepcopy(s)
        actual_batch = s.shape[0]
        reward = np.zeros(actual_batch, dtype=float)
        diag_triggered = np.zeros(actual_batch, dtype=bool)

        # If ent/threshold provided, we can maintain compatibility returning ent_
        ent_ = np.zeros(actual_batch)
        a_d_ = np.zeros(actual_batch, dtype=int)

        # Handle each sample separately
        for i in range(actual_batch):
            if done[i]:
                continue
            action = int(a_p[i])
            if action < self.symptom_size:
                # original symptom query behavior
                s_[i, action] = self.all_state[i, action] * 2 - 1
                # compute reward change like original reward_func (approx)
                # cost and earn only consider symptom indices
                reward[i] = -self.cost[action]
                reward[i] += np.sum(np.abs(s[i] - s_[i]) * self.earn[: self.symptom_size]) * 0.7
                reward[i] += np.sum(((s_[i] - s[i]) > 0) * self.earn[: self.symptom_size])
            else:
                # diagnosis action
                diag_idx = action - self.symptom_size
                diag_triggered[i] = True
                done[i] = True
                a_d_[i] = diag_idx
                if self.disease[i] >= 0 and diag_idx == self.disease[i]:
                    reward[i] += 10.0
                    right_diagnosis[i] = True
                else:
                    reward[i] -= 10.0

        return s_, reward, done, right_diagnosis, diag_triggered, ent_, a_d_


if __name__ == '__main__':
    pass
