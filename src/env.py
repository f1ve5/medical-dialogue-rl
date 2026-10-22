"""
Module: env
Description: 为强化学习智能体搭建交互环境
"""

import numpy as np
from tqdm import tqdm
import pickle
import copy
from operator import itemgetter
from scipy.stats import entropy


def sample_patients(sample_size, disease_dict, s2idx, prevalence=None):
    """合成患者数据
    这里的代码没有必要细看，文章做法其实不太科学
    返回一个包含所有患者信息的字典如下所示：
    {
      "0": {'d': np.str_('lambert-eaton syndrome'), 'self_report': [119, 828, 753], 'acquired': [772, 1172, 991, 744], 'test': [1361, 1670, 1897, 1782, 1561, 1866, 1846], 'all_sym': [119, 828, 753, 772, 1172, 991, 744, 1361, 1670, 1897, 1782, 1561, 1866, 1846]},
      "1": {'d': np.str_('lambert-eaton syndrome'), 'self_report': [119, 828, 753], 'acquired': [772, 1172, 991, 744], 'test': [1361, 1670, 1897, 1782, 1561, 1866, 1846], 'all_sym': [119, 828, 753, 772, 1172, 991, 744, 1361, 1670, 1897, 1782, 1561, 1866, 1846]},
    }
    key 为患者编号，d 为患者疾病，self_report 为患者主诉，acquired 为患者隐性症状，test 为患者检查项目，all_sym 为患者所有症状（acquired + test）
    我们把隐性症状和患者检查项目视为相同类型，基于self_report问诊出acquired和test
    """

    def sample_disease(sample_size, disease, prevalence=None):

        if prevalence is None:
            prevalence = np.ones(len(disease)) / len(disease)
        sampled_disease = np.random.choice(disease, sample_size, replace=True, p=prevalence)
        return sampled_disease

    sampled_disease = sample_disease(sample_size, list(disease_dict.keys()))
    patients = {}
    s_count = 0
    init_count = 0
    # 主诉平均2.38个，总的症状12.49
    s_index = np.random.poisson(2, len(sampled_disease)) + 1
    for i in tqdm(range(len(sampled_disease))):
        d = sampled_disease[i]
        num_s = np.min([np.random.poisson(8, 1)[0], len(disease_dict[d]["symptom"])])
        num_s = np.max([num_s, len(disease_dict[d]["symptom"]) // 2])
        s = np.random.choice(disease_dict[d]["symptom"], num_s, replace=False)
        s = list(itemgetter(*s)(s2idx))
        if len(disease_dict[d]["test"]) > 1:
            t = list(itemgetter(*disease_dict[d]["test"])(s2idx))
        elif len(disease_dict[d]["test"]) == 1:
            t = [s2idx[disease_dict[d]["test"][0]]]
        else:
            t = []
        patients[i] = {
            "d": d,
            "self_report": s[: s_index[i]],
            "acquired": s[s_index[i] :],
            "test": t,
        }
        patients[i]["all_sym"] = s + t
        s_count += len(s + t)
        init_count += len(patients[i]["self_report"])
    print("average symptom:", s_count / sample_size)
    print("average initial:", init_count / sample_size)
    return patients


class environment(object):
    def __init__(self, data_path, args):
        self.args = args
        self.load_disease_dict()

        # 读取真实数据集
        with open(data_path, "rb") as f:
            self.patients = pickle.load(f)

        self.sample_size = len(self.patients)
        self.idx = 0
        self.indexes = np.arange(self.sample_size)

        self.diag_size = len(self.disease_dict)  # 疾病数
        self.cost = np.ones(self.symptom_size + self.diag_size)
        self.earn = np.ones(self.symptom_size + self.diag_size)

    def idx_to_str(self, idx, idx_type="D"):
        if idx_type == "D":
            return list(self.d2idx.keys())[list(self.d2idx.values()).index(idx)]
        if idx_type == "S":
            return list(self.s2idx.keys())[list(self.s2idx.values()).index(idx)]

    def load_disease_dict(self):
        # 1. 真实疾病词典
        self.d2idx = {}
        with open("./dataset/disease.txt", "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    self.d2idx[line.strip()] = idx

        # 2. 真实症状词典
        self.s2idx = {}
        with open("./dataset/symptom.txt", "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    self.s2idx[line.strip()] = idx

        self.disease_dict = self.d2idx # 用于兼容原代码的 diag_size 计算
        self.ts_size = 0
        self.ss_size = len(self.s2idx)
        self.symptom_size = self.ss_size

    def initialize_state(self, batch_size):
        # 动态处理由于最后一个 batch 可能不足 batch_size 大小的情况
        self.batch_size = batch_size
        self.batch_index = self.indexes[self.idx : self.idx + batch_size]
        self.idx += batch_size
        actual_batch_size = len(self.batch_index)

        self.disease = []
        init_state = np.zeros((actual_batch_size, self.symptom_size))
        self.all_state = np.zeros((actual_batch_size, self.symptom_size))

        for i, item in enumerate(self.batch_index):
            patient = self.patients[item]

            # 读取标签（test.pk 没有 disease_tag，用 -1 代替）
            if "disease_tag" in patient:
                self.disease.append(self.d2idx[patient["disease_tag"]])
            else:
                self.disease.append(-1)

            # 显性主诉症状：阳性置 1， 阴性我们在 init 状态就告诉模型为 -1
            for sym, is_pos in patient.get("explicit_inform_slots", {}).items():
                if sym in self.s2idx:
                    sym_idx = self.s2idx[sym]
                    if is_pos:
                        init_state[i, sym_idx] = 1
                        self.all_state[i, sym_idx] = 1
                    else:
                        init_state[i, sym_idx] = -1
                        self.all_state[i, sym_idx] = 0

            # 隐性检查症状：等待被提问，不在 init_state 直接提供
            for sym, is_pos in patient.get("implicit_inform_slots", {}).items():
                if sym in self.s2idx:
                    sym_idx = self.s2idx[sym]
                    if is_pos:
                        self.all_state[i, sym_idx] = 1
                    else:
                        self.all_state[i, sym_idx] = 0

        self.disease = np.array(self.disease)
        return init_state, self.disease
        # init_state 病人的初始状态 (batchsize, symptom_size) 1表示阳性     disease 病人的疾病 (batchsize,)

    def step(self, s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent):
        # 更新 RL agent 的状态
        # a_p选择的动作 s旧状态，s_新状态
        s_ = copy.deepcopy(s)  # (200, 2112)
        ent_ = copy.deepcopy(ent)  # (200,)
        # 更新动作状态 1->1 0->1
        s_[~done, a_p[~done]] = self.all_state[~done, a_p[~done]] * 2 - 1
        a_d_, p_d_ = agent.choose_diagnosis(s_)  # a_d_为诊断的疾病

        ent_[~done] = entropy(p_d_[~done], axis=1)  # 更新entropy
        ent_ratio = (ent - ent_) / np.maximum(ent_init, np.finfo(float).eps)

        # diag表示某病人是否可以进行最后的诊断：当entropy小于threshold且未完成，可以进行诊断
        diag = (ent_ < threshold[a_d_]) & (~done)
        right_diag = (a_d_ == np.array(self.disease)) & diag  # 获取诊断正确的病历

        # reward_s 每次 action 后获得的奖励，包含 r_p 和 熵减奖励r_H，细节见论文
        reward_s = self.args.mu * self.reward_func(s, s_, a_p)
        reward_s[ent_ratio > 0] += self.args.nu * ent_ratio[ent_ratio > 0]  # 熵减奖励 nu为2.5 mu为1
        reward_s[diag] -= 1 * self.args.mu  # 诊断错误-1
        reward_s[right_diag] += 2 * self.args.mu  # 诊断正确+1
        reward_s[done] = 0

        done += diag  # 可以进行最后诊断的病人自然设为done
        right_diagnosis += right_diag
        return s_, reward_s, done, right_diagnosis, diag, ent_, a_d_

    def reward_func(self, s, s_, a_p):
        # 对应论文中的 r_p
        # s旧状态，s_新状态，ap
        reward = -self.cost[a_p]  # reward 从-1开始
        reward += (
            np.sum(
                np.abs(s - s_) *
                # 选择的动作是新动作
                self.earn[: self.symptom_size],
                axis=1,
            )
            * 0.7
        )
        reward += np.sum(((s_ - s) > 0) * self.earn[: self.symptom_size], axis=1)  # 选择的动作是阳性症状

        return reward


"""
你可以在 __main__ 中单独测试 env 代码
"""

if __name__ == "__main__":

    import json

    with open("./source_data/medlineplus.json", "r") as f:
        kb = json.load(f)
    d2idx = {}
    disease_dict = {}

    symptom_set = set()
    test_set = set()
    i = 0
    for disease in kb:
        d2idx[disease] = i
        disease_dict[disease] = {
            "test": [item[:10] for item in kb[disease]["test"]],
            "symptom": [item[:10] for item in kb[disease]["symptom"]],
        }
        i += 1
        symptom_set.update([item[:10] for item in kb[disease]["symptom"]])
        test_set.update([item[:10] for item in kb[disease]["test"]])

    s2idx = {}
    i = 0
    for item in symptom_set:
        s2idx[item] = i
        i += 1
    for item in test_set:
        s2idx[item] = i
        i += 1
    print(len(symptom_set), len(test_set))
    print(len(symptom_set.union(test_set)))
    print(len(disease_dict))

    count_s = 0
    count_t = 0
    for disease in disease_dict:
        count_s += len(disease_dict[disease]["symptom"])
        count_t += len(disease_dict[disease]["test"])
    print(count_t / 893, count_s / 893)
