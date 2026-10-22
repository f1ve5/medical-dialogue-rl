"""
Module: agent
Description: 设定 agent 模型
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import os


class sym_acquire_func(nn.Module):
    """RL 策略函数
    input：state_size 患者状态
    output：action 问诊动作
    """

    def __init__(self, state_size, action_size):
        super(sym_acquire_func, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024 * 5)
        self.fc2 = nn.Linear(1024 * 5, 2048 * 5)
        self.fc3 = nn.Linear(2048 * 5, 1024 * 5)
        self.out = nn.Linear(1024 * 5, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.relu(x)
        action_prob = torch.exp(F.log_softmax(self.out(x), dim=1))
        return action_prob


class diagnosis_func(nn.Module):
    """诊断分类器
    input：state_size 患者状态
    output：disease_size 疾病分布
    """

    def __init__(self, state_size, disease_size):
        super(diagnosis_func, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, disease_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.out(x)
        return output


class Policy_Gradient_pair_model(object):
    """核心问诊模型
    这里的问诊模型包含了问诊策略模型和疾病分类器
    问诊策略模型即前面的 RL 策略函数，采用 reinforce 算法
    疾病分类器是一个简单的 mlp
    """

    def __init__(self, state_size, disease_size, device, LR=1e-4, Gamma=0.99, Eta=0.01):

        self.policy = sym_acquire_func(state_size, state_size)
        self.classifier = diagnosis_func(state_size, disease_size)
        self.device = device
        self.lr = LR
        self.policy.to(self.device)
        self.classifier.to(self.device)
        self.optimizer_p = torch.optim.Adam(self.policy.parameters(), lr=LR / 5)
        self.optimizer_c = torch.optim.Adam(self.classifier.parameters(), lr=LR)
        self.counter = 1
        self.cross_entropy = nn.CrossEntropyLoss()
        self.gamma = Gamma
        self.eta = Eta

    def create_batch(self, states, rewards_s, action_s, true_d):
        # states 15*200*2112
        # rewards_s 15*200
        # action_s 15*200
        cumulate_R_s = []
        R_s = 0
        for r_s in rewards_s[::-1]:
            R_s = r_s + self.gamma * R_s
            cumulate_R_s.insert(0, R_s)

        rewards_s = np.array(rewards_s)
        ave_rewards_s = np.mean(np.sum(rewards_s, axis=0))

        cumulate_R_s = np.array(cumulate_R_s).T
        states = np.array(states).swapaxes(0, 1)
        action_s = np.array(action_s).T
        true_d = np.array(true_d).T
        # 即为r!=0的三元组
        valid_sample = cumulate_R_s != 0

        self.batch_rewards_s = torch.from_numpy(cumulate_R_s[valid_sample]).float()
        self.batch_states = torch.from_numpy(states[valid_sample]).float()
        self.batch_action_s = torch.from_numpy(action_s[valid_sample])
        self.batch_true_d = torch.from_numpy(true_d[valid_sample])

        return len(self.batch_rewards_s), ave_rewards_s

    def choose_action_s(self, state):
        # 挑选问诊动作
        self.policy.eval()
        state = torch.from_numpy(state).float()
        probs = self.policy.forward(state.to(self.device))  # (batchsize, symptom_size)
        m = Categorical(probs)
        action = m.sample().detach().cpu().numpy()

        return action

    def choose_diagnosis(self, state):
        # 获取诊断结果
        self.classifier.eval()
        state = torch.from_numpy(state).float()
        output = self.classifier.forward(state.to(self.device)).detach().cpu()

        return torch.max(output, dim=1)[1].numpy(), torch.softmax(output, dim=1).numpy()

    def update_param_rl(self):
        # 更新RL模型参数
        # reinforce 算法在这里实现
        # 注意这里的 reinforce 算法添加了熵正则化项，即 entropy Loss
        self.policy.train()
        self.optimizer_p.zero_grad()
        state_tensor = self.batch_states.to(self.device)
        reward_tensor = self.batch_rewards_s.to(self.device)
        action_s_tensor = self.batch_action_s.to(self.device)
        prob_tensor = self.policy.forward(state_tensor)
        # Policy Loss
        m = Categorical(prob_tensor)
        log_prob_tensor = m.log_prob(action_s_tensor)
        policy_loss = -(log_prob_tensor * (reward_tensor)).mean()
        # entropy Loss
        entropy_loss = -torch.max(torch.tensor([self.eta - self.counter * 0.00001, 0])) * m.entropy().mean()
        loss = policy_loss + entropy_loss
        loss.backward()
        self.optimizer_p.step()

        self.counter += 1

    def update_param_c(self):
        # 更新诊断模型参数
        # 使用收集的带标签状态训练分类器
        self.classifier.train()
        self.optimizer_c.zero_grad()
        state_tensor = self.batch_states.to(self.device)
        label_tensor = self.batch_true_d.to(self.device)
        output_tensor = self.classifier.forward(state_tensor)
        loss = self.cross_entropy(output_tensor, label_tensor)
        loss.backward()
        self.optimizer_c.step()

    def save_model(self, args):
        info = str(args.threshold) + "_" + str(args.mu) + "_" + str(args.nu) + "_" + str(args.trail)
        os.makedirs("./model_saved", exist_ok=True)
        torch.save(self.policy.state_dict(), "./model_saved/policy_" + info + ".pth")
        torch.save(self.classifier.state_dict(), "./model_saved/classifier_" + info + ".pth")

    def load_model(self, args):
        info = str(args.threshold) + "_" + str(args.mu) + "_" + str(args.nu) + "_" + str(args.trail)
        policy_path = "./model_saved/best_policy_" + info + ".pth"
        classifier_path = "./model_saved/best_classifier_" + info + ".pth"
        if not os.path.exists(policy_path):
            policy_path = "./model_saved/policy_" + info + ".pth"
        if not os.path.exists(classifier_path):
            classifier_path = "./model_saved/classifier_" + info + ".pth"
        map_location = self.device
        self.policy.load_state_dict(torch.load(policy_path, map_location=map_location))
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=map_location))
