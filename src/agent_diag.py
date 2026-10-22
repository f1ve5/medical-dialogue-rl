"""
agent_diag.py
Agent 实现：将诊断也作为动作（动作维度 = symptom_size + disease_size）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os


class sym_acquire_func_v2(nn.Module):
    def __init__(self, state_size, disease_size):
        super(sym_acquire_func_v2, self).__init__()
        self.state_size = state_size
        self.disease_size = disease_size
        self.fc1 = nn.Linear(state_size, 1024 * 5)
        self.fc2 = nn.Linear(1024 * 5, 2048 * 5)
        self.fc3 = nn.Linear(2048 * 5, 1024 * 5)
        # 输出维度变为 symptom_size + disease_size
        self.out = nn.Linear(1024 * 5, state_size + disease_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.relu(x)
        action_logprob = F.log_softmax(self.out(x), dim=1)
        action_prob = torch.exp(action_logprob)
        return action_prob


class Policy_Gradient_pair_model_diag(object):
    """包含策略网络（带诊断动作）和可选诊断分类器（保留用于对比/分析）。"""

    def __init__(self, state_size, disease_size, device, LR=1e-4, Gamma=0.99, Eta=0.01):
        self.state_size = state_size
        self.disease_size = disease_size
        self.action_size = state_size + disease_size

        self.policy = sym_acquire_func_v2(state_size, disease_size)
        # 保留分类器接口（可选），与原来一致
        self.classifier = nn.Sequential(nn.Linear(state_size, 1024), nn.GELU(), nn.Linear(1024, disease_size))

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
        # states: list of per-step arrays (steps, batch, state_dim)
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

        valid_sample = cumulate_R_s != 0

        self.batch_rewards_s = torch.from_numpy(cumulate_R_s[valid_sample]).float()
        self.batch_states = torch.from_numpy(states[valid_sample]).float()
        self.batch_action_s = torch.from_numpy(action_s[valid_sample])
        self.batch_true_d = torch.from_numpy(true_d[valid_sample])

        return len(self.batch_rewards_s), ave_rewards_s

    def choose_action_s(self, state):
        # state: numpy array (batch, state_size)
        self.policy.eval()
        state = torch.from_numpy(state).float().to(self.device)
        probs = self.policy.forward(state)
        m = Categorical(probs)
        action = m.sample().detach().cpu().numpy()
        return action

    def choose_diagnosis(self, state):
        # 仍保留分类器接口供分析使用
        self.classifier.eval()
        state = torch.from_numpy(state).float().to(self.device)
        output = self.classifier(state).detach().cpu()
        return torch.max(output, dim=1)[1].numpy(), torch.softmax(output, dim=1).numpy()

    def update_param_rl(self):
        self.policy.train()
        self.optimizer_p.zero_grad()
        state_tensor = self.batch_states.to(self.device)
        reward_tensor = self.batch_rewards_s.to(self.device)
        action_s_tensor = self.batch_action_s.to(self.device)
        prob_tensor = self.policy.forward(state_tensor)
        m = Categorical(prob_tensor)
        log_prob_tensor = m.log_prob(action_s_tensor)
        policy_loss = -(log_prob_tensor * (reward_tensor)).mean()
        entropy_loss = -torch.max(torch.tensor([self.eta - self.counter * 0.00001, 0])) * m.entropy().mean()
        loss = policy_loss + entropy_loss
        loss.backward()
        self.optimizer_p.step()
        self.counter += 1

    def update_param_c(self):
        self.classifier.train()
        self.optimizer_c.zero_grad()
        state_tensor = self.batch_states.to(self.device)
        label_tensor = self.batch_true_d.to(self.device)
        output_tensor = self.classifier(state_tensor)
        loss = self.cross_entropy(output_tensor, label_tensor)
        loss.backward()
        self.optimizer_c.step()

    def save_model(self, args):
        info = str(args.threshold) + "_" + str(args.mu) + "_" + str(args.nu) + "_" + str(args.trail)
        os.makedirs("./model_saved", exist_ok=True)
        torch.save(self.policy.state_dict(), "./model_saved/policy_diag_" + info + ".pth")
        torch.save(self.classifier.state_dict(), "./model_saved/classifier_diag_" + info + ".pth")

    def load_model(self, args):
        info = str(args.threshold) + "_" + str(args.mu) + "_" + str(args.nu) + "_" + str(args.trail)
        best_policy_path = "./model_saved/best_policy_diag_" + info + ".pth"
        best_classifier_path = "./model_saved/best_classifier_diag_" + info + ".pth"
        policy_path = "./model_saved/policy_diag_" + info + ".pth"
        classifier_path = "./model_saved/classifier_diag_" + info + ".pth"
        map_location = self.device
        if os.path.exists(best_policy_path):
            policy_path = best_policy_path
        if os.path.exists(best_classifier_path):
            classifier_path = best_classifier_path
        if not (os.path.isfile(policy_path) and os.path.isfile(classifier_path)):
            raise FileNotFoundError('Missing diagnosis checkpoints; train the diagnosis model first.')
        if os.path.exists(policy_path):
            self.policy.load_state_dict(torch.load(policy_path, map_location=map_location))
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=map_location))
