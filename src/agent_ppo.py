import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import os

# 从原有代码导入网络结构以保持一致性
from agent import sym_acquire_func, diagnosis_func

class critic_func(nn.Module):
    """新增 Critic 网络用于估算状态价值 V(s)"""
    def __init__(self, state_size):
        super(critic_func, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.out(x)

class PPO_agent_model(object):
    def __init__(self, state_size, disease_size, device, LR=1e-4, Gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.device = device
        self.gamma = Gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = sym_acquire_func(state_size, state_size).to(device)
        self.critic = critic_func(state_size).to(device)
        self.classifier = diagnosis_func(state_size, disease_size).to(device)

        self.optimizer_p = torch.optim.Adam(self.policy.parameters(), lr=LR / 5)
        self.optimizer_v = torch.optim.Adam(self.critic.parameters(), lr=LR / 5)
        self.optimizer_c = torch.optim.Adam(self.classifier.parameters(), lr=LR)

        self.MseLoss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def choose_action_s(self, state):
        self.policy.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().to(self.device)
            probs = self.policy(state_t)
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def choose_diagnosis(self, state):
        self.classifier.eval()
        state_t = torch.from_numpy(state).float().to(self.device)
        output = self.classifier(state_t).detach().cpu()
        return torch.max(output, dim=1)[1].numpy(), torch.softmax(output, dim=1).numpy()

    def update_param_ppo(self, states, actions, old_log_probs, returns):
        self.policy.train()
        self.critic.train()

        s_t = torch.FloatTensor(states).to(self.device)
        a_t = torch.LongTensor(actions).to(self.device)
        lp_t = torch.FloatTensor(old_log_probs).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        for _ in range(self.K_epochs):
            probs = self.policy(s_t)
            m = Categorical(probs)
            log_probs_new = m.log_prob(a_t)
            dist_entropy = m.entropy()
            state_values = self.critic(s_t).squeeze(-1)

            advantages = ret_t - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            ratios = torch.exp(log_probs_new - lp_t)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            loss_critic = self.MseLoss(state_values, ret_t)

            self.optimizer_p.zero_grad()
            loss_actor.backward()
            self.optimizer_p.step()

            self.optimizer_v.zero_grad()
            loss_critic.backward()
            self.optimizer_v.step()

    def update_param_c(self, states, true_labels):
        self.classifier.train()
        self.optimizer_c.zero_grad()
        s_t = torch.FloatTensor(states).to(self.device)
        l_t = torch.LongTensor(true_labels).to(self.device)
        output = self.classifier(s_t)
        loss = self.cross_entropy(output, l_t)
        loss.backward()
        self.optimizer_c.step()

    def save_model(self, args, is_best=False):
        prefix = "best_ppo_" if is_best else "ppo_"
        info = f"{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
        os.makedirs("./model_saved", exist_ok=True)
        torch.save(self.policy.state_dict(), f"./model_saved/{prefix}policy_{info}.pth")
        torch.save(self.classifier.state_dict(), f"./model_saved/{prefix}classifier_{info}.pth")
        torch.save(self.critic.state_dict(), f"./model_saved/{prefix}critic_{info}.pth")

    def load_model(self, args):
        info = f"{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
        path = f"./model_saved/best_ppo_policy_{info}.pth"
        if not os.path.exists(path): path = f"./model_saved/ppo_policy_{info}.pth"
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        c_path = path.replace("policy", "classifier")
        self.classifier.load_state_dict(torch.load(c_path, map_location=self.device))
