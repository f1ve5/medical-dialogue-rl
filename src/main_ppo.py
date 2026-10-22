from env import *
from agent_ppo import PPO_agent_model
import copy
import torch
import pickle
import numpy as np
from scipy.stats import entropy
import argparse
import os

def evaluate(env_path, agent, threshold, args):
    from evaluation import evaluate_agent
    return evaluate_agent(environment(env_path, args), agent, args.BATCHSIZE, args.MAXSTEP, threshold)[0]


def main():
    print("Initializing PPO Training....")
    env = environment('./dataset/train.pk', args)
    agent = PPO_agent_model(state_size=env.symptom_size, disease_size=env.diag_size, device=device, LR=args.lr, Gamma=args.gamma)

    # 初始化日志记录列表
    accuracy_list = []
    reward_list = []
    dev_accuracy_list = []

    threshold = args.threshold * np.ones(env.diag_size)
    best_dev_accu = -1

    for epoch in range(args.EPOCHS):
        env = environment('./dataset/train.pk', args)
        batches = env.sample_size // args.BATCHSIZE + (1 if env.sample_size % args.BATCHSIZE != 0 else 0)

        epoch_rewards = []
        epoch_accus = []

        for batch_idx in range(batches):
            s, true_disease = env.initialize_state(args.BATCHSIZE)
            actual_batch_size = s.shape[0]
            a_d, p_d = agent.choose_diagnosis(s)
            init_ent = ent = entropy(p_d, axis=1)
            done = ent < threshold[a_d]
            right_diag = (a_d == env.disease) & done

            # 轨迹暂存
            t_states, t_actions, t_log_probs, t_rewards, t_active = [], [], [], [], []

            for i in range(args.MAXSTEP):
                t_active.append((~done).copy())
                a_s, log_p = agent.choose_action_s(s)
                s_, r_s, done, right_diag, _, ent_, a_d_ = env.step(
                    s, a_s, done, right_diag, agent, init_ent, threshold, ent
                )
                t_states.append(s); t_actions.append(a_s); t_log_probs.append(log_p); t_rewards.append(r_s)
                s, ent = s_, ent_
                if all(done): break

            # 计算 PPO 需要的 Returns (G_t)
            returns = []
            discounted_reward = np.zeros(actual_batch_size)
            for r in reversed(t_rewards):
                discounted_reward = r + (args.gamma * discounted_reward)
                returns.insert(0, discounted_reward.copy())

            # 数据展平
            flat_s = np.concatenate(t_states, axis=0)
            flat_a = np.concatenate(t_actions, axis=0)
            flat_lp = np.concatenate(t_log_probs, axis=0)
            flat_ret = np.concatenate(returns, axis=0)
            flat_true = np.tile(true_disease, len(t_states))

            valid = np.concatenate(t_active, axis=0)
            if valid.any():
                agent.update_param_ppo(flat_s[valid], flat_a[valid], flat_lp[valid], flat_ret[valid])
                agent.update_param_c(flat_s[valid], flat_true[valid])

            # 计算当前 batch 表现
            cur_a_d, _ = agent.choose_diagnosis(s)
            batch_accu = (sum(right_diag) + sum((cur_a_d == env.disease) & (~done))) / actual_batch_size
            epoch_accus.append(batch_accu)
            epoch_rewards.append(np.mean(t_rewards))

        # Epoch 结束统计
        accuracy_list.append(np.mean(epoch_accus))
        reward_list.append(np.mean(epoch_rewards))

        dev_accu = evaluate('./dataset/dev.pk', agent, threshold, args)
        dev_accuracy_list.append(dev_accu)

        print(f"Epoch {epoch+1}/{args.EPOCHS} | Train Accu: {accuracy_list[-1]:.4f} | Dev Accu: {dev_accu:.4f}")

        # 保存日志文件 (防止冲突)
        info = f"ppo_{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
        with open(f"./log/stats_logging_{info}.pkl", "wb") as f:
            pickle.dump([accuracy_list, reward_list, dev_accuracy_list], f)

        if dev_accu > best_dev_accu:
            best_dev_accu = dev_accu
            agent.save_model(args, is_best=True)

def test():
    print("Testing PPO Model....")
    env = environment('./dataset/test.pk', args)
    agent = PPO_agent_model(env.symptom_size, env.diag_size, device)
    agent.load_model(args)

    # 此处假设 threshold 已在训练中保存或取默认
    threshold = args.threshold * np.ones(env.diag_size)
    predictions = []

    batches = env.sample_size // args.BATCHSIZE + (1 if env.sample_size % args.BATCHSIZE != 0 else 0)
    for episode in range(batches):
        s, _ = env.initialize_state(args.BATCHSIZE)
        a_d, p_d = agent.choose_diagnosis(s)
        init_ent = ent = entropy(p_d, axis=1)
        done = ent < threshold[a_d]
        for _ in range(args.MAXSTEP):
            a_s, _ = agent.choose_action_s(s)
            s_, _, done, _, _, ent_, a_d_ = env.step(s, a_s, done, np.zeros(s.shape[0], dtype=bool), agent, init_ent, threshold, ent)
            s, ent = s_, ent_
            if all(done): break

        a_d, _ = agent.choose_diagnosis(s)
        for d_idx in a_d:
            predictions.append(env.idx_to_str(d_idx.item(), idx_type="D"))

    with open("./test_prediction_ppo.txt", "w", encoding="utf-8") as f:
        for p in predictions: f.write(str(p) + "\n")
    print("Test predictions saved to test_prediction_ppo.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=str, default="gpu")
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-EPOCHS", type=int, default=200) # 改回 200 以对齐基线
    parser.add_argument("-BATCHSIZE", type=int, default=200)
    parser.add_argument("-MAXSTEP", type=int, default=15)
    parser.add_argument("-threshold", type=float, default=1.0)
    parser.add_argument("-gamma", type=float, default=0.99)
    parser.add_argument("-mu", type=float, default=1.0)
    parser.add_argument("-nu", type=float, default=2.5)
    parser.add_argument("-trail", type=int, default=2)
    parser.add_argument("-train", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device=="gpu" and torch.cuda.is_available() else "cpu")
    if args.train: main()
    else: test()
