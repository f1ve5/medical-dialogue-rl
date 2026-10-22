"""
main_diag.py
训练/测试脚本：Agent 的动作空间包含问诊动作与诊断动作
并保存与原版相近的日志、best checkpoint 和评估中间结果。
"""

import argparse
import copy
import os
import pickle
import random

import numpy as np
import torch

from agent_diag import Policy_Gradient_pair_model_diag
from env_diag import environment_diag


def evaluate(env_path, agent, args):
    from evaluation import evaluate_agent
    return evaluate_agent(environment_diag(env_path, args), agent, args.BATCHSIZE, args.MAXSTEP, diagnosis_actions=True)


def main():
    env = environment_diag("./dataset/train.pk", args)
    agent = Policy_Gradient_pair_model_diag(
        state_size=env.symptom_size, disease_size=env.diag_size, device=device, LR=args.lr, Gamma=args.gamma
    )

    info = str(args.threshold) + "_" + str(args.mu) + "_" + str(args.nu) + "_" + str(args.trail)
    os.makedirs("./log", exist_ok=True)
    os.makedirs("./model_saved", exist_ok=True)

    if args.train:
        accuracy_list = []
        step_list = []
        reward_list = []
        positive_list = []
        diag_rate_list = []
        dev_accuracy_list = []
        dev_step_list = []
        dev_positive_list = []
        best_dev_accu = -1.0

        for epoch in range(args.EPOCHS):
            env = environment_diag("./dataset/train.pk", args)
            batches = env.sample_size // args.BATCHSIZE + (1 if env.sample_size % args.BATCHSIZE != 0 else 0)

            for _ in range(batches):
                states = []
                rewards_s = []
                action_s = []
                true_d = []

                s, true_disease = env.initialize_state(args.BATCHSIZE)
                actual_batch_size = s.shape[0]
                s_init = copy.deepcopy(s)
                s_final = copy.deepcopy(s)
                done = np.zeros(actual_batch_size, dtype=bool)
                diag_triggered = np.zeros(actual_batch_size, dtype=bool)
                steps_used = np.zeros(actual_batch_size, dtype=int)
                final_diag = np.full(actual_batch_size, -1, dtype=int)

                for _ in range(args.MAXSTEP):
                    active = ~done
                    if not np.any(active):
                        break
                    steps_used[active] += 1

                    a_s = agent.choose_action_s(s)
                    s_, r_s, done, _, diag_mask, _, a_d_ = env.step(
                        s, a_s, done, np.zeros(actual_batch_size, dtype=bool), agent
                    )

                    s_final[diag_mask] = s_[diag_mask]
                    final_diag[diag_mask] = a_d_[diag_mask]
                    diag_triggered |= diag_mask

                    states.append(s)
                    rewards_s.append(r_s)
                    action_s.append(a_s)
                    true_d.append(true_disease)

                    s = s_
                    if all(done):
                        break

                ave_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d)
                agent.update_param_rl()
                agent.update_param_c()

                pred_disease, _ = agent.choose_diagnosis(s)
                final_pred = pred_disease.copy()
                final_pred[diag_triggered] = final_diag[diag_triggered]

                batch_accuracy = np.mean(final_pred == env.disease)
                ave_pos = (np.sum(s_final == 1) - np.sum(s_init == 1)) / max(actual_batch_size, 1)
                diag_rate = np.mean(diag_triggered)

                accuracy_list.append(batch_accuracy)
                step_list.append(np.mean(steps_used))
                reward_list.append(ave_reward_s)
                positive_list.append(ave_pos)
                diag_rate_list.append(diag_rate)

            agent.save_model(args)
            dev_accu, dev_step, dev_pos, _ = evaluate("./dataset/dev.pk", agent, args)
            print("DEV:", "Accu:", dev_accu, "Ave. Step:", dev_step, "Ave. Pos:", dev_pos)
            dev_accuracy_list.append(dev_accu)
            dev_step_list.append(dev_step)
            dev_positive_list.append(dev_pos)

            if dev_accu > best_dev_accu:
                best_dev_accu = dev_accu
                torch.save(agent.policy.state_dict(), "./model_saved/best_policy_diag_" + info + ".pth")
                torch.save(agent.classifier.state_dict(), "./model_saved/best_classifier_diag_" + info + ".pth")
                with open("./log/best_meta_diag_" + info + ".pkl", "wb") as f:
                    pickle.dump({"dev_accu": best_dev_accu}, f)

            with open("./log/stats_logging_diag_" + info + ".pkl", "wb") as f:
                pickle.dump(
                    [
                        accuracy_list,
                        step_list,
                        reward_list,
                        positive_list,
                        diag_rate_list,
                        dev_accuracy_list,
                        dev_step_list,
                        dev_positive_list,
                    ],
                    f,
                )

    else:
        agent.load_model(args)
        env = environment_diag("./dataset/test.pk", args)
        batches = env.sample_size // args.BATCHSIZE + (1 if env.sample_size % args.BATCHSIZE != 0 else 0)
        predictions = []

        for _ in range(batches):
            s, _ = env.initialize_state(args.BATCHSIZE)
            actual_batch_size = s.shape[0]
            done = np.zeros(actual_batch_size, dtype=bool)
            diag_triggered = np.zeros(actual_batch_size, dtype=bool)
            final_diag = np.full(actual_batch_size, -1, dtype=int)

            for _ in range(args.MAXSTEP):
                active = ~done
                if not np.any(active):
                    break

                a_s = agent.choose_action_s(s)
                s_, _, done, _, diag_mask, _, a_d_ = env.step(
                    s, a_s, done, np.zeros(actual_batch_size, dtype=bool), agent
                )
                diag_triggered |= diag_mask
                final_diag[diag_mask] = a_d_[diag_mask]
                s = s_
                if all(done):
                    break

            pred_disease, _ = agent.choose_diagnosis(s)
            final_pred = pred_disease.copy()
            final_pred[diag_triggered] = final_diag[diag_triggered]
            for d_idx in final_pred:
                predictions.append(env.idx_to_str(int(d_idx), idx_type="D"))

        with open("./test_prediction_diag.txt", "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(str(p) + "\n")
        print("Test predictions written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Settings for diag agent")
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument("-seed", type=int, default=514)
    parser.add_argument("-threshold", type=float, default=1.0)
    parser.add_argument("-BATCHSIZE", type=int, default=200)
    parser.add_argument("-EPOCHS", type=int, default=10)
    parser.add_argument("-MAXSTEP", type=int, default=15)
    parser.add_argument("-nu", type=float, default=2.5)
    parser.add_argument("-mu", type=float, default=1.0)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-gamma", type=float, default=0.99)
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-trail", type=int, default=1)
    parser.add_argument("-lamb", type=float, default=0.99)
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    main()
