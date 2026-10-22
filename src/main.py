"""
Module: main.py
Description: 强化学习模型训练与测试的主逻辑
Source: Yuan H, Yu S. Efficient symptom inquiring and diagnosis via adaptive alignment of reinforcement learning and classification[J]. Artificial Intelligence in Medicine, 2024, 148: 102748.
Source code: https://github.com/Yuanhy1997/Auto-Diagnosis-by-RL-and-Classification
"""

from env import *
from agent import *
import copy
import torch
import pickle
import numpy as np
from scipy.stats import entropy
import argparse
import random
import os


def evaluate(env_path, agent, threshold):
    from evaluation import evaluate_agent
    return evaluate_agent(environment(env_path, args), agent, args.BATCHSIZE, args.MAXSTEP, threshold)[:3]


def main():
    print("Initializing Environment and loading train patients....")
    env = environment('./dataset/train.pk', args)
    agent = Policy_Gradient_pair_model(
        state_size=env.symptom_size, disease_size=env.diag_size, device=device, LR=args.lr, Gamma=args.gamma
    )
    accuracy_list = []
    step_list = []
    reward_list = []
    positive_list = []
    threshold_list = []
    dev_accuracy_list = []
    dev_step_list = []
    dev_positive_list = []
    ave_step = 0

    if args.threshold_random_initial:
        threshold = np.random.rand(env.diag_size)
    else:
        threshold = args.threshold * np.ones(env.diag_size)
    best_a = 0
    best_dev_accu = -1
    best_threshold = threshold.copy()
    os.makedirs("./model_saved", exist_ok=True)

    for epoch in range(args.EPOCHS):
        env = environment('./dataset/train.pk', args)
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

            a_d, p_d = agent.choose_diagnosis(s)
            init_ent = entropy(p_d, axis=1)
            done = init_ent < threshold[a_d]
            right_diag = (a_d == env.disease) & done
            diag_ent = np.zeros(actual_batch_size)
            finl_diag = np.zeros(actual_batch_size).astype(np.int64) - 1
            diag_ent[right_diag] = init_ent[right_diag]  # 更新正确诊断疾病的诊断熵
            ent = init_ent  # ent是各个病人当前的分布熵

            for i in range(args.MAXSTEP):  # 设定最大问诊轮次
                a_s = agent.choose_action_s(s)  # a_s为选择问诊的症状 (batchsize,)
                s_, r_s, done, right_diag, final_idx, ent_, a_d_ = env.step(
                    s, a_s, done, right_diag, agent, init_ent, threshold, ent
                )  # a_d为各个病人当前判断的疾病 (batchsize,)

                s_final[final_idx] = s_[final_idx]  # final_idx为可以进行最后诊断的病人 (batchsize,)
                diag_ent[right_diag] = ent_[right_diag]  # 更新得到正确诊断患者的诊断熵
                finl_diag[right_diag] = a_d_[right_diag]  # 储存最终诊断结果

                if i == args.MAXSTEP - 1:
                    r_s[done == False] -= 1  # 没有在最大轮次前结束，惩罚

                states.append(s)
                rewards_s.append(r_s)
                action_s.append(a_s)
                true_d.append(true_disease)

                s = s_  # 更新s和ent
                ent = ent_

                if all(done):
                    break

            s_final[~done] = s_[~done]  # 更新那些没有done的病人的最终状态

            ave_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d)
            a_d, p_d = agent.choose_diagnosis(s)  # 获取200名病人最终的疾病分布

            t_d = (a_d == env.disease) & (~done)  # t_d 正确诊断但没有done
            # 更新正确诊断但是没有done疾病的entropy
            diag_ent[t_d] = entropy(p_d[t_d], axis=1)
            finl_diag[t_d] = a_d[t_d]
            # finl_diag中初始为-1，利用那些得到正确诊断的病人数据去更新各个疾病的熵参数 threshold[item]为某疾病的熵参数

            # 下面是根据论文里公式对疾病诊断熵进行更新
            # K_{d}=\lambda K_{d}+(1-\lambda) H_{\psi}\left(s_{\mathrm{fin}}\right)
            for idx, item in enumerate(finl_diag):
                if item >= 0 and abs(threshold[item] - diag_ent[idx]) > 0.01:
                    threshold[item] = args.lamb * threshold[item] + (1 - args.lamb) * diag_ent[idx]

            agent.update_param_rl()
            agent.update_param_c()

            # 计算每个 batch 中模型的准确率，平均问诊轮次，平均reward，平均命中的阳性症状
            ave_pos = (np.sum(s_final == 1) - np.sum(s_init == 1)) / ave_step if ave_step > 0 else 0
            step_list.append(ave_step / actual_batch_size)
            reward_list.append(ave_reward_s)
            positive_list.append(ave_pos)
            threshold_list.append(threshold.copy())
            accuracy_list.append(
                (sum(right_diag) + sum(a_d[done == False] == env.disease[done == False])) / actual_batch_size
            )  # 计算 batch 的诊断准确率
            best_a = np.max([best_a, accuracy_list[-1]])  # 最大的准确率

            print(
                "==Epoch:",
                epoch + 1,
                "\tAve. Accu:",
                accuracy_list[-1],
                "\tBest Accu:",
                best_a,
                "\tAve. Pos:",
                ave_pos,
            )
            print(
                "Threshold:",
                threshold[:5],
                "\tAve. Step:",
                ave_step / actual_batch_size,
                "\tAve. Reward Sym.:",
                ave_reward_s,
                "\n",
            )

        agent.save_model(args)
        info = str(args.threshold) + "_" + str(args.mu) + "_" + str(args.nu) + "_" + str(args.trail)
        with open("./log/stats_logging_" + info + ".pkl", "wb") as f:
            pickle.dump([accuracy_list, step_list, reward_list, positive_list, threshold_list], f)

        dev_accu, dev_step, dev_pos = evaluate('./dataset/dev.pk', agent, threshold)
        print("DEV:", "Accu:", dev_accu, "Ave. Step:", dev_step, "Ave. Pos:", dev_pos)
        dev_accuracy_list.append(dev_accu)
        dev_step_list.append(dev_step)
        dev_positive_list.append(dev_pos)
        if dev_accu > best_dev_accu:
            best_dev_accu = dev_accu
            best_threshold = threshold.copy()
            torch.save(agent.policy.state_dict(), "./model_saved/best_policy_" + info + ".pth")
            torch.save(agent.classifier.state_dict(), "./model_saved/best_classifier_" + info + ".pth")
            with open("./log/best_meta_" + info + ".pkl", "wb") as f:
                pickle.dump({"dev_accu": best_dev_accu, "threshold": best_threshold}, f)

        with open("./log/stats_logging_" + info + ".pkl", "wb") as f:
            pickle.dump(
                [
                    accuracy_list,
                    step_list,
                    reward_list,
                    positive_list,
                    threshold_list,
                    dev_accuracy_list,
                    dev_step_list,
                    dev_positive_list,
                ],
                f,
            )


def test():
    print("Loading test dataset....")
    env = environment('./dataset/test.pk', args)
    agent = Policy_Gradient_pair_model(
        state_size=env.symptom_size, disease_size=env.diag_size, device=device, LR=1e-4, Gamma=0.99
    )
    agent.load_model(args)

    info = str(args.threshold) + "_" + str(args.mu) + "_" + str(args.nu) + "_" + str(args.trail)
    best_meta_path = "./log/best_meta_" + info + ".pkl"
    if os.path.exists(best_meta_path):
        with open(best_meta_path, "rb") as f:
            best_meta = pickle.load(f)
        threshold = best_meta["threshold"]
    else:
        with open("./log/stats_logging_" + info + ".pkl", "rb") as f:
            buffer = pickle.load(f)
        threshold = buffer[4][-1]

    steps_on_ave = 0
    pos_on_ave = 0

    predictions = []

    batches = env.sample_size // args.BATCHSIZE + (1 if env.sample_size % args.BATCHSIZE != 0 else 0)
    for episode in range(batches):
        states = []
        action_m = []
        rewards_s = []
        action_s = []
        true_d = []

        s, _ = env.initialize_state(args.BATCHSIZE)
        actual_batch_size = s.shape[0]
        s_init = copy.deepcopy(s)
        s_final = copy.deepcopy(s)

        a_d, p_d = agent.choose_diagnosis(s)  # a_d为分类的疾病，p_d为疾病分布
        init_ent = entropy(p_d, axis=1)

        done = init_ent < threshold[a_d]
        ent = init_ent

        for i in range(args.MAXSTEP):
            a_s = agent.choose_action_s(s)
            s_, _, done, _, final_idx, ent_, a_d_ = env.step(
                s, a_s, done, np.zeros(actual_batch_size, dtype=bool), agent, init_ent, threshold, ent
            )

            s_final[final_idx] = s_[final_idx]

            s = s_
            ent = ent_

            if all(done):
                break

        s_final[~done] = s_[~done]

        a_d, _ = agent.choose_diagnosis(s)
        for d_idx in a_d:
            predictions.append(env.idx_to_str(d_idx.item(), idx_type="D"))

    print("Writing predictions to txt...")
    with open("./test_prediction.txt", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(str(p) + "\n")
    print("Done! Total predictions:", len(predictions))


if __name__ == "__main__":
    """
    设置超参
    """
    parser = argparse.ArgumentParser(description="Process Settings")
    parser.add_argument(
        "-device", type=str, default="cpu", help='Device to use for computation: "cpu" or "gpu"'
    )
    parser.add_argument("-seed", type=int, default=514, help="set a random seed")
    parser.add_argument("-threshold", type=float, default=1.0, help="set the initial threshold")
    parser.add_argument(
        "-threshold_random_initial",
        action="store_true",
        help="randomly initialize threshold",
    )
    # 这里的 EPISODES 其实就是每个 epoch 涉及多少个问诊患者
    parser.add_argument("-EPISODES", type=int, default=100000, help="episodes of MAD rounds per epoch")
    parser.add_argument(
        "-BATCHSIZE",
        type=int,
        default=200,
        help="games for each time onpolicy sample collection",
    )
    parser.add_argument("-EPOCHS", type=int, default=200, help="training epochs")
    parser.add_argument("-MAXSTEP", type=int, default=15, help="max inquiring turns of each MAD round")
    parser.add_argument("-nu", type=float, default=2.5, help="nu")
    parser.add_argument("-mu", type=float, default=1.0, help="mu")
    parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("-gamma", type=float, default=0.99, help="reward discount rate")
    parser.add_argument(
        "-train",
        action="store_true",
        help="whether test on the exsit result model or train a new model",
    )
    # 这个 trail 见 main 函数里的 info
    parser.add_argument("-trail", type=int, default=1, help="the trail number")
    parser.add_argument("-lamb", type=float, default=0.99, help="polyak factor for threshold adjusting")
    args = parser.parse_args()
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.train:
        main()
    else:
        test()
