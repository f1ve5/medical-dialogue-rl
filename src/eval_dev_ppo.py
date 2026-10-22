import argparse
import copy
import os
from collections import Counter
import numpy as np
import pickle
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 配置中文字体，防止图片乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from agent_ppo import PPO_agent_model
from env import environment

def load_model_and_threshold(args, env):
    # 初始化 PPO 模型类
    agent = PPO_agent_model(
        state_size=env.symptom_size,
        disease_size=env.diag_size,
        device=args.device_torch,
        LR=args.lr,
    )

    # 匹配模型文件名 (带 ppo_ 前缀)
    info = f"{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
    best_policy_path = f"./model_saved/best_ppo_policy_{info}.pth"
    best_classifier_path = f"./model_saved/best_ppo_classifier_{info}.pth"

    # 设置阈值，默认使用 args.threshold
    threshold = args.threshold * np.ones(env.diag_size)

    if os.path.exists(best_policy_path) and os.path.exists(best_classifier_path):
        map_location = "cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
        agent.policy.load_state_dict(torch.load(best_policy_path, map_location=map_location))
        agent.classifier.load_state_dict(torch.load(best_classifier_path, map_location=map_location))
        print(f"成功加载 PPO 最佳模型: {best_policy_path}")
    else:
        print(f"警告：未找到 PPO 最佳模型文件。将尝试加载普通存档...")
        agent.load_model(args)

    return agent, threshold

def evaluate_dev(args):
    env = environment("./dataset/dev.pk", args)
    agent, threshold = load_model_and_threshold(args, env)

    total_samples = 0
    total_correct = 0
    total_steps = 0
    total_recall = 0.0
    confusion = Counter()
    disease_names = list(env.d2idx.keys())

    batches = env.sample_size // args.BATCHSIZE + (1 if env.sample_size % args.BATCHSIZE != 0 else 0)

    for _ in range(batches):
        s, true_disease = env.initialize_state(args.BATCHSIZE)
        true_disease = np.asarray(true_disease)
        actual_batch_size = s.shape[0]

        _, p_d = agent.choose_diagnosis(s)
        init_ent = entropy(p_d, axis=1)
        a_d_init, _ = agent.choose_diagnosis(s)
        done = init_ent < threshold[a_d_init]
        ent = init_ent

        steps_used = np.zeros(actual_batch_size, dtype=int)
        discovered_sym_count = np.zeros(actual_batch_size, dtype=int)
        implicit_sym_count = np.array([len(env.patients[idx].get("implicit_inform_slots", {})) for idx in env.batch_index])

        for _ in range(args.MAXSTEP):
            active = ~done
            if not np.any(active):
                break

            steps_used[active] += 1
            a_s, _ = agent.choose_action_s(s)

            for i in range(actual_batch_size):
                if active[i]:
                    patient = env.patients[env.batch_index[i]]
                    sym_name = env.idx_to_str(a_s[i], idx_type="S")
                    if sym_name in patient.get("implicit_inform_slots", {}):
                        if patient["implicit_inform_slots"][sym_name] == 1 and s[i, a_s[i]] != 1:
                            discovered_sym_count[i] += 1

            s_, _, done, _, _, ent_, _ = env.step(
                s, a_s, done, np.zeros(actual_batch_size, dtype=bool), agent, init_ent, threshold, ent
            )
            s, ent = s_, ent_

        pred_disease, _ = agent.choose_diagnosis(s)
        total_samples += actual_batch_size
        total_correct += int((pred_disease == true_disease).sum())
        total_steps += int(steps_used.sum())

        for true_idx, pred_idx in zip(true_disease, pred_disease):
            confusion[(int(true_idx), int(pred_idx))] += 1

        for i in range(actual_batch_size):
            if implicit_sym_count[i] > 0:
                total_recall += (discovered_sym_count[i] / implicit_sym_count[i])

    # --- 计算统计指标 ---
    accuracy = total_correct / total_samples if total_samples else 0.0
    avg_steps = total_steps / total_samples if total_samples else 0.0
    avg_recall = total_recall / total_samples if total_samples else 0.0

    # ==================== 新增：计算每个疾病的准确率逻辑 (修复报错) ====================
    disease_num_correct = Counter()
    disease_num_total = Counter()
    disease_accuracy = {}

    for (true_idx, pred_idx), count in confusion.items():
        disease_num_total[true_idx] += count
        if true_idx == pred_idx:
            disease_num_correct[true_idx] += count

    for disease_idx in range(len(disease_names)):
        total = disease_num_total[disease_idx]
        correct = disease_num_correct[disease_idx]
        acc = correct / total if total > 0 else 0.0
        disease_accuracy[disease_names[disease_idx]] = {
            "accuracy": acc,
            "correct": int(correct),
            "total": int(total)
        }

    # ==================== 结果打印 ====================
    print(f"\n{'='*60}")
    print(f"PPO 算法开发集评估 (Trail {args.trail})")
    print(f"{'='*60}")
    print(f"整体准确率: {accuracy:.4f}")
    print(f"平均步数: {avg_steps:.4f}")
    print(f"平均症状召回率 (Recall): {avg_recall:.4f}")
    print(f"总样本数: {total_samples}")

    # 1. 混淆矩阵
    num_diseases = len(disease_names)
    cm = np.zeros((num_diseases, num_diseases))
    for (t, p), c in confusion.items():
        cm[t, p] = c

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Oranges',
                xticklabels=disease_names, yticklabels=disease_names)
    plt.title('PPO 开发集混淆矩阵')
    plt.tight_layout()
    plt.savefig('./log/dev_confusion_matrix_ppo.png')
    plt.close()

    # 2. 各疾病准确率柱状图 (降序排列)
    disease_names_sorted = sorted(disease_accuracy.keys(),
                                  key=lambda x: disease_accuracy[x]['accuracy'],
                                  reverse=True)
    accuracies = [disease_accuracy[name]['accuracy'] for name in disease_names_sorted]
    colors = ['#2ecc71' if acc >= accuracy else '#e74c3c' for acc in accuracies]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(disease_names_sorted)), accuracies, color=colors)
    plt.axhline(y=accuracy, color='#3498db', linestyle='--', linewidth=2, label=f'PPO 整体准确率: {accuracy:.4f}')
    plt.xlabel('疾病名称', fontsize=12)
    plt.ylabel('诊断准确率', fontsize=12)
    plt.title('PPO 各疾病诊断准确率对比 (降序排列)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(disease_names_sorted)), disease_names_sorted, rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('./log/dev_accuracy_by_disease_ppo.png', dpi=150, bbox_inches='tight')
    print("PPO 各疾病准确率排序图已保存: log/dev_accuracy_by_disease_ppo.png")
    plt.close()

    # 3. 保存文本报告
    with open("./log/dev_evaluation_report_ppo.txt", "w", encoding="utf-8") as f:
        f.write(f"PPO 评估报告 (Trail {args.trail})\n")
        f.write(f"{'='*30}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Avg Steps: {avg_steps:.4f}\n")
        f.write(f"Avg Recall: {avg_recall:.4f}\n\n")
        f.write(f"{'疾病名':<20} {'准确率':<12} {'正确/总数'}\n")
        for d_name in disease_names_sorted:
            info = disease_accuracy[d_name]
            f.write(f"{d_name:<20} {info['accuracy']:<12.6f} {info['correct']}/{info['total']}\n")

    print("\n评估完成！结果已保存至 log/ 文件夹。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=str, default="gpu")
    parser.add_argument("-trail", type=int, default=2)
    parser.add_argument("-threshold", type=float, default=1.0)
    parser.add_argument("-BATCHSIZE", type=int, default=200)
    parser.add_argument("-MAXSTEP", type=int, default=15)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-mu", type=float, default=1.0)
    parser.add_argument("-nu", type=float, default=2.5)
    parser.add_argument("-gamma", type=float, default=0.99)
    args = parser.parse_args()

    device_str = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    args.device_torch = torch.device(device_str)

    evaluate_dev(args)
