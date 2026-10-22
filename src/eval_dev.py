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

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入 REINFORCE 模型类
from agent import Policy_Gradient_pair_model
from env import environment


def load_model_and_threshold(args, env):
    # 使用 REINFORCE 模型类初始化
    agent = Policy_Gradient_pair_model(
        state_size=env.symptom_size,
        disease_size=env.diag_size,
        device=args.device_torch,
        LR=args.lr,
    )

    info = f"{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
    # 适配 REINFORCE 的文件命名习惯
    best_meta_path = f"./log/best_meta_{info}.pkl"
    best_policy_path = f"./model_saved/best_policy_{info}.pth"
    best_classifier_path = f"./model_saved/best_classifier_{info}.pth"

    # 尝试加载阈值
    if os.path.exists(best_meta_path):
        with open(best_meta_path, "rb") as f:
            best_meta = pickle.load(f)
        threshold = np.array(best_meta["threshold"])
    else:
        threshold = args.threshold * np.ones(env.diag_size)

    # 加载权重
    if os.path.exists(best_policy_path) and os.path.exists(best_classifier_path):
        map_location = "cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
        agent.policy.load_state_dict(torch.load(best_policy_path, map_location=map_location))
        agent.classifier.load_state_dict(torch.load(best_classifier_path, map_location=map_location))
        print(f"成功加载 REINFORCE 模型: {best_policy_path}")
    else:
        print("未找到最佳 REINFORCE 模型，加载默认保存点...")
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

        a_d, p_d = agent.choose_diagnosis(s)
        init_ent = entropy(p_d, axis=1)
        done = init_ent < threshold[a_d]
        ent = init_ent

        steps_used = np.zeros(actual_batch_size, dtype=int)
        implicit_sym_count = np.zeros(actual_batch_size, dtype=int)
        discovered_sym_count = np.zeros(actual_batch_size, dtype=int)

        for i, item in enumerate(env.batch_index):
            patient = env.patients[item]
            implicit_sym_count[i] = len(patient.get("implicit_inform_slots", {}))

        for _ in range(args.MAXSTEP):
            active = ~done
            if not np.any(active):
                break
            steps_used[active] += 1

            # --- 适配 REINFORCE 返回元组 (action, log_prob) ---
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
            s = s_
            ent = ent_

        pred_disease, _ = agent.choose_diagnosis(s)
        total_samples += actual_batch_size
        total_correct += int((pred_disease == true_disease).sum())
        total_steps += int(steps_used.sum())

        for true_idx, pred_idx in zip(true_disease, pred_disease):
            confusion[(int(true_idx), int(pred_idx))] += 1

        for i in range(actual_batch_size):
            if implicit_sym_count[i] > 0:
                recall = discovered_sym_count[i] / implicit_sym_count[i]
                total_recall += recall

    accuracy = total_correct / total_samples if total_samples else 0.0
    avg_steps = total_steps / total_samples if total_samples else 0.0
    avg_recall = total_recall / total_samples if total_samples else 0.0

    # ==================== 计算每个疾病的准确率 ====================
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
            "correct": correct,
            "total": total
        }

    # ==================== 打印结果 ====================
    print(f"\n{'='*60}")
    print(f"REINFORCE 算法开发集评估结果")
    print(f"{'='*60}")
    print(f"整体准确率: {accuracy:.4f}")
    print(f"平均步数: {avg_steps:.4f}")
    print(f"平均症状召回率: {avg_recall:.4f}")
    print(f"总样本数: {total_samples}")
    print(f"\n{'-'*60}")
    print(f"{'疾病名':<20} {'准确率':<10} {'正确数/总数':<15}")
    print(f"{'-'*60}")
    for disease_name in disease_names:
        info = disease_accuracy[disease_name]
        print(f"{disease_name:<20} {info['accuracy']:.4f}     {info['correct']}/{info['total']}")
    print(f"{'-'*60}\n")

    # ==================== 绘制混淆矩阵热力图 ====================
    num_diseases = len(disease_names)
    confusion_matrix = np.zeros((num_diseases, num_diseases))

    for (true_idx, pred_idx), count in confusion.items():
        confusion_matrix[true_idx, pred_idx] = count

    plt.figure(figsize=(10, 8))
    # REINFORCE 使用 Oranges 配色
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Oranges',
                xticklabels=disease_names, yticklabels=disease_names,
                cbar_kws={'label': '样本数'})
    plt.title('REINFORCE 开发集混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测疾病', fontsize=12)
    plt.ylabel('真实疾病', fontsize=12)
    plt.tight_layout()
    plt.savefig('./log/dev_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n混淆矩阵热力图已保存: log/dev_confusion_matrix.png")
    plt.close()

    # ==================== 绘制各疾病准确率柱状图 ====================
    disease_names_sorted = sorted(disease_accuracy.keys(),
                                  key=lambda x: disease_accuracy[x]['accuracy'],
                                  reverse=True)
    accuracies = [disease_accuracy[name]['accuracy'] for name in disease_names_sorted]
    # REINFORCE 使用 coral/orange 系配色
    colors = ['#e67e22' if acc >= accuracy else '#d35400' for acc in accuracies]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(disease_names_sorted)), accuracies, color=colors)
    plt.axhline(y=accuracy, color='#2c3e50', linestyle='--', linewidth=2, label=f'整体准确率: {accuracy:.4f}')
    plt.xlabel('疾病', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('REINFORCE 各疾病诊断准确率对比', fontsize=14, fontweight='bold')
    plt.xticks(range(len(disease_names_sorted)), disease_names_sorted, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('./log/dev_accuracy_by_disease.png', dpi=150, bbox_inches='tight')
    print("各疾病准确率图已保存: log/dev_accuracy_by_disease.png")
    plt.close()

    # ==================== 保存详细报告 ====================
    with open("./log/dev_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"REINFORCE 开发集评估详细报告\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"整体准确率: {accuracy:.6f}\n")
        f.write(f"平均步数: {avg_steps:.6f}\n")
        f.write(f"平均症状召回率: {avg_recall:.6f}\n")
        f.write(f"总样本数: {total_samples}\n\n")

        f.write(f"{'-'*60}\n")
        f.write(f"各疾病准确率\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'疾病名':<20} {'准确率':<12} {'正确数':<8} {'总数':<8}\n")
        f.write(f"{'-'*60}\n")
        for disease_name in disease_names:
            info = disease_accuracy[disease_name]
            f.write(f"{disease_name:<20} {info['accuracy']:<12.6f} {info['correct']:<8} {info['total']:<8}\n")

    print("详细评估报告已保存: log/dev_evaluation_report.txt\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate REINFORCE dev set")
    parser.add_argument("-device", type=str, default="gpu", help='Device: "cpu" or "gpu"')
    parser.add_argument("-seed", type=int, default=514)
    parser.add_argument("-threshold", type=float, default=1.0)
    parser.add_argument("-BATCHSIZE", type=int, default=200)
    parser.add_argument("-MAXSTEP", type=int, default=15)
    parser.add_argument("-nu", type=float, default=2.5)
    parser.add_argument("-mu", type=float, default=1.0)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-gamma", type=float, default=0.99)
    parser.add_argument("-trail", type=int, default=1, help="REINFORCE trial number")
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        args.device_torch = torch.device("cuda")
    else:
        args.device_torch = torch.device("cpu")

    evaluate_dev(args)
