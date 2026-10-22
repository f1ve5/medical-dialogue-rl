import argparse
import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import torch

from agent_diag import Policy_Gradient_pair_model_diag
from env_diag import environment_diag


matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_model(args, env):
    agent = Policy_Gradient_pair_model_diag(
        state_size=env.symptom_size,
        disease_size=env.diag_size,
        device=args.device_torch,
        LR=args.lr,
        Gamma=args.gamma,
    )

    info = f"{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
    best_policy_path = f"./model_saved/best_policy_diag_{info}.pth"
    best_classifier_path = f"./model_saved/best_classifier_diag_{info}.pth"
    policy_path = f"./model_saved/policy_diag_{info}.pth"
    classifier_path = f"./model_saved/classifier_diag_{info}.pth"

    if os.path.exists(best_policy_path) and os.path.exists(best_classifier_path):
        policy_path = best_policy_path
        classifier_path = best_classifier_path

    if not (os.path.isfile(policy_path) and os.path.isfile(classifier_path)):
        raise FileNotFoundError('Missing diagnosis checkpoints; train the diagnosis model first.')
    if os.path.exists(policy_path):
        agent.policy.load_state_dict(torch.load(policy_path, map_location=args.device_torch))
    if os.path.exists(classifier_path):
        agent.classifier.load_state_dict(torch.load(classifier_path, map_location=args.device_torch))

    return agent


def evaluate_dev(args):
    env = environment_diag("./dataset/dev.pk", args)
    agent = load_model(args, env)

    total_samples = 0
    total_correct = 0
    total_steps = 0.0
    total_pos = 0.0
    total_diag_rate = 0.0
    confusion = Counter()
    disease_names = list(env.d2idx.keys())

    batches = env.sample_size // args.BATCHSIZE + (1 if env.sample_size % args.BATCHSIZE != 0 else 0)
    for _ in range(batches):
        s, true_disease = env.initialize_state(args.BATCHSIZE)
        true_disease = np.asarray(true_disease)
        actual_batch_size = s.shape[0]
        s_init = np.copy(s)
        s_final = np.copy(s)

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
            s_, _, done, _, diag_mask, _, a_d_ = env.step(
                s, a_s, done, np.zeros(actual_batch_size, dtype=bool), agent
            )
            s_final[diag_mask] = s_[diag_mask]
            final_diag[diag_mask] = a_d_[diag_mask]
            diag_triggered |= diag_mask
            s = s_

            if all(done):
                break

        pred_disease, _ = agent.choose_diagnosis(s)
        final_pred = pred_disease.copy()
        final_pred[diag_triggered] = final_diag[diag_triggered]

        total_samples += actual_batch_size
        total_correct += int((final_pred == true_disease).sum())
        total_steps += float(np.mean(steps_used))
        total_pos += float((np.sum(s_final == 1) - np.sum(s_init == 1)) / max(actual_batch_size, 1))
        total_diag_rate += float(np.mean(diag_triggered))

        for true_idx, pred_idx in zip(true_disease, final_pred):
            confusion[(int(true_idx), int(pred_idx))] += 1

    accuracy = total_correct / total_samples if total_samples else 0.0
    avg_steps = total_steps / batches if batches else 0.0
    avg_pos = total_pos / batches if batches else 0.0
    avg_diag_rate = total_diag_rate / batches if batches else 0.0

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
            "total": total,
        }

    print(f"\n{'='*60}")
    print(f"Trail 3 诊断动作集成模型开发集评估结果")
    print(f"{'='*60}")
    print(f"整体准确率: {accuracy:.4f}")
    print(f"平均步数: {avg_steps:.4f}")
    print(f"平均阳性症状增量: {avg_pos:.4f}")
    print(f"诊断动作触发率: {avg_diag_rate:.4f}")
    print(f"总样本数: {total_samples}")
    print(f"{'-'*60}")
    print(f"{'疾病名':<20} {'准确率':<10} {'正确数/总数':<15}")
    print(f"{'-'*60}")
    for disease_name in disease_names:
        info = disease_accuracy[disease_name]
        print(f"{disease_name:<20} {info['accuracy']:.4f}     {info['correct']}/{info['total']}")
    print(f"{'-'*60}\n")

    num_diseases = len(disease_names)
    confusion_matrix = np.zeros((num_diseases, num_diseases))
    for (true_idx, pred_idx), count in confusion.items():
        if 0 <= true_idx < num_diseases and 0 <= pred_idx < num_diseases:
            confusion_matrix[true_idx, pred_idx] = count

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=disease_names,
        yticklabels=disease_names,
        cbar_kws={'label': '样本数'},
    )
    plt.title('Trail 3 开发集混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测疾病', fontsize=12)
    plt.ylabel('真实疾病', fontsize=12)
    plt.tight_layout()
    plt.savefig('./log/dev_confusion_matrix_diag.png', dpi=150, bbox_inches='tight')
    plt.close()

    disease_names_sorted = sorted(disease_accuracy.keys(), key=lambda x: disease_accuracy[x]['accuracy'], reverse=True)
    accuracies = [disease_accuracy[name]['accuracy'] for name in disease_names_sorted]
    colors = ['#3498db' if acc >= accuracy else '#2980b9' for acc in accuracies]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(disease_names_sorted)), accuracies, color=colors)
    plt.axhline(y=accuracy, color='#2c3e50', linestyle='--', linewidth=2, label=f'整体准确率: {accuracy:.4f}')
    plt.xlabel('疾病', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('Trail 3 各疾病诊断准确率对比', fontsize=14, fontweight='bold')
    plt.xticks(range(len(disease_names_sorted)), disease_names_sorted, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('./log/dev_accuracy_by_disease_diag.png', dpi=150, bbox_inches='tight')
    plt.close()

    info = f"{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
    with open(f"./log/dev_evaluation_report_diag_{info}.txt", "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Trail 3 诊断动作集成模型开发集评估报告\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"整体准确率: {accuracy:.6f}\n")
        f.write(f"平均步数: {avg_steps:.6f}\n")
        f.write(f"平均阳性症状增量: {avg_pos:.6f}\n")
        f.write(f"诊断动作触发率: {avg_diag_rate:.6f}\n")
        f.write(f"总样本数: {total_samples}\n\n")
        f.write(f"{'-'*60}\n")
        f.write(f"各疾病准确率\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'疾病名':<20} {'准确率':<12} {'正确数':<8} {'总数':<8}\n")
        f.write(f"{'-'*60}\n")
        for disease_name in disease_names:
            info_row = disease_accuracy[disease_name]
            f.write(f"{disease_name:<20} {info_row['accuracy']:<12.6f} {info_row['correct']:<8} {info_row['total']:<8}\n")

    print("混淆矩阵热力图已保存: log/dev_confusion_matrix_diag.png")
    print("各疾病准确率图已保存: log/dev_accuracy_by_disease_diag.png")
    print(f"详细评估报告已保存: log/dev_evaluation_report_diag_{info}.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trail 3 dev set")
    parser.add_argument("-device", type=str, default="gpu", help='Device: "cpu" or "gpu"')
    parser.add_argument("-seed", type=int, default=514)
    parser.add_argument("-threshold", type=float, default=1.0)
    parser.add_argument("-BATCHSIZE", type=int, default=200)
    parser.add_argument("-MAXSTEP", type=int, default=15)
    parser.add_argument("-nu", type=float, default=2.5)
    parser.add_argument("-mu", type=float, default=1.0)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-gamma", type=float, default=0.99)
    parser.add_argument("-trail", type=int, default=3, help="Trail 3 number")
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        args.device_torch = torch.device("cuda")
    else:
        args.device_torch = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    evaluate_dev(args)
