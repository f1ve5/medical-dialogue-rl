import argparse
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    parser = argparse.ArgumentParser(description="Plot Trail 3 metrics")
    parser.add_argument("-threshold", type=float, default=1.0)
    parser.add_argument("-mu", type=float, default=1.0)
    parser.add_argument("-nu", type=float, default=2.5)
    parser.add_argument("-trail", type=int, default=3)
    args = parser.parse_args()

    info_raw = f"{args.threshold}_{args.mu}_{args.nu}_{args.trail}"
    info_norm = f"{args.threshold:g}_{args.mu:g}_{args.nu:g}_{args.trail}"
    candidate_infos = [info_raw]
    if info_norm not in candidate_infos:
        candidate_infos.append(info_norm)

    log_path = None
    for info in candidate_infos:
        p = rf"./log/stats_logging_diag_{info}.pkl"
        if os.path.exists(p):
            log_path = p
            break

    if log_path is None:
        # 兜底：按 trail 自动选择最新的诊断版日志
        pattern = rf"./log/stats_logging_diag_*_{args.trail}.pkl"
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            log_path = matches[0]

    if log_path is None or not os.path.exists(log_path):
        report_path = None
        for info in candidate_infos:
            p = rf"./log/dev_evaluation_report_diag_{info}.txt"
            if os.path.exists(p):
                report_path = p
                break

        if report_path is None:
            report_matches = glob.glob(rf"./log/dev_evaluation_report_diag_*_{args.trail}.txt")
            if report_matches:
                report_matches.sort(key=os.path.getmtime, reverse=True)
                report_path = report_matches[0]

        if report_path is None:
            print("未找到对应的训练日志或评估报告。请先训练，或检查 -trail/-threshold/-mu/-nu 参数。")
            return

        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

        def extract(pattern, default=0.0):
            match = re.search(pattern, report_text)
            return float(match.group(1)) if match else default

        accuracy = extract(r"整体准确率:\s*([0-9.]+)")
        avg_steps = extract(r"平均步数:\s*([0-9.]+)")
        avg_pos = extract(r"平均阳性症状增量:\s*([0-9.]+)")
        diag_rate = extract(r"诊断动作触发率:\s*([0-9.]+)")

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Trail 3 评估摘要', fontsize=16)

        axs[0, 0].bar(['Accuracy'], [accuracy], color='steelblue')
        axs[0, 0].set_ylim([0, 1.05])
        axs[0, 0].set_title('整体准确率')

        axs[0, 1].bar(['Steps'], [avg_steps], color='mediumpurple')
        axs[0, 1].set_title('平均步数')

        axs[1, 0].bar(['Positive'], [avg_pos], color='indianred')
        axs[1, 0].set_title('平均阳性症状增量')

        axs[1, 1].bar(['Diag Rate'], [diag_rate], color='teal')
        axs[1, 1].set_title('诊断动作触发率')

        for ax in axs.flat:
            ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = 'trail3_metrics_summary.png'
        plt.savefig(out_path, dpi=300)
        print(f"未找到训练日志，已根据评估报告 {os.path.basename(report_path)} 生成摘要图：{out_path}")
        plt.show()
        return

    print(f"使用训练日志：{os.path.basename(log_path)}")

    with open(log_path, "rb") as f:
        buffer = pickle.load(f)

    acc_batch = buffer[0]
    step_batch = buffer[1]
    rew_batch = buffer[2]
    pos_batch = buffer[3]
    diag_rate_batch = buffer[4] if len(buffer) > 4 else []
    dev_acc_epoch = buffer[5] if len(buffer) > 5 else []

    if len(dev_acc_epoch) > 0:
        b_per_e = max(len(acc_batch) // len(dev_acc_epoch), 1)
        acc_epoch = [np.mean(acc_batch[i * b_per_e : (i + 1) * b_per_e]) for i in range(len(dev_acc_epoch))]
        step_epoch = [np.mean(step_batch[i * b_per_e : (i + 1) * b_per_e]) for i in range(len(dev_acc_epoch))]
        rew_epoch = [np.mean(rew_batch[i * b_per_e : (i + 1) * b_per_e]) for i in range(len(dev_acc_epoch))]
        pos_epoch = [np.mean(pos_batch[i * b_per_e : (i + 1) * b_per_e]) for i in range(len(dev_acc_epoch))]
        diag_rate_epoch = [np.mean(diag_rate_batch[i * b_per_e : (i + 1) * b_per_e]) for i in range(len(dev_acc_epoch))] if diag_rate_batch else []
        epochs = range(1, len(dev_acc_epoch) + 1)
    else:
        print("未发现验证集数据，无法自动转换 Epoch。请检查日志。")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Trail 3 训练指标 (按 Epoch 统计)', fontsize=16)

    axs[0, 0].plot(epochs, acc_epoch, label='Train Accu', color='blue', marker='.')
    axs[0, 0].plot(epochs, dev_acc_epoch, label='Dev Accu', color='orange', marker='o')
    axs[0, 0].set_title('诊断准确率 (Accuracy)')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, rew_epoch, color='green')
    axs[0, 1].set_title('平均奖励值 (Reward)')

    axs[1, 0].plot(epochs, pos_epoch, color='red')
    axs[1, 0].set_title('阳性症状命中数 (Positive Hits)')

    axs[1, 1].plot(epochs, step_epoch, color='purple', label='Steps')
    if diag_rate_batch:
        axs[1, 1].plot(epochs, diag_rate_epoch, color='teal', label='Diag Rate')
        axs[1, 1].legend()
    axs[1, 1].set_title('平均问诊轮次 (Steps)')

    for ax in axs.flat:
        ax.set_xlabel('Epoch')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = 'trail3_metrics_epoch.png'
    plt.savefig(out_path, dpi=300)
    print(f"Trail 3 Epoch 可视化完成：{out_path}")
    plt.show()


if __name__ == '__main__':
    main()
