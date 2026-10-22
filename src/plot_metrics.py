import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# 适配中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

log_path = "./log/stats_logging_1.0_1.0_2.5_1.pkl"

if not os.path.exists(log_path):
    print(f"日志文件 {log_path} 不存在。")
    exit()

with open(log_path, "rb") as f:
    buffer = pickle.load(f)

# 原始数据 (Batch 级别)
acc_batch, step_batch, rew_batch, pos_batch = buffer[0], buffer[1], buffer[2], buffer[3]
dev_acc_epoch = buffer[5] if len(buffer) > 5 else []

# 计算聚合步长 (每 Epoch 有多少个 Batch)
if len(dev_acc_epoch) > 0:
    b_per_e = len(acc_batch) // len(dev_acc_epoch)
    # 将 Batch 数据按 Epoch 取平均
    acc_epoch = [np.mean(acc_batch[i*b_per_e : (i+1)*b_per_e]) for i in range(len(dev_acc_epoch))]
    step_epoch = [np.mean(step_batch[i*b_per_e : (i+1)*b_per_e]) for i in range(len(dev_acc_epoch))]
    rew_epoch = [np.mean(rew_batch[i*b_per_e : (i+1)*b_per_e]) for i in range(len(dev_acc_epoch))]
    pos_epoch = [np.mean(pos_batch[i*b_per_e : (i+1)*b_per_e]) for i in range(len(dev_acc_epoch))]
    epochs = range(1, len(dev_acc_epoch) + 1)
else:
    print("未发现验证集数据，无法自动转换 Epoch。请检查日志。")
    exit()

# 开始绘图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('REINFORCE 训练指标 (按 Epoch 统计 - Trail 1)', fontsize=16)

# 1. 准确率
axs[0, 0].plot(epochs, acc_epoch, label='Train Accu', color='blue', marker='.')
axs[0, 0].plot(epochs, dev_acc_epoch, label='Dev Accu', color='orange', marker='o')
axs[0, 0].set_title('诊断准确率 (Accuracy)')
axs[0, 0].legend()

# 2. 奖励
axs[0, 1].plot(epochs, rew_epoch, color='green')
axs[0, 1].set_title('平均奖励值 (Reward)')

# 3. 阳性命中
axs[1, 0].plot(epochs, pos_epoch, color='red')
axs[1, 0].set_title('阳性症状命中数 (Positive Hits)')

# 4. 问诊步数
axs[1, 1].plot(epochs, step_epoch, color='purple')
axs[1, 1].set_title('平均问诊轮次 (Steps)')

for ax in axs.flat:
    ax.set_xlabel('Epoch')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('baseline_metrics_epoch.png', dpi=300)
print("基线 Epoch 可视化完成：baseline_metrics_epoch.png")
plt.show()
