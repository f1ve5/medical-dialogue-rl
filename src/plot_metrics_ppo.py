import pickle
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

log_path = r"./log/stats_logging_ppo_1.0_1.0_2.5_2.pkl"

if not os.path.exists(log_path):
    print(f"日志文件 {log_path} 不存在。")
    exit()

with open(log_path, "rb") as f:
    buffer = pickle.load(f)

# PPO 数据解包 (已经是 Epoch 级别)
acc_epoch = buffer[0]
rew_epoch = buffer[1]
dev_acc_epoch = buffer[2]
pos_epoch = buffer[3] if len(buffer) > 3 else []
epochs = range(1, len(acc_epoch) + 1)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PPO 训练指标 (按 Epoch 统计 - Trail 2)', fontsize=16)

# 1. 准确率
axs[0, 0].plot(epochs, acc_epoch, label='Train Accu', color='blue', marker='.')
if dev_acc_epoch:
    axs[0, 0].plot(epochs, dev_acc_epoch, label='Dev Accu', color='orange', marker='o')
axs[0, 0].set_title('诊断准确率 (Accuracy)')
axs[0, 0].legend()

# 2. 奖励
axs[0, 1].plot(epochs, rew_epoch, color='green')
axs[0, 1].set_title('平均奖励值 (Reward)')

# 3. 阳性命中
if len(pos_epoch) > 0:
    axs[1, 0].plot(epochs, pos_epoch, color='red')
    axs[1, 0].set_title('阳性症状命中数 (Positive Hits)')
else:
    axs[1, 0].text(0.5, 0.5, '数据未记录\n参考评估报告', ha='center', va='center', color='gray')
    axs[1, 0].set_title('阳性症状命中数')

# 4. 问诊步数 (PPO 逻辑中如果没记 step 列表)
axs[1, 1].text(0.5, 0.5, '数据未记录\n参考评估报告', ha='center', va='center', color='gray')
axs[1, 1].set_title('平均问诊轮次 (Steps)')

for ax in axs.flat:
    ax.set_xlabel('Epoch')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('ppo_metrics_epoch.png', dpi=300)
print("PPO Epoch 可视化完成：ppo_metrics_epoch.png")
plt.show()
