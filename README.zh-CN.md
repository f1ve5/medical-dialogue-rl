# 基于强化学习的医学问诊

[English](README.md) · [项目分析](docs/PROJECT_REPORT.md) · [复现说明](docs/REPRODUCIBILITY.md)

本项目是医学问诊强化学习课程项目，研究智能体如何根据初始主诉选择后续问题，并利用疾病分类器输出诊断。项目包含 REINFORCE 基线适配、PPO 扩展，以及将诊断作为动作的探索。系统运行在模拟环境中，没有临床验证。

## 展示重点

- **问题建模**：将问诊表示为序列决策过程，以 118 维三值症状状态区分阳性、阴性和未知，对应 12 类疾病。
- **算法实现**：REINFORCE + 分类器；PPO 增加 Critic、裁剪目标和优势归一化。
- **实验分析**：比较诊断准确率、问诊轮次及已有报告的症状召回指标，分析逐疾病表现。
- **工程整理**：统一命令入口、修复批次游标和单样本维度问题、增加使用合成数据的回归检查。

## 现有结果

| 方法 | 开发集准确率 | 平均问诊次数 | 已记录召回指标 |
| --- | ---: | ---: | ---: |
| REINFORCE | 64.85% | 9.4393 | 0.1148 |
| PPO | 66.95% | 9.1255 | 0.1260 |

PPO 历史结果相对基线提高 **2.10 个百分点**，平均问诊次数下降约 **3.3%**。这些来自原有开发集报告，本次未重新训练；不应表述为统计显著提升、测试集成绩或临床效果。两种方法的阈值策略不同，尚不能将差异完全归因于 PPO。

诊断动作实验存在重复读取首批样本的缺陷，原报告 7.25% 的结果不作为有效对照。代码已修复，需重新训练后再分析，不能据此断言联合诊断动作设计普遍无效。

## 使用

```bash
python -m pip install -r requirements.txt
python scripts/inspect_data.py
python -m unittest discover -s tests -v
python scripts/run.py ppo train --device cpu --epochs 200
python scripts/run.py ppo evaluate
python scripts/run.py ppo predict
```

后两条命令需要训练后的权重。原模型较大，必要时使用 `--batch-size 8` 降低内存占用。详细环境说明见英文首页和复现文档。

## 数据与贡献边界

本地数据核对为 **1,912 / 239 / 239**，总计 **2,390** 条，12 类疾病、118 个症状；与原报告的 2,374 条不一致。测试集没有疾病标签。

项目基于 [Yuan 和 Yu 的公开实现](https://github.com/Yuanhy1997/Auto-Diagnosis-by-RL-and-Classification)，基线不应宣称为独立原创算法。具体个人贡献需要结合课程提供的初始代码进一步确认，见 [NOTICE.md](NOTICE.md)。原始数据仍在本地，但默认不跟踪上传；准确的数据来源、分发权限和上游代码授权尚待确认。
