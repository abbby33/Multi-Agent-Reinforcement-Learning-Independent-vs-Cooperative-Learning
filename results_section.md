# Results（结果）

## 1. 实验概述

本研究进行了多组多智能体强化学习实验，比较了不同的学习算法在多个环境中的表现。实验主要分为两个部分：
- **Checkpoint 1**: 比较独立Q学习（Independent Q-Learning, IQL）和合作Q学习（Cooperative Q-Learning, CQL）
- **Final Project**: 评估深度Q网络（Deep Q-Network, DQN）方法，包括独立深度Q网络（IDQN）和双深度Q网络（DDQN）

## 2. Checkpoint 1: 表格化Q学习算法比较

### 2.1 实验设置
- **环境**: 自定义多智能体仓库环境（MultiAgentWarehouseEnv）
- **智能体数量**: 2个
- **算法**: 
  - Independent Q-Learning (IQL): 每个智能体独立学习
  - Cooperative Q-Learning (CQL): 智能体共享Q表，学习联合策略
- **训练参数**:
  - IQL: 15,000 episodes, learning_rate=0.1, discount_factor=0.98
  - CQL: 20,000 episodes, learning_rate=0.12, discount_factor=0.98

### 2.2 实验结果

| 指标 | Independent Q-Learning | Cooperative Q-Learning |
|------|----------------------|----------------------|
| **平均奖励** | 384.68 ± 163.38 | 414.78 ± 11.31 |
| **成功率** | 90.00% | 100.00% |
| **平均episode长度** | 37.89 ± 54.15 | 34.51 ± 7.07 |

### 2.3 结果分析

**主要发现**:
1. **成功率**: CQL达到100%的成功率，显著优于IQL的90%
2. **奖励稳定性**: CQL的奖励标准差（11.31）远小于IQL（163.38），表明CQL学习更加稳定
3. **Episode长度**: CQL的平均episode长度更短且更稳定（34.51 ± 7.07 vs 37.89 ± 54.15），说明CQL能够更高效地完成任务

**可视化结果**: 训练曲线图（`results/training_curves.png`）显示了两种方法的训练过程对比，包括：
- Episode奖励变化趋势
- 成功率随训练的变化
- Episode长度分布

## 3. Final Project: 深度强化学习实验

### 3.1 实验设置

#### 3.1.1 自定义仓库环境（WarehouseParallelEnv）
- **环境类型**: PettingZoo并行接口
- **网格大小**: 6×6
- **智能体数量**: 3个
- **最大步数**: 200
- **算法**: Independent Deep Q-Network (IDQN)
- **训练参数**: 
  - Episodes: 800
  - Learning rate: 3e-4
  - Buffer size: 250,000
  - Batch size: 256

#### 3.1.2 PettingZoo MPE Simple Spread环境
- **环境**: PettingZoo的simple_spread_v3
- **智能体数量**: 3个
- **最大cycles**: 80
- **算法**: 
  - IDQN: 1,500 episodes
  - Double DQN (DDQN): 1,800 episodes
- **训练参数**:
  - Learning rate: 3e-4
  - Buffer size: 300,000 (IDQN), 250,000 (DDQN)
  - Batch size: 512 (IDQN), 256 (DDQN)

### 3.2 实验结果

#### 3.2.1 Warehouse环境 - IDQN

| 指标 | 数值 |
|------|------|
| **平均Episode奖励** | -375.57 ± 246.24 |
| **成功率** | 1.38% (平均) |
| **训练Episodes** | 800 |
| **平均Loss** | 1.17 ± 0.29 |

**观察**: 
- 训练曲线: `results/training_curves_warehouse_idqn.png`
- 结果文件: `results/experiment_results_deep_warehouse_idqn.pkl`
- 训练过程中成功率较低（平均1.38%），表明在复杂多智能体环境中，简单的IDQN方法难以有效学习协作策略
- 奖励波动较大（标准差246.24），说明训练过程不稳定

#### 3.2.2 MPE Simple Spread环境 - IDQN vs DDQN

| 指标 | IDQN | Double DQN |
|------|------|------------|
| **平均Episode奖励** | -77.24 ± 26.17 | -77.21 ± 24.51 |
| **成功率** | 0.00% | 0.00% |
| **训练Episodes** | 1,500 | 1,800 |
| **平均Loss** | 0.31 ± 0.16 | 0.37 ± 0.18 |

**结果分析**:
- **IDQN训练曲线**: `results/training_curves_mpe_simple_spread.png`
- **DDQN训练曲线**: `results/training_curves_mpe_simple_spread_ddqn.png`
- **结果文件**: 
  - `results/experiment_results_deep_mpe.pkl` (IDQN)
  - `results/experiment_results_deep_mpe_ddqn.pkl` (DDQN)

**主要发现**:
1. **成功率**: 两种方法在MPE Simple Spread环境中均未能达到成功（成功率0%），表明该环境对多智能体协作要求较高
2. **奖励表现**: DDQN的奖励标准差略小（24.51 vs 26.17），显示训练过程相对更稳定
3. **训练挑战**: 深度Q学习方法在复杂多智能体环境中面临样本效率低、探索困难等问题

**Double DQN改进**: 为了解决IDQN在复杂环境中的过估计问题，我们实现了Double DQN算法。Double DQN使用策略网络选择动作，目标网络评估Q值，从而减少Q值过估计偏差。虽然在本实验中DDQN未能显著提升成功率，但其训练稳定性有所改善。

### 3.3 深度学习方法的关键改进

1. **神经网络架构**: 使用2层MLP作为Q网络，能够处理连续状态空间
2. **经验回放**: 使用Replay Buffer存储和采样经验，打破数据相关性
3. **目标网络**: 使用目标网络稳定训练过程
4. **Double DQN**: 通过解耦动作选择和动作评估，减少过估计偏差

## 4. 方法比较与讨论

### 4.1 表格化方法 vs 深度学习方法

| 特性 | 表格化Q学习 | 深度Q学习 |
|------|------------|----------|
| **状态表示** | 离散状态空间 | 连续状态空间 |
| **可扩展性** | 受状态空间限制 | 可处理高维状态 |
| **收敛速度** | 相对较快（简单环境） | 需要更多样本 |
| **适用环境** | 小规模离散环境 | 复杂连续环境 |

### 4.2 独立学习 vs 合作学习

**独立Q学习（IQL）**:
- 优点: 实现简单，计算效率高
- 缺点: 无法显式建模智能体间的协作关系

**合作Q学习（CQL）**:
- 优点: 能够学习联合最优策略，性能更稳定
- 缺点: 联合动作空间随智能体数量指数增长

### 4.3 深度强化学习的挑战

1. **样本效率**: 深度方法需要大量训练样本
2. **稳定性**: 训练过程可能不稳定，需要仔细调参
3. **过估计**: DQN存在Q值过估计问题，Double DQN可以缓解

## 5. 可视化结果

所有训练曲线图已保存在`results/`目录下：
- `training_curves.png`: IQL vs CQL对比
- `training_curves_warehouse_idqn.png`: Warehouse环境IDQN训练
- `training_curves_mpe_simple_spread.png`: MPE环境IDQN训练
- `training_curves_mpe_simple_spread_ddqn.png`: MPE环境DDQN训练

这些图表展示了：
- Episode奖励随训练的变化趋势
- 成功率随训练的变化
- Episode长度的变化
- Epsilon衰减过程（探索-利用平衡）

## 6. 结论

1. **合作学习优势**: 在表格化Q学习中，CQL显著优于IQL，特别是在任务完成率（100% vs 90%）和稳定性（标准差11.31 vs 163.38）方面
2. **深度学习方法**: 深度Q网络能够处理更复杂的环境，但在多智能体设置中面临挑战：
   - Warehouse环境IDQN成功率仅1.38%
   - MPE Simple Spread环境中IDQN和DDQN均未能达到成功
   - 需要更多的训练样本、更精细的超参数调优和更先进的算法
3. **算法改进**: Double DQN相比标准DQN在训练稳定性方面有所改善（奖励标准差从26.17降至24.51），但未能显著提升成功率
4. **环境适应性**: 不同算法在不同环境中的表现差异显著：
   - 表格化方法在简单离散环境中表现优异（CQL达到100%成功率）
   - 深度方法在复杂连续环境中需要更多优化和探索
5. **方法选择建议**: 
   - 对于小规模离散环境，表格化Q学习（特别是CQL）是更好的选择
   - 对于大规模连续环境，需要探索更先进的多智能体深度强化学习算法（如MADDPG、QMIX、VDN等）

## 7. 未来工作

1. 进一步优化深度Q网络的超参数
2. 探索其他多智能体深度强化学习算法（如MADDPG、QMIX等）
3. 在更多环境中验证算法的泛化能力
4. 研究更高效的探索策略

