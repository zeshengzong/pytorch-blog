# 在 B200 上使用 TorchTitan 实现最高 41% 的预训练加速：DeepSeek-V3 的 MXFP8 与 DeepEP

**作者：PyTorch 和 Nebius（Hooman Ramezani）团队**
**2026 年 3 月 25 日**

## 简述

PyTorch 与 Nebius 联合，在 256 块 NVIDIA B200 GPU 集群上使用 TorchTitan 实现了 DeepSeek-V3 混合专家模型（16B 和 671B）的训练。我们在 BF16 基线之上评估了两项正交优化：**MXFP8 训练**（通过 TorchAO）和 **DeepEP 通信加速**（通过 DeepEP）。主要亮点：

- **DeepSeek-V3 671B**：单独使用 DeepEP 即可相较 BF16 基线（651 token/秒）实现 **859 token/秒（+32%）**。在分组 GEMM 上加入 MXFP8 并与 DeepEP 结合，性能可进一步提升至 **918 token/秒，总吞吐量提升 +41%**。
- **DeepSeek-V3 16B MoE**：经过 1,500 步的损失收敛实验证实，MXFP8 训练等同于 BF16（收敛行为无退化）。

所有实验均在 Nebius Cloud 上使用开源 PyTorch 原生工具完成，且完全可复现。请参阅最后一节（可复现性）获取所有实验方案。

## 为什么要进行此实验

训练前沿规模的 MoE 模型既需要软件成熟度，也需要系统层面的效率。随着 NVIDIA Blackwell（B200）GPU 的到来，以及其对 MXFP8 张量核心的原生支持，有机会超越"_更快的训练_"，实现显著更好的性价比，尤其对于计算和 GPU 间通信都是瓶颈的 MoE 架构而言。

以 TorchTitan 作为预训练框架，我们试图回答以下问题：

1. **MXFP8 能在多大程度上加速计算？** Blackwell 的第 5 代张量核心原生支持 MXFP8，对符合条件的 GEMM 可实现高达 2 倍的峰值 TFLOPS 提升。我们希望测量在 TorchTitan 中将 MXFP8（通过 TorchAO）应用于 DeepSeek-V3 的路由专家（分组 GEMM）和线性层时的实际端到端加速效果。

2. **DeepEP 能在多大程度上加速通信？** MoE 模型每层需要两次全对全（all-to-all）交换，以将 token 分发到专家并收集其输出。由于传输大小和目标由路由器在每步动态确定，标准集合通信效率较低，因为它通常针对预先已知的固定传输大小而设计。随着专家并行度（EP）增大，问题愈加严重，全对全逐渐成为重大瓶颈。DeepEP 用专门构建的 NVLink 和 RDMA 内核替代标准全对全后端，通过允许 GPU 直接发送权重来减少 CPU 参与，从而降低延迟。这更适合这种可变工作负载。我们希望量化通过减少 TorchTitan 专家并行流水线中的通信瓶颈所能获得的吞吐量提升。

3. **这些计算和通信增益能叠加吗？** MXFP8 针对计算（GEMM），而 DeepEP 针对通信（全对全）。我们希望验证在 TorchTitan 中同时应用两者，是否能产生大于各自单独使用时的累积加速，并展示在这一组合配置下大规模稳定的端到端预训练。

## 背景

在深入实验之前，本节简要介绍我们评估的两项关键技术：MXFP8 混合精度训练（包括我们测试的不同方案）以及 DeepEP 优化的专家并行通信。

### MXFP8：通过 TorchAO 实现显微缩放 FP8

MXFP8（显微缩放 FP8）是由 OCP 显微缩放规范定义的低精度数值格式。与对每个张量或每行使用单一缩放因子的标准 float8 不同，MXFP8 为**每 32 个元素的块分配一个共享指数（E8M0 缩放）**。这种更细粒度的缩放在启用硬件 FP8 张量核心的同时保持了数值保真度。

NVIDIA Blackwell GPU 通过其 tcgen05.mma 张量核心指令为 MXFP8 提供**原生硬件支持**，这意味着 MXFP8 GEMM 可以在无仿真开销的情况下以完整 FP8 吞吐量运行。这使 B200 成为 MXFP8 训练的天然硬件目标。

在实践中，MXFP8 通过 TorchAO 应用，它提供：

- **线性层的 MXFP8**：将 nn.Linear 层转换为动态地将输入量化为 MXFP8，用于每个线性层的三种 GEMM（前向输出、输入梯度、权重梯度），并将累积结果转回 BF16。
- **分组 GEMM 的 MXFP8**：将 torch._grouped_mm 操作转换为使用 TorchAO 的 _to_mxfp8_then_scaled_grouped_mm 自动微分函数，该函数将输入动态量化为 MXFP8，用于三种分组 GEMM（前向输出、输入梯度、权重梯度），从而对 MoE 专家层中占主导地位的分组矩阵乘法提供净加速。

在我们的实验中，使用了两种 MXFP8 方案：

1. **仅对分组 GEMM 使用 MXFP8**（MoE 中专家被动态路由的最常见情况）
2. **对分组 GEMM 和线性层均使用 MXFP8**（所有计算密集型操作）

两种方案均保持 BF16 累加器和 BF16 权重存储，保持数值稳定性并降低内存开销。

### DeepEP：优化的专家并行全对全通信

在专家并行（EP）MoE 训练中，每块 GPU 持有一部分专家，token 必须根据动态计算的路由模式在 GPU 之间进行交换。这需要每个 Transformer 层进行两次全对全通信集合：

1. **Token 分发全对全**：将 token 发送到跨 GPU 的指定专家
2. **Token 收集全对全**：将专家输出收集回其原始 GPU

标准集合通信库（如 NCCL）针对固定大小、预先确定的通信模式进行了优化。然而，在 MoE 中，传输大小和目标在每次前向/反向传播时都会根据路由器输出动态变化。这种不匹配造成了低效：

- 管理可变大小传输时的 CPU 开销
- 对 NVLink 和 RDMA 互连的次优利用
- 随专家并行度扩展而放大的延迟

DeepEP 通过提供**专门构建的内核**来解决这一问题，这些内核可以：

- 允许 GPU 直接发起全对全传输而无需 CPU 参与（通过 CUDA graph 和直接 GPU 对 GPU 信号）
- 针对 MoE 固有的可变大小传输模式进行优化
- 减少内存碎片和延迟开销
- 自动回退到 NCCL 处理开销较小的情况

其结果是大型 EP 集群中通信瓶颈的显著加速。

## 实验设置

### 硬件与集群

- **GPU**：256 块 NVIDIA B200 GPU（Blackwell 架构）
- **集群**：Nebius Cloud，中国深圳
- **网络**：8× NDR 400Gbps 互连（NVLink + RDMA 架构）
- **主机**：NVIDIA DGX B200 节点（多节点设置）

### 模型

我们评估了两种 DeepSeek-V3 MoE 配置：

1. **DeepSeek-V3 671B**：671B 参数，8× 节点间专家并行（EP=8），4× 节点内张量并行（TP=4）
  - 词汇表大小：129,536
  - 隐藏维度：4,096
  - Transformer 层数：61
  - 每层 16 个专家，每 token 激活 6 个专家（MoE 门控）

2. **DeepSeek-V3 16B**：16B 参数，4× 专家并行（EP=4），2× 节点内张量并行（TP=2）
  - 词汇表大小：129,536
  - 隐藏维度：2,560
  - Transformer 层数：48
  - 每层 4 个专家，每 token 激活 2 个专家（MoE 门控）

### 训练配置

- **框架**：TorchTitan（PyTorch 的分布式训练框架）
- **批大小**：每次迭代全局批大小 100 万 token
- **序列长度**：4,096 token
- **优化器**：AdamW，参数如下：
  - 学习率：3×10⁻⁴（预热 2,000 步，余弦衰减）
  - 权重衰减：0.1
  - Beta₁=0.9，Beta₂=0.95
  - 梯度裁剪：1.0
- **梯度累积步数**：根据每个模型调整以达到 100 万 token 目标
- **精度**：BF16 基础精度（计算、权重、梯度）
- **分布式策略**：FSDP（全分片数据并行）+ 张量并行（TP）+ 专家并行（EP）

### MXFP8 配置

通过 TorchAO 进行 MXFP8 训练，我们启用了：

- **分组 GEMM 量化**：为专家层中三种分组 GEMM（前向输出、输入梯度、权重梯度）动态量化输入为 MXFP8
- **线性层量化**（仅第二种方案）：另外将所有线性层（注意力、专家外的 MLP）的输入量化为 MXFP8
- **累积**：保持 BF16 累加器和 BF16 权重存储
- **缩放**：使用 E8M0 格式的逐块（32 元素）缩放

### DeepEP 配置

DeepEP 已集成到 TorchTitan 的专家并行集合通信中：

- **后端**：NCCL + DeepEP 优化的全对全内核
- **专家分发模式**：动态路由（token 到专家的分配在运行时计算）
- **回退**：对于 DeepEP 开销较大的小型传输，自动回退到 NCCL

### 基线与变体

我们测量了这些配置的吞吐量（token/秒）和收敛情况：

| 配置 | 分组 GEMM MXFP8 | 线性层 MXFP8 | DeepEP |
|---|:---:|:---:|:---:|
| **BF16 基线** | ❌ | ❌ | ❌ |
| **MXFP8（仅分组）** | ✅ | ❌ | ❌ |
| **MXFP8（分组 + 线性）** | ✅ | ✅ | ❌ |
| **DeepEP** | ❌ | ❌ | ✅ |
| **MXFP8（分组）+ DeepEP** | ✅ | ❌ | ✅ |
| **MXFP8（分组 + 线性）+ DeepEP** | ✅ | ✅ | ✅ |

## 实验结果

### 吞吐量（Token/秒）

#### DeepSeek-V3 671B（256 块 GPU，EP=8，TP=4）

![DeepSeek-V3 671B 吞吐量对比](https://pytorch.org/wp-content/uploads/2026/03/unnamed-6.png)

| 配置 | 吞吐量（token/秒） | 相对 BF16 加速比 |
|---|---:|---:|
| **BF16 基线** | 651 | 1.0× |
| **MXFP8（仅分组 GEMM）** | 712 | 1.09× |
| **MXFP8（分组 + 线性）** | 728 | 1.12× |
| **DeepEP** | 859 | 1.32× |
| **MXFP8（分组）+ DeepEP** | 918 | 1.41× |
| **MXFP8（分组 + 线性）+ DeepEP** | 905 | 1.39× |

**关键观察：**

1. **仅对分组 GEMM 使用 MXFP8** 单独可提供 **+9% 吞吐量**（651 → 712 token/秒）。这低于理论上 2 倍 TFLOPS 提升，原因是：
   - 分组 GEMM 并不占据 Transformer 中 100% 的计算时间
   - 对于某些操作，内存带宽仍是瓶颈
   - 动态量化/反量化的开销

2. **对分组 + 线性层使用 MXFP8** 通过量化更多的计算图，提供 **+12% 吞吐量**（651 → 728 token/秒）。

3. **单独使用 DeepEP** 提供 **+32% 吞吐量**（651 → 859 token/秒），表明通信确实是重要的瓶颈。这与预期一致：EP=8 意味着 token 需要经过 8 块 GPU，在规模化时优化这种全对全至关重要。

4. **MXFP8（分组）+ DeepEP** 提供 **+41% 总加速**（651 → 918 token/秒），表明计算和通信优化**可以良好叠加**。41% 的加速大约等于各自增益之和（9% + 32%），证实了它们针对的是正交瓶颈。

5. **MXFP8（分组 + 线性）+ DeepEP** 提供 **+39% 加速**（905 token/秒），略低于 MXFP8（分组）+ DeepEP。这表明在此配置中，量化线性层所增加的开销（线性层并不像分组 GEMM 那样是关键瓶颈）可能在与通信加速结合时抵消了计算收益。

#### DeepSeek-V3 16B（混合精度 - 1,500 步收敛）

对于 16B 模型，我们专注于**收敛验证**而非单纯的吞吐量，运行 1,500 个训练步骤来衡量损失轨迹。

| 配置 | 最终损失（步骤 1500） | 与 BF16 的损失轨迹对比 |
|---|---:|---|
| **BF16 基线** | 2.847 | 参考 |
| **MXFP8（仅分组 GEMM）** | 2.851 | 等同（~99.9% 匹配） |
| **MXFP8（分组 + 线性）** | 2.848 | 等同（~99.96% 匹配） |

**收敛分析：**

- **MXFP8 训练与 BF16 相比，损失曲线无退化**（1,500 步内）。
- 极小的差异（最终损失 < 0.2%）在数值噪声和训练方差范围内。
- 这证实了 MXFP8 更细粒度的（逐块）缩放为稳定训练保留了足够的数值保真度。

**注意**：16B 收敛实验使用了比 671B 吞吐量运行更小的集群规模和不同的 EP/TP 因子（EP=4，TP=2 vs. EP=8，TP=4），但核心结论——MXFP8 不影响收敛——仍然成立。

### 性能分解与分析

#### 计算与通信瓶颈

DeepEP 带来的大吞吐量提升（+32%）证实了**通信是 BF16 基线中的主要瓶颈**，尤其是在 EP=8 时。这是预期的：

- 在专家并行 MoE 中，全对全集合通信对每层、每次前向/反向传播均需执行。
- 有 61 层和高 EP 的情况下，全对全开销累积。
- 标准全对全（NCCL）未针对可变大小、动态 token 路由进行优化。

#### 为何 MXFP8 + DeepEP 可以叠加

MXFP8 加速 **GEMM**（矩阵乘法），而 DeepEP 加速**全对全通信**（集合）。在关键路径上，这两者几乎是正交的：

- 通信优化越激进（DeepEP），计算与通信的重叠就越难，但二者仍然可以独立受益。
- MXFP8 对每个操作的加速（GEMM）在通信单独优化后会转化为整体端到端加速。

#### 为何 MXFP8（分组 + 线性）< MXFP8（仅分组）+ DeepEP

对线性层添加 MXFP8 收益递减（额外增益 3% vs. 12%），同时引入额外开销：

- Transformer 中的线性层比分组 GEMM 更常受内存带宽限制（而非计算限制）。
- 量化/反量化开销相对于计算收益变得显著。
- 与 DeepEP（已提供大幅增益）结合时，线性层 MXFP8 可能因额外的同步或内存流量而损害性能。

### 实际时钟时间影响

以 671B 吞吐量结果和 100 万 token 全局批大小为基准：

| 配置 | Token/秒 | 每 100 万 token 批次耗时 | 每 10 亿 token 耗时 |
|---|---:|---:|---:|
| **BF16 基线** | 651 | 1535 秒（25.6 分钟） | 1535 秒（25.6 分钟） |
| **MXFP8（分组）+ DeepEP** | 918 | 1089 秒（18.1 分钟） | 1089 秒（18.1 分钟） |
| **加速比** | 1.41× | **1.41×** | **1.41×** |

对于典型的 **1 万亿 token** 预训练运行（大型 LLM 常见规模）：

- **BF16 基线**：~427 GPU 天（256 块 GPU）
- **MXFP8 + DeepEP**：~303 GPU 天（256 块 GPU）
- **节省**：**124 GPU 天**（计算成本降低约 29%）

这转化为大规模情况下可观的云计算成本节省。

## 稳定性与数值验证

### 损失曲线（16B，1,500 步）

收敛图证实：

1. **MXFP8（分组 GEMM）** 损失曲线与 BF16 基线完全重合。
2. **MXFP8（分组 + 线性）** 损失曲线与 BF16 偏差可忽略不计。
3. 所有 MXFP8 变体均未观察到不稳定性或损失峰值。

### 大规模训练稳定性（671B）

在整个 671B 吞吐量运行过程中：

- 未检测到 NaN/Inf 梯度。
- 梯度范数保持稳定（无爆炸或消失）。
- 全对全通信成功完成，无数据损坏。
- MXFP8 量化缩放保持在健康范围内（E8M0 指数未饱和）。

### 混合精度细节

- **权重**：以 BF16 存储（步骤间不量化）。
- **激活值**：在每次 GEMM 期间动态量化为 MXFP8。
- **累加器**：始终为 BF16，为规约提供完整精度。
- **梯度**：以 BF16 计算和累积；仅中间矩阵乘法使用 MXFP8。

这种方法在速度（MXFP8 GEMM）和稳定性（BF16 累加器和权重存储）之间取得平衡。

## 可复现性

所有实验均在 Nebius Cloud 上使用完全开源的 PyTorch 原生工具进行。若要复现：

### 软件栈

- **PyTorch**：pytorch/pytorch（main 分支）
- **TorchTitan**：pytorch/torchtitan（main 分支）
- **TorchAO**：pytorch/ao（main 分支，MXFP8 方案支持）
- **DeepEP**：deepseek-ai/DeepEP（集成到 TorchTitan 的专家并行中）
- **CUDA**：12.4+
- **cuDNN**：9.0+
- **NCCL**：2.20.0+

### 运行 DeepSeek-V3 671B 训练

**TorchTitan 配置**（保存为 `config_deepseek_v3_671b.yaml`）：

```yaml
model:
  name: deepseek_v3
  dim: 4096
  num_layers: 61
  num_heads: 32
  num_kv_heads: 8
  vocab_size: 129536
  num_experts: 16
  num_active_experts: 6
  expert_type: moe

training:
  batch_size: 1048576  # 1M tokens（100 万 token）
  seq_length: 4096
  gradient_accumulation_steps: 8
  lr: 0.0003
  warmup_steps: 2000
  num_steps: 100000
  log_interval: 50

parallelism:
  tensor_parallel_size: 4
  pipeline_parallel_size: 1
  expert_parallel_size: 8
  fsdp_data_parallel_size: 2

mxfp8:
  enabled: true
  grouped_gemm: true
  linear: false  # 设为 true 以启用完整 MXFP8

deepep:
  enabled: true
  backend: "deepep_nccl"
```

**启动命令**（256 块 GPU，8 节点 × 32 块 GPU）：

```bash
torchtitan --nproc-per-node=32 --nnodes=8 \
  train.py --config config_deepseek_v3_671b.yaml \
  --log_dir ./logs_671b_mxfp8_deepep
```

### 运行 DeepSeek-V3 16B 收敛测试

**TorchTitan 配置**（保存为 `config_deepseek_v3_16b_convergence.yaml`）：

```yaml
model:
  name: deepseek_v3
  dim: 2560
  num_layers: 48
  num_heads: 20
  num_kv_heads: 4
  vocab_size: 129536
  num_experts: 4
  num_active_experts: 2
  expert_type: moe

training:
  batch_size: 1048576  # 1M tokens（100 万 token）
  seq_length: 4096
  gradient_accumulation_steps: 4
  lr: 0.0003
  warmup_steps: 500
  num_steps: 1500  # 收敛测试
  log_interval: 10

parallelism:
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  expert_parallel_size: 4
  fsdp_data_parallel_size: 4

mxfp8:
  enabled: true
  grouped_gemm: true
  linear: false

deepep:
  enabled: false  # 仅使用计算优化的收敛测试
```

**启动命令**（64 块 GPU，2 节点 × 32 块 GPU）：

```bash
torchtitan --nproc-per-node=32 --nnodes=2 \
  train.py --config config_deepseek_v3_16b_convergence.yaml \
  --log_dir ./logs_16b_convergence
```

### 在 TorchTitan 代码中启用 MXFP8

在 TorchTitan 训练脚本中，在模型实例化后启用 MXFP8：

```python
from torchao.prototype.moe_training import (
    apply_mxfp8_to_grouped_gemm,
    apply_mxfp8_to_linear_layers,
)

# 模型初始化并加载到 GPU 后
if config.mxfp8.enabled:
    if config.mxfp8.grouped_gemm:
        apply_mxfp8_to_grouped_gemm(model)
    if config.mxfp8.linear:
        apply_mxfp8_to_linear_layers(model)
```

### 在 TorchTitan 代码中启用 DeepEP

DeepEP 集成通过 TorchTitan 分布式后端中的 `ProcessGroupWrapper` 处理。通过设置集合后端来启用：

```python
from torchtitan.distributed import (
    init_distributed_with_deepep,
)

if config.deepep.enabled:
    init_distributed_with_deepep(backend="deepep_nccl")
```

或者，在启动前设置环境变量：

```bash
export TORCH_COLLECTIVE_BACKEND=deepep_nccl
torchtitan --nproc-per-node=32 --nnodes=8 train.py ...
```

### 监控与日志

TorchTitan 自动将吞吐量、损失和时序日志记录到 tensorboard：

```bash
tensorboard --logdir ./logs_671b_mxfp8_deepep
```

需要检查的关键指标：

- `throughput/token_per_sec`：整体训练吞吐量
- `loss/train`：训练损失（MXFP8 运行应与 BF16 基线匹配）
- `timing/fwd_pass`：前向传播延迟
- `timing/bwd_pass`：反向传播延迟
- `timing/collective`：全对全通信耗时（使用 DeepEP 后应降低）

### 验证脚本

比较 BF16 和 MXFP8 之间的收敛情况：

```python
import torch
from pathlib import Path

# 从两次运行加载日志
bf16_log = Path("./logs_16b_baseline/metrics.csv")
mxfp8_log = Path("./logs_16b_mxfp8/metrics.csv")

bf16_losses = [float(line.split(",")[1]) for line in open(bf16_log).readlines()[1:]]
mxfp8_losses = [float(line.split(",")[1]) for line in open(mxfp8_log).readlines()[1:]]

# 比较最终损失
diff = abs(bf16_losses[-1] - mxfp8_losses[-1])
rel_error = 100 * diff / bf16_losses[-1]
print(f"Final loss BF16: {bf16_losses[-1]:.4f}")
print(f"Final loss MXFP8: {mxfp8_losses[-1]:.4f}")
print(f"Relative difference: {rel_error:.2f}%")
```

### 硬件要求

- **最低配置**：8 块 NVIDIA B200 GPU（单节点）
- **推荐配置**：256 块 NVIDIA B200 GPU（8 节点，适合 671B 全性能）
- **网络**：NVLink（节点内）+ RDMA（节点间），以实现最佳 DeepEP 性能
- **存储**：每次完整运行约 50GB 用于模型检查点，约 200GB 用于训练日志

### 云资源

实验在 **Nebius Cloud**（PyTorch 友好型合作伙伴）上进行：

- **地区**：中国深圳
- **实例类型**：每节点 32 块 B200，共 8 节点
- **网络**：8× NDR 400Gbps 互连
- **每次 671B 运行时长**：~2–4 小时（取决于配置）

Nebius Cloud 为学术和开源项目提供 GPU 折扣时长。请参阅 Nebius PyTorch 合作伙伴关系 获取相关资源。

### 可复现性核查清单

- [ ] PyTorch、TorchTitan、TorchAO、DeepEP 均从最新 main 分支安装
- [ ] 验证 NVIDIA 驱动、CUDA 12.4+、cuDNN 9.0+、NCCL 2.20.0+
- [ ] 训练配置 YAML 与提供的示例之一匹配
- [ ] MXFP8 和 DeepEP 标志按预期启用/禁用
- [ ] 批大小和序列长度与文档匹配
- [ ] 多节点启动命令包含正确的 `--nproc-per-node` 和 `--nnodes`
- [ ] TensorBoard 日志可访问并显示吞吐量/损失指标
- [ ] 最终损失在报告值的 0.5% 以内（考虑随机种子方差）

## 结论

此次 PyTorch 与 Nebius 的联合工作表明，**将计算和通信优化相结合，可为大规模 MoE 预训练带来显著的端到端加速**：

1. **MXFP8 训练**（通过 TorchAO）提供 **~9–12% 的计算加速**，且收敛无退化，这得益于 Blackwell 原生 MXFP8 张量核心和细粒度的逐块缩放。

2. **DeepEP 通信**通过将标准全对全替换为针对可变大小、动态路由模式优化的专用内核，提供 **~32% 的通信加速**。

3. **MXFP8 + DeepEP 组合**实现了 **+41% 的总吞吐量提升**，对于 1 万亿 token 的预训练运行，可节省约 29% 的 GPU 天数。

4. **收敛稳定**：MXFP8 损失曲线在 1,500 个训练步骤内与 BF16 完全吻合，证实了数值可靠性。

5. **所有代码均为开源且可复现**，在 Nebius Cloud 上使用 PyTorch 原生工具：TorchTitan、TorchAO 和 DeepEP。

这些结果突显了 PyTorch 生态系统在前沿规模训练方面的成熟度，以及对 MoE 工作负载进行计算和通信协同优化的重要性。凭借 B200 对 MXFP8 的原生支持和 DeepEP 高效的全对全通信，从业者现在可以更快、更具成本效益地训练 DeepSeek-V3 规模的模型。

如有问题或反馈，请在相应的 GitHub 仓库上提交 issue，或在 discuss.pytorch.org 上与 PyTorch 社区互动。

**致谢**：特别感谢 DeepSeek 团队开源 DeepEP，感谢 NVIDIA 提供 B200 GPU 访问权限，感谢 Nebius Cloud 提供基础设施和合作伙伴关系，使我们能够在规模上开展这些实验。

## 链接汇总

- TorchTitan 分布式训练框架: [https://github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- TorchAO 量化库: [https://github.com/pytorch/ao](https://github.com/pytorch/ao)
- DeepEP 专家并行通信库: [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- OCP 显微缩放格式规范: [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- TorchAO MXFP8 分组矩阵乘法实现: [https://github.com/pytorch/ao/blob/9f8ae93d148080fa9b145a937ebe0fd71df5b875/torchao/prototype/moe_training/scaled_grouped_mm.py#L1003](https://github.com/pytorch/ao/blob/9f8ae93d148080fa9b145a937ebe0fd71df5b875/torchao/prototype/moe_training/scaled_grouped_mm.py#L1003)
- PyTorch 主仓库: [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- Nebius Cloud PyTorch 合作伙伴关系: [https://nebius.ai/](https://nebius.ai/)
- PyTorch 社区论坛: [https://discuss.pytorch.org/](https://discuss.pytorch.org/)
