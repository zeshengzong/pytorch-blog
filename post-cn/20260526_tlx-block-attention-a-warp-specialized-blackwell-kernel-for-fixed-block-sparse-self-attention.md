# TLX Block Attention：面向固定块稀疏自注意力的 Warp 专用 Blackwell 内核

作者：Jake Siso、Dev (Devashish) Shankar、Jackie (Jiaqi) Xu、Jacky Zhou、Darren Liu、Han Xu、Yasmine Badr、Dan Chanpuriya、Hongtao Yu、Max Leung

2026 年 5 月 26 日

_代码地址：https://github.com/facebookresearch/ads_model_kernel_library_

_本文介绍了 TLX Block Attention 的设计——这是一个面向 NVIDIA Blackwell GPU 的 Triton 内核，利用对块对角注意力模式的编译时先验知识，消除了通用注意力实现中存在的整类算法开销。在 NVIDIA B200 GPU 上，该内核相比 Flash Attention v2 实现了约 1.85× 的前向加速和约 2.50× 的反向加速；当旋转位置嵌入融入注意力尾声（epilogue）时，组合注意力与旋转反向传播的速度提升约 3.5×。_

本工作构建于 TLX（Triton 语言扩展）之上——TLX 是一组面向 Triton 编译器的底层扩展，在 NVIDIA Blackwell GPU 上暴露了对 warp 专用化、异步张量核心操作以及内存层次管理的硬件原生控制能力。TLX 弥合了 Triton 高层 Python 生产力与传统上需要原始 CUDA 或 CUTLASS 才能实现的细粒度硬件控制之间的差距。更多关于 TLX 的信息，请参见 triton-ext 仓库。

## 1. 引言

自注意力是一种让模型衡量序列中每个元素对其他所有元素重要性的机制——本质上是在问"这段输入的哪些部分应该相互影响理解？"它是 Transformer 架构的核心构建块，也是这些模型能够捕获数据中丰富的上下文相关关系的原因。一个直观的类比是：过去的决策如何影响当前和未来的决策？

块对角自注意力——将序列划分为固定大小的组，每组仅在内部进行注意力计算——是推荐系统和特征交互模型中广泛使用的模式（BlockBERT，Qiu 等人，EMNLP 2020）。在我们的广告排序栈中，生产工作负载通常以 1152 的批量大小运行，序列长度最长约 4k tokens，注意力头维度为 64 或 128，随着序列长度增加，注意力结构中约有 70% 的稀疏度。随着这些模型越来越深、越来越宽，注意力计算成本逐渐成为主要瓶颈。

目前这些工作负载运行在通用内核上，如带块掩码或滑动窗口的 Flash Attention v2。FlexAttention（FA4）支持块稀疏模式，但最小 tile 大小为 256——与这些模型所需的 64 token 块不兼容。带块掩码的 Flash Attention v2 在该 tile 大小下仍是最强的可用基线，但留下了大量性能空间。Flash Attention 的分块迭代、在线 softmax 修正、logsumexp 记账以及辅助内核启动对于任意长度的因果注意力是必不可少的——但当模式是块对角且在编译时已知时，这些都是纯粹的开销。

**本工作的核心论点：当你在编译时知道注意力模式时，你可以构建出快得多的实现。** 我们利用每个 Q tile 恰好关注一个 K/V tile 这一固定约束，将这一知识贯穿整个算法，将多迭代累加器折叠为单次 GEMM，消除修正阶段，并移除辅助内核启动。

## 2. 为什么选择块注意力？

### 2.1 固定块约束及其级联简化

标准 Flash Attention 通过将一个 Q tile 在多个 K/V tile 上迭代来处理任意长度的序列，维护运行时统计量（逐行最大值和 log-sum-exp），并在每一步应用修正因子以保持数值稳定性：

_代码清单 1：标准 Flash Attention 内循环，展示多 tile 迭代和在线 softmax 修正。_

```python
# Flash Attention 内循环（标准版）
for k_tile in K_tiles:
    S = Q @ k_tile.T                   # 部分得分
    m_new = max(m_old, rowmax(S))
    alpha = exp(m_old - m_new)         # 修正因子
    O = alpha * O + exp(S - m_new) @ v_tile
    l = alpha * l + rowsum(exp(S - m_new))
O = O / l                              # 最终归一化
# 将 L = m + log(l) 存储到 HBM 供反向传播使用
```

这对于任意序列是正确且优雅的。但对于固定 64 token 块大小的块对角注意力，整个 Q-tile 遍历 K-tiles 的循环被简化为**单次迭代**。每个 Q tile 与其对应的 K/V tile 是同一个 tile。这一约束在整个算法中级联展开：

1. **无多 tile 迭代。** 得分矩阵 S = Q · Kᵀ ∈ ℝ^{64×64} 在一次 GEMM 后即完整。不存在需要跨迭代维护状态的循环。

2. **无在线 softmax 修正。** 由于只有一个 tile，对 S 计算的逐行最大值和行和在全局范围内立即正确。修正因子 α = exp(m_old − m_new) 恒等于 1，可以完全省略。

3. **无 logsumexp（L）存储。** Flash Attention 将每行的 log-sum-exp L 存储到 HBM，以便反向传播重新计算 softmax。有了单个 tile，反向传播可以直接从 Q、K、V 重新计算 P = softmax(S)，无需任何辅助张量——消除了每个前向/反向对中的一整次 HBM 写入和读取。

4. **无 Di 预处理内核。** 标准 Flash Attention 反向传播在主反向传播之前会启动一个独立的内核来计算 Di = rowsum(dO ⊙ O)。在 TLX Block Attention 中，Di 在 dP/dS 反向传播阶段**内联**计算，消除了一次内核启动及其相关的内存流量。

5. **无带重缩放的输出累加。** 有了单个 tile，输出 O = P · V 是来自单次 GEMM 的全新结果，而非多个重缩放部分结果的累加。这使得所有 async_dot 调用都能使用 `use_acc=False`——告知张量核心硬件 TMEM 累加器无需跨调用保留，可以自由重用。

_代码清单 2：`use_acc=False` 向硬件发出信号，表明不需要跨 tile 累加，从而实现 TMEM 复用。_

```python
# 来自内核：use_acc=False 表示不需要累加
tlx.async_dot(
    q_tile[buff_idx],
    k_tile_T,
    TMEMqk[tmem_idx],
    use_acc=False,           # 全新结果——无需累加
    mBarriers=[qk_SMEM_free[buff_idx], qk_TMEM_full[tmem_idx]],
)
```

### 2.2 与标准 Flash Attention 的比较

下表总结了算法差异：

| 方面 | 标准 Flash Attention | TLX Block Attention |
|------|----------------------|---------------------|
| 每个 Q tile 的 K tile 数 | 多个（完整序列） | 恰好 1 个（同一块） |
| 得分矩阵 | 多个 tile 累加 | 单个 [64, 64]——完整 |
| Logsumexp L 张量 | 存储到 HBM 供反向使用 | 不需要 |
| 运行时最大值/行和 | 跨 tile 维护 | 计算一次，寄存器内消费 |
| 修正因子 α | 每次迭代都需要 | 不需要（已省略） |
| 输出累加 | 带重缩放的增量累加 | 单次 P·V GEMM |
| use_acc 模式 | True（跨 tile 累加） | False（全新结果） |
| Di 预处理 | 独立内核启动 | 内联计算 |

_表 1：标准 Flash Attention 与 TLX Block Attention 的算法差异。_

这些不是微优化——它们代表着整个算法阶段的消除。反向传播尤其受益显著：不存储 L 张量消除了每批次 × 头数 × 序列的一次 HBM 往返，而内联 Di 计算则消除了一次内核启动及其相关的驱动开销和内存带宽。

## 3. 内核架构：Warp 专用流水线

### 3.1 TLX

我们选择 Triton 作为编写框架，因为它提供了一种 Python 原生、面向 tile 的编程模型，自然地映射到下文描述的 warp 专用流水线结构——同时避免了原始 CUDA 或 CUTLASS 的样板代码，并在编译器演进过程中保持可移植性。Triton 的 TLX（Triton 语言扩展）进一步暴露了 Blackwell 特定的原语，如 async_dot、local_trans 以及显式的 TMEM/SMEM 屏障管理，在硬件控制与开发效率之间实现了良好平衡。根据我们的经验，TLX 提供了与低级替代方案相当（甚至超越）的性能，同时由于其 Python 原生的简洁性，使得迭代速度显著更快。

具体而言，本内核依赖于几个超越基础 Triton 的 TLX 原语：tlx.async_dot 用于发出带显式累加器控制的 warp 专用 tcgen05 MMA 操作；tlx.async_descriptor_load 用于 TMA 驱动的 SMEM 填充；tlx.local_trans 用于 TMEM 到寄存器的传输；以及协调 warp 组间生产者-消费者流水线的 mBarrier 同步模型。这些扩展可在 triton-ext 仓库中获取。

### 3.2 Warp 专用化

TLX Block Attention 使用 **warp 专用化**——同一 CTA 内的不同 warp 被永久分配到不同的硬件单元，并在内核的整个生命周期内执行不同的代码路径。这与传统 CUDA 模型形成对比，在传统模型中，所有 warp 执行相同的代码，仅通过条件分支产生差异。

| 阶段 | Warp 数 | 寄存器数 | 硬件单元 | 职责 |
|------|---------|---------|---------|------|
| 加载 | 1 | 48 | TMA 引擎 | 对 Q、K、V 执行 async_descriptor_load |
| QK MMA | 1 | 48 | tcgen05 张量核心 | async_dot(Q, Kᵀ) → TMEMqk |
| Softmax | 4 | 120 | CUDA 核心 + SFU | 掩码 / 缩放 / exp2 / 归一化 → P 到 SMEM |
| PV MMA | 1 | 48 | tcgen05 张量核心 | async_dot(P, V) → TMEMpv |
| 尾声 | 8 | 200 | CUDA 核心 + L2 + TMA 引擎 | TMEM → 寄存器 → BF16 → SMEM → TMA 存储 |
| **合计** | **15** | — | — | **每 CTA 480 线程** |

_表 2：前向流水线阶段配置。寄存器分配有意设计为非对称——硬件加速阶段分配最少寄存器；CUDA 核心阶段分配最多。_

```
图 1 — 前向流水线 warp 时间线（概念图，一次迭代）：

时间 →
加载     [─ TMA Q,K ─][─ TMA V ─]
QK MMA         [── async_dot Q·Kᵀ ──]
Softmax                  [── exp2/归一化 → P ──]
PV MMA                            [── async_dot P·V ──]
尾声                                   [── local_load → BF16 → 存储 ──]
```

每个阶段的输出触发一个屏障，解除对下一阶段的阻塞，从而在硬件单元之间形成生产者-消费者流水线。当尾声 warp 将 tile _i_ 写入全局内存时，MMA warp 正在计算 tile _i+1_，而加载 warp 则通过 TMA 预取 tile _i+2_——同时有三个 tile 在飞行中。

### 3.3 Roofline 上下文

在 BLOCK_D=64、HEAD_DIM=128 时，算术强度约为 33 FLOP/byte——远低于 B200 的峰值强度约 281 FLOP/byte。该内核在设计上受内存带宽限制。这就是为什么通过 TMA 进行延迟隐藏以及最小化不必要的内存流量（已消除的 L 张量、融合的旋转位置嵌入）是主要的优化杠杆。

### 3.4 缓冲区管理

为了让硬件单元持续保持繁忙，内核使用三重缓冲的 SMEM（3 个槽位）和双重缓冲的 TMEM（2 个槽位），消耗了 256 KB SMEM 预算中约 169 KB。有了三个 SMEM 槽位，加载 warp 可以在 MMA warp 处理 tile i+1 和尾声 warp 排空 tile i 的同时预取 tile i+2。反向内核降为双重缓冲的 SMEM（约 162 KB），以便在相同的 256 KB 预算内容纳额外的梯度 tile。

## 4. 反向传播：不需要 Logsumexp 张量的梯度计算

在标准 Flash Attention 中，反向传播需要前向传播将 logsumexp 张量（L）保存到高带宽内存（HBM）。该张量在反向传播期间用于重建注意力概率（P）。此外，标准注意力需要一个独立的预处理内核来计算 Δᵢ（dO ⊙ out 的逐行求和）。

由于块对角注意力在单个 tile 中计算完整的 64×64 得分矩阵，我们可以完全绕过这两个要求。反向内核不读取任何 logsumexp 张量，也不需要独立的预处理步骤。相反，它完全内联重新计算 S = Q · Kᵀ 和 P = softmax(S)——当 tile 在单次遍历中就能适应时，这是一个代价低廉的操作。

这一级联简化使我们能够构建一个完全融合的、7 阶段 warp 专用反向流水线：

| 阶段 | Warp 数 | 寄存器数 | 硬件单元 | 职责 |
|------|---------|---------|---------|------|
| 加载 | 1 | 48 | TMA 引擎 | 加载 Q、K、V、dO（+ 旋转位置编码的 sin/cos） |
| QK MMA | 1 | 48 | tcgen05 张量核心 | 重新计算 S = Q · Kᵀ |
| Softmax/P | 4 | 120 | CUDA 核心 + SFU | 重新计算 P = softmax(S) |
| dV MMA | 1 | 48 | tcgen05 张量核心 | dV = Pᵀ · dO |
| dP/dS | 4 | 120 | TC + CUDA 核心 | dP = dO · Vᵀ，Δᵢ，dS |
| dQ/dK MMA | 1 | 48 | tcgen05 张量核心 | dQ = dS · K，dK = dSᵀ · Q |
| 尾声 | 8 | 200 | CUDA 核心 + L2 + TMA 引擎 | 存储 dQ、dK、dV（+ 融合旋转位置编码） |
| **合计** | **20** | — | — | **每 CTA 640 线程** |

_表 4：7 阶段反向流水线配置。_

反向传播本质上比前向传播更复杂。它需要 20 个 warp（每 CTA 640 线程）来平衡密集的计算需求。最值得注意的是，它完全饱和了 SM 上的 256 KB 张量内存。五个不同的 TMEM 缓冲区——TMEMqk、TMEMdv、TMEMdp、TMEMdq 和 TMEMdk——共同达到了 100% 的 TMEM 利用率。为了适应这一点，反向内核从前向传播中的三重缓冲 SMEM 降为双重缓冲 SMEM（约 162 KB / 256 KB，63%），同时保持双重缓冲 TMEM。

## 5. 可变长度序列的调度

现实世界的推荐和特征交互模型处理的序列长度并不是整齐划一的。相反，流量以锯齿状、可变长度的序列为主，这些序列被打包到单个扁平化缓冲区中。天真地将每个序列映射到一个 CTA，会导致短序列早早完成时 SM 空闲，而其他 SM 正在处理长序列——这是严重的负载不均衡。

为了最大化 SM 占用率，内核启动 `min(NUM_SMS, total_blocks)` 个持久程序——每个 SM 恰好一个持久线程块。工作负载通过两个预计算数组来平衡：

1. **BLOCK_PER_BATCH**：每个序列的 64-token tile 数量的前缀和。
2. **BLOCK_PER_PROGRAM**：分配给每个 SM 的平衡 tile 范围——使用封闭形式的除余算术而非累积求和计算。

为了消除 GPU 同步开销，当 CPU 端偏移张量可用（`cpu_offsets`）时，所有标量调度算术（tile 计数、除余、前缀和）都在内核启动前在 CPU 上计算——零 GPU 同步点。

在内核内部，每个 SM 必须确定给定的全局 tile 索引属于哪个序列（批次索引）。这使用了一个无分支的二分搜索，恰好执行 32 次迭代（对于任何合理的批量大小都足够），零线程同步。

## 6. 融合旋转反向传播：更高精度，更高速度

对于自注意力层，自注意力之前是投影 + 正弦信号。在反向传播中，这变为注意力反向传播 -> 正弦信号，传统上需要 2 次不同的内核启动。

### 6.1 基线：双内核反向传播

传统反向传播需要两次独立的内核启动：

1. 注意力反向内核——通过张量核心以 FP32 累加 dQ、dK、dV，然后在存储到全局内存时截断为 BF16。
2. 旋转反向内核——从全局内存重新加载 BF16 梯度，应用旋转共轭 R(−θ)，并存储最终的 BF16 结果。

这种分离有三个代价：

| 问题 | 影响 |
|------|------|
| **精度损失** | FP32 梯度在旋转变换_之前_被截断为 BF16——然后在最终存储时再次截断。两个量化点，每个注入约 0.4% 的相对误差（BF16 只有 7 个尾数位）。下游投影 GEMM 放大了累积误差。 |
| **内存带宽浪费** | dQ、dK、dV 被写入后立即重新读取——在 [total_seq_len, 1152] 张量（head_dim=128，3 个 KV 头）上完整往返。对于长达数百万的序列，这种流量相当可观。 |
| **内核启动开销** | 两次独立分发，而一次就足够了。 |

### 6.2 融合方案

注意力反向内核已经为梯度存储尾声专门分配了一个 warp 组。我们利用这一点，在梯度仍在 FP32 寄存器中时，将旋转共轭注入该尾声：

1. 张量核心以 FP32（TMEM）存储 dQ、dK、dV。
2. 将 FP32 值加载到寄存器中。
3. 以完整 FP32 精度应用 R(−θ)——轻量级的 sin/cos 加载 + 逐元素乘法。
4. 强制转换为 BF16 并发出单次全局存储。

逐步比较：

| 方面 | 基线（分离） | 融合内核 |
|------|------------|--------|
| 注意力反向计算 | FP32 | FP32 |
| 中间存储 | BF16 → 全局内存 | FP32 寄存器 |
| 旋转 sin/cos 操作 | BF16 | FP32 |
| BF16 量化点 | 2 | 1（仅最终存储） |
| 全局内存往返 | 2 | 0 |
| 内核启动次数 | 2 | 1 |

在反向尾声中融合旋转共轭。交织操作在仍处于 FP32 时将 R(−θ) 应用于成对的 [cos, sin] 分量。

```python
# 对 dV 应用旋转共轭（neg_sin 处理共轭）
dv0, dv1 = dvLocal.reshape(BLOCK_D, HALF_DIM, 2).split()
dvLocal = tl.interleave(
    dv0 * cos_local - dv1 * neg_sin,
    dv1 * cos_local + dv0 * neg_sin,
)
```

## 7. 性能结果

所有基准测试均在 NVIDIA B200 GPU（x86 CPU）上以 BF16 精度进行。主要配置使用 B=1152 序列，HEAD_DIM=128，H=4 个头，max_seq_len=2000，稀疏度=0.7——离散均匀分布（代表生产流量分布）。

### 7.1 内核级加速比

| 传播方向 | Flash Attention v2 带块注意力（ms） | TLX Block Attention（ms） | 加速比 |
|---------|-----------------------------------|--------------------------|--------|
| 前向 | 1.81 | 0.98 | **1.85×** |
| 反向 | 5.89 | 2.36 | **2.50×** |
| **合计** | **7.70** | **3.33** | **2.31×** |

_表 5：内核级性能比较（B=1152，D=128，H=4，BF16，B200，max_seq_len=2000，稀疏度=0.7）。_

反向加速比（2.50×）大于前向加速比（1.85×），主要是因为反向传播受益于两个独立的简化：(1) 消除了 logsumexp 存储和 Di 预处理；(2) 内联 P 重新计算避免了标准 Flash Attention 反向传播所需的 L 张量 HBM 往返。

### 7.2 跨工作负载的扩展性

_表 6：跨序列长度和稀疏度比例的扩展性能。无论分布形状如何，加速比都保持一致（batch=1152，对于 >7000 则 batch=768）。相对于 flash attention v2（jfa）的内核加速比。_

### 7.3 融合旋转反向传播

将旋转反向传播融合到注意力尾声中的影响尤为显著：

| 配置 | 时间（ms） |
|------|----------|
| 注意力反向传播（独立） | 1.556 |
| 旋转反向传播（独立） | 4.880 |
| **未融合总计** | **6.436** |
| **融合注意力+旋转反向传播** | **1.819** |
| **加速比** | **3.54×** |

_表 7：融合与非融合旋转反向传播的时间分解。独立旋转内核主导了未融合总时间。seq_len=1735537，heads=3，head_dim=128，batch=1152。_

独立旋转反向传播的代价是注意力反向传播本身的 3 倍以上——它纯粹受内存带宽限制，以无实质计算的方式读写 [M, D] 张量。将其融合到注意力尾声中，通过现有的 TMEM → 寄存器流水线摊销了这一带宽成本，将组合操作从 6.436 ms 降低到 1.819 ms。

端到端来看，将该内核集成到自注意力层中，这些层的 **模型 FLOPs 利用率（MFU）提升了 +30.6%**。

### 7.4 数值精度

将旋转反向传播融合到 FP32 尾声中也带来了可量化的精度改善。与高精度 PyTorch 参考实现相比，TLX Block Attention 将查询梯度（dQ）的最大梯度误差降低了 2 倍以上：

| 指标 | Flash Attention v2 | TLX Block Attention | 更精确 |
|------|-------------------|---------------------|--------|
| 最大 dQ 误差 | 0.2559 | 0.1201 | **TLX** |
| 最大 dK 误差 | 0.1689 | 0.1689 | 持平 |
| 最大 dV 误差 | 0.0112 | 0.0112 | 持平 |
| 平均 dQ 误差 | 0.000309 | 0.000220 | **TLX** |

_表 8：相对于 PyTorch 参考实现的梯度数值精度。由于单量化点融合旋转路径，TLX Block Attention 将最大 dQ 误差降低了 53%。_

dQ 受益最多，因为查询梯度（dQ = dS · K）通过融合旋转共轭时只有 1 个量化点而非 2 个。dK 也经过旋转共轭（RoPE 同时旋转 Q 和 K），但其最大绝对误差恰好由 MMA 累加本身主导，而非旋转内存往返，因此消除中间 BF16 强制转换带来的逐元素改善并未体现在最大值上。

## 8. 适用性

如果你的模型使用块对角注意力——每个 token 只关注固定本地组内的其他 token——这个内核非常合适。

* **在 NVIDIA Blackwell GPU 上训练。** 该内核使用 tcgen05 MMA 指令、TMEM 分配以及 Blackwell 时代的 TMA 描述符——这些在 Ampere 或 Hopper 上均不存在。async_dot / local_trans / tlx API 专门针对 Blackwell 架构（sm_100+）。
* **HEAD_DIM ∈ {64, 128}。** 这些是支持的注意力头维度；其他值需要重新编译，并可能需要重新计算 SMEM/TMEM 预算。

## 9. 结论

TLX Block Attention 展示了单一架构约束的复合威力。通过认识到大量特征交互和序列模型只需要严格的块对角注意力，一系列简化得以实现。

消除跨块注意力意味着无需多 tile 累加。无多 tile 累加意味着无需在线 softmax 修正因子。无在线 softmax 修正意味着 logsumexp 张量可以在反向传播中完全丢弃。不需要独立的 logsumexp 张量，就能释放足够的寄存器和内存带宽预算，将旋转嵌入直接融合到反向尾声中，这独立地同时提升了速度和数值精度。

结果是一个专为 Blackwell 架构的 TMA 和 TMEM 硬件原语量身定制的 warp 专用内核：前向传播使用 15 个 warp，反向传播使用 20 个，每个 warp 组永久分配到与其瓶颈匹配的硬件单元。这种设计相比 Flash Attention v2 实现了 2.3× 的内核级加速，融合旋转时的组合反向加速为 3.5×，以及生产自注意力层上 +30.6% 的 MFU 提升。

该内核在 github.com/facebookresearch/ads_model_kernel_library 以开源形式发布——在你自己的块稀疏注意力工作负载上试试，让我们知道你的发现。

## 致谢

作者感谢 Triton 和 PyTorch 团队持续开发使本内核成为可能的 tlx Blackwell 扩展。特别感谢更广泛的 GPU 内核研究社区，他们在 Flash Attention、warp 专用流水线和持久内核调度方面的工作为这些优化奠定了基础。

## 参考文献

1. Qiu, J., Ma, H., Levy, O., Yih, S. W., Wang, S., & Tang, J. (2020). BlockBERT: Efficient Attention Using Block Structures. EMNLP Findings 2020. https://arxiv.org/abs/1911.02972

2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. https://arxiv.org/abs/2205.14135

3. Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024. https://arxiv.org/abs/2307.08691

4. NVIDIA Corporation. (2024). NVIDIA Blackwell Architecture Technical Brief. https://resources.nvidia.com/en-us-blackwell-architecture

5. Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL 2019. https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

6. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864

7. He, H. & Guessous, D. (2024). FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention. PyTorch Blog. https://pytorch.org/blog/flexattention/

8. Yu, H., Ren, M., Maher, B., Nay, S., Zhu, G., & Jiang, S. (2024). Enabling Advanced GPU Features in PyTorch – Warp Specialization. PyTorch Blog. https://pytorch.org/blog/warp-specialization/

---
原文链接: https://pytorch.org/blog/tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/
