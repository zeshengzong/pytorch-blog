# 在 DeepSpeed 中使用 Muon 优化器

作者：Zhipeng Wang、Guokai Ma、Peng Du 和 Chi McIsaac，DeepSpeed 团队

2026 年 6 月 3 日

## 摘要

DeepSpeed 现在支持 Muon 优化器！Muon 优化器已获得极大的关注，并被多个前沿 AI 实验室广泛采用。其中一个 AI 实验室是 Moonshot AI，它已将 Muon 优化器用于训练其大型基础模型，如 Kimi-K2-Thinking。本文深入介绍了 Muon 优化器是什么，以及它在 DeepSpeed 上的表现。

## 什么是 Muon 优化器？

Muon 是一种专为神经网络隐藏层二维权重设计的优化器。它获取权重的梯度，计算其动量，然后应用 Newton-Schulz 迭代对动量矩阵进行正交化，再使用这个正交化后的矩阵来更新权重。由于 Muon 只需维护一个动量缓冲区（相比 Adam 的两个），它在优化器状态上使用的内存更少。

正交化步骤是 Muon 在预训练中具备收敛优势的关键。在实践中，transformer 中二维权重的梯度更新往往具有非常高的条件数——它们几乎是低秩的，被少数几个较大的奇异方向所主导。通过对动量矩阵进行正交化，Muon 均衡了所有奇异值，有效地放大了那些原本会被遮蔽的稀少但重要的更新方向。这带来了更好的样本效率：在 NanoGPT 竞速基准测试中，Muon 相比 AdamW 将训练速度提高了 35%；在 1.5B 参数规模下，它达到 GPT-2 XL 级别性能的速度比 AdamW 快约 25%。

与需要为每个参数维护两个动量缓冲区的 Adam 优化器不同，Muon 优化器只需要一个动量缓冲区。这意味着对于使用 Muon 优化器的参数，我们只需为动量分配一个缓冲区，与 Adam 相比可以节省内存。

Muon 被 Keller Jordan 的 NanoGPT 改良版和 Andrej Karpathy 的 nanochat 所使用，Muon 的一个变体（MuonClip）也被 MoonShot 的生产级大语言模型 Kimi-K2 采用。最近，智谱 AI 的 GLM-5（7440 亿参数）确认在 GLM-4.5 和 GLM-5 的预训练中使用了 Muon 优化器，以及一种"Muon Split"技术——按注意力头拆分 MLA 上投影矩阵并对每个头独立进行正交化，解决了使用 Muon 时 MLA 和 GQA 之间的性能差距。DeepSeek-V4（1.6 万亿参数）也采用了 Muon 优化器，以实现更快的收敛和更高的训练稳定性。

## DeepSpeed 中的 Muon 优化器支持

将 Muon 优化器应用于 DeepSpeed 的挑战之一在于，以前的优化器（SGD、Adam）将梯度视为扁平化的缓冲区。因此，由于梯度缓冲区已经被扁平化，很难在同一位置直接替换为 Muon 优化器。我们将 Muon 更新步骤移至 stage 1 和 stage 2 的 `DeepSpeedZeroOptimizer` 的 `get_flat_partition` 函数中，在该函数中每个参数的梯度仍处于未扁平化阶段，因此我们可以轻松地应用 Muon 更新。

Muon 优化器适用于二维权重矩阵（注意力层和 MLP 权重）。它对动量矩阵应用 Newton-Schulz 正交化，这要求权重是二维的。非二维参数（嵌入层、层归一化、偏置、lm_head）回退到 AdamW。我们在模型引擎初始化器中应用解析，为模型参数标记 `use_muon`——当且仅当模型参数是二维的且属于隐藏层时。使用 Muon 优化器时，任何标记了 `use_muon` 的参数都将使用 Muon 优化器更新权重。

请注意，Muon 是一种混合优化器：它仅对二维隐藏权重使用 Muon 更新，对所有其他参数（嵌入层、层归一化、偏置、lm_head）回退到 Adam。DeepSpeed 配置通过 `muon_lr`（用于 Muon 参数）和 `adam_lr`（用于 Adam 参数）支持分别设置学习率。

## 使用 Muon 优化器运行 DeepSpeed 微调

Deepspeed finetune demo 是一个在单一环境中使用不同 DeepSpeed 训练功能并比较其性能的演示。你可以用它来测试使用 Muon 优化器微调大语言模型：

```
git clone https://github.com/delock/deepspeed_finetune_demo
cd deepspeed_finetune_demo
./finetune.sh z2_muon.json
```

## Muon 优化器收敛实验结果

我们通过微调 Moonlight-16B-A3B（一个混合专家模型，总参数量 16B，激活参数量 3B）来测试 Muon 优化器，并在代码生成（MBPP/MBPP+）、通用知识（MMLU）和数学推理（GSM8K）基准上进行评估。每个基准使用其各自领域的专属训练集。

**训练配置：**
- 模型：Moonlight-16B-A3B（MoE，总参数量 16B / 激活参数量 3B）
- 训练数据集：MBPP/MBPP+ 使用 sahil2801/CodeAlpaca-20k，MMLU 使用 cais/mmlu（auxiliary_train，约 9.5 万个样本），GSM8K 使用 meta-math/MetaMathQA（sample_rate=0.1，约 3.95 万个样本）
- ZeRO Stage 2，bf16，专家并行（autoep_size=4）
- 批量大小：16，梯度累积：2，4 块 GPU
- 训练 1 个 epoch，梯度裁剪：1.0

## 评估结果

| 优化器 | 学习率 | adam_lr（用于 Muon） | MBPP | MBPP+ | MMLU | GSM8K |
|--------|--------|---------------------|------|-------|------|-------|
| 基线（微调前） | — | — | 0.495 | 0.431 | 0.401 | 0.526 |
| AdamW | 2e-6 | — | 0.661 | 0.534 | 0.660 | 0.805 |
| Muon | 1e-4 | 2e-6 | 0.646 | 0.548 | 0.678 | 0.810 |

Muon 在 4 项指标中的 3 项上优于 AdamW：MBPP+（0.548 vs 0.534，+1.4pp）、MMLU（0.678 vs 0.660，+1.8pp）和 GSM8K（0.810 vs 0.805，+0.5pp）。在 MBPP 基础测试上，AdamW 略胜 Muon（0.661 vs 0.646，-1.5pp），但 Muon 在包含更多测试用例的更严格的 MBPP+ 上取得了更高分（0.548 vs 0.534），这表明其具有更好的泛化能力。

## Muon 优化器内存节省

Muon 优化器比 Adam 使用更少的优化器状态内存，因为它为每个参数只维护一个动量缓冲区，而不是两个（第一动量和第二动量）。

**内存使用对比**

| 优化器 | 每参数状态缓冲区数 | 每参数内存占用 |
|--------|-----------------|--------------|
| Adam | 2 个（m, v） | 8 字节 |
| Muon | 1 个（动量） | 4 字节 |

请注意，Muon 是一种混合优化器：二维隐藏权重使用 Muon（1 个缓冲区），而其余参数（嵌入层、层归一化、lm_head）仍使用 Adam（2 个缓冲区）。实际的内存节省取决于二维隐藏权重占总参数的比例。对于典型的 transformer 模型，约 90% 的参数是二维隐藏权重，因此优化器状态内存减少了约 45%。然而，由于 GPU 总内存还包括模型权重、梯度和激活，端到端的内存减少幅度较小（见下方实测结果）。

## 实测 GPU 内存：Qwen2.5-3B 微调

我们在使用上述相同的 8xA100（40GB）配置（批量大小 32，ZeRO Stage 2，bf16）对 tatsu-lab/alpaca 进行 Qwen2.5-3B 微调期间测量了峰值 GPU 内存。

| 优化器 | 每 GPU 峰值内存 | 相比 AdamW 节省 |
|--------|--------------|---------------|
| AdamW | 34.5 GiB | — |
| Muon | 31.4 GiB | 9% |

与 AdamW 相比，Muon 将每 GPU 内存减少了约 3 GiB（9%）。节省完全来自优化器状态：Muon 参数存储一个动量缓冲区（4 字节），而不是 Adam 的两个（8 字节）。然而，由于优化器状态只是 GPU 总内存的一个组成部分（还包括模型权重、梯度和激活），端到端的减少是适度的。对于更大的模型或更紧张的内存预算，这 9% 的节省可能决定了工作负载是否能在设备上运行，还是需要 CPU 卸载。

## 下一步计划

Muon 正在社区中迅速获得关注，Kimi-K2（1 万亿参数）和 GLM-5（7440 亿参数）的生产级采用表明，它是替代 Adam 成为大规模训练默认优化器的有力竞争者。我们正在积极构建 DeepSpeed 中对 Muon 的完整支持，一系列改进已在推进中：

- ZeRO Stage 2 支持——已合并
- ZeRO Stage 3 支持——已合并
- 基于 Gram-Schmidt 的 Newton-Schulz 迭代——一种更快的正交化内核，正在审查中
- CPU 卸载——进行中
- MuonClip——Kimi-K2 使用的变体，已计划

我们欢迎任何关于 DeepSpeed 中 Muon 优化器支持的想法、反馈和贡献——请创建一个 issue 进行讨论或向 DeepSpeed 提交 PR。让我们让 Muon 在 DeepSpeed 中既稳健又闪电般快速！

---
原文链接: https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/

## 链接汇总

- DeepSpeed 微调演示 GitHub 仓库: https://github.com/delock/deepspeed_finetune_demo
