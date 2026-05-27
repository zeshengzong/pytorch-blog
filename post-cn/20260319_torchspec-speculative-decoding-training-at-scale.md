# TorchSpec：规模化投机解码训练

- 作者：TorchSpec team, Mooncake team
- 日期：2026年3月19日
- 原文： https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/

## 介绍

在过去一年里，大语言模型在规模和能力上都快速扩展。像 Kimi K2.5、GLM 5 和 Qwen 3.5 这样的前沿模型，已经具备数千亿参数、可扩展到百万 token 的上下文窗口，从而支持长上下文推理、智能体工作流以及复杂工具调用。随着模型能力持续增强，高效推理成为 LLM 部署中最关键的系统挑战之一。

投机解码是加速 LLM 生成最有效的技术之一。在投机解码中，轻量草稿模型会一次向前提出多个 token，再由更大的目标模型通过一次前向计算进行验证。当预测被接受时，就可以一次生成多个 token，从而提升吞吐与降低时延。近期方法如 MTP（Multi Token Prediction）和 [EAGLE-3](https://arxiv.org/abs/2503.01840) 表明，经过良好训练的草稿模型能够稳定带来加速收益。

草稿模型训练中的一个关键点，是把目标模型的中间隐状态传递给草稿模型。随着前沿 LLM 规模不断增大，一个新的系统瓶颈出现了：如何高效传输目标模型产生的海量隐状态。例如，EAGLE-3 依赖目标模型的 3 层隐状态。在为 Kimi K2.5 训练 EAGLE-3 草稿模型时，单个 128K token 训练样本大约就需要 7 GB 的隐状态数据。

现有流水线通常有两种方案。一种是预先计算隐状态并落盘，这会带来巨大的存储需求和严重 I/O 压力。另一种是把推理与训练共置，在训练过程中在线生成隐状态，虽然避免了落盘，但要求目标模型与草稿训练进程同机部署，会显著增加 GPU 显存压力。

为解决这些问题，团队提出了 [TorchSpec](https://github.com/torchspec-project/TorchSpec)：一个基于 torch 的解耦式投机解码训练框架。TorchSpec 将生成隐状态的推理系统与消费隐状态的训练系统分离。不再把隐状态写入磁盘，而是通过 [Mooncake](https://github.com/kvcache-ai/Mooncake) 中间存储，基于 RDMA（Remote Direct Memory Access）或 TCP，将隐状态从推理工作进程直接流式传输到训练工作进程。

借助 TorchSpec，团队报告称已完成 [Kimi K2.5 EAGLE-3 草稿模型](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) 训练，耗时 1500 H200 GPU 小时，规模达到 [60 万训练样本](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)（60 亿 token）。

![来自原文页面的基准图 1](https://pytorch.org/wp-content/uploads/2026/03/unnamed-2.png)

*草稿模型训练配置：lookahead=4*

训练完成后，在 lookahead 为 3 token 的设置下，输出吞吐在 batch size 1 时提升超过 +60%，在 batch size 8 时提升 +30%，在 batch size 16 时提升 +26%。

![来自原文页面的基准图 2](https://pytorch.org/wp-content/uploads/2026/03/unnamed-3.png)

## 背景

当前投机解码训练常见的两种方式是：

- 推理训练共置（Inference co-located training）
- 离线隐状态准备（Offline hidden states preparation）

这两种方式在中等规模下可行，但随着草稿模型规模和上下文长度增加，都会遭遇瓶颈。

### 推理训练共置

在共置训练中，目标模型和草稿模型共享同一批 GPU。目标模型先前向产生隐状态和 logits，再立即供草稿模型训练使用。这种紧耦合方式会带来以下约束：

- 分片刚性：草稿模型并行策略受目标模型绑定。例如目标模型使用 TP=4，草稿模型也必须使用 4 个 rank，即使其他配置更高效。
- 推理与训练无法独立扩展：共置框架往往缺乏跨节点分片支持，训练常被限制在单节点，同时推理与训练的资源规模被强绑定。
- 显存压力：目标模型占用大量 GPU 显存，导致草稿模型可用训练显存受限。

基于 [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)（1T 参数 MoE，384 experts，约 575 GB 模型权重）的共置训练显存分析如下：

| GPU | 总显存（8 卡） | 模型权重 | 单卡分片 | 单卡剩余显存 |
| --- | --- | --- | --- | --- |
| 8xH200 | 1,128 GB | ~575 GB | ~72 GB | ~69 GB |
| 8xH100 | 640 GB | ~575 GB | ~72 GB | ~8 GB |

*虽然草稿模型通常较小，但像 Training-Time Testing（TTT）这样的 SOTA 训练方法需要保留多个投机步的中间激活，显著增加显存占用。在仅 8 GB 可用显存下，上下文长度可能只能训练到 4096。*

### 离线隐状态准备

离线方案会先从目标模型预计算隐状态，序列化到磁盘，再在草稿模型训练时读回。这实现了推理与训练解耦，但在大模型和长上下文下会造成严重存储压力。

*Kimi K2.5 存储分析（hidden_size=7168，vocab_size=163,840）：*

*当上下文长度为 131,072 token 时，单样本开销：*

| 张量 | 形状 | 数据类型 | 大小 |
| --- | --- | --- | --- |
| 隐状态（3 层辅助层） | (131072, 21504) | bf16 | 5.25 GB |
| 最后一层隐状态 | (131072, 7168) | bf16 | 1.75 GB |
| 输入 IDs | (131072,) | int64 | 1 MB |
| 单样本总计 |  |  | ~7.0 GB |

*注：目标 logits 可以由最后一层隐状态通过 lm_head 重新计算，因此无需额外存储。*

| 数据集规模 | 所需存储 |
| --- | --- |
| 10,000 样本 | 70 TB |
| 30,000 样本 | 210 TB |
| 100,000 样本 | 700 TB |

在这个规模下，分布式文件系统将承受巨大压力，尤其是多个投机训练任务并行时会争抢 I/O 带宽；同时序列化与反序列化开销也会显著拖慢训练。

## TorchSpec：解耦式草稿模型训练

TorchSpec 采用了不同路线：将推理与训练完全解耦。目标模型运行在专用推理 GPU 上，草稿模型运行在独立训练 GPU 上，二者通过 [Mooncake](https://github.com/kvcache-ai/Mooncake) 以及 RDMA 或 TCP 进行张量流式传输。

该架构解决了关键挑战：

1. 灵活的独立扩展：推理和训练的 GPU 数量完全独立，允许为更高的隐藏状态生成吞吐量配置更多推理引擎，或添加更多训练 GPU 以实现更大的 FSDP 分片和更大的全局批次。
2. 训练显存最大化：训练 GPU 完全用于草稿模型，能够支持更长序列与更大 batch。
3. 无额外存储开销：隐状态直接从推理流向训练，无需落盘，消除文件系统压力与序列化成本。

![来自原文页面的架构图](https://pytorch.org/wp-content/uploads/2026/03/unnamed-4.png)

### 为什么是 Mooncake？

Mooncake 最初由 Moonshot AI 与清华大学开发，是面向生产 LLM 服务的 KV cache 传输引擎，现已发展为 PyTorch 生态中的活跃社区项目。Mooncake 能在不同网络协议下实现高吞吐跨节点传输，并管理内存生命周期，这正是 TorchSpec 高效可靠传输隐状态所需的核心能力。

关键特性：

- RDMA + TCP 统一 API：在 InfiniBand/RoCE 集群上接近线速传输；无 RDMA 时可无代码改动回退到 TCP。
- GPU Direct RDMA：数据可直接进入 GPU 显存，绕过 CPU staging。
- 零拷贝传输：张量打包到预注册的 pinned-memory buffer 后直接传输，无需序列化和中间复制。
- 生产级稳定性：已在大规模生产环境验证，为长时间多节点训练提供可靠基础。

### 长上下文支持

由于训练显存完全留给草稿模型，TorchSpec 在 EAGLE-3 训练中可支持共置方案难以达到的序列长度。原文指出，Kimi K2.5 在共置训练中会消耗 72 GB 显存；在 lookahead=4 的解耦训练下，单张 H100 可训练到 44K token，单张 B200 可扩展到 200K token。

![来自原文页面的长上下文图](https://pytorch.org/wp-content/uploads/2026/03/unnamed-5.png)

除了架构解耦，TorchSpec 还采用推理引擎原生实现，即由生产推理引擎直接生成隐状态。这带来两点收益：

- 推理训练一致性：模板格式、分词器和内核在训练与部署环境中保持一致。
- 借助引擎获得原生模型支持：新增目标模型架构时，训练侧改动很小。

当前 TorchSpec 已支持 vLLM 和 SGLang，后续计划支持 TensorRT LLM。只要推理引擎支持某模型，TorchSpec 通常即可直接训练其草稿模型，包括：

- 新模型架构（MoE、多模态等）
- 量化模型（FP8、INT4 等）
- 稀疏注意力、RoPE 变体及其他模型特性

### 边解码边训练（Train with Decode）

草稿模型通常在贴近目标模型 token 分布的数据上训练效果更好。常见做法是保留原始 prompt，并先用目标模型再生成 response 作为训练前处理。TorchSpec 的引擎原生设计允许在训练过程中直接从仅含 prompt 的输入自回归生成输出，简化整体流程。

### 案例：为 Kimi K2.5 训练 EAGLE-3 模型

Kimi K2.5 是一个能体现解耦式方案价值的典型高难度训练场景。

#### 挑战

- 模型规模：仅服务目标模型就至少需要 8xH200 或 16xH100，若共置训练则草稿模型可用显存非常有限。
- 长上下文：其面向长上下文智能体和推理任务，训练序列长度需达到 200,000 token。
- 大词表：词表规模 163,840，隐藏维度 7,168。

### TorchSpec 方案

TorchSpec 建议将 Kimi K2.5 部署在专用的 8xH200 推理集群上，再用另一组 8xH200 训练 EAGLE-3 草稿模型。这样推理与训练显存预算分离，可在 60 万样本规模下将训练上下文长度提升到 100,000 token。

**脚本**

- 3 节点 8xH100，TP=16 推理 + TP=8 训练： [kimi-k25-3node-h100](https://github.com/torchspec-project/TorchSpec/tree/main/examples/kimi-k25-3node-h100)
- 2 节点 8xH200，TP=8 推理 + TP=8 训练： [kimi-k25-2node-h200](https://github.com/torchspec-project/TorchSpec/tree/main/examples/kimi-k25-2node-h200)

**训练数据集**： [kimi-600k-training-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)

**草稿模型**： [kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3)

## 路线图

TorchSpec 仍在积极开发中，重点方向包括：

- 扩展模型覆盖：计划支持 Minimax M2.5、Qwen 3.5，以及 GLM 5 的 MTP 层持续训练。
- Packed sequence 训练：将多个短序列打包为单训练样本，提升 GPU 利用率并减少 padding 浪费。
- 增加训练算法：除 EAGLE-3 外，支持 DFlash、MTP 等更多投机训练方法。
- 引擎集成：对接更多推理引擎（如 TensorRT LLM），方便用户按部署栈选择。

## 致谢

感谢以下团队与协作者：

- TorchSpec 团队与社区：*Yubo Wang, *Yinghui Liu, Shirley Wu, Junxiong Wang, Qingyang Wu, Bobbie Bie, Fan Yin, Chao Wang, Weicong Wu, Jue Wang
- Mooncake 团队：*Jiaqi Liao, Mingxing Zhang

## 链接汇总

- 原文地址: [https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)
- EAGLE-3 论文: [https://arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840)
- TorchSpec 项目: [https://github.com/torchspec-project/TorchSpec](https://github.com/torchspec-project/TorchSpec)
- Mooncake 项目: [https://github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- Kimi K2.5 EAGLE-3 草稿模型: [https://huggingface.co/lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3)
- 60 万训练样本数据集: [https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)
- Kimi K2.5 目标模型: [https://huggingface.co/moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)
- 训练脚本（3 节点 H100）: [https://github.com/torchspec-project/TorchSpec/tree/main/examples/kimi-k25-3node-h100](https://github.com/torchspec-project/TorchSpec/tree/main/examples/kimi-k25-3node-h100)
- 训练脚本（2 节点 H200）: [https://github.com/torchspec-project/TorchSpec/tree/main/examples/kimi-k25-2node-h200](https://github.com/torchspec-project/TorchSpec/tree/main/examples/kimi-k25-2node-h200)
