# 在 GB300 上使用 SGLang 部署 DeepSeek-V4：自首日以来在相同交互性下实现 5 倍吞吐量提升

By SGLang 团队和 NVIDIA 团队 | 2026 年 6 月 23 日

**摘要：** DeepSeek-V4 的支持在首日（Day-0）就已上线 SGLang，但首日的技术栈仅仅是起点。自发布以来，我们协调推进了一系列内核、运行时和加固方面的改进：MHC 融合与 token-bucket 预热、KV Compression V2、W4A4 MegaMoE、更强的 SWA 预算和驱逐行为、更好的解耦解码准入、DeepSeek-V4 prefill 路径中的可中断 CUDA graph 支持，以及 SGLang 和 Dynamo 中消除服务前沿不稳定性的 bug 修复。

在本博客中，我们来看看性能结果。在公开的 SemiAnalysis InferenceX GB300 解耦通道上（DeepSeek-V4 Pro，FP4，ISL=8192，OSL=1024，dynamo-sglang），2026 年 6 月的 MTP 曲线在约 50 tok/s/user 时达到约 11,200 tok/s/GPU，而首日（2026 年 4 月）无 MTP 曲线约为 2,200 tok/s/GPU——在相同用户可见交互性下实现了 5 倍的提升。

## 性能结果

以下是来自公开的 SemiAnalysis InferenceX 仪表板的性能结果。每张图表固定了模型、GPU 系列、精度、工作负载、服务框架和服务模式，并比较了 SGLang 随时间推移的自身表现。

### NVIDIA GB300 解耦 8K/1K

![DeepSeek V4 Pro on NVIDIA GB300](https://pytorch.org/wp-content/uploads/2026/06/sglang-dsv4-gb300-disagg-8k1k.png)

NVIDIA GB300 解耦通道展示了显著的性能提升。

有两点值得关注。首先，这不是单点胜利：无 MTP 和 MTP 曲线在整个交互性范围内都有提升。其次——对实际部署更为重要的是——曲线现在在高交互性区域能够维持更深的吞吐量。首日曲线在约 40 tok/s/user 之后急剧下降；6 月的曲线在 40 tok/s/user（无 MTP）时维持了 **2.1 倍**的吞吐量，在 80 tok/s/user（MTP）时维持了 **2.6 倍**的吞吐量，这正是大多数部署所针对的交互性范围。

无 MTP 和 MTP 前沿建立在相同的 DeepSeek-V4 服务栈之上。当我们降低了 prefill 侧的内核开销、使 DeepSeek-V4 运行时编译行为更加可预测、改进了 decode 侧的 SWA 核算、将 FP4 MoE 路径与部署方案对齐，并修复了扭曲推测路径的正确性问题后，两条曲线都得到了提升。

### NVIDIA Blackwell Ultra 聚合 8K/1K

![DeepSeek-V4 Pro on NVIDIA Blackwell Ultra](https://pytorch.org/wp-content/uploads/2026/06/sglang-dsv4-blackwell-agg-8k1k.png)

NVIDIA Blackwell Ultra 聚合通道同样得益于相同的优化，展示了显著的性能提升。在 30 tok/s/user（无 MTP）时吞吐量提升了 **2.91 倍**，在 90 tok/s/user（MTP）时提升了 **2.85 倍**。

此外，无 MTP 峰值吞吐量与首日曲线相比提升了超过 **6 倍**。这是因为早期的无 MTP 曲线始于一个较低吞吐量的回退方案：仅 TP 执行、无 DP attention、无推测解码以及更窄的搜索空间。到后来的公开提交时，Blackwell Ultra 聚合通道已经转向了更强大的方案族，具有更好的按并发调度策略、decode worker 上更高的可持续批处理大小，以及更成熟的 FP4/MoE 路径。

## 首日已有的能力

首日的支持已经相当完善。在发布窗口期间，SGLang 已经拥有功能完备的 DeepSeek-V4 服务路径、通过 DeepSeek-V4 方案族提供的部署指导，以及跨多个平台的经过验证的启动配置。发布时的技术栈已经涵盖了使 DeepSeek-V4 服务具有挑战性和趣味性的主要要素：FP4 推理、使用 DeepGEMM MegaMoE 内核实现高吞吐量和 FlashInfer TRT-LLM MoE 后端实现低延迟的 MoE 执行、FlashMLA 相关的 attention/内核基础设施、DP/EP/TP 方案变体、解耦部署、推测解码支持，以及 decode 侧的完整 CUDA graph 支持。

这个基础至关重要。我们不是从"模型能否运行？"开始，而是从一个已经能在真实硬件上服务 DeepSeek-V4 的工作系统出发，包括首日发布后首批 NVIDIA Blackwell Ultra 聚合提交和首批 NVIDIA GB300 解耦提交。后续工作的目标是将首日的技术栈转变为更快、更稳定、更面向生产的服务路径。

## 首日之后的变化

### 内核相关优化

最显眼的内核侧工作是在 MHC 流水线中。在 [#24775](https://github.com/sgl-project/sglang/pull/24775) 中，SGLang 围绕更深层的融合实现重新构建了 DeepSeek-V4 的 MHC 路径：大型 `mhc_pre path` 迁移到了更强的 DeepGEMM 支持流程上，RMSNorm 工作被融合到 MHC 路径中而不是作为单独的阶段边界保留，SGLang 还添加了专用的融合 `hc_head kernel`。这一变更减少了 DeepSeek-V4 prefill 路径中最昂贵部分之一的中间张量流量和调度器可见的管道操作。延续同一主题，[#25976](https://github.com/sgl-project/sglang/pull/25976) 添加了融合的 `mhc_fused_post_pre` 内核，缩小了 MHC 路径中昂贵边界的数量，而不是将 MHC 视为一堆独立的点操作。

KV Compression V2 是另一个重大的内核方面的飞跃。[#24890](https://github.com/sgl-project/sglang/pull/24890) 添加了 V2 DeepSeek-V4 压缩内核，包括新的 c4、c128 和在线 c128 压缩内核、更新的压缩器管道，以及新的融合 norm/rope V2 组件。这很重要，因为 DeepSeek-V4 不仅需要快速的矩阵乘法，还需要在并发增加时仍保持高效的压缩和索引器内核。

FP4 MoE 路径也得到了实质性改进。在 [#25052](https://github.com/sgl-project/sglang/pull/25052) 之前，DeepSeek-V4 的 DeepGEMM MegaMoE 路径使用的是 W4A8 内核，这意味着专家权重被量化为 MXFP4，但激活路径仍然量化为 MXFP8。在 [#25052](https://github.com/sgl-project/sglang/pull/25052) 之后，SGLang 添加了启用 W4A4 MegaMoE 路径的选项，使激活路径也量化为 MXFP4 而非 MXFP8，精度损失可忽略不计。这在高吞吐量操作范围内最明显地提升了 MoE 效率。

综合来看，这就是首日之后的内核故事：更少的未融合边界、更好的 DeepSeek-V4 专用压缩内核、更好的 FP4 MoE 内核，以及隐藏在 prefill 和压缩路径中更少的非矩阵乘法开销。

### 运行时相关优化

运行时的工作同样重要。在 DeepSeek-V4 上，服务前沿通常取决于运行时能否在真实请求混合下正确地预算、分配、图捕获和回收状态。这就是为什么运行时级别的优化和解耦工作如此显著地推动了曲线提升。

[#24036](https://github.com/sgl-project/sglang/pull/24036) 通过将全长核算与实际需要保留在滑动窗口池中的 SWA 尾部分离，修复了解耦 decode SWA 预分配大小问题。[#24857](https://github.com/sgl-project/sglang/pull/24857) 进一步推进了这一点，提供了更准确的全 token 和 SWA token 预算、更好的跨等待、运行和传输状态的预留逻辑，以及更真实的 SWA 尾部预分配行为。实际上，这些变更使 decode worker 能够在达到内存限制之前以更高的有效批处理大小或更多的并发请求运行。这对吞吐量直接有价值，因为能够维持更大批次的 decode worker 可以让 GPU 保持更忙碌的状态。

我们还优化了运行时方案以更好地匹配不同的服务场景。对于 Blackwell Ultra 聚合通道，我们从一刀切的方法转向按并发调度策略，使每个并发级别都能使用定制的方案。对于 NVIDIA GB300 解耦通道，我们引入了一组经过调优的方案，带来了显著的性能提升。这些改进来自于仔细调整关键参数，如 prefill 与 decode 的比例、并行执行计划（例如宽 EP 配置）、全窗口和滑动窗口 attention 的 KV 缓存内存分配、token 和请求大小限制，以及分词器配置。

CUDA graph 的故事也得到了改善。DeepSeek-V4 运行时路径包含足够多的不规则行为，使得图捕获具有挑战性，运行时不得不过于频繁地退回到 eager 孤岛。[#25195](https://github.com/sgl-project/sglang/pull/25195) 中针对 DeepSeek-V4 DP attention 的可中断 CUDA graph 工作，以及 [#25795](https://github.com/sgl-project/sglang/commit/c4a7d1209231e662c4447fe3d3326d8c3d1087b7) 中后续的推测路径启用，将更多的 prefill 路径推回到了图友好的执行方式下。这通过减少主机绑定和让 GPU 在 prefill 阶段保持更好的利用率来改善 prefill worker 的性能。Decode 在首日就已经有了完整的 CUDA graph 支持，因此这里的主要附加价值在 prefill 侧，它有更多的人不规则路径会中断图捕获。

还有一些优化未被 SemiAnalysis InferenceX 基准测试场景覆盖，但对实际部署仍然重要。使用 TP、DP 和 EP 等常见并行方式的 PD 部署在首日就已存在。[#24704](https://github.com/sgl-project/sglang/pull/24704) 通过为 DeepSeek-V4 PD 部署添加流水线并行支持扩展了该路径。DeepEP 路径也通过 [#25391](https://github.com/sgl-project/sglang/pull/25391) 中的 DeepEP Waterfill 支持等变更得到了进一步收紧。

### Bug 修复和加固

这一时期一些最有价值的性能改进是以 bug 修复的形式出现的。它们不是通过添加新算法来改善基准测试表现，而是通过消除阻止运行时维持更好曲线的正确性和稳定性问题来实现的。

在推测和解耦方面，[#23919](https://github.com/sgl-project/sglang/pull/23919) 修复了 PD-MTP 元数据缓冲区的 hidden-size bug，[#25805](https://github.com/sgl-project/sglang/pull/25805) 修复了在使用 MTP 推测的解耦 decode 下 SWA 内存处理中的 double-free 问题。这类问题可能使推测服务路径在某个狭窄的方案中看起来正常，然后在真实并发扫描中崩溃。

运行时编译行为也需要加固。[#25810](https://github.com/sgl-project/sglang/pull/25810) 添加了代表性的 MHC token-count bucket 预热，使第一个真实请求不会持续为 DeepSeek-V4 实际访问的 token-count bucket 支付延迟编译成本。这是一个性能变更，但也是一个可靠性变更：一个已经知道其热点形状的服务系统不应该在关键路径上花费时间在运行时重新发现它们。

还有一些正确性修复。[#25733](https://github.com/sgl-project/sglang/pull/25733) 通过将 fp8\_einsum 输入缩放转换为 ue8m0，修复了 DeepSeek-V4-Pro 在 NVIDIA Blackwell 上的 NaN 问题。该变更主要是一个正确性修复，但它也对推测路径产生了实际的服务侧影响：一旦 Blackwell FP8-einsum 缩放不再破坏 DeepSeek-V4 MTP 路径，接受长度也随之恢复。在一次观察到的运行中，这一行修复将接受率从 0.57 提高到了 0.70。这是一个很好的例子，说明一个不以"性能 PR"面貌出现的 bug 修复仍然能在实践中推动 MTP 前沿的提升。

[Dynamo](https://github.com/ai-dynamo/dynamo) 也需要一个重要修复才能与 SGLang 良好配合。[ai-dynamo/dynamo#9080](https://github.com/ai-dynamo/dynamo/pull/9080) 将 bootstrap-room 生成与所选的 prefill DP rank 对齐，减少了跨 DP rank 的工作负载不平衡。这不在 SGLang 主仓库中，但它仍然是 GB300 解耦通道公开服务路径的一部分，因此也是我们在实时前沿看到的性能故事的一部分。

因此，bug 修复部分是性能故事的一部分，而非独立于它。当 DeepSeek-V4 运行时不再错误地调整元数据大小、不再 double-free SWA 状态、不再为已知形状支付延迟 MHC 编译成本、不再在 NVIDIA Blackwell 上触发数值问题、不再跨 DP rank 偏斜工作时，服务前沿变得更快也更可信。

## 如何复现

以下是 SemiAnalysis InferenceX 运行所使用的脚本和方案的公开参考点，这些运行生成了本文中的结果。

NVIDIA GB300 解耦运行使用 [srt-slurm](https://github.com/NVIDIA/srt-slurm) 在 Slurm 管理的集群中启动 Dynamo 前端和 SGLang prefill/decode 服务器。用于 GB300 解耦性能扫描的配置可在以下位置找到：[https://github.com/SemiAnalysisAI/InferenceX/tree/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/multi\_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k](https://github.com/SemiAnalysisAI/InferenceX/tree/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k)

`srt-slurm` 的复现流程如下：

```bash
git clone https://github.com/NVIDIA/srt-slurm
cd srt-slurm
pip install -e .
srtctl apply -f benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml
```

NVIDIA Blackwell Ultra 聚合脚本可在以下位置找到：

- **NVIDIA Blackwell Ultra 聚合无 MTP 脚本：** [https://github.com/SemiAnalysisAI/InferenceX/blob/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/single\_node/fixed\_seq\_len/dsv4\_fp4\_b300\_sglang.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/single_node/fixed_seq_len/dsv4_fp4_b300_sglang.sh)
- **NVIDIA Blackwell Ultra 聚合 MTP 脚本：** [https://github.com/SemiAnalysisAI/InferenceX/blob/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/single\_node/fixed\_seq\_len/dsv4\_fp4\_b300\_sglang\_mtp.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/single_node/fixed_seq_len/dsv4_fp4_b300_sglang_mtp.sh)

要获取本文中展示的确切 NVIDIA GB300 和 NVIDIA Blackwell Ultra 曲线，请访问 [SemiAnalysis InferenceX 网站](https://inferencex.semianalysis.com/inference)获取原始性能数据和最新曲线。

## 路线图

下一步的重点不再是证明 DeepSeek-V4 的可行性，而是继续收紧生产路径：

- 优化 DeepSeek-V4 模型的内核性能，包括 DeepEP v2、ragged 索引器内核，以及更多小内核的融合。
- 扩展 DeepSeek-V4 在 SM120/SM121 硬件以及 NVFP4 检查点上的支持。
- 继续改进缓存和路由敏感的 DeepSeek-V4 服务路径。
- 优化 DeepSeek-V4 在智能体基准测试上的性能。

这些待优化的项目正在此路线图 issue 中跟踪：[#23602](https://github.com/sgl-project/sglang/issues/23602)。

## 致谢

感谢推动 DeepSeek-V4 内核、运行时和服务路径工作的 SGLang 和 NVIDIA 贡献者；感谢 SemiAnalysis InferenceX 团队保持基准测试工作流和变更日志的公开；以及围绕 GB300 和 Blackwell Ultra 部署的更广泛的 DeepSeek-V4 服务工作。

**SGLang 团队和社区贡献者：** Yuhao Yang, Cheng Wan, Baizhou Zhang, Pranjal Shankhdhar, Tom Chen, Ziyi Xu, Qiaolin Yu, Ke Bao, Liangsheng Yin, Yuwei An, Chunan Zeng, Shangming Cai, Yanbo Yang, Lianmin Zheng, Banghua Zhu, Ying Sheng

**NVIDIA 团队：** Yangmin Li, Weiliang Liu, Ishan Dhanani, Hao Lu, Ian Wang, Trevor Morris, Elvis Chen, Shu-Hao Yeh, Julien Lin, Akhil Goel, Nicolas Castet, Kedar Potdar, Ankur Singh, Harshika Shrivastava, Kaixi Matteo Chen, Xuting Zhou, Po-Han Huang, Triston Cao，以及更多

---
原文链接: https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/
