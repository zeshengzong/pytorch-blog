# 从分钟到秒：LLM 引导的 Helion 内核自动调优

By Jongsok Choi, Ethan Che, Jason Ansel, Oguz Ulgen | 2026 年 6 月 18 日

### 摘要

Helion 是 PyTorch 的领域专用语言（DSL），用于编写性能可移植的机器学习内核，其性能高度依赖于自动调优。目前 Helion 的搜索采用无似然贝叶斯优化（LFBO）来寻找最优配置。LFBO 是一个强大的基线方法，效果良好，但每个内核仍需经历数百次编译和基准测试循环。为此，我们引入了 LLM 引导的自动调优器，在基准测试约 10 倍少的配置的情况下达到了 LFBO 级别的内核性能（几何平均 1.009X），且实际耗时减少约 6.7 倍。对于 LLM 落后超过 5% 的少数内核，混合策略（LLM 种子配置后接 LFBO 精调）可以弥合差距，同时成本仍比完整的 LFBO 搜索低约 3 倍。最后，结果在很大程度上与 LLM 模型无关——Opus-4.8、gpt-5.5 和 Sonnet-4.6 的表现相差仅几个百分点——表明 LLM 引导的自动调优是一种在实际生产质量下大幅加速内核调优的实用方法。

## 引言

自动调优是 Helion 的基石——Helion 是 PyTorch 用于编写高性能、可移植 ML 内核的 DSL。每个 Helion 内核都在庞大的高维配置空间（tile 大小、block 大小、num\_warps、num\_stages，详见[文档](https://helionlang.com/api/autotuner.html)）中进行调优，以在目标硬件上达到峰值性能。减少调优时间对于开发者效率和生产部署同样至关重要，这直接影响 Helion 的采用率。

自动调优器在组合空间中搜索配置，对每个配置进行基准测试，并保留最优配置。使搜索更快、更智能是一个活跃的研究方向。Helion 当前默认的自动调优器使用 LFBO（无似然贝叶斯优化），在搜索过程中动态训练一个轻量级随机森林分类器，基于已基准测试的数据学习预测哪些配置是有前景的候选。它利用预测结果聚焦于最重要的参数，在空间中进行有针对性的跳跃。LFBO 搜索现在是默认选项，因为它在 NVIDIA 和 AMD GPU 上的内核性能和调优时间方面均显示出显著改进。详见我们的 PyTorch 博客"[使用贝叶斯优化加速 Helion 自动调优](https://pytorch.org/blog/accelerating-autotuning-in-helion/)"。

LFBO 是一个强大的基线方法，效果良好，但每个内核仍需经历数百次编译和基准测试循环。如果不用盲目地开始搜索，而是让 LLM 对内核进行推理并提出配置方案呢？这就是 LLM 引导的自动调优器——在每一轮自动调优中，LLM 会看到内核、工作负载和当前最优配置，然后提出新的配置进行尝试。在本文中，我们描述 LLM 引导的自动调优器的工作原理，并展示在 B200 上 33 个案例（11 个内核 x 3 种形状）中将 LLM 引导搜索与 LFBO 搜索进行对比的基准测试结果。结果表明，新的 LLM 方法在编译/基准测试 10 倍少配置的情况下达到了 LFBO 级别的内核性能，实际耗时减少 6.7 倍。我们还引入了混合搜索来结合两者的优势，即使用 LLM 快速达到高性能配置，然后使用 LFBO 进行精细搜索。

## LLM 引导的自动调优器工作原理

通过多轮提示和反馈循环，新的 LLM 自动调优器执行基于种群的搜索。在初始阶段，Helion 将内核及相关细节提供给 LLM，请求一组候选配置。LLM 响应后，Helion 编译并基准测试这些配置，保留性能最优的配置。随后进行多轮精调，LLM 会获得最成功的配置、其性能指标以及对成功模式的分析，以指导特定的变异方向。如果未检测到显著的性能提升，过程将提前终止。以在 B200 GPU 上运行的 rms\_norm 内核为例展示了该过程。

初始提示

初始提示设定 LLM 的角色，提供可调参数，并给出输出约定：

```
You are an expert GPU kernel autotuner for Helion/Triton kernels.

Use the provided Configuration Space and Default Configuration as the source of truth for allowed field names, scalar-vs-list, required list lengths, valid ranges and defaults.

Output contract:
- Return minified JSON on a single line: {"configs":[...]}. No markdown/fences/comments.
- Only specify fields you want to change; unspecified = default.
- For list-valued fields, emit a JSON array of the exact required length shown in the space.
- If unsure about a field's structure, length, or allowed values, omit it instead of guessing.
```

Helion 还提供内核、目标硬件和配置空间。rms\_norm 内核的提示包含：

- 内核源码：实际的 @helion.kernel 源代码
- 输入张量：例如：arg\[0\]: shape=\[4096, 1024\], dtype=torch.float16, …
- GPU 硬件：例如：NVIDIA B200, 148 SMs, 178.4 GB, 2048 threads/SM
- 配置空间：每个可调字段的类型/范围
- 默认配置：基线配置。

Helion 编译器还会分析内核，将启发式信息添加到提示中。对于 rms\_norm：

```
## Compiler Analysis
Helion's compiler statically analyzed this kernel's structure and derived the following structural priors. Treat them as strong starting points.
Compiler-derived seed config(s):
{"block_sizes":[1],"load_eviction_policies":["last","last","last","last","last"],"reduction_loops":[null]}
```

## 模型返回的内容

一个包含 15 个配置的压缩 JSON。Opus-4.8 对 rms\_norm 的首次回复：

```
{"configs":[
  {"block_sizes":[1],"load_eviction_policies":["last","last","last","last","last"]},
  {"block_sizes":[1]},
  {"block_sizes":[4],"load_eviction_policies":["last", "..."],"num_warps":8},
  {"block_sizes":[8],"load_eviction_policies":["last", "..."],"num_warps":8,"num_stages":2},
  {"block_sizes":[16],"load_eviction_policies":["last", "..."],"num_warps":8,"num_stages":4},
  {"block_sizes":[1],"load_eviction_policies":["last", "..."],"pid_type":"persistent_blocked"}
  ….
]}
```

Helion 的执行框架解析这些内容，丢弃格式错误和重复的配置，然后编译并基准测试它们。

## 精调轮次

基准测试后，每一轮后续精调都会发送基于搜索状态构建的精调提示：

- 搜索状态：轮次编号、种群大小、当前最优性能。
- 锚点配置：用于变异的最优配置。
- 结果：已基准测试配置的实测性能。
- 成功/失败配置模式：哪些字段值与快速配置或失败/慢速配置相关。
- 下一步：基于配置失败率的建议

这样模型会以最优配置为锚点，并避免失败的模式。如果每轮的相对改进低于约 0.5%，反馈循环将提前停止。

## LLM 种子 LFBO：两全其美

我们还探索了一种混合策略（LLM 种子 LFBO 搜索），以解决 LLM 倾向于忽略微架构参数探索的问题。这种方法融合了 LLM 和 LFBO 的互补优势：LLM 提供强大的起点，而 LFBO 擅长局部搜索。

### 混合工作流程

1. **阶段 1 – LLM 种子**：过程从单轮 LLM 引导搜索开始，如上文"初始提示"所述。Helion 进行基准测试以验证并保留最优配置。
2. **交接：**最成功的 LLM 生成配置作为下一阶段的起点，用于训练 LFBO 的代理模型。这使得 LFBO 能够立即了解有前景的区域，而非从零开始。
3. **阶段 2 – LFBO 精调**：LFBO 搜索以种子化的初始种群执行。LFBO 执行其迭代循环：更新随机森林分类器、预测最优候选、基于特征重要性变异关键参数。该循环持续进行，直到性能增益停滞或达到最大迭代次数（20 次）。

系统返回在两个阶段中找到的最优配置。通过利用高质量的起点和信息充分的代理模型，混合搜索比冷启动的 LFBO 搜索收敛速度显著更快。这种效率使搜索预算能够集中在 LLM 可能未找到的微架构参数的精调上。

## 基准测试结果

### 方法论

我们在 NVIDIA B200 上使用 Opus 4.8 将 LFBO（全力 LFBOTreeSearch）与 LLM 引导搜索在 11 个内核上进行对比——matmul（方阵 + split-K）、grouped-GEMM、attention、fp8-attention、softmax、rms\_norm、rope、swiglu、mamba2 和 gated-delta-net——分别使用小、中、大三种形状。

### 结果 1：效率优势

这是 LLM 大放异彩的地方，以更小的搜索成本匹配 LFBO 的质量。

- **几何平均基准测试配置数：LLM 引导的自动调优器减少了 9.8 倍（每个内核约 55 个 vs 约 546 个）**。这是一个与机器无关的指标，展示了新方法的有效性。
- **几何平均实际耗时：端到端调优时间减少 6.7 倍（39 秒 vs 261 秒），在 384 线程主机上测量**。端到端调优时间包括配置生成（对于 LLM，包括其 API 往返时间）、每个候选配置的 Triton/ptxas 编译以及每个候选配置的 GPU 基准测试。

![搜索效率：每个内核评估的配置数](https://pytorch.org/wp-content/uploads/2026/06/Search-efficiency-From-Minutes-to-Seconds-blog-PyTorch-1024x512.png)

![每个内核的自动调优成本](https://pytorch.org/wp-content/uploads/2026/06/Autotuning-cost-per-kernel-From-minutes-to-seconds-PyTorch-blog-1024x512.png)

端到端调优时间主要由编译候选配置占据，Helion 在多个 CPU 核心上并行预编译这些配置。由于该主机拥有数百个线程，编译高度并行化。在核心数较少的机器上，LLM 约 10 倍少的配置将转化为更大比例的实际耗时减少。与机器无关的指标是基准测试的配置数以及最优配置收敛到最佳结果的速度（如下所示）。

## 结果 2：LLM 在 LFBO 预算的前约 7% 内收敛

绘制当前最优配置随搜索投入变化的曲线使差异一目了然。LLM 在几十个配置内就达到了其性能平台期。

![收敛 vs 搜索投入](https://pytorch.org/wp-content/uploads/2026/06/Convergence-vs-Search-Effort-From-Minutes-to-Seconds-PyTorch-blog-1024x683.png)

在所有 12 个收敛内核中，LLM 在 LFBO 预算的前约 7% 内就达到了其平台期。在 grouped GEMM（g=4, m=512）上，在相同内核性能下，配置数比 LFBO 少 18 倍。

## 结果 3：LLM 达到 LFBO 级别的性能

在内核性能方面，LLM 与 LFBO 大致持平，LLM 内核/LFBO 内核延迟的几何平均性能比为 1.009X。

![每个内核的性能](https://pytorch.org/wp-content/uploads/2026/06/Per-kernel-performance-From-minutes-to-seconds-PyTorch-blog-1024x1024.png)

因此 LLM 能快速给出好的配置，而 LFBO 可以通过更多的配置探索和调优时间获得更优性能。有 8 个案例中 LLM 在内核性能上落后 LFBO 超过 5%。

## 混合搜索能否弥合差距？

如果 LLM 的弱点在于未能精调参数，我们可以使用混合搜索策略，即 LLM 种子 LFBO TreeSearch。

我们在 LLM 落后 LFBO 超过 5% 的 8 个案例上运行了混合搜索。

![混合搜索能否弥合差距？](https://pytorch.org/wp-content/uploads/2026/06/Does-the-hybrid-close-the-gap-from-minutes-to-seconds-PyTorch-blog-1024x661.png)

混合策略在所有案例中都提升了内核性能，并在 8 个案例中的 6 个弥合了与 LFBO 的差距。mamba2 系列仍然表现不如 LFBO，我们正在研究改进 LLM 启发式方法以弥合这一差距。

在自动调优时间方面，混合搜索比 LFBO 显著更高效。在 8 个内核中，它探索的配置数比 LFBO 少 4 倍，端到端自动调优时间快 3 倍。LLM-only、混合和 LFBO 的几何平均对比结果如下所示。

| | **仅 LLM** | **混合** | **完整 LFBO** |
|---|---|---|---|
| **自动调优时间** | 44 秒 | 111 秒 | 328 秒 |
| **探索的配置数** | 59 | 186 | 686 |

我们还展示了下面各个内核的结果，对比了探索的配置数量及其调优时间：

![仅 LLM vs 混合 vs LFBO-自动调优](https://pytorch.org/wp-content/uploads/2026/06/LLM-only-vs-Hybrid-vs-LFBO-Autotuning-from-minutes-to-seconds-PyTorch-Blog-1024x410.png)

## 模型选择是否重要？

以上所有结果都使用了一个模型：Claude Opus-4.8。人们可能会问，执行工作的模型是否至关重要，还是任何能力足够的 LLM 都能达到相同效果。为此，我们在全部 33 个内核实例上使用另外两个模型——OpenAI gpt-5.5 和 Claude Sonnet-4.6——对 LLM-only 搜索（LLM 引导搜索）进行了基准测试，与 Opus 4.8 基线进行对比。

| **模型** | **相对于 Opus-4.8 的几何平均性能** | **几何平均探索配置数** |
|---|---|---|
| **Opus-4.8** | 1.00（基线） | 55 |
| **gpt-5.5** | 0.98 | 61 |
| **Sonnet-4-6** | 1.03 | 51 |

在几何平均上，所有 3 个模型的表现非常相似，有趣的是，Sonnet-4.6 以最少的配置数完成了任务。

## 结论

我们要回答的问题是："LLM 能否像 LFBO 搜索一样好地自动调优 Helion 内核，但成本远低于 LFBO？"在 B200 上基准测试的 33 个内核套件中，答案是肯定的。

**效率提升显著：**LLM 引导的自动调优器在 **LFBO 预算的 7% 内收敛到 LFBO 质量的结果，探索约 10 倍少的配置，实际耗时减少约 6.7 倍**，为开发者效率提供了巨大提升。

**LLM 达到 LFBO 级别性能**：LLM 引导的自动调优器在大多数内核上与 LFBO 搜索持平，甚至在某些内核上胜出。LFBO 在某些情况下以更长的自动调优时间为代价胜出。

**混合策略弥合差距**：混合方法（LLM 种子后接 LFBO 精调）可以恢复剩余性能，同时成本仍比 LFBO 搜索低约 3 倍。

**实用方案**：对于简化的工作流程，我们建议先尝试 LLM-only 搜索以快速识别高性能内核。为了最大化性能，用户可以使用混合搜索来精调并获取最终的性能提升。展望未来，我们计划增强启发式方法，以进一步提升 LLM 引导和混合自动调优器的有效性。

---
原文链接: https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/
