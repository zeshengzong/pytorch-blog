# 利用 ExecuTorch 与 Arm SME2 加速端侧 ML 推理

**作者：** Jason Zhu、Tyler Mullenbach、Damien Dooley、Gian Marco Idoice，Arm
**日期：** 2026 年 1 月 29 日

交互式图像分割已成为全球最受欢迎应用中的标志性移动体验。简而言之，你在图像上点击（或大致勾画），应用便会即时"抠出"目标对象，生成像素掩码。这支撑了一系列熟悉的功能，例如制作个性化贴纸、为图像替换背景，或对图像局部进行选择性增强。如果你用过 Instagram 的抠图工具，就能体会到对象掩码直接在设备上生成是多么轻松。这些效果背后，是通过 ExecuTorch（PyTorch 的开源端侧推理运行时）和 Arm SME2（可扩展矩阵扩展 2）驱动的紧凑型分割模型。

本文探讨这些硬件与软件进步如何在 SqueezeSAM（Instagram 抠图功能背后的端侧交互式分割模型）上实现高达 **3.9 倍的加速**，以及对移动应用开发者的广泛影响。SqueezeSAM 已在 Meta 系列应用中部署，详见此博客文章。

**本文面向读者：** 从事移动及边缘设备端侧 AI 部署的机器学习工程师与开发者，希望了解 SME2 对推理性能的影响，以及如何利用算子级性能分析优化模型。

### 端侧 AI 在移动设备上的崛起

随着端侧 AI 的发展，一个关键问题是：当更强大的模型能在严格的移动功耗与时延约束下更快运行时，会带来哪些新的可能性？实际上，许多交互式移动 AI 功能和工作负载已经在 CPU 上运行，原因在于 CPU 始终可用、与应用无缝集成，同时在多样化场景下提供高灵活性、低延迟和强性能。对于这类部署，性能往往取决于 CPU 执行矩阵密集型内核的效率，以及在算力不再是瓶颈之后还剩哪些制约因素。

SME2 是 Armv9 架构中引入的一组高级 CPU 指令，专为直接在设备上加速面向矩阵的计算工作负载而设计。我们量化了 SME2 在 ExecuTorch 与 XNNPACK 部署中对端到端推理的加速效果，并通过算子级分析揭示了哪些方面得到了提升。搭载 SME2 的新一代 Arm CPU 已应用于面向旗舰智能手机和下一代 PC 的 Arm Lumex 计算子系统（CSS），支持 SME2 的设备列表可在此处查看。

### 案例研究：利用 SME2 加速交互式图像分割

我们以 ExecuTorch 和 XNNPACK 为后端，测量 SME2 对 SqueezeSAM 端到端推理延迟的影响。该后端使用 Arm KleidiAI 优化内核来发挥 SME2 的加速优势。

启用 SME2 后，8 位整数（INT8）和 16 位浮点（FP16）推理均获得显著提升（图 1）。在单 CPU 核心、默认功耗设置下，INT8 延迟提升 1.83 倍（从 556 ms 降至 304 ms），FP16 提升 3.9 倍（从 1,163 ms 降至 298 ms）。未启用 SME2 时，这些延迟对于响应式交互体验来说过高；启用 SME2 后，单核端到端推理延迟降至约 300 ms，使端侧执行切实可行，同时为应用的其余部分留有余量。

上述结果表明，SME2 显著加速了 CPU 上的量化 INT8 模型。与此同时，在本案例研究中，SME2 使 FP16 延迟接近 INT8，这一点尤为值得关注，因为它拓宽了实际部署选项，而非取代 INT8。这为开发者提供了更大的灵活性，可根据精度和工作流需求选择最合适的数值精度，特别适用于精度敏感型工作负载，例如图像超分辨率、图像抠图、低光去噪和高动态范围（HDR）增强。若没有这种程度的 FP16 加速，移动端部署往往被迫选用 INT8 以满足延迟目标，尽管这意味着引入量化流程及精度损失风险。

除基准数字外，这些加速直接转化为 CPU 算力余量，可用于更丰富的体验，例如：在保持相机预览和界面响应的同时并行运行分割与增强（如去噪或 HDR），或将抠图从单张图片扩展到具有跨帧目标追踪的实时视频抠图，或降低功耗。

![图 1：SME2 启用与禁用时 SqueezeSAM 的端到端延迟对比](https://pytorch.org/wp-content/uploads/2026/01/1-5.png)

**图 1.** SME2 启用与禁用时 SqueezeSAM 在 1 个 CPU 核心正常模式（默认移动功耗设置）下的端到端延迟。INT8 从 556 ms 降至 304 ms（1.83 倍），FP16 从 1,163 ms 降至 298 ms（3.90 倍），在本案例研究中 FP16 延迟接近 INT8。

本文所有结果均为在搭载 SME2 Arm CPU 的 vivo X300 Android 旗舰智能手机上的受控测量。实际性能可能因模型、硬件和具体设备设置而有所不同。

## 软件栈：PyTorch、ExecuTorch、XNNPACK、Arm KleidiAI 与 SME2

### 框架间的协作关系

![CPU 执行软件栈：PyTorch → ExecuTorch → XNNPACK → Arm KleidiAI → SME2](https://pytorch.org/wp-content/uploads/2026/01/2-4.png)

上图概括了本案例研究所用的 CPU 执行软件栈。模型在 PyTorch 中定义，由 ExecuTorch 导出并运行，CPU 算力通过 XNNPACK 作为后端进行委托。XNNPACK 使用 Arm KleidiAI——Arm 面向 Arm CPU 上 ML 工作负载加速的轻量优化内核库。这些内核在受支持的设备上可自动利用 SME2 加速，同时也为非 SME2 系统提供针对其他 CPU 特性的优化实现。

当 ExecuTorch 启用 XNNPACK 委托运行模型时，XNNPACK 会在运行时根据底层硬件能力选择合适的内核实现。在 SME2 设备上，模型架构和应用代码无需任何改动，即可让矩阵乘法计算受益于 SME2 加速。一旦这些操作被加速，推理流水线的其他部分——如数据移动、布局转换和未委托算子——往往会成为新的瓶颈。这正是算子级分析对于理解端到端性能不可或缺的原因。

### 案例研究模型

我们的评测使用 SqueezeSAM，它采用轻量级、以 conv2d 为主的 UNet 架构，代表了众多移动视觉模型的典型结构。

模型结构自然地映射为对端到端推理时间影响最大的两类工作：

- **数学密集型操作：** 卷积（iGEMM，隐式通用矩阵乘法）以及注意力/MLP 层（GEMM，通用矩阵乘法）
- **数据移动操作：** 转置、形状变换和布局转换

平台说明：在许多基于 Armv9 架构的设备上，SME2 作为 CPU 核心间的共享执行资源实现，其扩展行为可能因片上系统（SoC）和 CPU 微架构而异。我们在评测中明确考虑了这一点，并在解读单核与多核结果时讨论其影响。

## 结果：INT8 与 FP16（1 CPU 核心 vs 4 CPU 核心）

我们以两种精度（INT8 和 FP16）在 SME2 启用与禁用两种状态下对同一模型进行基准测试。重点关注单核执行（SME2 相对收益最大），同时报告四核结果，以体现 SME2 作为共享硬件资源时的绝对延迟和扩展行为。所有测量均报告仅模型延迟。

模型在 Android 智能手机上使用 ExecuTorch 执行，SME2 启用与禁用两种状态下软件和系统条件完全相同。除非另有说明，结果反映的是无热降频的稳态性能。

所有结果以"正常模式 | 无约束模式（ms）"格式报告。正常模式对应启用系统功耗策略的默认移动功耗设置，代表典型用户行为。无约束模式对应通电常亮配置，CPU 频率上限实际被取消；单核测量的无约束模式结果固定在最高性能（Ultra/Prime，本次为 4.2 GHz）CPU 核心上。

在两种模式下，SME2 均表现出一致的相对加速趋势，表明其优势对系统功耗策略具有鲁棒性，尽管绝对延迟有所不同。除非明确说明，本文其余部分重点关注**正常模式**结果，因其更能反映典型智能手机使用条件下的用户感知延迟。**无约束模式**结果用于说明性能上限和硬件极限，应视为最优情况，而非日常用户体验的预期表现。

| 精度 | 核心数 | SME2 关闭（ms） | SME2 开启（ms） | 加速比 |
|------|--------|----------------|----------------|--------|
| INT8 | 1 | 556 \| 334 | 304 \| 172 | 1.83× \| 1.95× |
| INT8 | 4 | 195 \| 106 | 180 \| 104 | 1.08× \| 1.03× |
| FP16 | 1 | 1,163 \| 735 | 298 \| 173 | 3.90× \| 4.26× |
| FP16 | 4 | 374 \| 176 | 193 \| 124 | 1.94× \| 1.42× |

**表 1.** SqueezeSAM 在 Android 手机上 SME2 启用与禁用时的端到端延迟结果，分别在 1 个和 4 个 CPU 核心下测量（仅模型延迟）。数值以"正常模式 | 无约束模式"格式报告。

关于四核扩展性的说明：四核的较小加速比（例如 INT8 为 1.08×，而单核正常模式为 1.83×）与 SME2 是共享资源这一特性一致，同时也受内存带宽和缓存行为等其他共享系统效应影响。扩展特性可能因 SoC 和 CPU 实现而异。在生产部署中，若能满足延迟目标，优先使用 1 至 2 个核心以提升功耗效率；若需要更低绝对延迟且功耗预算允许，可使用更多核心。

### 为何算子级分析至关重要

端到端延迟告诉我们性能提升了多少，却无法解释原因，也无法指引下一步优化方向。为了理解 SME2 在哪里发挥作用以及之后的瓶颈在哪里，我们使用了算子级分析。

我们使用 ETDump 收集每个算子的计时信息。ETDump 是 ExecuTorch DevTools 中的一款性能分析工具，可在推理过程中记录各算子的执行时间。这使我们能够将端到端加速归因于模型的具体部分，如图 2 和表 2 所示。

为使分析具有可操作性，我们将算子归纳为少量类别，与常见模型结构清晰对应：

- **卷积（Convolution）：** Conv2d 层（通常使用 iGEMM 实现）
- **GEMM：** 矩阵乘法和线性层（注意力机制和 MLP 投影）
- **逐元素（Elementwise）：** ReLU、GELU、Add、Mul 及其他逐点操作
- **数据移动（Data Movement）：** 转置、复制、转换、形状变换和填充
- **其他（Other）：** 未委托算子和框架开销

有了这个分类框架，我们就能解释 SME2 在哪些地方帮助最大，以及矩阵算力被加速后还剩什么。

![图 2：FP16 和 INT8 在 SME2 启用与禁用时的算子类别耗时分解](https://pytorch.org/wp-content/uploads/2026/01/3-3.png)

**图 2.** FP16 和 INT8 在 SME2 启用与禁用时的算子类别耗时分解（绝对时间），测量于 Android 智能手机（1 个 Arm CPU 核心，默认移动功耗设置）。SME2 大幅减少了卷积和 GEMM 时间，数据移动在运行时中占比提升。

| 类别 | INT8 SME2 关（ms） | INT8 SME2 开（ms） | INT8 加速比 | INT8 占比（开启） | FP16 SME2 关（ms） | FP16 SME2 开（ms） | FP16 加速比 | FP16 占比（开启） |
|------|-------------------|-------------------|------------|-----------------|-------------------|-------------------|------------|-----------------|
| 卷积 | 309.7 | 69.8 | 4.4× | 23.0% | 881.2 | 98.1 | 9.0× | 32.9% |
| GEMM | 27.3 | 8.1 | 3.4× | 2.7% | 31.6 | 7.6 | 4.1× | 2.6% |
| 逐元素 | 2.1 | 2.2 | 1.0× | 0.7% | 1.6 | 1.7 | 0.9× | 0.6% |
| 数据移动 | 123.0 | 125.8 | 1.0× | 41.4% | 139.0 | 119.1 | 1.2× | 39.9% |
| 其他 | 93.7 | 98.2 | 1.0× | 32.3% | 109.6 | 71.7 | 1.5× | 24.0% |
| **端到端** | **555.8** | **304.1** | **1.83×** | **–** | **1,163.0** | **298.2** | **3.90×** | **–** |

**表 2.** INT8 和 FP16 在 SME2 关闭与开启时的算子级分解（Android 手机，单 CPU 核心，默认移动功耗设置）。非矩阵乘法算子的变化主要来自运行时波动。

## 端到端与算子级结果的三点洞察

### 洞察 1：SME2 加速矩阵算力，将瓶颈转移至数据移动

SME2 显著降低了 INT8 和 FP16 的端到端延迟。在单个 Arm CPU 核心上，INT8 提升 1.83 倍（556 ms 降至 304 ms），FP16 提升 3.90 倍（1,163 ms 降至 298 ms）。即使在四核上，SME2 也大幅降低了 FP16 延迟（374 ms 降至 193 ms）。这些收益将单核执行延迟拉入约 300 ms 区间，使交互式端侧执行切实可行，同时为应用其余部分保留了 CPU 余量。

算子级分析显示，SME2 大幅加速了矩阵密集型算子。禁用 SME2 时，卷积和 GEMM 主导推理时间，分别占 INT8 运行时的 55.7% 和 FP16 运行时的 75.8%。启用 SME2 后，GEMM 加速约 3–4 倍，卷积/iGEMM 加速约 4–9 倍，这是端到端加速的主要驱动力。

矩阵算力被加速后，数据移动和框架开销的相对成本上升，优化重心随之转移。

### 洞察 2：转置驱动的数据移动约占运行时的 40%

SME2 加速后，数据移动成为运行时的主要组成部分之一。在 INT8 启用 SME2 的运行中，数据移动占总运行时的 41.4%（FP16 为 39.9%）。ETDump 追踪显示，约 85% 的数据移动时间来自转置算子，仅两种转置节点类型就消耗了该类别超过 80% 的时间。

这一开销由模型和运行时的布局不匹配驱动，而非由算术强度导致。在实践中，当具有不同布局偏好的算子顺序组合时，会强制进行反复的 NCHW↔NHWC 转换。在本模型中，归一化操作以可移植的 NCHW 算子执行，但在某些上下文中无法与相邻卷积融合（例如，当非线性激活位于 Conv2d 和 BatchNorm 之间时），而 XNNPACK 的卷积内核偏好 NHWC 布局。这导致在 UNet 编解码块中出现反复的布局变换：

```
BatchNorm/GroupNorm (NCHW) → Transpose (NCHW→NHWC) → Convolution (NHWC) → Transpose (NHWC→NCHW) → BatchNorm/GroupNorm (NCHW)
```

由于这一成本源于模型和运行时的布局选择，而非算术强度，分析是将其浮出水面并使其成为可操作优化目标的关键。

值得一提的是，这一分析洞察已被证明具有实际价值。作为初步措施，我们在 ExecuTorch 中实现了一项针对性的图级优化，以减少归一化周围不必要的布局转换。实验表明，在 SME2 加速基础上，INT8 额外减少约 70 ms（23%）延迟，FP16 额外减少约 30 ms（10%）。

这些结果证实，转置密集型数据移动是有意义的优化机会，随着我们持续分析整个计算图的布局行为，进一步改进的空间仍然存在。更广泛的发现及其影响将在后续文章中介绍。

### 洞察 3：启用 SME2 后，FP16 在本案例研究中接近 INT8 级延迟

尽管 INT8 每个张量元素使用一半的内存带宽，但这并不自动转化为等比例的端到端加速。启用 SME2 后，FP16 在本案例研究中实现了接近 INT8 的延迟（单核 298 ms vs 304 ms）。

算子分解揭示了原因。FP16 的卷积加速尤为突出（9.0× vs INT8 的 4.4×），弥补了 INT8 的内存效率优势。与此同时，INT8 矩阵路径因量化、缩放和更复杂的内核分发逻辑而存在额外开销，削弱了 INT8 的有效带宽优势。

最终结论是：SME2 拓宽了可行精度选项的范围。INT8 仍是有效选择，而 FP16 对于不希望引入量化复杂性或精度损失的精度敏感型工作负载而言变得更加实用。尽管在本案例研究中 FP16 接近 INT8 的性能，但这一行为取决于具体工作负载，可能因算子组合、张量形状和内存压力而有所不同。

## 动手实践：复现工作流

如果你想亲自尝试这套工作流，我们提供了一个基于开源 SAM 模型的实践教程，内容涵盖模型导出、使用 SME2 运行推理以及使用 ETDump 进行算子级分析。完整的设置说明和代码示例见代码仓库和学习路径。

你将学到：

- 如何将分割模型导出为启用 XNNPACK 委托的 ExecuTorch 格式
- 如何将模型构建并部署到支持 SME2 的 Android、iOS 和 macOS 设备
- 如何运行 ETDump 分析以收集每个算子的计时信息
- 如何识别和量化自有模型中的数据移动及其他非数学瓶颈

## 结论：SME2 在实践中改变了什么

在本 SqueezeSAM 案例研究中，SME2 为 INT8 和 FP16 均带来了显著的端侧 CPU 加速，从根本上改变了交互式移动工作负载的可行边界。

对开发者和产品团队的意义：

- **端侧 ML 在 CPU 上更加可行**：SME2 实现高达 3.9 倍的端到端推理加速，在默认 Android 功耗设置下将真实交互式移动模型的单核延迟从超过 1 秒降至约 300 ms。这将基于 CPU 的端侧 ML 从勉强可用提升至切实可行，同时为应用其余部分保留余量。

- **FP16 在某些场景下成为更可行的部署选项**：通过大幅加速 FP16 并缩小与 INT8 的延迟差距，SME2 让开发者在选择最符合精度、工作流和延迟要求的数值精度时拥有更大灵活性，特别适用于精度敏感型工作负载。

- **节省的算力余量支持更丰富的体验**：释放的 CPU 资源可用于增加端侧功能，例如同时运行分割与增强（如去噪或 HDR），或将抠图从单张图片扩展到具有跨帧目标追踪的实时视频。

- **分析揭示下一个优化目标**：SME2 加速矩阵密集型算子（卷积/iGEMM 和 GEMM）后，瓶颈往往转移至数据移动和未委托算子。使用 ETDump 进行算子级分析可使这些成本可见且可操作。

两点具体建议，视出发点而定：

- 如果你目前尚未部署端侧 ML，SME2 加速的 CPU 推理可作为数学密集型模型移动 CPU 部署的可行起点，分析提供了验证性能并迭代优化的清晰路径。
- 如果你已经部署了端侧模型，SME2 可为扩展功能和改善用户体验创造余量，而分析能够指出影响最大的下一步改进方向（对于 SqueezeSAM，转置驱动的布局转换约占总运行时的 40%）。

SME2 加速与算子级分析相结合，为端侧 AI 提供了一套实用工作流，既能即时释放性能收益，又能持续定位最高价值的优化方向。

### 致谢

感谢 Meta ExecuTorch 团队 Bilgin Cagatay、Mergen Nachin、Digant Desai、Gregory Comer 和 Andrew Caples 在真实用例上的指导以及对推理优化实现的贡献。同时感谢 Arm 的 Ray Hensberger、Ed Miller、Mary Bennion 和 Shantu Roy 在整个工作过程中给予的支持与指导。

## 链接汇总

- 原文地址: [https://pytorch.org/blog/accelerating-on-device-ml-inference-with-executorch-and-arm-sme2/](https://pytorch.org/blog/accelerating-on-device-ml-inference-with-executorch-and-arm-sme2/)
- Instagram 抠图工具: [https://help.instagram.com/1382185835750156](https://help.instagram.com/1382185835750156)
- ExecuTorch 项目: [https://github.com/pytorch/executorch](https://github.com/pytorch/executorch)
- Arm SME2 技术页: [https://www.arm.com/technologies/sme2](https://www.arm.com/technologies/sme2)
- SqueezeSAM 论文: [https://arxiv.org/abs/2312.06736](https://arxiv.org/abs/2312.06736)
- Meta 应用中的 ExecuTorch 部署博客: [https://engineering.fb.com/2025/07/28/android/executorch-on-device-ml-meta-family-of-apps/](https://engineering.fb.com/2025/07/28/android/executorch-on-device-ml-meta-family-of-apps/)
- Armv9 架构: [https://www.arm.com/architecture/cpu/a-profile/armv9](https://www.arm.com/architecture/cpu/a-profile/armv9)
- 新一代 SME2 Arm CPU: [https://newsroom.arm.com/blog/arm-c1-cpu-cluster-on-device-ai-performance](https://newsroom.arm.com/blog/arm-c1-cpu-cluster-on-device-ai-performance)
- Arm Lumex 计算子系统（CSS）: [https://newsroom.arm.com/news/announcing-lumex-css-platform-ai-era](https://newsroom.arm.com/news/announcing-lumex-css-platform-ai-era)
- 支持 SME2 的设备列表: [https://learn.arm.com/learning-paths/cross-platform/multiplying-matrices-with-sme2/1-get-started](https://learn.arm.com/learning-paths/cross-platform/multiplying-matrices-with-sme2/1-get-started)
- Arm KleidiAI（GitHub）: [https://github.com/ARM-software/kleidiai](https://github.com/ARM-software/kleidiai)
- Arm KleidiAI（产品页）: [https://www.arm.com/markets/artificial-intelligence/software/kleidi](https://www.arm.com/markets/artificial-intelligence/software/kleidi)
- ETDump 文档: [https://github.com/pytorch/executorch/blob/main/docs/source/etdump.md](https://github.com/pytorch/executorch/blob/main/docs/source/etdump.md)
- 动手实践代码仓库: [https://github.com/ArmDeveloperEcosystem/sme-executorch-profiling](https://github.com/ArmDeveloperEcosystem/sme-executorch-profiling)
- 动手实践学习路径: [https://learn.arm.com/learning-paths/cross-platform/sme-executorch-profiling/](https://learn.arm.com/learning-paths/cross-platform/sme-executorch-profiling/)
