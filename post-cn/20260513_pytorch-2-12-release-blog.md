# PyTorch 2.12 发布博客

我们很高兴地宣布 PyTorch® 2.12 正式发布（版本说明）！

PyTorch 2.12 版本包含以下变化：

* CUDA 上的批量 `linalg.eigh` 因 cuSolver 后端选择逻辑更新，速度最高提升 100 倍
* 新增 `torch.accelerator.Graph` API，统一了 CUDA、XPU 及第三方后端的图捕获与回放接口
* `torch.export.save` 现已支持 Microscaling（MX）量化格式，可完整导出经过激进压缩的模型
* Adagrad 现在支持 `fused=True`，加入 Adam、AdamW 和 SGD，实现单内核优化器实现
* `torch.cond` 控制流现在可以在 CUDA Graph 内部进行捕获和回放
* ROCm 用户新增可扩展内存段、rocSHMEM 对称内存集合通信以及 FlexAttention 流水线

本次版本自 PyTorch 2.11 以来，共包含来自 457 位贡献者的 2,926 次提交。我们衷心感谢社区的每一份贡献。一如既往，我们鼓励大家试用并及时反馈问题，帮助我们持续改进 2.12。关于如何开始使用 PyTorch 2 系列的更多信息，请访问我们的入门页面。

有问题吗？欢迎在 5 月 20 日（周三）太平洋时间上午 10 点加入我们的直播问答，嘉宾包括 Joe Spisak、Andrey Talman 和 Alban Desmaison，主持人为 Chris Gottbrath。我们将对本次发布进行简要介绍并现场解答大家的问题。立即注册。

在整个 2.x 系列的演进过程中，PyTorch 正在从以研究为先的框架，转变为统一的、硬件无关的生产训练和推理规模化平台。PyTorch 2.10 奠定了基础，引入了跨后端性能原语并正式弃用了 TorchScript。PyTorch 2.11 在此基础上进一步扩展，新增了用于分布式训练的可微分集合通信、面向下一代 GPU 的 FlashAttention-4，以及更广泛的导出覆盖。

PyTorch 2.12 延续了这一方向：新的设备无关 torch.accelerator.Graph API 统一了 CUDA、XPU 及第三方后端的图捕获与回放；批量特征值分解速度最高提升 100 倍；torch.export 现已支持 Microscaling 量化格式，用于部署经过激进压缩的模型。纵观这些版本，PyTorch 正在各后端上变得更快，可适用于更广泛的平台，持续推动 AI 创新。

### 性能特性

**CUDA 上批量特征值分解最高提速 100 倍（`linalg.eigh`）**

CUDA 上 linalg.eigh 的后端选择逻辑已全面重构。遗留的 MAGMA 后端已被弃用，转而使用 cuSolver（PR #174619，由 Grayson Derossi 贡献），同时 cuSolver 的调度启发式策略已更新为无条件使用 syevj_batched（PR #175403，由 Johannes Z 贡献）。对于批量对称/厄米特征值问题，与上一版本相比，速度最高可提升 100 倍，解决了与 CuPy 长期存在的性能差距。

此前需要数分钟才能完成的工作负载（因为 PyTorch 在低效地逐一分派每个矩阵求解），现在通过使用 cuSolver 的 syevj_batched 内核可在数秒内完成——该内核专为将多个小/中型矩阵作为单个 GPU 操作一次性处理而设计。这些性能收益对依赖批量矩阵特征值分解的科学计算和机器学习工作负载尤为重要。（文档中的使用示例）

**融合 Adagrad 优化器**

Adagrad 优化器现已支持 fused=True，将整个优化器步骤在单个 CUDA 内核中完成，而不是为每个操作分别启动内核。这减少了内核启动开销和内存带宽占用。Adagrad 加入了 Adam、AdamW 和 SGD，共同提供融合变体。底层 CUDA 内核由 @MeetThePatel 在 2.11 周期中贡献（PR #159008），向用户暴露该功能的 Python 前端由 Jane Xu 在 2.12 中完成（PR #177672）。

### 跨硬件的编译与导出

**`torch.accelerator.Graph`：设备无关的加速器图捕获与流 API**

`torch.accelerator.Graph` 是一个新的设备无关 API，用于图捕获和回放，在 `torch.xpu.XPUGraph` 等特定后端实现之上提供统一抽象。每个后端可以通过轻量级的 GraphImplInterface 注册自己的实现，在保持后端自主性的同时提供一致的用户接口。

与此同时，`c10::Stream` 和 `torch.Stream` 现已暴露 `is_capturing()` 方法，以设备无关的替代方案取代了特定设备的 `is_current_stream_capturing`。流上下文管理器的可重入问题也已修复。这些变化共同实现了流和图管理的跨后端对等，初步支持 XPU 后端，并可通过 `PrivateUse1` 扩展至第三方后端。
由 Guangye Yu（Intel）贡献，跨六个 PR 完成，核心为 C++ 接口（PR #171269）和 Python 前端（PR #171285）。（文档字符串中的使用示例）

**`torch.export` 现已支持 Microscaling（MX）量化格式**

随着模型从研究走向生产，`torch.export` 是序列化 PyTorch 模型以进行部署的标准路径。然而，使用 Microscaling（MX）量化的模型——这是一种日益流行的用于缩小模型体积和降低推理成本的技术——此前无法导出，因为 `torch.export.save` 无法处理 MX 格式（MXFP4、MXFP6、MXFP8）中用作共享块尺度指数的 `float8_e8m0fnu` 数据类型。

在 PyTorch 2.12 中，`torch.export.save` 和 `torch.export.load` 现已能够正确序列化和反序列化具有该数据类型的张量，解除了利用 Microscaling 量化的模型从导出到部署全流程的阻塞。这对于在成本受限或边缘环境中部署大型语言模型、需要激进量化的团队尤为重要。由 Chizkiyahu Raful（ARM）贡献（PR #176270）。

**在 CUDA Graph 内捕获 torch.cond 控制流**

使用 torch.cond 的控制流区域现在可以作为 CUDA Graph 的一部分进行捕获和回放。此前，数据相关的控制流会因为分支在 CPU 上求值而强制回退到 CUDA 图树。通过利用 CUDA 12.4 的条件 IF 节点，torch.cond 分支现在完全在 GPU 上、在单次图捕获内求值。

由 Daniel Galvez 和 Ting-Yang Kuei（NVIDIA）贡献（PR #168912），Paul Zhang（Meta）添加了 Inductor 排序支持（PR #179457）。目前适用于 eager 和 cudagraphs 后端；Inductor 支持计划在未来版本中推出。

**XPU 的基于 FMA 的 addcdiv 下降**

Inductor 现在对 addcdiv 操作使用融合乘加（FMA）指令，在保持 Triton 内核融合优势的同时，实现了与 eager CUDA 执行的逐位数值一致性。

`addcdiv` 是一种融合算术运算（`result = input + value × (tensor1 / tensor2)`），是许多优化器更新规则（包括 Adam、AdamW 和 RMSprop）的核心。此前，Inductor 的下降使用独立的乘法和除法指令，与 eager 模式相比引入了细微的浮点舍入差异。这些差异在数千个训练步骤中累积，导致难以验证编译后的模型是否产生数值相同的结果。

该功能最初由 Michael Lazos（Meta）为 CUDA 实现（PR #174912），随后由 Guangye Yu（Intel）扩展至 XPU（PR #176163），修复了 Intel GPU 上的若干数值正确性问题。任何在优化器密集型训练循环中使用 `torch.compile` 的用户，现在都能在不牺牲数值可复现性的前提下获得编译后的性能提升——无论是 NVIDIA 还是 Intel 硬件。

### 分布式训练

**自定义算子中的 ProcessGroup 支持**

自定义算子现在可以直接接受 ProcessGroup 对象作为参数，而不再要求调用者将其转换为字符串组名并在全局注册表中查找。所有 c10d 函数式集合算子（all_reduce、reduce_scatter 等）已更新为同时接受直接传入的 ProcessGroup 对象和字符串名称。由 Aaron Orenstein（Meta）贡献（PR #172795）。

**多 GPU/多节点性能分析改进**

PyTorch Profiler Events API 现在暴露了流 ID、流类型、活动类型、未完成事件和 Python 函数事件——使 events() 与 Chrome 追踪 JSON 输出达到同等功能，并支持更丰富的程序化事后分析。此外，现在可以使用新的 seq_num 字段在不同 rank 之间关联 NCCL 集合追踪——参与同一集合操作的所有 rank 在同一进程组内共享相同的序列号。这些改进显著提升了跨多 GPU 和多节点调试分布式训练性能的工具能力。API 增强由 Ryan Zhang（Meta）贡献（PR #177888），NCCL seq_num 由 Marvin Dsouza（Meta）添加（PR #177148）。

**FlightRecorder：ncclx + gloo 后端**

FlightRecorder 的追踪分析器现在除了现有的 nccl 和 xccl 后端外，还支持 ncclx 和 gloo 后端，可在更广泛的集合后端上进行分布式通信追踪。此外，FlightRecorder 现在可以识别此前未被追踪的 torchcomms 操作（例如 all_gather_single、reduce_scatter_v、barrier）。本周期还修复了一个在多个进程组并发访问 FlightRecorder 单例时可能导致无限循环的竞争条件。后端允许列表由 Lily Janjigian（Meta）添加（PR #180268），torchcomms 操作支持由 Tushar Jain 贡献（PR #178359）。

## 平台相关更新

### CUDA

**CUDA Graph 内核注解**

`torch.cuda.graph` 现在接受 enable_annotations 关键字参数，可将注解元数据（例如集合算子名称、进程组、消息大小）注入捕获的 CUDA Graph 中的各个内核。通过配套的后处理脚本（python -m torch.cuda._annotate_cuda_graph_trace）对追踪器进行后处理后，注解会合并进追踪文件。这些注解会显示在 Perfetto/Chrome 性能分析器追踪中，使理解回放图中每个内核的功能变得更加容易。由 Shangdi Yu（Meta）贡献（PR #179768）。

**CUDA Green Context 工作队列限制**

CUDA Green Context 现在支持指定工作队列限制，对 GPU 资源分区提供更细粒度的控制。这项实验性功能允许用户限制 green context 内并发工作提交的数量，从而在并发工作负载之间实现更可预测的资源共享。由 Matthias Jouanneaux（NVIDIA）贡献（PR #177242）。

### ROCm

**ROCm：可扩展内存段**

AMD GPU（ROCm >= 7.02）现在在 PyTorch 的缓存分配器中支持可扩展内存段，与通过虚拟内存 API 动态扩展分配以减少内存碎片化的 CUDA 功能相匹配。由 Prachi Gupta（AMD）添加（PR #173330）。

**ROCm：rocSHMEM 支持**

rocSHMEM 支持在 AMD GPU 上启用了对称内存集合操作（torch.ops.symm_mem.*），将基于 NVSHMEM 的 GPU 内通信原语（包括点对点、广播、全对全以及面向 MoE 的 2D AllToAllv）移植到了 ROCm。rocSHMEM 实现使用专用编译单元来处理 NVSHMEM 和 rocSHMEM 之间的 API 和 warp 大小差异。由 Prachi Gupta 贡献（PR #173518）。

**ROCm：hipSPARSELt 和 FP8 半结构化稀疏**

hipSPARSELt 现已在 ROCm >= 7.12 的 PyTorch 构建中默认启用，为 AMD GPU 带来了半结构化（2:4）稀疏支持。FP8（float8_e4m3fn）输入现在也通过 MI350X（gfx950）上的 hipSPARSELt 获得支持，输出为 FP32。这启用了与之前仅支持 CUDA 的相同的 torch._cslt_sparse_mm 稀疏加速路径。hipSPARSELt 由 rraminen（AMD）启用（PR #170852），FP8 半结构化稀疏由 Benji Beck（Meta）添加（PR #179310）。

**ROCm：Inductor FlexAttention 流水线**

AMD GPU 上的 FlexAttention 现在在 Triton 后端使用两阶段流水线，在 MI350X 上针对多种注意力模式（因果、alibi、滑动窗口）和形状实现了 5-26% 的速度提升。这是一个单行配置更改（num_stages 从 1 改为 2），解锁了更高效的内存-计算重叠。由 nithinsubbiah 贡献（PR #176676）。

### Apple MPS

**MPS：Metal-4 离线着色器编译**

Apple Silicon 二进制 wheel 现在随附提前编译好的 Metal-4 着色器，基于 macOS 26 使用 metal-4 标准构建。这消除了首次运行时的运行时着色器编译开销，降低了 MPS 工作负载的启动延迟。同时还新增了一个伴随 API（`torch._C._mps_loadMetalllib`），用于直接加载预编译的 .metallib 二进制文件，支持 Triton Apple MPS 后端的编译时 metallib 工作流。由 Isalia20（Irakli Salia）贡献（PR #179378）。

## 弃用与破坏性变更

#### 分布式：torchcomms 计划中的破坏性变更

我们一直在努力将 torchcomms 直接集成到 PyTorch Distributed 中，以便所有人都能开箱即用地获得其带来的好处。在即将发布的版本（2.13+）中，我们计划默认使用 torchcomms，这包括对 ProcessGroup 运作方式的一些破坏性变更。我们的目标是使这些更改对大多数模型自动生效，并修复生态系统中的任何不兼容问题，但仍然有部分模型会受到影响。

我们仍在完善 torchcomms，但你现在就可以使用它，并获得新 API、容错、窗口、可扩展性和可调试性功能。要开始使用，请执行 `pip install torchcomms` 并设置 `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1`。

更多详情请参见 https://github.com/meta-pytorch/torchcomms。

主要变更：

* 即时初始化：我们将要求所有 ProcessGroup/通信子在 dist.init_process_group 期间即时初始化，且只支持单个后端设备。这意味着必须在初始化时指定设备。
* P2P 操作：我们的目标是使每个 ProcessGroup/通信子与底层通信子一一对应。这意味着在同一组/流上发出的 P2P 操作将不再保证并发运行。并发 P2P 操作将需要使用批处理 API 或单独的组/通信子。
* torchcomms 依赖：我们计划将 torchcomms 作为 PyTorch Distributed 的必要包，并弃用现有的 c10d::Backends，转而采用单一的、更现代的通信定义。

torchcomms 集成由 PyTorch Distributed 团队主导，2.12 中的基础工作包括 Yifan Mao 的后端包装器重构（PR #177157）和 Tushar Jain 的 FlightRecorder 集成（PR #175270）。

**Torchscript 现已弃用**

Torchscript 在 2.10 中已弃用，应使用 torch.export 替代 jit trace 和 script API，并使用 Executorch 替代嵌入式运行时。更多详情请参见 PTC 上的这个演讲。

**弃用 CUDA 12.8 Wheel**

从 PyTorch 2.12 开始，CUDA 12.8 二进制 wheel 已被弃用，将不再作为标准发布矩阵的一部分发布。默认 wheel 仍为 CUDA 13.0（通过 PyPI 的 `pip install torch` 安装），CUDA 13.2 已作为实验性构建添加。

在旧架构（例如 Pascal、Volta）上运行的用户应切换到 CUDA 12.6 wheel，该版本在本次发布中仍受支持。在新型 GPU（例如 Blackwell）上运行的用户应使用 CUDA 13.0+ wheel；请注意这需要将 NVIDIA 驱动程序升级到 580.65.06（Linux）或 580.88（Windows）。

## 链接汇总

- 版本说明: https://github.com/pytorch/pytorch/releases/tag/v2.12.0
- PyTorch 2 系列入门页面: https://pytorch.org/get-started/pytorch-2-x/
- 直播问答注册: https://pytorch.org/event/pytorch-2-12-release-live-qa/
- PyTorch 2.10 发布博客: https://pytorch.org/blog/pytorch-2-10-release-blog/
- PyTorch 2.11 发布博客: https://pytorch.org/blog/pytorch-2-11-release-blog/
- linalg.eigh 文档使用示例: https://docs.pytorch.org/docs/2.11/generated/torch.linalg.eigh.html
- torch.accelerator.Graph 文档字符串示例: https://github.com/pytorch/pytorch/blob/1d803512199040e98738e95d0dc074acbde9fb5c/torch/accelerator/graphs.py#L11-L48
- torch.export 文档: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- Executorch 文档: https://docs.pytorch.org/executorch/stable/index.html
- PTC 演讲视频: https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903
