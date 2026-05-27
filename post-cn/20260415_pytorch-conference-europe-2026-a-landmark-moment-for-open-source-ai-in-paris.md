# PyTorch 欧洲大会 2026：巴黎开源 AI 的里程碑时刻

作者：PyTorch 基金会 | 2026年4月15日 | 2026年4月17日

首届 PyTorch 欧洲大会于 2026 年 4 月 7-8 日在巴黎举行，吸引了 600 余名研究人员、开发者、从业者和学者，两天日程密集，涵盖主题演讲、技术深度解析、闪电演讲、海报展示以及社区交流活动。从裸金属基础设施到智能体 AI，各环节覆盖了完整的 AI 技术栈，并传递出一个明确信号：开源 AI 生态系统正以前所未有的速度加速发展。

所有环节的录像将在未来一周内发布至我们的 YouTube 频道。以下是本次大会亮点回顾。

## 重要公告：

在 PyTorchCon EU 期间，PyTorch 基金会宣布新项目加入社区，与 PyTorch、vLLM、DeepSpeed 和 Ray 共同成长。Helion 和 Safetensors 现已作为基金会托管项目正式加入。ExecuTorch 也正式成为 PyTorch Core 的组成部分。

* **Helion** 由 Meta 贡献，是一种嵌入 Python 的领域特定语言（DSL），能够以极少的样板代码轻松编写快速、可扩展的 ML 内核。通过将内核编写提升为 PyTorch 的一等公民，Helion 增强了自定义内核的创建能力，并通过自动调优减少了手动编码工作量。了解更多。

* **Safetensors** 由 Hugging Face 贡献，是一种安全的模型文件格式，能够防止任意代码执行，并提升多 GPU 和多节点部署的性能。它已成为模型分发中使用最广泛的元数据格式之一。了解更多。

**ExecuTorch** 正式成为 PyTorch Core 的组成部分。ExecuTorch 最初由 Meta 开发，简化了在边缘和设备端环境（包括手机、AR/VR 头显和微控制器）上运行 PyTorch 模型的流程。其设计围绕四个核心原则：端到端开发体验、跨硬件可移植性、小型模块化效率以及默认开放性。了解更多。

这些新增项目进一步巩固了基金会作为开源 AI 技术栈中立枢纽的地位，覆盖了从训练到推理的完整生命周期。

## 第一天：4月7日（周二）

### 开幕主题演讲

上午的主舞台以一系列主题演讲拉开序幕，为整场活动定下基调。

**Mark Collier**，PyTorch 基金会执行董事、Linux 基金会 AI 与基础设施总经理，率先登台，带来题为《共同进化：开源智能技术栈的复利效应》的演讲。他的核心观点明确：PyTorch 生态系统的力量不在于任何单一项目，而在于各组件如何相互强化、协同构建。随着基金会持续扩展项目组合，这种复利效应只会愈发显著。

**Edward Yang**，Meta 研究工程师，随后带来《PyTorch 最新进展》主题演讲，向社区展示了核心框架的最新动态。

**Joe Spisak**，Reflection AI 产品副总裁兼开源负责人，接着发表《社区主导的开源强化学习》演讲，重点介绍了强化学习社区如何通过开放协作突破边界。

**Ramine Roane**，AMD AI 产品管理与生态系统发展企业副总裁，带来题为《从单节点到分布式训练与推理：PyTorch 生态如何改变 AI》的主题演讲，追溯了从单节点实验到全球规模部署的发展历程。

**Patrick von Platen**，Mistral AI 研究工程师，发表《流式处理一切：从请求输入到流式输入的转变》演讲，展望了 AI 系统处理连续数据流而非离散请求的未来愿景。

**Maryam Tahhan**，Red Hat 首席工程师，与 **Nicolo Lucchesi**，Red Hat 高级机器学习工程师，联合发表主题演讲《任意[智能体 | 模型 | 加速器 | 云]：开源 AI 释放世界潜能》，阐述了开源是实现真正灵活、可移植 AI 的必由之路。

上午主题演讲环节在 **Besmira Nushi** 的收尾中落幕——她是 NVIDIA AI 研究高级经理，带来《（智能体）评估的难以承受之轻》演讲，深刻探讨了评估智能体 AI 系统所面临的挑战。

### 与开发者面对面

"与开发者面对面"环节是本次大会的一大亮点，参会者可以与项目背后的开发者近距离交流。第一天，**与 PyTorch 模块维护者面对面**环节汇聚了 Driss Guessous、Mergen Nachin、Natalia Gimelshein、Jason Ansel、Edward Yang 和 Alban Desmaison。下午还专门安排了**与 Helion 开发者面对面**环节，Jason Ansel、Oguz Ulgen、Will Feng 和 Markus Hoehnerbach 悉数参与。

### 框架与编译器

这一分会场是本次大会最为密集的专题之一。亮点环节包括：

* **《Helion 1.0：面向性能可移植内核的高级 DSL》** — Meta 的 Oguz Ulgen，将新贡献的 Helion 项目作为 PyTorch 内核编写的一等工具予以介绍。
* **《PyTorch 中的参数化 CUDA 图启动：无痛使用 CUDA 图》** — NVIDIA 的 Daniel Galvez，应对最棘手的性能优化挑战之一。
* **《FlexAttention + FlashAttention-4：快速且灵活》** — Meta 的 Driss Guessous，展示下一代注意力机制。
* **《Combo Kernels：Torch.compile 中的水平融合优化》** — Meta 的 Karthick Panner Selvam 和 Elias Ellison。
* **《使用 Torch.compile 进行模型变换》** — Lightning AI 的 Thomas Viehmann。
* **《Brevitas 量化库》** — AMD 的 Pablo Monteagudo Lago。

### 推理与生产部署

推理专题反映了社区日益关注高效将模型投入生产：

* **《逐力突围：从简单到复杂的 LLM 推理优化》** — 微软的 Christin Pohl，在主舞台进行全面演示。
* **《ExecuTorch 在微控制器上的应用：将 PyTorch 部署到最小边缘设备》** — Meta 的 RJ Ascani 和 Matthias Cremon，拓展 PyTorch 的运行边界。
* **《一次编写，随处运行：使用 PyTorch Transformers》** — Hugging Face 的 Pedro Cuenca。
* **《为何 WideEP 推理需要数据并行感知调度》** — IBM 的 Maroon Ayoub 与 Red Hat 的 Tyler Michael Smith。
* **《令牌切片：通过分块解码实现抢占式调度》** — IBM 的 Maroon Ayoub 与 Google 的 Kellen Swain。
* **《跨区域模型服务：PyTorch 推理、可观测性与 LLMOps》** — 亚马逊云科技的 Suraj Muraleedharan。
* **《使用 ExecuTorch 和 Arm SME2 加速设备端 ML 推理》** — Arm 的 Jason Zhu。

### 生成式 AI 与多模态

* **《灯光，摄像，推理！基于 VLLM-Omni 的视频生成即服务》** — Red Hat 的 Ricardo Noriega 和 Doug Smith。
* **《开放且可扩展的 LLM 评估的科学与实践》** — NVIDIA 的 Grzegorz Chlebus。
* **《torch.compile 与 Diffusers：峰值性能实战指南》** — Hugging Face 的 Sayak Paul。

### 训练系统

* **《为多模态模型推理路由弹性训练嵌入模型》** — Red Hat 的 Huamin Chen 与 AMD 的 Haichen Zhang。
* **《TorchJD：PyTorch 中的雅可比梯度下降》** — EPFL 的 Pierre Quinton 与 Simplex Lab 的 Valerian Rey。
* **《将谷歌的 Colossus 引入 PyTorch：通过 fsspec 实现快速存储以保持 GPU 繁忙》** — Google 的 Ankita Luthra 和 Trinadh Kotturu。
* **《Jigsaw：高分辨率输入训练中的领域与张量并行》** — 卡尔斯鲁厄理工学院的 Deifilia Kieckhefen。

### 应用与案例研究

* **《野外深度学习：面向真实世界野生动物保护生物声学的嵌入式 PyTorch》** — OWL Integrations 的 Taraqur Rahman 和 Owen O'Donnell。
* **《DeepInverse 如何用 PyTorch 解决科学与医疗健康中的成像问题》** — DeepInverse 的 Andrew Wang 与图卢兹大学的 Minh Hai Nguyen。
* **《使用 ExecuTorch 在 MCU 级设备上灵活部署 PyTorch 模型》** — NXP 的 Robert Kalmar 和 Martin Pavella。

### 负责任 AI、安全与隐私

* **《面向欧盟 AI 法案的工程实践：PyTorch 应原生暴露哪些能力？》** — AffectLog 的 Roy Saurabh 主持的鸟类同好会（Birds of a Feather）环节，探讨监管与框架设计的交汇点。
* **《从预训练到个性化：AI PC 上以隐私优先的微调》** — AMD 的 Daniel Holanda Noronha 和 Iswarya Alex。
* **《PyTorch 系统中的伦理、隐私与可持续性考量》** — Pau&Company 的 Paula Mesa Macias。
* **《为何经典 IAM 在智能体场景下失效：重思智能体系统的身份与访问管理》** — Red Hat 的 Parul Singh。

### 社区活动

第一天还设有 **PyTorch 女性及非二元性别成员午餐** 活动，以及晚间充满活力的 **Flare 派对**，美食、美酒、美好的陪伴一应俱全。两天活动均设有海报展示，覆盖各专题方向，让研究人员和工程师有机会分享进行中的工作，并进行一对一技术交流。海报专题涵盖应用与案例研究、框架与编译器、生成式 AI 与多模态、推理与生产部署，以及负责任 AI 与合规。

## 第二天：4月8日（周三）

### 上午主题演讲

第二天在主舞台以又一场强劲的主题演讲阵容拉开帷幕。

**Matt White**，PyTorch 首席技术官、Linux 基金会全球 AI 首席技术官，为当天的议程奠定基调。

**Tyler Michael Smith**，Red Hat 推理工程首席架构师，与 **Artur Niederfahrenhorst**，Anyscale 技术成员，联合发表 **vLLM 与 Ray 最新进展**主题演讲，这两个项目已成为生产级 AI 基础设施的基石。

**Lysandre Debut**，Hugging Face 首席开源官，发表题为《Hub 即基础设施：从开放 PyTorch 模型，到安全高性能的分发枢纽》的演讲，强调 Hugging Face Hub 已从模型仓库演变为关键 AI 基础设施。

**Jonathan Bryce**，云原生计算基金会（CNCF）执行董事，发表《AI 原生时代的开源基础设施》主题演讲，将云原生与 AI 原生开发紧密联系起来。

**Leonard Hussenot**，Google DeepMind 研究科学家，以《Gemma 4：为边缘压缩智能》压轴主题演讲，介绍了 Google DeepMind 如何使强大的模型小到足以在边缘设备上运行。

### 框架与编译器（第二天）

* **《Monarch：通往超级计算机的 API》** — Meta 的 Marius Eriksen。
* **《2026 年如何编写 C++ 扩展》** — Meta 的 Jane Xu 和 Mikayla Gawarecki。
* **《实现最优 GEMM 性能：PyTorch Inductor 的 CuTeDSL 后端》** — Meta 的 Nikhil Patel。
* **《使用 Torch.compile 的 C++ 包装模式加速 PyTorch 模型》** — Meta 的 Bin Bao。
* **《将 PyTorch Monarch 引入 AMD GPU：基于 ROCm 的单控制器分布式训练》** — AMD 的 Liz Li 和 Zachary Streeter。
* **《PyTorch 在 RISC-V 上的应用：从交叉编译到原生 CI》** — Meta 的 Ludovic Henry。
* **《PyTorch 对称内存 + NCCL 设备 API：多 GPU 内核的新路径》** — NVIDIA 的 Ke Wen 和 Sylvain Jeaugey。
* **《无缝集成：在 Torch.compile 栈中无图断点地使用自定义内核》** — NVIDIA 的 Kshiteej Kalambarkar、Masaki Kozuki 和 Pawel Gadzinski。
* **《在 Torch.compile 中通过子图融合与自定义算子自动调优实现超越最优的内核性能》** — Meta 的 Elias Ellison 和 Paul Zhang。
* **《揭秘 ASIC 上的 PyTorch：何时（以及为何）将开发迁移到 AI 加速器》** — Kollab Philippines 的 Alpha Romer Coma。

### 推理与生产部署（第二天）

* **《以 KV 缓存为中心的推理：使用 Llm-d 和 VLLM 构建状态感知服务平台》** — IBM Research 的 Martin Hickey 和 Maroon Ayoub。
* **《优化 NVIDIA Blackwell 上的大型 MoE 推理：NVFP4、ADP 和 DualPipe 策略》** — NVIDIA 的 Julien Demouth。
* **《可移植的高性能 LLM 服务：面向 VLLM 的 Triton 后端》** — IBM 的 Burkhard Ringlein 和 Jan van Lunteren。
* **《使用 Transformers.js 将 PyTorch 模型部署到浏览器及更多环境》** — Hugging Face 的 Joshua Lochner。
* **《深入 VLLM 的 KV 卸载连接器：用于更高推理吞吐量的异步内存传输》** — Red Hat 的 Nicolo Lucchesi。
* **《在 PyTorch 中优化 CPU LLM 推理：来自 VLLM 的经验》** — Arm 的 Crefeda Rodrigues 和 Fadi Arafeh。
* **《从 Hugging Face 到手持设备：使用 LiteRT 生成式 API 扩展 LLM 部署》** — Google 的 Cormac Brick 和 Weiyi Wang。
* **《全栈 PyTorch 机器人 VLA：通过 ExecuTorch/OpenVINO 从数据到边缘》** — Intel 的 Samet Akcay 和 Dmitriy Pastushenkov。
* **《超越理论：扩展分解式 PyTorch 模型时真正会出现什么问题》** — NVIDIA 的 Ekin Karabulut 和 Ron Kahn。
* **《通过预分发 GPU 缓存大幅降低 LLM 冷启动时间》** — Red Hat 的 Billy McFall 和 Maryam Tahhan。

### 训练系统（第二天）

* **《从 Hopper 到 Blackwell 的 FP8 训练》** — Meta 的 Luca Wehrstedt。
* **《PyTorch 中的无反向传播优化》** — 波兰科学院的 Andrii Krutsylo。
* **《调试不可调试的问题：引入 Torch.distributed.debug》** — Meta 的 Tristan Rice。
* **《将推荐系统扩展到 2000 个 GPU 及以上》** — Meta 的 Zain Huda。
* **《从响应到轨迹：多轮和多环境强化学习》** — Hugging Face 的 Kashif Rasul 和 Sergio Paniego Blanco。
* **《Trinity Large：在 2000+ 台 B300 上运行 Torchtitan》** — Prime Intellect 的 Matej Sirovatka。
* **《从头实现 DualPipe：在 PyTorch 中实现 DeepSeek 的 5D 并行》** — ING Bank 的 Dev Jadhav。
* **《容错训练：我们如何为分布式 AI 工作负载构建可靠集群》** — Nebius 的 Cyril Konkratenko 和 Maurits de Groot。
* **《日志远远不够：如何在实践中使 PyTorch 训练回归可见》** — Wayve 的 Sahana Venkatesh。

### 智能体与互操作性

* **《超越 JSON-RPC：在 PyTorch 生态中使用 gRPC 扩展模型上下文协议》** — Google 的 Ashesh Vidyut 和 Madhav Bissa。
* **《用于编译器构建的编码智能体：超越 AI 助手范式》** — yasp.ai 的 Reza Rahimi 和 Stefan Krassin。
* **《通过 Hugging Face Kernels Hub 上的代码测试套件弥合硬件差距》** — Hugging Face 的 Ben Burtenshaw。

### 安全与隐私

* **《将 PyTorch GPU 节点从 Azure 实时迁移到欧洲云》** — Acf Cyber Solutions 的 Mike Krom。
* **《使用 PyTorch 保护智能体 AI：威胁建模与 LLM 红队测试实践》** — VamiSec GmbH 的 Valeri Milke。

### 负责任 AI 与合规

* **《从梯度到治理：让 PyTorch 具备血统感知能力》** — Red Hat 的 Kateryna Romashko 和 Clodagh Walsh。
* **《为用户和监管机构建立信任：通往合规即代码的高性价比 PyTorch 路径》** — Zoho Corporation 的 Raja Gopal Hari Vijay。
* **《弥合差距：使用 PyTorch 构建符合合规要求的"玻璃箱"医疗 AI》** — Neurosonic 的 Muhammad Saqib Hussain 和 Mohaddisa Maryam。

### 应用与案例研究（第二天）

* **《足球视频中的球跟踪与检测：VLM 与传统流程的对比》** — Future Processing 的 Maciej Szymkowski。

### 鸟类同好会与与维护者面对面

第二天设有两场引人入胜的鸟类同好会：**《分解式令牌化：迈向令牌输入/令牌输出的 LLM 推理》**，由 IBM Research 的 Maroon Ayoub、阿里云的 Hang Yin 与 Xi Ning Wang、IBM 的 Nili Guy 以及 Moreh 的 Hyunkyun Moon 共同参与；以及 **《NCCL 在野：将通信扩展到数千个 GPU》**，由 NVIDIA 的 Jeff Hammond、Gabrielle Talavera、Ke Wen 和 Asma Farjallah 参与。

参会者还有机会参加**与 vLLM 维护者面对面**（Tyler Michael Smith 和 Nicolo Lucchesi）以及**与 Ray 维护者面对面**（Artur Niederfahrenhorst）环节。

### 社区博览会与交流

社区博览会贯穿两天活动，参会者有机会探索演示、与项目维护者交流，并发现生态系统中的新工具。赞助商活动包括《在 CPU 上验证 AI：vLLM 三阶段评估框架》和《龙虾陷阱：容器中的 OpenClaw》两场专题活动。

## 核心主题与要点

**开源 AI 技术栈正在走向成熟。** 随着 Helion 和 Safetensors 作为托管项目加入，以及 ExecuTorch 成为 PyTorch Core 的一部分，PyTorch 生态系统如今覆盖了 AI 生命周期中更广泛的环节。从安全的模型分发到边缘部署，再到高性能内核编写，这些项目填补了关键空白。

**推理是新的前沿阵地。** 大量环节聚焦于生产部署，从 vLLM 和 LLM 服务优化，到微控制器上的 ExecuTorch，再到基于 Transformers.js 的浏览器内推理。社区正明显地从训练阶段转向让模型在现实世界中真正发挥作用。

**智能体 AI 需要新的基础设施。** 多个环节深入探讨了智能体系统的独特挑战，包括重新思考身份与访问管理（IAM）、通过评估建立信任，以及保护智能体流水线的安全。这一领域值得持续关注。

**欧洲正在崛起。** 从欧盟 AI 法案合规专题到数据主权和云迁移讨论，本次大会折射出欧洲在塑造负责任 AI 发展方面日益重要的角色。首届欧洲大会对于全球社区而言是一个重要里程碑。

**硬件多样性正在扩展。** 各环节涵盖了 AMD ROCm、Arm SME2、Google TPU、NVIDIA Blackwell、RISC-V、微控制器、Qualcomm QNN、Intel OpenVINO 等众多平台。PyTorch 的硬件可移植性故事从未如此精彩。

## 感谢，巴黎

PyTorch 欧洲大会 2026 是社区的里程碑盛事。感谢每一位赞助商、演讲者、海报展示者，以及所有来到巴黎与我们共聚的参会者。那份活力、那种技术深度、以及开放协作的精神，令人由衷叹服。向 PyTorchCon Europe 项目主席 Alban Desmaison、Lysandre Debut 和 Luca Antiga，以及整个项目委员会致以崇高的敬意，感谢他们精心组织了这场成功的盛会！

所有环节录像将在 PyTorch YouTube 频道上发布。在 Flickr 上浏览本次活动的完整相册。

期待在我们接下来的 PyTorch 大会上再次相聚——包括今年晚些时候在中国上海举办的 PyTorch 中国大会（9月8-9日）和在美国加州圣何塞举办的 PyTorch 北美大会（10月20-21日）。

## 链接汇总


