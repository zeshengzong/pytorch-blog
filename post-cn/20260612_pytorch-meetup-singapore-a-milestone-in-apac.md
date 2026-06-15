# PyTorch 新加坡 Meetup：亚太地区的里程碑

作者：Sudhir Dharanendraiah | 2026 年 6 月 12 日 | 无评论

## 摘要

八十名工程师、研究人员和社区建设者齐聚首届 [PyTorch 新加坡 Meetup](https://luma.com/8scaib4z)。活动在 Red Hat 亚太办公室举办，由 [Sudhir Dharanendraiah](https://www.linkedin.com/in/sudhir-dharanendraiah-80a0867/)、[Ayush Satyam](https://www.linkedin.com/in/ayushsatyam146/)、[Sumantro Mukherjee](https://www.linkedin.com/in/sumantrom/) 和 [Daniel Kang（AER Labs）](https://www.linkedin.com/in/aer-dk/) 共同组织，聚集了来自亚太地区的 AI 从业者，共度一个充满技术交流、开源分享以及就如何将 AI 从研究笔记本推向生产环境进行坦诚对话的夜晚。

在这篇博客中，我们将深入探讨讨论的主题：推理、分布式训练、持续集成、社区治理，以及技术主权这一更宏观的议题。

## 主权智能：构建亚太 AI 未来

Sudhir Dharanendraiah（Red Hat）以一个引人深思的命题开场：亚太地区必须从 AI 技术的消费者转变为建设者。他的演讲《主权智能：构建亚太 AI 未来》主张，真正的技术独立不仅需要政策抱负，还需要在使主权 AI 切实可行的工程"管道"上投入资源。Sudhir 带领听众了解了 PyTorch 生态系统中的几个基础构建模块，包括用于硬件无关设备注册的 OpenReg、用于可移植性能的 torch.compile，以及用于在本地硅芯片上进行可扩展训练的完全分片数据并行（FSDP）。演示文稿可在[此处](https://drive.google.com/file/d/1yTga_plbL-Ay_EwTsLy5jf4DmCrtnJ1t/view)找到。

他强调了 vLLM 等高吞吐量服务工具如何能够在主权硬件上有效部署，论证了开源社区在弥合全球 AI 研究与区域生产就绪之间差距方面具有独特优势。这场演讲在来自积极投资国内 AI 能力国家的听众中引发了强烈共鸣，并为整个夜晚定下了协作基调：主权并非孤立，而是在共享的开放基础上自由构建的权利。

## 介绍 vLLM 的全新 Rust 前端

[Ziqi Zhao](https://www.linkedin.com/in/bugenzhao/)，Inferact 技术团队成员——该公司由 vLLM 最初的创建者资助，展示了推理引擎新的基于 Rust 的前端工作。Zhao 解释说，随着每一代 GPU 速度的不断提升，Python 全局解释器锁、动态类型和垃圾回收带来的 CPU 端开销已成为日益明显的瓶颈。Rust 前端通过在单个进程内提供更高的并发性和可预测的内存管理来解决这一问题，同时不替换 Python 引擎本身。

该架构遵循清晰的分层设计——从通过 ZMQ 和 MessagePack 通信的底层引擎核心客户端，经过词元化和聊天渲染层，到兼容 OpenAI 的 HTTP 服务器。流式传输是主要设计路径而非事后补充，每一层都作为流变换发挥作用。在配备 Qwen3-0.6B 的四 GPU GB200 设置上的基准测试结果显示，在解码敏感和预处理密集型工作负载上均有显著改善。Zhao 指出，一个集成拉取请求预计很快将并入 vLLM 主仓库，最初通过 git 子模块实现，并通过环境变量控制以便于选择启用。演示文稿可在[此处](https://drive.google.com/file/d/1f4pGriEaZOTlTapPz3GvhgbC0dONaFRU/view)找到。

## vLLM 介绍与项目更新

[Pin Siang Tan](https://www.linkedin.com/in/tanpinsiang/)，Embedded LLM 联合创始人及 vLLM 贡献者，全面介绍了 vLLM 的核心架构及其 2026 年的发展轨迹。他首先将 vLLM 与 Ollama 等轻量级工具区分开来：后者面向单用户笔记本电脑推理，而 vLLM 专为并发、生产级服务而构建，具备持续批处理、张量/流水线/专家/数据并行以及预填充-解码分离功能。该引擎现已支持超过一百种模型架构，可在从 NVIDIA 和 AMD GPU 到 TPU 和昇腾 NPU 的各类硬件上运行，GitHub 星标超过七万七千个，拥有来自五十多个组织的两千多名贡献者。

Tan 随后介绍了 vLLM 的"秘密武器"——持续批处理、torch.compile 集成、融合流程、量化（FP8、INT4、INT8、MXFP4、NVFP4）、投机解码以及用于高效资源使用的新型休眠模式。展望 2026 年第二季度，他概述了将 Model Runner V2 固化为默认版本、实现自动调优、引入弹性专家并行（允许在不重启的情况下向实时部署添加或移除 GPU）以及扩展到强化学习推理和全模态服务的计划。他的最后一张幻灯片总结了项目的信心："如果你在 2026 年为 LLM 提供服务，你要么在使用 vLLM，要么在解释为什么没用。"幻灯片可在[此处](https://drive.google.com/file/d/1tmOQABWvHGo6UzA_-6K11GQSwRl6Skv9/view)找到。

## vLLM-Omni：全模态 LLM 统一平台

来自乐天亚洲的 [Wang Zhipeng](https://www.linkedin.com/in/%E5%BF%97%E9%B9%8F-%E6%B1%AA-537882216/) 介绍了 vLLM-Omni，这是一个从 vLLM 核心代码库演化而来的独立框架，用于在单个兼容 OpenAI 的端点后为跨模态（图像、视频、语音和文本）运行的模型提供服务。该框架原生支持自回归和基于扩散的模型、统一的多阶段流水线，以及根据每种模态独特的延迟和吞吐量特性分配计算资源的模态感知调度。

Wang 描述了 vLLM-Omni 如何已在支持从视觉-语言助手（VLA）到世界模型的各类工作负载，并分享了该项目通过与 verl-omni 集成向具身 AI 和多模态强化学习扩展的计划。这场演讲强调了当晚的一个更广泛主题：推理堆栈正在迅速演进，超越纯文本范畴，而工具链需要跟上这一步伐。

## torch.compile 的实战应用

[Ayush Satyam](https://www.linkedin.com/in/ayushsatyam146/)（Red Hat）的演讲堪称一次跨越十二个热门 PyTorch 生态项目的源码探索之旅，涵盖推理、分布式训练、强化学习和特定领域框架。他的核心论点是：torch.compile 不是一个简单地开关即可的标志，它是一个重塑代码库的架构决策，而受益最多的项目是那些愿意主动配合编译器的项目。

Ayush 将他的发现组织成四个"幕"。在推理方面，他展示了 Hugging Face Transformers 如何仅编译解码步骤（因为预填充以可变长度运行一次），而 Diffusers 更进一步，仅编译模型内的重复块，并支持在不重新编译的情况下热切换 LoRA 适配器。在分布式训练方面，他将 Lightning AI 重新排序包装器以使 compile 能够穿透分布式层的方法，与 DeepSpeed 大胆的 DeepCompile 子框架进行了对比——后者剥离所有 ZeRO-3 钩子，将自定义算子注入编译图，甚至替换了 autograd 元类。在强化学习方面，他展示了 TorchRL 如何仅编译数学更新函数，同时将环境交互保留在 Eager Python 中。最后，在特定领域工作中，他重点介绍了 PyG 在导入时进行 Jinja2 代码生成以避免图断裂、MONAI 针对医学影像元数据的剥离和重附加策略，以及 TorchVision 对数据依赖形状使用无支撑符号整数的做法。

演讲以一套编译模式分类法作为总结——图分割、编译器猴子补丁、渐进式编译、不透明包装、双执行模式和形状稳定化——与会者可将其应用于自己的项目。幻灯片可在[此处](https://drive.google.com/file/d/1SbNrWLO-ALMXl52pA6xl2gWixk1XfQQq/view)找到。

## PyTorch 社区与持续集成

Sumantro Mukherjee（Red Hat）以一场将社区治理与工程严谨性相结合的演讲为技术议程画上句号。他从贡献者旅程开始讲起——从提交 GitHub issue、经历分类、拉取请求、维护者审查，到 @pytorchbot 合并流程——强调中位数分类周转时间不到两个工作日，且自动化机器人驱动的回滚周期保持主干绿色。

演讲的主要部分聚焦于 PyTorch 的多云持续集成基础设施，该基础设施现在每天跨五个云提供商（AWS、Azure、GCP、IBM Cloud 以及 Linux 基金会的 OSDC）运行超过九万个 CI 任务，涵盖 x86_64、aarch64 和 ppc64le 架构。他还介绍了多云工作组，这是 PyTorch 基金会 TAC 的一项倡议，由 Meta、Red Hat、IBM、NVIDIA、AMD、Google、Intel、华为和 Linux 基金会共同参与，其章程是开发可持续的、社区管理的 CI/CD 基础设施。

除基础设施外，Mukherjee 还概述了 PyTorch 基金会的治理结构：技术顾问委员会（TAC）、五个工作组（CI 基础设施、多云、生态系统、加速器和安全）以及 RFC 流程。他以开放邀请作为结语：TAC 每月第二个星期二召开会议，多云工作组每周四 UTC 下午 5 点召开会议，任何[希望贡献](https://lists.pytorch.org/g/tac/messages)的人均可参与。幻灯片可在[此处](https://drive.google.com/file/d/1clFICwZb5zjS_xxzkDxmcx9-rlqn5CRy/view)找到。

## 展望未来

随着正式议程转变为餐饮交流（我们有披萨！），关于编译器内部原理与服务架构、认证路径与贡献者工作流程，以及该地区主权 AI 技术栈可能呈现的面貌等话题的讨论一直延续到深夜。对许多与会者而言，这次 meetup 是他们首次有机会与在同一时区使用相同开源工具的同行建立联系。

首届 PyTorch 新加坡 Meetup 证明，东南亚对深度、有技术含量的社区活动的需求是真实存在的。近百名与会者、来自四个组织的六位演讲者，以及一个让参与者字面意义上俯瞰整座城市的场地，这个夜晚既是一个起点，也是一种意图的宣示。组织者已表示将举办更多 meetup，而若以现场的热情为参照，下一次活动将需要一个更大的场地。即将举办的活动详情将添加至 [AI SGP](https://www.ai.engineer/singapore/2026#side-events)。

活动页面：[https://luma.com/8scaib4z](https://luma.com/8scaib4z)

---
原文链接: https://pytorch.org/blog/pytorch-meetup-singapore-a-milestone-in-apac/
