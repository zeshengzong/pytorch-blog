# vLLM 与 PyTorch 携手改善 aarch64 平台的开发者体验

作者：Kaichao You（Inferact）| 2026 年 5 月 18 日

## 简介

PyTorch 2.11 使得在 aarch64 Linux 上可以直接从 PyPI 安装支持 CUDA 的 PyTorch wheel 包，消除了以往在 NVIDIA GH200、GB200 和 GB300 等系统上部署时需要自定义包索引和各种变通方案的麻烦。本文介绍了这一打包改进如何提升 vLLM 用户的安装体验，并重点介绍了 vLLM 与 PyTorch 通过 PyTorch Foundation 的协作是如何将这一修复推向生产环境的。

## 我在一次黑客马拉松中首次遭遇的问题

这个故事要追溯到 2024 年 10 月。当时我在 CUDA MODE（现为 GPU MODE）IRL 黑客马拉松上，试图在 GH200 机器上运行 vLLM。这本应是一个五分钟的任务，结果却让我花了一整天大部分时间盯着一个 `pip install`——表面上看起来一切正常，wheel 已解析、依赖已满足、安装完成且没有报错——但运行时 `torch.cuda.is_available()` 却顽固地返回 `False`。

深入排查后，原因近乎荒谬地平凡：在 `aarch64` Linux 上，`pip install torch` 拉取的是 PyPI 上的**仅 CPU** wheel。默认 PyPI 索引根本没有发布 `aarch64` 的 GPU wheel。要获得支持 CUDA 的构建版本，必须显式地将 pip 指向 PyTorch 下载索引：

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

这本身只是个小麻烦。真正的麻烦在于它与传递依赖之间的交互。PyPI 不允许软件包为其依赖项指定自定义索引。因此，如果 vLLM 依赖树中的某个软件包声明了 `torch==<某个版本>` 的依赖，且版本不匹配，pip 就会愉快地回到默认 PyPI 索引，找到 CPU wheel，**静默卸载**我精心安装的 GPU 构建版本，并用 CPU 版本替换它。你会以为一切正常，直到你的模型无法找到 GPU。

对于任何试图在 GH200（以及后来的 GB200 / GB300）上部署 vLLM 的人来说，这将一行安装命令变成了一个充满 `--index-url` 标志、版本固定和安装后健全性检查的迷宫。

## vLLM 期间采用的临时解决方案

在等待上游正式修复的同时，vLLM 不得不推出自己的变通方案，以免 aarch64 用户陷入困境。

第一个方案是 `use_existing_torch.py`，于 2024 年 9 月通过 [vllm-project/vllm#8713](https://github.com/vllm-project/vllm/pull/8713) 添加——PR 标题明确写道 _"启用现有 pytorch（针对 GH200、aarch64、nightly）"_。其流程正如名称所示：你自己安装正确的 `torch` 构建版本（从 PyTorch 索引、nightly 或自定义构建），然后运行 `python use_existing_torch.py`，该脚本会从 vLLM 的 `requirements/*.txt`、`requirements/*.in` 和 `pyproject.toml` 中剥离所有 `torch`/`torchvision`/`torchaudio` 依赖声明。去掉这些约束后，后续的 vLLM 安装就无法再触发 pip "好心地"回到默认 PyPI 索引，将你支持 CUDA 的 `torch` 静默换成 CPU wheel 了。这很丑陋——我们实际上是在安装时重写自己的依赖文件——但它让 GH200 用户顺利工作了一年多。

后来，随着 `uv` 的成熟，我们有了更优雅的方案。在 [vllm-project/vllm#24303](https://github.com/vllm-project/vllm/pull/24303) 中，我们向 `pyproject.toml` 添加了以下内容：

```
[tool.uv]
no-build-isolation-package = ["torch"]
```

这告诉 `uv` 不要在隔离环境中构建 `torch`——实际效果是 uv 会复用当前环境中已有的 torch，而不是尝试解析并重新安装自己的版本。结合从正确索引先安装 `torch`，这比文件重写方案更符合人体工程学：只需在 `pyproject.toml` 中加一行配置，`uv pip install vllm`（或 `uv sync`）就会在 aarch64 上尊重预先安装的支持 CUDA 的 `torch`。

vLLM 的变通方案是社区在打包标准缺口面前的即兴创作。[Wheel Variants](https://developer.nvidia.com/blog/streamline-cuda-accelerated-python-install-and-packaging-workflows-with-wheel-variants/) 则是 NVIDIA 和 Astral 将这一修复正式化，使即兴创作不再必要。

## 从黑客马拉松的头疼问题到 TAC 议程

时间来到 2025 年。vLLM 加入了 PyTorch Foundation，我成为其在技术顾问委员会（TAC）的代表之一。aarch64 wheel 的问题不断出现——无论是在我自己的工作中，还是来自其他在 Grace Hopper 和 Grace Blackwell 系统上使用 vLLM 的用户。2025 年 8 月，我提交了 [pytorch/pytorch#160162](https://github.com/pytorch/pytorch/issues/160162) 来正式追踪这个问题，今年早些时候，在 2026 年 1 月的 TAC 会议上，我代表 vLLM 用户直接提出了这一问题。

诉求很简单：将 aarch64 GPU wheel 发布到默认 PyPI 索引，让 `pip install torch` 在 GB200 级别的机器上像在 x86 上一样"开箱即用"。这些 wheel 会动态链接到 NCCL 和 cuBLAS 等库——与 x86 上已采用的方式相同——从而不会导致体积膨胀。如此庞大的二进制文件对用户来说下载困难，对 PyPI 项目维护者来说托管费用高昂，因此 PyPI 维护者对此有严格限制并强烈不鼓励。

NVIDIA 工程团队要求将 CUDA SBSA wheel 发布到 PyPI，并推动了链接这些 wheel 的小 wheel 方案。

这正是 PyTorch Foundation 最适合协调的跨项目、基础设施层面的问题。vLLM 和 PyTorch 都是 Foundation 项目，拥有一个共同的论坛来暴露生态系统摩擦——而不是每个项目各自独立应对——最终确实发挥了重要作用。

## 修复已落地

2026 年 4 月，在另一次 TAC 会议上，我得知该问题已得到解决：从 **PyTorch 2.11.0** 开始，在 aarch64 Linux 上执行默认的 `pip install torch` 现在会拉取支持 CUDA 的 wheel，而不再是仅 CPU 的版本。NVIDIA 的 Piotr Bialecki 确认该变更已在 2.11.0 版本中生效。

我在 GB200 上进行了验证，结果正是你所期望的——平淡无奇，却是最好的方式：

```
$ uv run --no-project --python 3.12 --with 'torch==2.11.0' -- python -c "import torch; print(torch.cuda.is_available())"
True

$ uv run --no-project --python 3.12 --with 'torch==2.10.0' -- python -c "import torch; print(torch.cuda.is_available())"
False
```

升级一个版本，整个变通方案栈就消失了。不再需要在 requirements 文件中传播自定义索引 URL。不再有静默的 CPU wheel 替换破坏正常安装。新用户不再需要调试"为什么我的 GB200 找不到 GPU"。

对于 vLLM 而言，这意味着在 GB200 / GB300 上的安装现在真正顺畅了。使用 Grace Blackwell 系统的新用户可以按照标准安装说明操作，第一次就能成功——当你试图在全新平台上快速启动推理服务时，这一点非常重要。

vLLM 中的变通方案——`use_existing_torch.py` 和 `[tool.uv] no-build-isolation-package = ["torch"]` 设置——将继续保留。它们对于运行自定义 PyTorch 构建（nightly、补丁版 fork，或与 vLLM 源码构建配合使用的从源码构建版本）并需要 vLLM 安装完全不触碰该 `torch` 的高级用户仍然有用。改变的是_默认_路径：aarch64 上的普通用户不再需要了解这些的存在。他们可以直接 `pip install` 然后开始工作，而这些变通方案悄然成为高级用户的工具，而不再是对所有人的额外负担。

## 为什么值得写这篇文章

从宏观来看，这是一个微小的变化——一次打包调整，而非新功能。但我认为值得花一点时间去欣赏它，原因有两点。

第一，这是 vLLM 和 PyTorch 在 PyTorch Foundation 旗帜下富有成效协作的具体案例。TAC 不只是一个治理仪式；它是一个让下游项目的痛点能够传递到真正能解决问题的人面前的平台，协调跨项目工作是默认发生的，而非偶然。这个问题走完了完整的路径——从一名开发者在黑客马拉松上对着终端抱怨，到 TAC 讨论，到被追踪的 GitHub issue，到正式发布——而 Foundation 正是使这条路径变得短暂的关键。

第二，开发者体验会复利增长。每一个不必在 `--index-url` 标志上浪费的小时，都是真正用于在 vLLM 和 PyTorch 之上构建东西的小时。aarch64 GPU 系统只会越来越普及，现在在枯燥的基础设施层面解决这个问题，远比让每位用户自己去发现并绕过它要好得多。

uv 侧的变通方案（构建隔离透传）是更广泛的 [WheelNext 努力](https://wheelnext.dev/proposals/pepxxx_build_isolation_passthrough/)的一部分——这是重新思考 Python 打包如何在 AI 时代处理加速器绑定依赖的非常受欢迎的推动。

特别感谢让这一切成为可能的人们：PyTorch 核心团队的 Alban Desmaison、Nikita Shulga 和 Andrey Talman，他们接受了最初的请求并推动其落地；NVIDIA PyTorch 团队，他们推动了 aarch64 构建工作，并由 Piotr Bialecki 确认修复已在 2.11.0 中落地，Piotr Bialecki 在 NVIDIA 和上游之间的这些问题上一直是稳定的联络人；PyTorch 发布工程团队，负责构建和发布 wheel；以及幕后众多工程师——来自 PyTorch、NVIDIA 和 Arm——他们在工具链、CI 基础设施和打包方面的工作使这一切成为可能。也感谢 TAC 的每一位成员，为这类对话保持大门敞开。

继续前行。

---
原文链接: https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/
