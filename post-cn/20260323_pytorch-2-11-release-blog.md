# PyTorch 2.11 发布博客

**作者：PyTorch 基金会 | 2026 年 3 月 23 日 | 暂无评论**

---

## 概述

我们很高兴地宣布 PyTorch® 2.11 正式发布（发布说明）！

PyTorch 2.11 版本包含以下主要变更：

- **分布式训练的可微集合通信操作**
- **FlexAttention 现已在 Hopper 和 Blackwell GPU 上支持 FlashAttention-4 后端**
- **MPS（Apple Silicon）全面扩展算子覆盖**
- **RNN/LSTM GPU 导出支持**
- **XPU Graph**

自 PyTorch 2.10 以来，本次发布包含来自 432 位贡献者的 2723 个提交。我们衷心感谢社区的每一位成员的贡献。一如既往，我们鼓励大家尝试这些新特性，并反馈遇到的问题，帮助我们持续改进 2.11。有关如何开始使用 PyTorch 2 系列的更多信息，请访问我们的入门页面。

3 月 31 日（周二）上午 10 点，Andrey Talman 和 Nikita Shulga 将主持一场在线直播，介绍 2.11 的新特性，包括分布式训练的可微集合通信、FlexAttention 的 FlashAttention-4 后端、MPS 扩展等内容，并进行现场问答。点击注册参与。

---

## API-UNSTABLE 特性

### 分布式训练的可微集合通信操作

为函数式集合通信操作添加了可微性支持，使训练工作流能够通过集合操作进行反向传播。这对于分布式深度学习研究和高级训练技术是一项重大进展，无需编写自定义 autograd 函数即可实现相关功能。

### FlexAttention 在 Hopper 和 Blackwell GPU 上支持 FlashAttention-4 后端

该后端支持自动生成 CuTeDSL 评分/掩码修改函数，并从 PyTorch JIT 实例化 FlashAttention-4 内核，在计算密集型工作负载上比现有 Triton 实现提速 1.2× 至 3.2×。该特性仍在积极开发中，可能随着稳定化进程而有所变化；有关设置详情和当前限制，请参阅 FlexAttention + FlashAttention-4 博客文章。

### MPS（Apple Silicon）开发改进 / 算子扩展

本次发布支持从 MPS 后端报告错误，并持续扩展算子覆盖范围，包括新的分布函数（log_normal、cauchy、geometric）、算子迁移（erfcx、grid_sampler_2d 支持所有操作模式），以及为整数和复数类型扩展 baddbmm/addbmm。

异步错误报告能够检测 GPU 索引操作期间发生的越界访问，例如：

```python
import torch
x=torch.rand(10, 1, 10, device='mps')
y=x[:, [1]]
torch.mps.synchronize()  # 将抛出索引越界错误
```

### RNN/LSTM GPU 导出支持

RNN 模块（LSTM、GRU 等）现在可以在 GPU 上导出，并且支持对带动态形状的 LSTM 进行追踪。这极大地扩展了可使用 torch.export 部署用于生产推理的模型类型。GRU API 保持不变；新 API 适用于 LSTM。

### ROCm 设备端断言与 TopK 优化

在 ROCm 上添加了设备端断言支持，以改善调试体验，同时对 TopK 算子进行了重大优化，并通过将数据缓存到共享内存改进了基数选择性能。提升了 AMD GPU 上的开发体验和运行性能。

### XPUGraph 支持以优化 Intel GPU 上的执行

XPUGraph 允许用户将一系列 XPU 操作捕获为 Intel GPU 上的运行时执行图，并多次重放。这减少了 CPU 开销，例如内核启动和 Python 运行时开销，从而提升 Intel GPU 上的工作负载性能。使用详情请参阅 API 文档。

### 通过 OpenBLAS 在 CPU 上支持 FP16 半精度 GEMM

通过 OpenBLAS 在 CPU 上添加了 FP16 半精度 GEMM 支持，为基于 CPU 的部署提供更快的 FP16 推理速度。这对边缘设备和纯 CPU 推理场景非常有价值。

---

## 非特性更新

### CUDA 版本

从本次发布开始，CUDA 13 现已成为 x86_64 和 ARM 平台默认安装的版本。需要替代构建版本的用户仍可从相应的 https://download.pytorch.org/whl 子目录获取纯 CPU 版本以及 CUDA 12.8 构建版本。

### Torchscript 已弃用

Torchscript 已在 2.10 版本中被弃用，应使用 torch.export 替代 jit trace 和 script API，并使用 Executorch 替代嵌入式运行时。更多详情，请参阅 PTC 的此演讲。

### 2026 年发布节奏

2026 年，发布节奏已从每季度一次提高到每两个月一次。请参阅已发布的发布计划。

## 链接汇总

- 发布说明: [https://github.com/pytorch/pytorch/releases/tag/v2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)
- 入门页面: [https://pytorch.org/get-started/pytorch-2.0/](https://pytorch.org/get-started/pytorch-2.0/)
- 直播注册: [https://streamyard.com/watch/zHmCTfH6Y3zQ](https://streamyard.com/watch/zHmCTfH6Y3zQ)
- FlexAttention + FlashAttention-4 博客文章: [https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- XPUGraph API 文档: [https://docs.pytorch.org/docs/2.11/xpu.html#graphs](https://docs.pytorch.org/docs/2.11/xpu.html#graphs)
- torch.export 文档: [https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)
- Executorch 文档: [https://docs.pytorch.org/executorch/stable/index.html](https://docs.pytorch.org/executorch/stable/index.html)
- PTC 演讲（Torchscript 弃用说明）: [https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903)
- 已发布的发布计划: [https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence)
