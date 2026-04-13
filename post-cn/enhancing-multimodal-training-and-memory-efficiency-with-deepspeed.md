# 使用 DeepSpeed 增强多模态训练与内存效率

**作者：** Masahiro Tanaka（Anyscale）和 Olatunji Ruwase（Snowflake）
**日期：** 2026 年 2 月 24 日

## 概述

本博客介绍了两项关键的 DeepSpeed 更新：（1）与 PyTorch 完全兼容的 backward API，支持多模态、多组件模型的高效训练（包括非标量 backward 调用）；（2）低精度模型训练，可显著降低峰值内存，尤其适用于大规模场景。

对于多模态工作负载（如将视觉编码器与 LLM 结合），训练循环可能变得复杂且涉及多个组件。第一项更新引入了与 PyTorch 完全兼容的 backward API，使编写此类循环变得简单直接，能够以简洁的代码实现复杂的并行方案，同时 DeepSpeed 透明地管理各种性能优化。举一个例子，该 API 的灵活性使得解耦混合并行成为可能，在多模态 AI 模型训练中实现了 30% 的加速，同时使基于 DeepSpeed 的模型开发体验更接近"原生 PyTorch"。

与此同时，对于 LLM 微调，一个新选项允许将所有模型状态（参数、梯度和优化器状态）保持在较低精度（如 BF16 或 FP16），从而大幅减少内存占用，使研究人员能够在更受限的硬件上训练更大的模型。低精度训练在广泛的应用场景中具有重要价值，包括监督微调（SFT）、强化学习（RL）和多模态训练。我们的实验表明，在保持数值稳定性的同时，峰值内存减少了 40%（基准测试脚本）。数值稳定性通过与 torch.autocast 的集成来实现，从而确保模型质量得以保持。

本博客的其余部分将详细说明这些更新如何直接促进前沿训练工作负载的开发。

## 1. 与 PyTorch 完全兼容的 backward API

DeepSpeed 现在在保留所有优化的同时，支持 PyTorch 的原生 `backward()` 语法。传统上，DeepSpeed 的训练循环依赖于引擎的 backward API：

```python
loss = model_engine(batch)
model_engine.backward(loss)
model_engine.step()
```

引擎的 `backward` API 对于传统的预训练和微调流程已足够。然而，近期复杂的训练流程需要更高的灵活性。主要存在两个限制：

1. 它只接受标量 loss。
2. 必须调用 `model_engine.backward(loss)`，而不能使用常见的 PyTorch `loss.backward()` 风格。

由于这些限制，用户无法简单地实现原生 PyTorch 允许的模式。以下是一些示例：

```python
# 1. 组合多个模型和 loss
output1 = model1(batch1)
output2 = model2(batch2)
loss = criterion(output1, output2)
loss.backward()

# 2. 将 loss 函数与主模型分开定义
output = model(batch)
loss = loss_fn(output)
loss.backward()

# 3. 通过非标量张量以自定义梯度调用 backward
output = model(batch)
output.backward(grad)
```

DeepSpeed 引擎之前可以通过内部 API 处理这些用例，但这需要大量的代码修改，且容易引入错误。随着与 PyTorch 完全兼容的 backward API 的加入，我们现在可以使用与原生 PyTorch 相同的代码，同时保留 DeepSpeed 的强大优化，包括 ZeRO 和卸载功能。

PyTorch 兼容 backward API 的一个典型用例是使用 Ray 为多模态模型实现解耦混合并行。在该训练流水线中，两个 Ray Actor 组分别处理视觉编码器和 LLM。在反向传播时，LLM 向视觉编码器传递梯度，视觉编码器使用该梯度调用 backward 函数。然而，由于梯度是非标量张量，这种用例以前并不被 DeepSpeed API 官方支持。解耦混合并行证明，backward API 的灵活性与 DeepSpeed 的优化以及 DeepSpeed-Ulysses（高效序列并行）的结合，实现了训练 30% 的加速。

以下是运行在不同 actor 上的两个模型的伪代码。由于它们运行在不同的进程中，我们通过 Ray actor 通信传递梯度。可以看到，视觉嵌入的梯度是一个非标量张量。尽管该代码与 PyTorch API 完全相同，但它会根据您的配置激活各种 DeepSpeed 优化。

```python
# 运行在 LLM actor 上
def text_backward_step(self):
# ...
  self.loss.backward()
  return self.vision_embeddings.grad.detach().clone()

# 运行在 Vision actor 上
def vision_backward_step(self, vision_embedding_grad):
  self.vision_output.backward(gradient=vision_embedding_grad)
```

请查看代码仓库以获取完整的训练流水线。

## 2. 内存高效的低精度模型状态

您现在可以将所有模型状态（参数、梯度和优化器状态）保持在 BF16 或 FP16 精度，从而显著降低内存消耗。

传统上，DeepSpeed 的混合精度保留 FP32 主参数、梯度和优化器状态，这在技术上更安全但内存消耗大。虽然 DeepSpeed 已通过配置支持 `torch.autocast`（请参阅 API 文档），但缺少绕过创建 FP32 状态的选项，限制了在受限硬件上训练大模型的能力。在实践中，许多训练工作负载无需 FP32 状态也能稳定收敛。

通过低精度模型状态选项，您可以轻松跳过创建 FP32 状态，并将低精度选项与 `torch.autocast` 支持结合使用（配置详情请参阅文档和示例）。这种组合在不牺牲收敛性的情况下大幅提升了内存效率。

```json
{
...
  "zero_optimization": {
    "stage": 3,
    ...
  },
  "bf16": {
    "enabled": true,
    "bf16_master_weights_and_grads": true,
    "bf16_optimizer_states": true
  },
  "torch_autocast": {
    "enabled": true,
    "dtype": "bfloat16"
  }
}
```

我们的示例脚本展示了显著的内存节省效果：

| 配置 | 已分配内存 | 峰值内存 | 平均步骤时间 |
|---|---|---|---|
| 基线（fp32 主权重） | 25.74 GB | 31.38 GB | 0.6016s |
| BF16 低精度（主权重 + 优化器状态） | 16.17 GB | 18.93 GB | 0.6427s |

该实验（7B 模型，ZeRO3，4 块 GPU）展示了**峰值内存降低 40%**。为了验证 BF16 低精度训练保持数值稳定性，我们在 Wikitext-103 数据集上训练了 1000 步：

| 配置 | 最终 Loss | 平均 Loss |
|---|---|---|
| 基线（fp32 主权重） | 3.09 | 2.78 |
| BF16 低精度 | 3.12 | 2.90 |

## 相关测试

我们在 CI 中持续测试这些新 API，您可以在测试中看到各种用例模式。

- PyTorch 兼容 backward API
- 低精度主参数/梯度/优化器状态
- 与 torch.autocast 结合使用

## 总结

本次 DeepSpeed 更新带来了关键进展：

- **支持复杂的多模态工作负载：** 全新的 PyTorch 兼容 backward API 支持以简洁的代码实现复杂的多组件训练循环，例如多模态模型所需的训练循环。例如，PyTorch 兼容 backward API 已为解耦混合并行实现了 30% 的加速。
- **扩展到更大模型：** 低精度模型状态与 `torch.autocast` 的结合可将峰值内存减少多达 40%，而不会牺牲收敛性，使您能够用相同的硬件训练更大的模型。

我们很高兴看到您如何在自己的训练设置中使用本博客中描述的新 API 和功能，并欢迎您在试用过程中通过 GitHub 提交反馈和问题。

## 链接汇总

- DeepSpeed GitHub 仓库: https://github.com/deepspeedai/DeepSpeed
- 解耦混合并行博客: https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid
- BF16 低精度训练基准测试脚本: https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/bf16_master_weight
- Ray 框架 GitHub 仓库: https://github.com/ray-project/ray
- DeepSpeed-Ulysses 论文: https://arxiv.org/abs/2309.14509
- 多模态训练代码仓库: https://github.com/ray-project/multimodal-training
- PyTorch 兼容 backward API 测试: https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/zero/test_zero_user_backward.py
- 低精度主参数/梯度/优化器状态测试: https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/half_precision/test_bf16.py
- torch.autocast 结合使用测试: https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/half_precision/test_with_autocast.py
