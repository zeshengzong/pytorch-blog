# 在 Blackwell 上加速扩散模型：使用 Diffusers 和 TorchAO 实现 MXFP8 与 NVFP4

作者：Vasiliy Kuznetsov（Meta）和 Sayak Paul（Hugging Face）

2026 年 4 月 8 日

用于图像和视频生成的扩散模型正迅速普及，能够生成极为逼真的视觉媒体。然而，这些模型的采用往往受到内存和计算资源的严苛限制。量化对于高效部署这些模型至关重要。

在本文中，我们展示了使用 diffusers 和 torchao 在 NVIDIA B200 上对 Flux.1-Dev、QwenImage 和 LTX-2 模型进行可复现的端到端推理加速，MXFP8 可实现最高 1.26 倍的加速，NVFP4 可实现最高 1.68 倍的加速。我们还概述了如何使用选择性量化、CUDA Graphs 和 LPIPS 作为评估指标，来迭代优化这些模型的精度和最佳性能。复现本文实验的代码在这里。

目录：

- MXFP8 和 NVFP4 背景介绍
- 使用 Diffusers 和 TorchAO 的基本用法
- 基准测试结果
- 技术注意事项

## MXFP8 和 NVFP4 背景介绍

MXFP8 和 NVFP4 是 NVIDIA Blackwell 架构（如 B200 GPU）原生支持的微缩放格式。与对整个张量进行缩放的标准量化不同，微缩放将元素分组为共享高精度缩放因子的小块（如 16 或 32 个值）。这样可以在保持动态范围和精度的同时显著降低位宽。

- MXFP8（OCP 微缩放 FP8）：来自开放计算项目（OCP）的 8 位行业标准格式（E4M3/E5M2）。它使用 8 位缩放的块大小为 32。它提供了"最佳平衡点"，推理速度比 BF16 更快，视觉质量几乎没有损失（更低的 LPIPS），并且通常在较小的批次大小下实现最低延迟。
- NVFP4（NVIDIA FP4）：由 Blackwell Tensor Core 独特加速的 4 位浮点格式（E2M1）。它使用 FP8 缩放因子的块大小为 16。它提供最高的理论吞吐量和最低的内存占用（约为 BF16 的 3.5 倍小），是高批次、计算密集型工作负载的理想选择。

请参阅此文章了解更多信息。

## 使用 diffusers 和 TorchAO 的基本用法

### 先决条件

NVFP4 需要至少 10.0 的 CUDA 计算能力。请确保您的 GPU 满足要求。本文中的基准测试在 B200 机器（B200 DGX）上进行。

对于虚拟环境，您可以使用 conda：

```
conda create -n nvfp4 python=3.11 -y

conda activate nvfp4

pip install --pre torch --index-url
https://download.pytorch.org/whl/nightly/cu130

pip install --pre torchao --index-url
https://download.pytorch.org/whl/nightly/cu130

pip install --pre mslk --index-url
https://download.pytorch.org/whl/nightly/cu130

pip install diffusers transformers accelerate sentencepiece protobuf av imageio-ffmpeg
```

在撰写本文时，PyTorch、TorchAO 和 MSLK 的 nightly 版本分别为 2.12.0.dev20260315+cu130、0.17.0.dev20260316+cu130 和 2026.3.15+cu130。

部分模型需要用户在 Hugging Face Hub 平台上进行身份验证。因此，如果尚未登录，请在运行示例前先运行 `hf auth login`。

## 基本用法

在 Diffusers 中使用 TorchAO 的 NVFP4 量化配置非常简单，因为它已原生集成：

```python
from diffusers import DiffusionPipeline, TorchAoConfig, PipelineQuantizationConfig

import torch

from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)

config = NVFP4DynamicActivationNVFP4WeightConfig(
    use_dynamic_per_tensor_scale=True, use_triton_kernel=True,
)
pipe_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(config)}
)

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,
    quantization_config=pipe_quant_config
).to("cuda")
pipe.transformer.compile_repeated_blocks(fullgraph=True)

pipe_call_kwargs = {
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "max_sequence_length": 512,
    "num_images_per_prompt": 1,
    "generator": torch.manual_seed(0),
}
result = pipe(**pipe_call_kwargs)
image = result.images[0]
image.save("my_image.png")
```

上述代码片段对模型的每个 `torch.nn.Linear` 层进行量化。

在本文中，我们始终使用带 `fullgraph=True` 的区域编译，因为它能显著减少编译时间，并且产生的结果几乎与完整模型编译一样好。请从这里了解更多关于区域编译的信息。

### 方案选择

以下代码片段展示了如何使用 TorchAO 配置 MXFP8 和 NVFP4 推理：

```python
# MXFP8

quant_config = MXDynamicActivationMXWeightConfig(
    activation_dtype=torch.float8_e4m3fn,
    weight_dtype=torch.float8_e4m3fn,
    kernel_preference=KernelPreference.AUTO,
)

# NVFP4

quant_config = NVFP4DynamicActivationNVFP4WeightConfig(
    use_dynamic_per_tensor_scale=True,
    use_triton_kernel=True,
)
```

## 基准测试结果

### Flux.1-Dev

以下推理参数用于对 FLUX.1-dev 进行基准测试：

```
{
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "max_sequence_length": 512,
}
```

### 性能与峰值内存

首先，我们展示不同设置和基准下的延迟和峰值内存消耗，MXFP8 加速最高可达 1.26 倍，NVFP4 最高可达 1.59 倍。请注意，这些结果使用了选择性量化，即我们排除某些层不进行量化。我们将在本文后面讨论更多关于选择性量化的内容。

### Flux-1.dev 在 MXFP8 和 NVFP4 量化下的性能与峰值内存

| 量化模式 | 批次大小 | 延迟（秒） | 内存（GB） | 相对 BF16 加速比 |
|---------|---------|-----------|-----------|----------------|
| 无 | 1 | 2.10 | 38.34 | 1.00 |
| MXFP8 | 1 | 1.75 | 26.90 | 1.21 |
| NVFP4 | 1 | 1.41 | 21.33 | 1.50 |
| 无 | 4 | 7.87 | 44.39 | 1.00 |
| MXFP8 | 4 | 6.36 | 32.95 | 1.24 |
| NVFP4 | 4 | 5.09 | 27.39 | 1.55 |
| 无 | 8 | 15.57 | 53.00 | 1.00 |
| MXFP8 | 8 | 12.40 | 41.56 | 1.26 |
| NVFP4 | 8 | 9.81 | 36.00 | 1.59 |

NVIDIA B200，选择性量化，torch.compile 区域编译；batch_size=1 使用 `torch.compile(..., mode='reduce-overhead')`。量化模式"无"表示不进行量化。

### 精度

在测试提示词下，MXFP8 和 NVFP4 生成的图像与 bfloat16 基准接近：

为了进行更全面的精度评估，我们计算了 bfloat16 图像（基准）与 MXFP8|NVFP4 图像（实验）之间的平均 LPIPS 分数，对 Drawbench 数据集中的提示词取平均值：

### Flux-1.dev 在 MXFP8 和 NVFP4 量化下的平均 LPIPS 分数

| 量化模式 | Drawbench 平均 LPIPS |
|---------|---------------------|
| 无 | 0 |
| MXFP8 | 0.11 |
| NVFP4 | 0.44 |

NVIDIA B200，选择性量化，torch.compile 区域编译。

LPIPS 分数为零意味着"图像完全相同"，较低的 LPIPS 分数对应更高的感知相似度。我们用于计算平均 LPIPS 分数的代码在这里。请参阅本文后面的 LPIPS 部分，了解更多有关使用 LPIPS 进行精度评估的详情。

### LTX-2

对于 LTX-2，我们启用了 VAE 上的分块处理以使内存需求可控。以下推理时间参数用于获取结果：

```
{
        "prompt": (
              "INT. HOME OFFICE - DAY. Soft natural daylight lights a desk with an open laptop. The camera holds a steady medium shot. A small real house cat sits naturally on all fours in front of the laptop, much smaller than the desk and computer. The cat looks at the screen curiously. Suddenly, with a soft magical sparkle effect, a pair of tiny reading glasses appears in midair and gently lands on the cat's face. A faint whimsical chime sound plays. The cat pauses for a split second, then begins pressing the keyboard clumsily with one paw, producing rapid typing sounds. The laptop screen glow reflects softly on the cat's fur while light playful music continues."
        ),
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 768,
        "height": 512,
        "num_frames": 121,
        "frame_rate": 24.0,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
}
```

### 性能与峰值内存

### LTX-2 在 MXFP8 和 NVFP4 量化下的性能与峰值内存

| 量化模式 | 批次大小 | 延迟（秒） | 内存（GB） | 加速比 |
|---------|---------|-----------|-----------|-------|
| 无 | 1 | 16.230 | 72.77 | 1.00 |
| MXFP8 | 1 | 13.724 | 54.54 | 1.18 |
| NVFP4 | 1 | 10.374 | 45.72 | 1.56 |
| 无 | 4 | 61.591 | 87.61 | 1.00 |
| MXFP8 | 4 | 50.956 | 69.38 | 1.21 |
| NVFP4 | 4 | 36.963 | 60.56 | 1.67 |
| 无 | 8 | 122.427 | 107.40 | 1.00 |
| MXFP8 | 8 | 102.546 | 89.18 | 1.19 |
| NVFP4 | 8 | 72.689 | 80.36 | 1.68 |

NVIDIA B200，选择性量化，torch.compile 区域编译。量化模式"无"表示不进行量化。

### 精度

请查看此链接，对比测试提示词的视频结果。对提示词数据集计算评估分数（如我们对 Flux-1.dev 所做的那样）留待未来研究。

### QwenImage

以下推理时间参数用于获取结果：

```
{
    "prompt": "A cat holding a sign that says hello world",
    "negative_prompt": " ",
    "height": 1024,
    "width": 1024,
    "true_cfg_scale": 4.0,
    "num_inference_steps": 50,
}
```

### 性能与峰值内存

### QwenImage 在 MXFP8 和 NVFP4 量化下的性能与峰值内存

| 量化模式 | 批次大小 | 延迟（秒） | 内存（GB） | 加速比 |
|---------|---------|-----------|-----------|-------|
| 无 | 1 | 7.454 | 62.21 | 1.00 |
| MXFP8 | 1 | 6.430 | 55.65 | 1.16 |
| NVFP4 | 1 | 5.369 | 52.45 | 1.39 |
| 无 | 4 | 26.779 | 75.52 | 1.00 |
| MXFP8 | 4 | 21.835 | 68.97 | 1.23 |
| NVFP4 | 4 | 18.279 | 65.76 | 1.47 |
| 无 | 8 | 52.095 | 92.47 | 1.00 |
| MXFP8 | 8 | 41.569 | 85.91 | 1.25 |
| NVFP4 | 8 | 34.969 | 82.7 | 1.49 |

NVIDIA B200，选择性量化，torch.compile 区域编译，batch_size=1 使用 `torch.compile(..., mode='reduce-overhead')`。量化模式"无"表示不进行量化。

### 精度

在测试提示词下，MXFP8 和 NVFP4 生成的图像与 bfloat16 基准接近，但 NVFP4 与 MXFP8 相比差异稍大：

在以下表格中，我们报告了与 Flux.1-Dev 类似的 LPIPS 分数。

### QwenImage 在 MXFP8 和 NVFP4 量化下的平均 LPIPS 分数

| 量化模式 | Drawbench 平均 LPIPS |
|---------|---------------------|
| 无 | 0 |
| MXFP8 | 0.34 |
| NVFP4 | 0.41 |

注意：在我们的实验中，我们发现 QwenImage 比 Flux.1-Dev 对量化更敏感，这体现在 QwenImage 的平均 MXFP8 LPIPS 分数较高（0.34），而 Flux-1.Dev 上 MXP8 的平均 LPIPS 分数为 0.11。通过更激进的选择性量化或更先进的数值算法（GPTQ、QAT 等）进一步降低 QwenImage 的平均 LPIPS 分数留待未来研究。

## 技术注意事项

在本节中，我们分享了如何使用选择性量化、CUDA Graphs 和 LPIPS 来迭代本文所呈现的性能和精度指标。

## 通过选择性量化优化精度和性能

我们使用选择性量化来优化延迟（所有模型）和 LPIPS（Flux-1.dev），根据两个简单启发规则跳过某些层：

- 如果 `torch.nn.Linear` 的权重或激活形状太小而无法从量化中获益（`min(M, K, N) < 1024)`），则跳过。这是为了确保量化矩阵乘法带来的加速大于量化激活的额外开销（更多背景：这里）。

  如何使用 torchao 工具在您的模型中找到权重和激活形状的教程在这里。请注意，即使权重较大，较小的激活形状也可能使量化无法获益。

- 如果该层可能对模型精度有显著贡献（如嵌入层、归一化层），则跳过。

要在您的模型上应用此方法，您可以打印出模型（`print(model)`）并手动检查 FQN，然后根据您对模型架构的了解跳过可能影响精度的 FQN。

我们为每个模型使用的精确启发规则如下：

- Flux-1.dev
- QwenImage
- LTX-2

为了量化选择性量化的影响，我们测量了纯 Bfloat16 图像与使用 NVFP4 和 MXFP8 生成的图像之间的性能、内存和平均 LPIPS（使用 AlexNet）。

### 完全量化与选择性量化对 Flux-1.dev 的影响

| 量化模式 | LPIPS | 延迟（秒） | 内存（GB） |
|---------|-------|-----------|-----------|
| MXFP8 + 完全量化 | 0.138128 | 1.774 | 26.84 |
| MXFP8 + 选择性量化 | 0.107562 | 1.746 | 26.90 |
| NVFP4 + 完全量化 | 0.479679 | 2.112 | 21.25 |
| NVFP4 + 选择性量化 | 0.438337 | 2.076 | 21.33 |

（LPIPS 越低越好，LPIPS 约为 0.1 通常意味着图像几乎无法区分。LPIPS 计算代码在这里）。

从上述结果可以看出，排除某些层不进行量化（即"选择性量化"）在延迟、峰值内存消耗和 LPIPS 之间提供了最佳权衡。因此，我们在本文报告的其余两个模型中遵循选择性量化的方案。

我们使用简单的启发规则来找到我们的选择性量化方案。还有更高级的选择性量化方法，例如此层敏感性研究。

请注意，在迭代我们的选择性量化方案时，我们发现 TorchAO 的内核在将张量量化为 NVFP4 时存在性能差距。我们在此 PR 中通过将 `to_nvfp4` 内核升级为使用 MSLK 来改善了 NVFP4 的性能。

### 使用 CUDA Graphs 改善 CPU 开销

我们注意到，当使用 NVFP4 和较小的批次大小（如 1）时，CPU 开销对延迟改善有不可忽视的影响。为了显著减少此开销，我们使用了"reduce-overhead"编译模式，该模式启用了 CUDA graphs。以下我们提供了应用 CUDA Graphs 前后的性能分析追踪。

为了将 `torch.compile(..., mode='reduce-overhead')` 与 diffusers 库的逐块编译干净地组合，我们必须将每个 Transformer 块包装在一个克隆其输入的函数中。执行此操作的 PR 在这里，展示了 QwenImage + nvfp4 在 batch_size==1 时的 1.81 倍加速。

### 使用 LPIPS 评估图像生成精度

我们使用 LPIPS（GitHub）指标来比较量化模型生成的图像与基准（bfloat16）模型生成的图像有多相似。以伪代码表示：

```python
lpips_scores = []

for text_prompt in dataset:
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {"prompt": prompt, "generator": generator, ...}
    image_baseline = pipe_bf16(**kwargs)
    image_quantized = pipe_quantized(**kwargs)
    lpips_score = calculate_lpips_score(image_baseline, image_quantized)
    lpips_scores.append(lpips_score)

lpips_mean = lpips_scores.sum() / len(lpips_scores)
```

我们使用的实际代码在这里。

### 图像对的 LPIPS 分数示例

本节提供图像对的 LPIPS 分数示例，以帮助将上述报告的 LPIPS 指标置于背景中，并使读者能够推断"什么是好的 LPIPS 分数"。

以下图像使用 FLUX.1-dev 生成。左侧图像是基准（bfloat16），右侧图像来自使用 MXFP8 对模型每个 `torch.nn.Linear` 进行量化。LPIPS 分数基于右侧图像（实验）与左侧图像（基准）的比较。

以下我们提供了类似的比较，但右侧为 NVFP4 图像。

## 结论

在本文中，我们研究了 NVFP4 和 MXFP8 量化方案在流行图像和视频生成模型上的性能。我们提供了在速度、质量和内存之间提供合理权衡的方案。我们还发现了一些阻碍最佳性能的重要问题以及解决方法。我们希望这些方案能帮助提升您的图像和视频生成工作负载的性能。

## 资源

- 代码仓库：https://github.com/sayakpaul/diffusers-blackwell-quants
- TorchAO 文档：
  - MXFP8：https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig.html
  - NVFP4：https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.NVFP4DynamicActivationNVFP4WeightConfig.html
- Diffusers x TorchAO 集成：https://huggingface.co/docs/diffusers/main/en/quantization/torchao

所有输出可在此处找到：https://huggingface.co/datasets/sayakpaul/diffusers-blackwell-quants

## 链接汇总

- diffusers: https://github.com/huggingface/diffusers
- torchao: https://github.com/pytorch/ao
- Flux.1-Dev: https://huggingface.co/black-forest-labs/FLUX.1-dev
- QwenImage: https://huggingface.co/Qwen/Qwen-Image
- LTX-2: https://huggingface.co/Lightricks/LTX-2
- 代码仓库（复现实验）: https://github.com/sayakpaul/diffusers-blackwell-quants
- NVFP4 背景文章: https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/
- 区域编译博客: https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
- LPIPS GitHub: https://github.com/richzhang/PerceptualSimilarity
- Drawbench 数据集: https://huggingface.co/datasets/sayakpaul/drawbench
- 计算 LPIPS 分数代码: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/5354691e2f171e86245468cbda57af56dd2c606a/README.md?plain=1#L26
- LTX-2 视频对比: https://gist.github.com/sayakpaul/ed83f505b6fbed4f4d874826773a891a
- TorchAO 推理文档（microbenchmarks）: https://docs.pytorch.org/ao/main/workflows/inference.html#microbenchmarks-and-roofline-model
- TorchAO 调试权重和激活教程: https://docs.pytorch.org/ao/main/eager_tutorials/debugging_weights_and_activations.html
- Flux-1.dev 选择性量化代码: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/f313fe7dcb44f55dae4dd5191239bad15fa2a5b6/benchmark.py#L190-L201
- QwenImage 选择性量化代码: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L137
- LTX-2 选择性量化代码: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L160
- LPIPS 计算代码: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/f313fe7dcb44f55dae4dd5191239bad15fa2a5b6/compute_lpips.py
- 层敏感性研究: https://huggingface.co/blog/badaoui/sensitivity-aware-mixed-precision-quantizer-v1#layer-sensitivity-estimation
- NVFP4 性能改进 PR: https://github.com/pytorch/ao/pull/4031
- MSLK: https://github.com/meta-pytorch/MSLK
- CUDA Graphs 封装 PR: https://github.com/sayakpaul/diffusers-blackwell-quants/pull/1
- compute_lpips.py: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/main/compute_lpips.py
- TorchAO MXFP8 API 文档: https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig.html
- TorchAO NVFP4 API 文档: https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.NVFP4DynamicActivationNVFP4WeightConfig.html
- Diffusers x TorchAO 集成文档: https://huggingface.co/docs/diffusers/main/en/quantization/torchao
- 所有实验输出: https://huggingface.co/datasets/sayakpaul/diffusers-blackwell-quants
