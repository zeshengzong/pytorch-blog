# vLLM 模型推理中的可移植 Helion 内核

**作者：** Sean Chen（Red Hat）和 Yanan Cao（PyTorch，Meta Platforms）
**日期：** 2026 年 6 月 10 日

## 摘要

Helion 内核已集成到 vLLM 中，用于基于 Qwen3 模型的 FP8 推理，并在 NVIDIA H100 和 B200 GPU 上进行了评估。实验表明，Helion 提供了一种高效的 PyTorch 原生工作流，用于开发融合 GPU 内核，同时在众多量化、归一化以及融合密集型推理内核上实现了性能提升。端到端基准测试在多种服务场景下均展现了吞吐量增益，并且针对 Blackwell GPU 上 GEMM 性能的进一步优化工作仍在持续推进中。

## vLLM 与 Helion 简介

**vLLM** 是一个高性能的大语言模型（LLM）推理与服务框架。它因出色的吞吐量性能、高效的 KV 缓存管理、持续批处理架构以及对投机解码、量化和分布式服务等高级推理特性的支持，而被广泛应用于生产级 LLM 服务。在内部，vLLM 大量依赖自定义 GPU 内核、TorchInductor 融合以及 CUTLASS 和 DeepGEMM 等优化 GEMM 后端，以在不同硬件平台上实现高推理效率。

**Helion** 是一种 PyTorch 原生、硬件无关的内核 DSL，专为使用分块编程模型编写高性能内核而设计。与底层 CUDA 编程不同，Helion 提供了更自然的以 PyTorch 语法为中心的开发体验，同时仍保留了对内存布局、分块策略和内核调度的低级控制。你可以将其理解为带有分块能力的 PyTorch。如果你了解 PyTorch 或 Triton，你已经掌握了 Helion 的大部分内容。除了流畅的编写体验外，Helion 的另一大优势是其强大的预先（AOT）自动调优基础设施，它可以探索大规模内核配置空间，并为特定工作负载和硬件目标自动选择优化实现。

## 使用 Helion 内核的 vLLM 模型推理

我们首先专注于使用 Qwen3 模型家族进行无张量并行推理，并启用了 FP8 激活量化。

我们的目标是评估 Helion 内核相比现有 vLLM 实现能否提升推理性能。

在本次实验中，我们将量化推理中涉及的几乎所有前向传播内核替换为 Helion 实现，并在内核级别和端到端服务级别对它们进行了基准测试。

### vLLM 前向传播融合模式

对于 Qwen3 模型，vLLM 中未融合的前向传播按以下内核序列执行：

1. input_norm
2. fp8_quant
3. scaled_mm (qkv_proj)
4. split_qkv
5. q_norm
6. k_norm
7. rope
8. attention
9. fp8_quant
10. scaled_mm (out_proj)
11. post_attention_norm
12. fp8_quant
13. scaled_mm (gate_up)
14. silu_and_mul
15. fp8_quant
16. scaled_mm (down_proj)

### 动态逐词元激活量化

在应用 torch.compile 和 TorchInductor 融合流程后，执行模式变为：

1. rms_norm + fp8_quant
2. scaled_mm (qkv_proj)
3. split_qkv + q_norm + v_norm
4. rope
5. attention
6. fp8_quant
7. scaled_mm (out_proj)
8. rms_norm + fp8_quant
9. scaled_mm (gate_up)
10. silu_and_mul + fp8_quant
11. scaled_mm (down_proj)

请注意，`scaled_mm` 和 attention 均被注册为 PyTorch 自定义算子。由于这些算子对 TorchInductor 不透明，它们形成了阻止编译器进一步融合的硬边界。

### 动态逐组激活量化

当启用动态逐组激活量化并为 `scaled_mm_blockwise` 选择 DeepGEMM 时，执行模式变为：

1. rms_norm
2. fp8_quant (ue8m0)
3. scaled_mm (qkv_proj, DeepGEMM)
4. split_qkv + q_norm + v_norm
5. rope
6. attention
7. fp8_quant (ue8m0)
8. scaled_mm (out_proj, DeepGEMM)
9. rms_norm
10. fp8_quant (ue8m0)
11. scaled_mm (gate_up, DeepGEMM)
12. silu_and_mul
13. fp8_quant (ue8m0)
14. scaled_mm (down_proj, DeepGEMM)

DeepGEMM 内部使用 UE8M0 激活量化。在当前的 vLLM 实现中，`fuse_act_quant` 和 `fuse_norm_quant` 流程不支持 UE8M0 量化，这阻止了这些额外的融合操作。

如果 DeepGEMM 不可用且使用基于 CUTLASS 的内核，则执行模式与动态逐词元量化情况类似。

### Helion 内核实现

在本工作中，我们实现了以下 Helion 内核：

- dynamic_per_token_scaled_fp8_quant
- rms_norm_dynamic_per_token_quant
- silu_and_mul_dynamic_per_token_quant
- fused_qk_norm_rope
- per_token_group_fp8_quant
- rms_norm_per_block_quant
- silu_and_mul_per_block_quant
- scaled_mm
- scaled_mm_blockwise

`scaled_mm` 和 `scaled_mm_blockwise` 内核遵循 vLLM 中现有的 Triton 实现。`silu_and_mul_dynamic_per_token_quant` 是一个新的融合内核，它将 `silu_and_mul` 和 `dynamic_per_token_quant` 合并为单次内核启动。其余内核是对 vLLM 使用的现有 `torch.ops._C` CUDA 内核的 Helion 重新实现。

### vLLM Helion 内核集成

我们通过 vLLM Helion 内核集成框架集成了这些内核，该框架提供了：

- 自动调优基础设施
- 配置管理
- 内核注册
- 运行时调度

为启用 Helion 内核，我们手动更新了 vLLM 融合流程，将相应内核替换为对应的 Helion 融合内核。融合后，前向传播执行模式变为以下内容：

**对于逐词元激活量化：**

1. rms_norm_dynamic_per_token_quant (helion)
2. scaled_mm (helion)
3. fused_qk_norm_rope (helion)
4. attention (default)
5. dynamic_per_token_scaled_fp8_quant (helion)
6. scaled_mm (helion)
7. rms_norm_dynamic_per_token_quant (helion)
8. scaled_mm (helion)
9. silu_and_mul_dynamic_per_token_quant (helion)
10. scaled_mm (helion)

**对于逐组激活量化：**

1. rms_norm_per_block_quant (helion)
2. scaled_mm_blockwise (helion)
3. fused_qk_norm_rope (helion)
4. attention (default)
5. per_token_group_fp8_quant (helion)
6. scaled_mm_blockwise (helion)
7. rms_norm_per_block_quant (helion)
8. scaled_mm_blockwise (helion)
9. silu_and_mul_per_block_quant (helion)
10. scaled_mm_blockwise (helion)

### 自动调优

我们使用 Helion 默认的 LFBOTreeSearch 算法，配置如下：

```
initial_population=FROM_RANDOM, copies=5, max_generations=20, similarity_penalty=1.0
```

为了最大化性能，我们使用与每个模型编译时静态维度（如隐藏层大小和中间层大小）完全匹配的形状对内核进行自动调优。这是 vLLM-Helion 集成的优势所在——它允许 Helion 为许多不同的形状自动调优/存储/调度配置，这一优势同样适用于实际生产用例。

对于动态维度（`num_tokens`），我们在 1 到 8192 的 2 的幂次值范围内进行自动调优。

例如，我们对输入张量 `[M, K] x [K, N]` 的 `scaled_mm` 内核进行了自动调优，其中

- M 范围从 1 到 8192
- (K, N) 对应每个 Qwen3 模型的投影层。

| 模型 | qkv_proj | out_proj | gate_up | down_proj |
|-------|----------|----------|---------|-----------|
| Qwen3-1.7B | [2048, 4096] | [2048, 2048] | [2048, 12288] | [6144, 2048] |
| Qwen3-8B | [4096, 6144] | [4096, 4096] | [4096, 24576] | [12288, 4096] |
| Qwen3-32B | [5120, 10240] | [5120, 5120] | [5120, 51200] | [25600, 5120] |

*表 1：每个 Qwen3 模型的投影层 [K, N] 维度。*

我们对测试的每个硬件平台独立进行了所有内核的自动调优。

### 运行时调度

在运行时，Helion 集成框架将请求调度到最适合输入形状的自动调优配置。

例如，scaled_mm 的调度基于两个输入矩阵的形状 (M, K, N)，其中 M 根据每批请求的运行时 `num_tokens` 向上取整到下一个 2 的幂次。类似的策略也应用于其他内核。

## 性能评估——内核级别

内核级别基准测试旨在评估每个 Helion 内核相对于其基准线所产生的局部加速比。具体而言，我们使用 CUTLASS 作为 `scaled_mm` 和 `scaled_mm_blockwise` 的基准线，而其他算子则与经过 torch.compile 的 vLLM 实现和现有的 `torch.ops._C` 内核进行比较。原因如下：

- vLLM 中逐词元量化默认使用 `torch.compile`，
- 逐组量化因性能问题默认使用 `torch.ops._C` CUDA 实现。

对于 torch.compile 基准线，我们匹配了 vLLM 的编译设置：

```python
torch.compile(
    native_torch_impl,
    fullgraph=True,
    dynamic=False,
    backend="inductor",
    options={
        'enable_auto_functionalized_v2': False,
        'size_asserts': False,
        'alignment_asserts': False,
        'scalar_asserts': False,
        'combo_kernels': True,
        'benchmark_combo_kernel': True
    }
)
```

值得注意的是，启用 `'combo_kernels': True` 非常重要，因为它允许 TorchInductor 将多个独立内核融合为单次启动。

对于内核级别基准测试，我们通过 `triton.testing.do_bench_cudagraph` 启用了 `CudaGraph` 模式，并进行了适当的预热和重复测试，以消除调度开销、缓存冷启动和时序波动等噪声。

| 内核 | 相对 torch.compile 加速比 (H100) | 相对 torch.ops._C 加速比 (H100) | 相对 CUTLASS 加速比 (H100) | 相对 torch.compile 加速比 (B200) | 相对 torch.ops._C 加速比 (B200) | 相对 CUTLASS 加速比 (B200) |
|--------|------|------|------|------|------|------|
| dynamic_per_token_scaled_fp8_quant | 1.237x | 1.405x | N/A | 1.311x | 1.495x | N/A |
| rms_norm_dynamic_per_token_quant | 1.180x | 1.802x | N/A | 1.240x | 1.969x | N/A |
| silu_and_mul_dynamic_per_token_quant | 1.256x | N/A | N/A | 1.420x | N/A | N/A |
| fused_qk_norm_rope | 1.383x | 1.204x | N/A | 1.133x | 1.155x | N/A |
| per_token_group_fp8_quant | 1.423x | 1.408x | N/A | 1.150x | 1.446x | N/A |
| rms_norm_per_block_quant | 1.674x | 2.055x | N/A | 1.424x | 2.128x | N/A |
| silu_and_mul_per_block_quant | 1.731x | 2.269x | N/A | 1.483x | 2.325x | N/A |
| scaled_mm | N/A | N/A | 1.080x | N/A | N/A | 0.739x |
| scaled_mm_blockwise | N/A | N/A | 0.957x | N/A | N/A | 0.782x |

*表 2：Helion 内核实现的几何平均加速比汇总。*

对于非 GEMM 内核，Helion 持续展现出强劲性能，优于 TorchInductor 生成的内核和现有的 vLLM CUDA 实现。

对于 GEMM 工作负载（`scaled_mm` 和 `scaled_mm_blockwise`），结果则更为复杂：

- 在 H100 上，scaled_mm 优于 CUTLASS。
- 在 B200 上，两个 GEMM 内核目前均落后于 CUTLASS。

B200 上的主要限制因素是 Triton 生成的 GEMM 内核在 Blackwell GPU 上的性能，而非 Helion 编程模型本身。Helion 目前依赖 Triton 代码生成来实现这些内核，观察到的性能差距在很大程度上反映了 Triton GEMM 在 Blackwell 硬件上的当前状态。Helion 的 CuteDSL 后端的持续工作预计将进一步提升 Blackwell 上的 GEMM 性能。

## 性能评估——端到端模型级别

端到端模型级别基准测试突出了 Helion 内核对用户可见的影响。我们为此选取了 3 种不同规模的 Qwen3 模型变体：

- Qwen3-1.7B
- Qwen3-8B
- Qwen3-32B

所有三个 Qwen3 模型的模型级别基准测试流量模式均启用了 `CudaGraph`，num_tokens 值在 1 到 8192 的 2 的幂次区间内变化。

为构建流量模式，我们使用了 vLLM 内置的服务基准测试工具和随机输入数据。

为了最小化前缀缓存效应带来的噪声，我们：

- 禁用了提示词随机化，
- 在每次基准测试运行前重启了 vLLM 服务器。

以下是一个示例命令：

```bash
vllm serve --model $MODEL --max-num-seqs $BATCH_SIZE --tensor-parallel-size 1 --compilation-config '{"max_cudagraph_capture_size": 8192, "custom_ops": ["+quant_fp8"], "pass_config": {"fuse_norm_quant": true, "fuse_act_quant": true, "enable_qk_norm_rope_fusion": true}}'

vllm bench serve \
  --backend vllm \
  --model $MODEL \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts $NUM_PROMPTS \
  --max-concurrency $BATCH_SIZE \
  --input-len 512 \
  --output-len 600 \
  --num-warmups $NUM_WARMUPS \
  --disable-shuffle
```

`max_cudagraph_capture_size` 设置为 8192 以匹配默认的 `max_num_batched_tokens`，确保所有执行路径均被 CUDA graph 捕获。

所有工作负载均在两个 NVidia GPU 平台上进行评估：

- NVIDIA H100
- NVIDIA B200

为了更深入地了解性能提升的来源，我们将 Helion 内核分为三类，分别独立进行基准测试，也进行组合测试。

- **fp8_quant**：fp8 量化内核和融合量化内核
- **qk_norm_rope**：`fused_qk_norm_rope` 内核
- **scaled_mm**：`scaled_mm` 或 `scaled_mm_blockwise` 内核

### 动态逐词元激活量化

我们使用了以下检查点：

- RedHatAI/Qwen3-1.7B-FP8-dynamic
- RedHatAI/Qwen3-8B-FP8-dynamic
- RedHatAI/Qwen3-32B-FP8-dynamic

*图 1：在 H100 上启用逐词元激活量化时的总吞吐量加速比，以默认 vLLM 设置为基准线。*

对于 1.7B 模型，结果显示在启用所有 Helion 内核组时，H100 上的端到端吞吐量提升约 1.05 倍。对于 8B 模型，提升在批大小 32 附近最为显著，这与内核级别观察结果一致——Helion scaled_mm 在 `num_tokens = 32` 附近实现了最强性能。

我们还评估了投机解码场景，其中解码阶段的有效 `num_tokens` 自然落在这一性能甜点范围内。

使用：

- RedHatAI/Qwen3-8B-speculator.eagle3
- RedHatAI/Qwen3-32B-speculator.eagle3

我们观察到在启用所有 Helion 内核时，端到端吞吐量提升最高约 1.09 倍。

| 批大小 | 模型 | 投机词元数（每位置接受率） | Helion TTFT（均值，ms） | 默认 TTFT（均值，ms） | **TTFT 加速比** | Helion TPOT（均值，ms） | 默认 TPOT（均值，ms） | **TPOT 加速比** | Helion 总吞吐量（tok/s） | 默认总吞吐量（tok/s） | **总吞吐量加速比** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | Qwen3-8B | 1 (47%) | 34.75 | 39.93 | **1.15x** | 4.63 | 5.01 | **1.08x** | 6,314.86 | 5817.23 | **1.09x** |
| 16 | Qwen3-8B | 3 (35%, 25%, 15%) | 38.46 | 51.18 | **1.33x** | 4.40 | 4.63 | **1.05x** | 6,616.60 | 6261.1 | **1.06x** |
| 8 | Qwen3-32B | 2 (24%, 10%) | 81.92 | 100.93 | **1.23x** | 13.29 | 14.37 | **1.08x** | 1,101.61 | 1018.32 | **1.08x** |
| 8 | Qwen3-32B | 3 (24%, 10%, 4%) | 83.01 | 104.73 | **1.26x** | 13.33 | 14.21 | **1.07x** | 1,100.04 | 1030.51 | **1.07x** |

*表 3：在 H100 上启用逐词元激活量化和投机解码的端到端基准测试结果。括号中报告了投机词元的接受率。*

在 NVIDIA B200 上，我们在端到端评估中仅启用了 `fp8_quant` 内核组。其余内核组要么：

- 相对于基准线性能不足（Triton 对 Blackwell GEMM 的限制），
- 要么在不同流量模式下增益不稳定。

即使只启用与量化相关的内核，我们仍在所有测试的 Qwen3 模型规模上观察到了有意义的吞吐量提升。

*图 2：在 B200 上启用逐词元激活量化时的总吞吐量加速比，以默认 vLLM 设置为基准线。*

### 动态逐组激活量化

对于逐组激活量化，我们使用了以下检查点：

- Qwen/Qwen3-1.7B-FP8
- Qwen/Qwen3-8B-FP8
- Qwen/Qwen3-32B-FP8

对于逐组激活量化，DeepGEMM 是 H100 和 B200 上块状 FP8 GEMM 的默认后端。然而，我们当前的逐组 Helion 量化内核与 DeepGEMM 所需的 UE8M0 量化格式尚不兼容。因此，在本次实验中，我们强制 vLLM 使用 CUTLASS 作为线性后端。

这意味着本节的基准线**不是**默认的 vLLM 配置。然而，这一比较仍然有意义，因为我们能够在所有运行中为线性层使用一致的 CUTLASS 内核。因此，测量到的差异来自被评估的非 GEMM 内核，如 FP8 量化和融合量化内核，而非线性后端的变化。

以下图表显示，仅启用小型 Helion 内核仍在所有工作负载中产生了约 1.05 倍的端到端吞吐量提升。

*图 3：在 H100 和 B200 上启用逐组激活量化时的总吞吐量加速比，以将线性层后端替换为 CUTLASS 的默认 vLLM 设置为基准线。*

## 资源

为便于复现和进一步探索，本文讨论的所有 Helion 内核实现均链接在相应的 GitHub issue 中。同一 issue 还包括我们实验中使用的 vLLM 分支，用于复现报告的端到端基准测试结果。

## 注意事项

在我们的实验中，大部分工程时间花费在内核自动调优上。对于 scaled_mm 等大型内核，在所有三个模型规模上运行全力度的自动调优扫描，共涵盖 168 种不同的输入形状，可能需要整整一天时间，因为 Helion 会为每种形状自动生成并基准测试数千种候选内核实现。初步研究表明，详尽的逐形状自动调优和调度并非总是必要的，减少特化桶的数量可能以最小的性能损失实现自动调优成本与运行时性能之间更好的权衡。Helion 团队正在积极探索更多技术以进一步减少调优时间，包括搜索空间缩减策略和基于 LLM 的自动调优方法。

另一个注意事项是，Helion 运行时调度本身在每次内核启动时引入了数十微秒的 CPU 开销。对于小型内核，这种开销可能主导端到端延迟。因此，CUDA graph 的捕获和回放对于使用 Helion 内核实现最优性能至关重要。Helion 团队正在积极降低非 CudaGraph 模式下的调度延迟。

## 结论

Helion 提供了一种自然的、以 PyTorch 语法为中心的方式，以分块编程风格编写内核。它显著简化了内核开发并减少了实现工作量。在我们的实验中，大多数内核可以在一天内完成实现和验证，这表明 Helion 是一种用于快速开发新内核和探索内核融合机会的实用 DSL。

结合其强大的 AOT 自动调优能力，Helion 展现出实现高性能的强大潜力。我们的实验表明，Helion 内核在众多内核上提供了强劲的性能，并在大多数情况下持续优于默认的 vLLM 实现。对于 GEMM 内核，特别是在 Blackwell GPU 上，仍有改进空间以达到或超越 CUTLASS 性能，相关团队正在积极通过改进 Triton 代码生成和引入 CuteDSL 等替代后端来解决这一问题。

## 致谢

本工作得到了 Red Hat 的 OCTO 和 vLLM 团队以及 Meta 的 Helion 团队众多贡献者的支持。我们特别感谢我们的同事：Luka Govedič、Richard Zou 和 Will Feng 在整个工作过程中提供的反馈和支持。

---
原文链接: https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/
