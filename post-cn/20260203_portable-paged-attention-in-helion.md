# Helion 中的可移植分页注意力机制

**作者：** Burkhard Ringlein（IBM Research）和 IBM Research 的 vLLM 团队

最近，PyTorch 团队发布了 Helion，这是一种新的基于 PyTorch 的领域特定语言，旨在使高性能且可移植内核的开发更加便捷。Helion 内置了广泛的自动调优功能，有望将性能可移植性的前沿推进到比 Triton 更远的地方。

为了检验这一承诺（同时学习 Helion），我们着手迎接在 Helion 中编写 AI 最关键性能内核之一的挑战：分页注意力（Paged Attention），即 vLLM 的核心。

在过去一年中，我们为 vLLM 贡献了一个完全用 Triton 编写的高性能且平台可移植的注意力后端，该后端没有外部依赖，可在 NVIDIA、AMD 和 Intel GPU 上运行（参见我们的 PyTorch 大会演讲）。因此，我们将其中一个内核（unified_attention_2d）用 Helion 实现，作为 vLLM 中的一个新实验性后端（PR#27293）。

## vLLM、Triton 和 Helion 的简要背景

vLLM 被广泛用于 LLM 推理，是 PyTorch 基金会的一部分。vLLM 在生产环境中的采用日益增多，可在 NVIDIA、AMD 和 Intel GPU 上运行，也支持 Google TPU、华为 Ascend NPU、AWS Inferentia 或 IBM Spyre 等自定义加速器。vLLM 通过其精心设计的软件架构以及与 `torch.compile` 的深度集成，为几乎所有 LLM 模型提供高效、高性能的推理。

Triton 是一种可以用 Python 编写的领域特定语言（DSL），提供对 AMD、Intel 和 NVIDIA GPU 的即时（JIT）编译。Triton 内核已被证明能展现出最先进的性能并且具有可移植性。例如，我们用 Triton 编写了分页注意力，完全相同的内核代码在 NVIDIA H100 和 AMD MI300 上均达到了最先进的性能（您可以阅读我们的详细论文或相关博客文章）。为此，我们也在有限程度上利用了 Triton 的自动调优器。然而，Triton 中的自动调优存在严重局限性，尽管其对性能可移植性有积极影响，但这些局限性阻止了其在生产环境中的使用。因此，目前我们的 Triton 注意力后端使用简单的 if-else 语句作为启发式规则。

此外，Triton 也是 PyTorch Inductor 的输出语言，Inductor 是 `torch.compile` 的编译组件。

Helion 是另一种 DSL，于 10 月底进入 beta 阶段。Helion 将自身定位为"分块的 PyTorch"，并有两大目标：第一，将分块（tiling）引入 PyTorch，使得分块程序可以使用 PyTorch API 编写；第二，通过广泛的自动调优增强可移植性。与 Triton 相比，Helion 的自动调优器不仅具有可用的缓存机制，还拥有更多的自由度。这种更大的自由度来自于 Helion 的自动调优器不仅可以调整实现的算法层面，还可以调整编译标志（如 warp 数量或流水线深度）。它还具有先进的搜索算法，这是我们之前在 Triton 背景下研究过的内容。

## 实现细节：如何在分块 PyTorch 中编写分页注意力

### 启动网格与并行化方法

作为起点，我们希望重新实现更简单的"2D"版本的统一注意力 Triton 内核。它被称为"2D"是因为该内核具有二维启动网格（详情参见相关博客），我们选择这个内核版本是因为我们认为并行分块 softmax 实现在初期会过于复杂。

然而，由于 Helion 和 Triton 处理启动网格的方式不同，我们没有 1:1 地遵循 2D 方法，而是围绕"Q 块"这一核心概念构建了 Helion 内核。这个概念在下图中说明：

图 1：我们 Helion 内核中"Q 块"的概念。

在这张图中，我们看到需要计算的*一个*请求的三个维度。注意力内核需要遍历所有查询词元，直至 query_length（底轴）。在我们的内核中，我们同时获取多个查询词元。这个分块大小 TILE_Q 是可调的。接下来，对于每个词元，有多个查询头和 KV 头（左轴）。我们重新实现了 QGA 优化，使得一个 KV 头的所有查询头同时被获取。每个 KV 头的查询头数（QpKV）是该方向的分块大小，称为 TILE_M。最后，我们需要以可调的 TILE_N 块大小（对角轴）遍历该查询的 KV 缓存，直至当前上下文长度。在这个内层循环中，包括矩阵乘法（hl.dot）在内的实际注意力计算正在进行，使用了在线 softmax 实现。在内核中，在所有这些之外还有一个额外的循环，用于遍历批次中的所有请求（图中未显示）。

然而，vLLM 处理的输入张量，其第一个维度是序列数量和查询长度的组合（通常称为"扁平化 varlen"布局）。因此，vLLM 提供一个额外的张量作为索引，用于知道哪个词元属于哪个序列。

因此，在尝试了一些实现方案后，我们确定了上述四层循环方法：

```python
# 注册可调块大小
q_block_size = hl.register_block_size(1, q_block_padded_size)
num_pages_at_once = hl.register_block_size(1, 32)

# 外层循环 -> 变为启动网格
for seq_tile, tile_m in hl.tile([num_seqs, num_query_heads],
      block_size=[1, num_queries_per_kv],):

    seq_len = t_seq_lens[seq_tile]
    query_start = t_query_start_lens[seq_tile]
    query_end = t_query_start_lens[seq_tile + 1]
    query_len = query_end - query_start

    # 遍历一个请求的查询
    for tile_q in hl.tile(query_len, block_size=q_block_size):
    ...

       # 遍历 KV 缓存
        for tile_n in hl.tile(num_blocks,                  
              block_size=num_pages_at_once):

        ...
```

如上所示，外层循环是一个融合循环，具有两个维度：批次中的序列和 QpKV。这个外层循环将成为 Triton 中的启动网格（Helion 建议对外层循环也使用 hl.tile 而非 hl.grid）。由于我们在循环之前也需要调好的块大小进行边界计算，我们显式地预先注册块大小。此外，为了简化启动网格，我们改变了实现中循环的顺序（相对于上面描述的顺序），让外层循环遍历查询头。

接下来，第二层循环遍历所选序列的查询长度，使用可调的分块大小。但请注意，我们填充了该分块大小的上界（参见 q_block_padded_size），这样 JIT 编译器和自动调优器就不会针对所有可能的查询长度组合被触发。相反，我们在这里只提供 2 的幂次方的填充长度，这减少了运行时的 JIT/自动调优开销。最内层的循环遍历所选序列中 KV 缓存的页数。因此，相应已注册块大小的上界意味着 32 页 KV 缓存内存（每页例如 16 个词元）。

由此生成的 Triton 代码可能如下所示：

```python
# src[helion_unified_attention.py:129]: for seq_tile, tile_m in hl.tile(
# src[helion_unified_attention.py:130]:     [num_seqs, num_query_heads],
# src[helion_unified_attention.py:131]:     block_size=[1, num_queries_per_kv],
# src[helion_unified_attention.py:129-132]: ...

num_pid_m = num_seqs
num_pid_n = tl.cdiv(32, _BLOCK_SIZE_3)
inner_2d_pid = tl.program_id(0)
num_pid_in_group = 4 * num_pid_n
group_id = inner_2d_pid // num_pid_in_group
first_pid_m = group_id * 4
group_size_m = min(num_pid_m - first_pid_m, 4)
pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
offset_2 = pid_0
offset_3 = pid_1 * _BLOCK_SIZE_3
...

# src[helion_unified_attention.py:141]: for tile_q in hl.tile(query_len, block_size=q_block_size):
# src[helion_unified_attention.py:141-252]: ...

for offset_9 in tl.range(0, v_0.to(tl.int64), _BLOCK_SIZE_0, loop_unroll_factor=2, num_stages=2, disallow_acc_multi_buffer=False, flatten=False):

    indices_9 = offset_9 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int64)

    # src[helion_unified_attention.py:174]: for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
    # src[helion_unified_attention.py:174-244]: ...

    for offset_10 in tl.range(0, v_19.to(tl.int64), _BLOCK_SIZE_1, loop_unroll_factor=1, num_stages=1, disallow_acc_multi_buffer=True):

        indices_10 = offset_10 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int64)
        mask_1 = indices_10 < v_19
```

如上所示，这使用了由 Helion 自动调优器确定的 `pid_type="flat"` 程序启动版本。在这种程序类型中，内核只有一个"真实"的 PID（`tl.program_id(0)`），并从中推导出所有其他"局部" id。

### Helion 中的分块

一般来说，Helion 要求分块必须是通用的，即我们不能假设它们始终对应于 vLLM 的 KV 缓存块大小，因此需要相应地编写我们的程序。

Helion 中的分块功能非常强大，hl.tile 生成的分块会自动调整为不完整的分块。例如，如果分块大小为 32 而张量形状为 63，则第二个分块将只有 31 个元素，所有掩码都会自动生成。

然而，在分页注意力中存在额外的约束：我们始终必须从 KV 缓存中加载完整页面，因为在编译时无法确定是否需要完整页面。这对 Helion 来说不是问题，但这意味着我们需要创建自己的掩码。

分块的动态性还给我们带来了另一个问题：想象一个批次中查询长度为 7、2、1 的情况，如下图所示：

图 2：使用固定分块大小访问扁平化的"varlen"张量需要手动掩码。

因此，如果我们始终以相同的分块大小（本例中为 4）遍历，第二个分块中将会有第二个请求的词元，以及第一个请求的最后三个词元！同样，下一个分块也会混合第二和第三个请求的词元。然而，我们无法为批次中的每个序列更改分块大小，因为每个编译的内核只能有一个分块大小（记住，这是编译时常量）。一种解决方案是在这里只使用块大小为 1，但从我们在 vLLM 中开发 Triton 注意力后端的经验来看，这通常性能很差。

因此，唯一的其他选择是调整分块的索引并使用 hl.load 进行手动掩码。

```python
adjusted_tile_q_index = query_start + tile_q.begin + hl.arange(q_block_size)
query_head_offset = tile_m.begin + hl.arange(num_queries_per_kv)
q_load_mask = adjusted_tile_q_index[:, None, None] < query_end

# (tile_q, tile_m, HEAD_SIZE)
q = hl.load(t_query,
            [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
            extra_mask=q_load_mask)
```

总体而言，Helion 中这个内核实现需要 133 行代码（vLLM 格式含注释），而 Triton 需要 295 行。请在 Helion 内核实现代码处查看。

对我们来说，尽管由于 Helion 的编程模型我们不得不进行算法上的修改，但与编写 Triton 版本相比，编写 Helion 内核非常简单。在 Triton 中，大量的时间和精力（以及代码行数）被用于确保所有分块具有正确的掩码和边界，这需要手动处理所有维度的张量步幅和偏移。Helion 会自动且正确地完成这些操作，为开发者节省了大量精力（尤其是调试时！）。

此外，Helion 还通过自动调优处理其他底层细节，如实际的启动网格实现、分块大小发现或张量内存分配。

### 自动调优

Helion 的优势之一和核心功能是自动调优。它不仅可以搜索各种调优旋钮，还能自动检测这些旋钮的所有可能有效值。用户只需定义分块或块大小的上下界。也可以这样做：

```python
for tile_n in hl.tile(seqlen//page_size, block_size=None):
```

这与 Triton 形成强烈对比，在 Triton 中，用户需要列出所有可能的有效配置（通常通过大量嵌套的 `num_stages=s` for s in [1, 2, 3, 4, 5, 6, 8, 42] ... 来完成）。除了不那么友好的 API 之外，这还存在用户容易遗漏某些在特定平台上非常有效的组合的风险。Helion 的方法通过要求用户仅在符号层面上定义形状，然后推导出所有可能的组合来解决这个问题。

如果不是最外层的 hl.tile 循环（它将成为启动网格），分块的边界可以从张量形状中推导出来。

Helion 还会检查自动调优过程中创建的每个内核变体的正确性（与默认配置相比），并丢弃所有数值误差过高的实验。在我们实验的初期，这个基线（或 Helion 称之为"默认配置"）是自动推导的，取搜索空间的"中间"位置。然而，这种自动发现给我们带来了一些问题，因为结果是一个对我们内核无效的配置，无法与 vLLM 的页大小 16 配合使用。因此，自动调优完全无法工作，我们不得不为 Helion 打补丁并手动定义默认配置。

是的，Helion 仍然只是 beta 版并在积极开发中，这个问题后来通过允许用户将外部函数定义为自动调优基线来修复：`autotune_baseline_fn=callable()`。这个功能解决了我们的问题，我们可以将现有的 Triton 实现定义为基线，因为我们知道它能给出非常好的性能和正确的结果。这大大简化和增强了我们的自动调优过程。

我们非常欣赏的另一个功能是自动调优的"工作量级别"，这是 Helion 添加的一个用户自定义设置：`autotune_effort=`，可以是 *'none'*、*'quick'* 或 *'full'*。根据我们在 vLLM 中开发 Triton 注意力后端的经验，我们知道由于分页内存和随之而来的有限有效配置数量，自动调优过程是受约束的，通常不值得进行数天的自动调优。因此，一旦此功能可用，我们将工作量设置为 quick。但与往常一样，"快速"和"慢速"是相对的，Helion 自动调优器的快速模式仍然需要 10 小时来调优我们的内核在 72 种不同场景下（如批大小、序列长度、头大小等）。这比"full"（或默认）设置所需的 25 小时更快，但可能没有我们希望的那么"快速"。

一旦自动调优完成，它还会打印出推荐的最佳配置：

```
[2752s] Autotuning complete in 2752.1s after searching 5353 configs.

One can hardcode the best config and skip autotuning with:

@helion.kernel(config=helion.Config(
     block_sizes=[32, 4], 
     indexing=['pointer', 'pointer', 'pointer', 'pointer', 
'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 
'tensor_descriptor'], 
l2_groupings=[2],
load_eviction_policies=['', '', '', '', '', 'last', 'last', ''], 
loop_orders=[[1, 2, 0], [1,0]], 
num_stages=6, num_warps=8, pid_type='flat', 
range_flattens=[None, True, True, True], 
range_multi_buffers=[None,  None, None, False], 
range_num_stages=[], range_unroll_factors=[0, 1,  2, 1], 
range_warp_specializes=[]), static_shapes=False)
```

这再次展示了 Helion 可以探索的所有不同配置旋钮。因此，Helion 在不同自动调优算法上投入大量精力是重要且有意义的。目前，遗传算法被用作默认算法。

在上面的示例中，Helion 找到了 32 个词元（TILE_Q）和 4 页（TILE_N，相当于 64 个词元）作为最佳分块大小。它还确定了如何寻址所涉及的所有张量，以及循环是否应该被展开和重新排序。

所有旋钮的详细讨论超出了本博客文章的范围。

我们在调优 Triton 内核方面的经验告诉我们，一个好的权衡是在广泛的场景（微基准测试设置）下进行调优，然后选择少数几个配置用于 vLLM，并使用决策树或其他启发式规则在它们之间进行选择。

然而，当前版本 Helion 的一个缺点是它期望整个内核有一个配置，我们无法像在 Triton 中那样区分对预填充批次或解码批次有利的配置。因此，在本博客的实验中，我们确定了 6 种配置，在 NVIDIA 平台上部署 vLLM 时进行"实时"调优，AMD GPU 为 7 种。

## 性能评估

良好的开发体验固然不错，但如果性能比 Triton 差，我们可能仍然不会使用 Helion。因此，我们在 NVIDIA H100 和 AMD MI300X 上对我们在 Helion 中的新分页注意力与我们成熟的 Triton 内核进行了基准测试。对于推理服务器等用例，我们始终需要关注两个方面：内核单独的性能表现，以及新内核如何影响完整系统（即 vLLM）。因此，我们首先在微基准测试中单独分析内核性能，然后在第二步中进行端到端分析。

图 3：H100 上的微基准测试。延迟按批次中解码请求的比例排序。批次中包含的序列具有可变长度，中位数为最大序列长度的 40%，这可能出现在实际的在线推理场景中。批次中的词元总数表示为 x 轴。

图 4：H100 上的微基准测试。设置和结果与上图相同，但延迟按批次中的最大序列长度排序，包含预填充、部分预填充和解码的组合。

从 H100 的结果可以看出，我们的 Helion 分页注意力实现在解码方面已经超越了 Triton 2D 注意力内核，在预填充的大批次上也持平。我们的数据显示，对于预填充请求，Helion 内核的性能相对于 Triton 在 29% 到 137% 之间变化，对于仅解码请求则在 132% 到 153% 之间。此外，图表显示 Helion 静态形状变体与不使用静态形状的变体之间几乎没有差异。这一事实对端到端测量很重要，将在下一节中讨论。

预填充中的差距可以用 Helion 内核相对于 Triton 内核较小的启动网格来解释。如上所述，Helion 内核只在查询头和批次维度上并行化，而不沿查询本身并行化。这与 Triton 2D 内核形成对比，在 Triton 2D 内核中，两个维度是批次维度（与 Helion 相同）以及查询头和查询词元的混合作为第二维度。因此，优化 Helion 内核的启动网格是另一个优化方向。

图 5：MI300X 上的微基准测试。延迟按批次中解码请求的比例排序。批次中包含的序列具有可变长度，中位数为最大序列长度的 40%，这可能出现在实际的在线推理场景中。批次中的词元总数表示为 x 轴。

图 6：MI300X 上的微基准测试。设置和结果与上图相同，但延迟按批次中的最大序列长度排序，包含预填充、部分预填充和解码的组合。

对于 MI300X，结果略有不同，预填充的差距更大。在这里，Helion 内核相对于 Triton 内核的性能，对于预填充请求在 13% 到 75% 之间变化，对于仅解码批次在 58% 到 107% 之间变化。

然而，在这里 Helion 内核对于仅解码请求也持平或超越了 Triton 内核，因此我们可以认为我们的内核实现是平台和性能可移植的。

### vLLM 端到端测试

作为 vLLM 的忠实拥趸，我们当然希望评估我们的 Helion 分页注意力算法在实际且相关的端到端场景中的表现。

因此，我们编写了一个"helion_attn"后端，类似于我们的 Triton 注意力后端，并评估了其性能。我们也提交了该草案 PR。

为了评估，我们使用了 vLLM 内置的服务基准测试脚本和流行的 ShareGPT 基准测试。我们也禁用了前缀缓存。

```bash
VLLM_ATTENTION_BACKEND=EXPERIMENTAL_HELION_ATTN vllm serve \
     meta-llama/llama3.1-8b-instruct/ \
    --disable-log-requests --no-enable-prefix-caching

vllm bench serve --model meta-llama/llama3.1-8b-instruct/ \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --ignore_eos
```

这种设置"端到端"地评估 vLLM 推理服务器的性能，因为性能由客户端测量，客户端像真实用户一样向服务器发送请求。这意味着客户端也不假设对推理服务器状态的任何了解，即当前正在运行多少请求或请求应该具有什么"形状"才能获得良好的内核性能。在这些实验使用的特定基准测试中，客户端一次发送 1000 个请求，vLLM 服务器必须尽快处理它们，这包括调度非常大的批次，尤其是仅解码的批次。

我们将我们的实验性 Helion 注意力后端与当前 vLLM 中的 Triton 后端进行了比较。两个后端都对混合批次和仅解码批次使用完整的 CUDA/HIP 图。一个区别是，当前 vLLM 中的 Triton 注意力后端不进行"实时"调优，而是使用 if-else 语句在四种不同配置之间进行选择。这与我们的概念验证 Helion 注意力后端形成对比，后者在运行时使用自动调优器从 6 或 7 种配置中进行选择，具体取决于平台。为了对两种实现公平，我们始终进行两次预热基准测试运行，以允许自动调优器运行，并让 Helion 和 Triton JIT 编译器编译大多数相关内核版本。每张图显示三个结果：当前 vLLM 中 triton_attn 后端的性能作为基线，我们具有静态形状的 Helion 内核的性能，以及我们具有动态形状的 Helion 内核的性能。在这些实验中，我们也用 Triton 结果对结果进行了归一化。

图 7：在 H100 上使用 vllm bench 与 Llama3.1 8B 和 ShareGPT 数据集的端到端性能测量。

从图中可以看出，具有静态形状的 Helion 仅实现了 Triton 总吞吐量的约 26%，而具有动态形状的 Helion 实现了 96% 的总词元吞吐量，在 TTFT（首词元时间，即预填充时间）上也持平，在 ITL（词元间延迟，即解码下一个词元的时间）上也非常接近。

这个实验揭示了推理服务器的一个重要现实：请求形状多种多样、数量众多，且事先不可知。此外，调度的批次形状还取决于请求到达顺序等各种其他方面。因此，即使相同的基准测试作为预热运行过，使用具有静态形状的 Helion 也会对几乎每个请求触发重新编译，因为查询张量的形状很少完全相同。由于这是一个端到端实验，这些编译时间反映在测量的延迟和吞吐量中。这种评估内核实现性能的方式与微基准测试中查看原始内核性能的方式不同，但它反映了 vLLM 用户将实际体验到的真实世界性能。

因此，具有静态形状的 Helion 由于巨大的 JIT 开销超过了（小幅）内核运行时性能提升而表现不佳。请注意，由于在"实时"自动调优期间生成的 Triton 代码发生了不可恢复的崩溃，我们不得不在具有静态形状的端到端实验中禁用它，并使用了在微基准测试中确定的具有最佳解码性能的配置。这一实验限制可以解释静态形状和动态形状之间 TTFT 差距的一小部分，但无法解释 ITL 中的差距以及吞吐量的巨大差异。静态形状是默认启用的，允许 Helion 使用生成的 Triton 代码中硬编码的张量形状来优化性能。静态形状通常被提及为 Helion 中的性能优化，但对于像 vLLM 这样高度动态的使用场景，它们并不适用。

更令人惊讶的是微基准测试的相应结果：即使查看纯内核性能，优化静态形状与否之间几乎没有差异。我们怀疑，分页注意力内核的输入实际上是相当受形状约束的这一事实，大大促成了 Helion 中静态和动态形状编译之间最小的性能差异。例如，矩阵乘法的大小始终需要与 vLLM 的 KV 缓存页大小（或块大小）对齐，而编译时优化（如循环融合）无法改变这一点。

这些端到端结果中另一个令人惊讶之处是，在这个特定基准测试中，Helion 的 TTFT 实际上与 Triton 持平，因为这里的预填充批次比我们的微基准测试设置中更大且更均匀。

图 8：在 MI300X 上使用 vllm bench 与 Llama3.1 8B 和 ShareGPT 数据集的端到端性能测量。

在 MI300X 上，我们的后端同样可以工作，但使用动态形状仅实现了 59% 的总词元吞吐量。这个结果并不令人惊讶，因为我们的微基准测试已经显示，在 MI300X 上，Helion 内核和 Triton 内核之间对于预填充的差距更大。

**经验教训与总结**

我们享受这次实验，并将 Helion 视为既实用的语言又强大的自动调优器。总体而言，我们在这篇博客文章描述的实验上花费了不到三周的时间，并对令人印象深刻的结果感到非常惊喜。

当然，在 Helion 中进一步优化这个分页注意力实现还有很多选择。例如，平衡长预填充、长解码或非常大的批次。这需要在 Helion 中实现不同的启动网格或沿查询的并行化，以及进一步研究确定在不同内核版本之间选择的最优启发式规则，类似于在我们的 Triton 注意力后端中的实现方式。这种缺失的细粒度优化也解释了报告实验中 Triton 和 Helion 实现之间的性能差距。在最好的情况下，我们可以教会 Helion 自动调优器自动进行这种平衡（参见进一步的讨论）。由于这些权衡都依赖于平台，我们认为 Helion 的自动调优器非常适合可靠且快速地自动化这一过程。然而，在此背景下，我们还需要找到一个好的平衡点，即何时重新触发 Helion 调优和 JIT，以及何时使用启发式规则选择配置来实现已编译内核的低延迟执行（参见进一步的讨论）。

我们实验性 vLLM 后端的另一种可能优化是在 CUDA/HIP 图捕获期间使用静态形状，因为在那里额外的 JIT 开销无关紧要，而且记录的 CUDA/HIP 图的形状是静态的。因此，在这里让编译器考虑形状进行更激进的优化是安全的。然而，我们之后在运行时必须切换到动态形状。

在我们的实验过程中，我们意识到，对于 Helion 内核的开发，我们还需要一个广泛、自动化且可靠的微基准测试套件，以了解内核在大量用例中的详细性能。这类似于我们学习开发 Triton 内核的方式，因此，我们改编了最初为 Triton 工作而构建的微基准测试套件。

最有用的"Helion 命令"原来是 `tensor.view`，用于及早了解 Helion 编译器是否认为张量的形状与我们预期的相同。这使得调试只打印出符号形状的编译器错误变得容易得多。

最后，我们希望在 Helion 中添加一些预训练的启发式规则或决策树，在 vLLM 等低延迟场景中，在长时间自动调优和所有情况下只有一种配置之间取得平衡。

总之，我们认为 Helion 是 PyTorch 生态系统中一个令人兴奋且相当实用的补充，我们对它将如何影响 vLLM 感到好奇。

## 致谢

这项工作得到了 IBM Research AI 平台团队的支持，特别感谢我们的同事：Thomas Parnell、Jan van Lunteren、Mudhakar Srivatsa 和 Raghu Ganti。我们还要感谢 Jason Ansel 和 Meta 的 Helion 团队的反馈和支持，尤其是对我们报告的 bug 的快速修复，有时甚至在 24 小时内完成。

## 链接汇总

- Helion 介绍博客: https://pytorch.org/blog/helion/
- PyTorch 大会演讲（vLLM Triton 后端）: https://www.youtube.com/watch?v=5vmbRVXBvVM
- vLLM Helion 后端 PR#27293: https://github.com/vllm-project/vllm/pull/27293
- Triton 分页注意力详细论文: https://ibm.biz/triton-attention-anatomy
- 在 AMD GPU 上用 Triton 启用 vLLM v1 博客: https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
- Triton 自动调优局限性（triton-dejavu）: https://github.com/IBM/triton-dejavu
- Triton 自动调优对性能可移植性影响论文: https://ibm.biz/triton-autotuning-paper
- triton-dejavu 贝叶斯优化自动调优: https://github.com/IBM/triton-dejavu?tab=readme-ov-file#bayesian-optimization-to-speed-up-autotune-process
- 2D 统一注意力 Triton 内核源码: https://github.com/vllm-project/vllm/blob/5d6ce2b9601f3251487e44eb9e00c098101c4af6/vllm/attention/ops/triton_unified_attention.py#L57-L352
- Helion 内核实现代码（PR diff）: https://github.com/vllm-project/vllm/pull/27293/changes#diff-64a408b7cd48a2ab00590b745c0b2f71723be74f3ee491d2d76e029d22b0c71f
- Helion 自动调优平衡讨论 #1286: https://github.com/pytorch/helion/issues/1286
- Helion JIT 与启发式讨论 #1242: https://github.com/pytorch/helion/issues/1242
- Helion bug 24 小时修复 #1249: https://github.com/pytorch/helion/issues/1249
