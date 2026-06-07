# Up to 580tps! New Speed Record of Qwen3.5-397B-A17B on GPU for Agentic Workloads with TokenSpeed

**By:** TokenSpeed Team, Qwen Team
**Date:** May 27, 2026

## TL;DR

The TokenSpeed inference engine achieved a record-breaking 580 tps running the Qwen3.5-397B-A17B model on GPUs. This extreme performance for agentic workloads is driven by systematic elimination of memory copies, advanced kernel fusions, and fully overlapped CPU-GPU execution-keeping the GPU saturated at all times. On the functionality side, TokenSpeed also supports hybrid prefix caching and unified Prefill-Decode state transfers to handle complex agentic serving scenarios.

## 1. Introduction

The Qwen open-source models represent a highly capable family of large language models designed for broad accessibility and flexible deployment. They feature a comprehensive matrix of open-source versions with varying parameter sizes, catering to diverse scenarios from resource-efficient edge devices to complex cloud environments. Trained on extensive, high-quality corpora, these models demonstrate exceptional proficiency in natural language understanding, advanced logical reasoning, full-stack coding, and ultra-long context processing. Furthermore, with built-in support for autonomous agent planning, multi-step task execution, and tool calling, the Qwen open-source lineup empowers developers and researchers worldwide to efficiently build, customize, and deploy powerful AI applications.

Qwen3.5 models, the flagship of Qwen open-source lineup, push the boundaries further by adopting a **hybrid attention mechanism** that interleaves standard full attention layers with linear attention layers based on the Gated Delta Network (GDN). Unlike traditional pure-Transformer architectures, this hybrid design maintains strong modeling capabilities while significantly reducing computational complexity for long-sequence inference.

TokenSpeed is a high-performance, open-source LLM inference engine released by the LightSeek Foundation under the MIT license, purpose-built for agentic workloads. It aims to deliver "speed-of-light" performance comparable to TensorRT-LLM while maintaining the developer-friendly usability of vLLM. Built from the ground up with a native SPMD architecture and static compilation, it significantly accelerates the execution of complex multi-step agent tasks, empowering developers to efficiently deploy ultra-fast, production-grade AI applications.

This post presents the complete design, implementation, and optimization of Qwen3.5 models in the TokenSpeed inference framework, covering runtime architecture design (PD disaggregation, prefix caching, scheduler), key performance optimizations, and performance benchmarks.

## 2. Runtime Designs and Features

Qwen3.5 uses a hybrid architecture: most layers are GDN (linear attention with per-layer conv_state and temporal_state), with every N-th layer being standard full attention with a conventional KV cache. TokenSpeed provides full GDN-aware support across prefix caching, scheduling, and prefill-decode disaggregation, enabling efficient serving of the entire hybrid stack.

### 2.1 GDN/Mamba prefix cache

Prefix cache is critical to agentic workloads, where multi-turn tool-calling sequences frequently share long contexts and conversation histories. TokenSpeed's prefix cache is split across two layers. C++ owns the logical cache: radix-tree matching, page IDs, eviction, and Mamba slot lifetime. Python owns the physical tensors: GPU KV pages, Mamba `conv_state` / `ssm_state`, stream ordering, copy-on-write, zeroing, and snapshot copies.

For the normal KV cache, a prefix hit means reusing cached page IDs. For Mamba, that is not enough. A reusable prefix must also carry the recurrent state at the same prefix boundary. TokenSpeed solves this by attaching a `MambaSlot` to the same radix-tree node that represents the cached KV prefix.

#### Slot Lifecycle

Each active Mamba request may hold two slot types:

- `working` slot: mutable state used by the current forward step.
- `checkpoint` slot: snapshot destination that can later be published to the prefix tree.

The scheduler allocates these slots in C++, but Python writes the actual tensor contents.

A checkpoint slot becomes reusable only after two things are true: Python has populated it with a clean state, and C++ has attached it to a block-aligned radix-tree node.

#### Prefix Match and Copy-on-Write

When a future request matches the tree, `HybridPrefixCache` first performs the normal KV prefix match, then finds the nearest Mamba checkpoint node. If such a node exists, the scheduler returns `mamba_cow_src_index`.

Python then copies that cached checkpoint into the request's private working slot before running forward. The cached tree slot is not mutated; only the request's working slot changes.

#### Keeping Checkpoints Clean

The main correctness risk is stale data in reused slots: `MambaChunkAllocator` hands out integer slot IDs without clearing GPU memory. TokenSpeed prevents stale state through runtime rules.

Concretely, a newly allocated working slot is guaranteed safe in exactly two ways: it either receives a copy-on-write copy from a known-clean checkpoint, or Python explicitly zeroes it before use. Checkpoints are published only at aligned boundaries, so the tree never advertises arbitrary intermediate state as reusable prefix state.

#### Chunked Prefill Under Overlap Scheduling

Chunked prefill introduces a subtlety in overlap mode: the CPU may schedule the next chunk before it commits the previous chunk's output. The checkpoint is still safe because of CUDA stream ordering.

The previous chunk's Mamba forward writes the checkpoint on `execution_stream`. At the start of the next loop iteration, the default stream waits for `execution_stream`. Only after that does C++ insert the previous chunk into the tree and detach its checkpoint slot. The next chunk then gets a fresh checkpoint slot.

The key invariant is:
_C++ may publish the checkpoint slot ID during overlap scheduling, but any later GPU consumer is ordered after the previous chunk's snapshot write, and the published slot is no longer reused as the next checkpoint destination._

#### Decode Overlap

Decode has a different hazard: the next decode may mutate the same working slot before the CPU has committed the previous result. TokenSpeed handles this by snapshotting block-aligned decode states before dispatching the next decode.

This preserves the clean state before the working slot advances.

#### Summary

TokenSpeed's Mamba prefix cache is safe because it treats Mamba state as a tree-owned checkpoint, not as an incidental side effect of a request. C++ controls when a slot becomes part of the prefix tree. Python controls when the tensor contents are copied, zeroed, and snapshotted. Together they maintain one central invariant: every Mamba slot reachable from the prefix tree contains a clean, aligned state for the prefix represented by that tree node.

### 2.2 Scheduler

The hybrid architecture places unique requirements on the scheduler: it must simultaneously manage KV Cache (full attention layers) and Mamba State (linear attention layers) as two separate resource pools.

#### Mamba State and Hybrid Model Management

TokenSpeed's scheduler implements the following key mechanisms:

- **Dual Resource Pool Management:** Each request holds both KV Cache block indices and Mamba Pool slot indices (`mamba_pool_indices`), with the scheduler managing allocation and release for both.
- **State Lifecycle:**
  - On request arrival: allocate mamba_pool slot
  - During prefill: populate initial state (or load from prefix cache)
  - During decode: update state in-place each step
  - On completion or preemption: release slot
- **Speculative Decoding Support:** The scheduler maintains intermediate state cache (`spec_cache`) storing Conv/SSM state snapshots for each speculative step, enabling rollback upon verification failure.
- **Layer-Level Routing:** `HybridLinearAttnBackend` routes forward calls to the appropriate backend (full attention or linear attention) based on `layer_id`, with separate metadata initialization for each backend type.

### 2.3 GDN PD

#### 2.3.1 The Challenge

For hybrid models, mamba layers maintain state tensors beyond conventional key-value pairs. These states must be transferred from prefill nodes to decode nodes alongside KV caches, requiring correct layer-wise alignment between full-attention and Mamba layers.

#### 2.3.2 What We Built

We introduce end-to-end Mamba cache support for PD disaggregation, including:

**1. Unified State Transfer: Two Worlds, One Wire**

The core insight is that Mamba states, despite their different semantics, can be transferred using the same RDMA machinery as KV caches — as long as the system knows how to address them.

We designed a dual-tensor pool on each node: one pool holds convolutional states (the short-term memory of causal convolutions), the other holds recurrent SSM states (the long-term compressed history). Both are pre-allocated as contiguous GPU memory, with each request owning exactly one slot per layer. At registration time, the prefill and decode nodes exchange buffer descriptors — base addresses, per-slot sizes, and a mapping from each physical buffer to its corresponding global layer ID.

When transfer begins, the system maps each request's slot indices into physical byte offsets, groups contiguous slots into scatter-gather blocks, and issues them as bulk RDMA writes. From the network's perspective, Mamba states are just another set of memory regions — no serialization, no intermediate staging. The key difference is the addressing: KV caches are indexed by page tables, while Mamba states are indexed by flat slot IDs assigned by the scheduler.

**2. Cross-Layer Scheduling: A Unified Heartbeat**

The most subtle piece of the puzzle is **when** to transfer each layer's state.

In layerwise transfer mode, the prefill node doesn't wait for the entire forward pass to finish before starting data movement. Instead, it begins shipping data as soon as each layer group completes — overlapping computation with communication. But for a hybrid model, this means the transfer thread must track progress across both attention layers and Mamba layers as if they were one continuous pipeline.

We introduced a **unified step counter** that ticks once after every layer's forward pass — regardless of type. The transfer thread watches this counter and, for each layer window, sends whichever data belongs to that window: KV pages for full-attention layers, state slots for Mamba layers. The model's layer-type pattern becomes invisible to the transfer logic — it simply asks "which buffers map to layers 4 through 7?" and sends them all once the counter reaches 7.

On the decode side, the mirror of this mechanism is a **layer-done barrier**: the model forward can begin executing layer 0 before layer 15's state has arrived. Each layer's computation calls into the state pool, which blocks only if that specific layer hasn't been loaded yet. This allows decode to overlap network reception with early-layer execution, hiding transfer latency behind useful work.

**3. PD-Aware Token Lifecycle: The Three-Phase Handshake**

The final piece connects state transfer to the token generation lifecycle. In a disaggregated system, the prefill node doesn't just produce states — it also produces the first output token. The decode node needs both before it can begin generation.

We designed a three-phase handshake:

1. **Transfer completes:** All KV pages and Mamba states for the final layer group are shipped. But the transfer thread doesn't declare success yet — it holds at a barrier, waiting for the forward pass to finish.
2. **Token produced:** The prefill forward completes and emits the first output token. The event loop records this token and signals the waiting transfer thread.
3. **Status delivered:** The transfer thread sends a lightweight status message (carrying the bootstrap token) to the decode endpoint via a side channel. Only when the decode node receives both the bulk state data and this token does it emit a "remote prefill done" event to its scheduler.

This protocol ensures an invariant: **the decode node never begins generation with incomplete state, and never wastes a step re-deriving the first token**. The Mamba states, the KV cache, and the bootstrap token arrive as a logically atomic unit — even though they travel through different paths and at different times.

## 3. Performance Optimizations

### 3.1 Mamba State Update Optimization

#### Eliminating Mamba State Copies with Index Indirection

In speculative decoding with Mamba-style linear attention, the target-verify phase traditionally carries a hidden memory cost. After the draft model produces speculative tokens, the base model runs forward to validate them. Because each draft token advances the Mamba state by one step, the engine needs to preserve intermediate states for every speculative position, then recover the correct one based on how many tokens were accepted.

The previous pipeline handled this with a dedicated intermediate state cache: the kernel wrote per-step Mamba states into a side buffer during verify, and a post-verify `fused_mamba_state_scatter_with_mask` kernel copied the state at the accepted position back into the scheduler-owned working slot. The scatter itself was a full tensor copy across `num_layers × state_dim` —not free, and executed on every decoding step.

#### The Core Idea: Move Pointers, Not Data

Instead of buffering intermediate states in a separate cache and scattering the accepted one afterward, we let the kernel write each step's output directly to a dedicated physical row, then simply remember _which_ row holds the canonical state.

The state buffer is extended with a **draft region** appended after the scheduler-allocated base slots; each request owns a private slice of draft rows indexed by its `req_pool_index`. A lightweight table `current_input_indices` records, for each request, which physical row currently holds its canonical Mamba state.

During target-verify:

- **Input redirection:** The kernel reads its initial state from the row recorded in `current_input_indices` (which may be a working slot, a COW-forked slot, or a draft row from a previous step). No data movement happens here—only an index lookup.
- **Output routing:** A per-request `output_state_indices` tensor tells the kernel exactly where to write each step's output: slot 0 is the working row, slot 1..N are the request-private draft rows. The kernel writes directly into these pre-assigned locations, eliminating the intermediate cache entirely.
- **Post-verify bookkeeping:** Once the accepted length is known, we simply update `current_input_indices[req]` to point at the draft row corresponding to the last accepted token. This is an **O(1) integer write**, not an O(L·D) tensor copy.

### 3.2 Runtime Optimization

#### 3.2.1 Overlap Is All You Need

Following common practice in modern inference engines, TokenSpeed employs CUDA multi-stream parallelism to overlap non-sequential operations. By executing independent workloads concurrently across multiple streams, TokenSpeed effectively reduces scheduling overhead and improves end-to-end latency.

**Shared Expert and Routed Expert Overlap**

Qwen3.5 MoE layers contain shared experts and routed experts. Shared experts process all tokens while routed experts only handle TopK-selected tokens. The two are naturally parallelizable and are implemented via the `StreamFork` class for stream forking and synchronization:

1. Main stream executes TopK routing, expert dispatch, and MoE GEMM
2. Auxiliary stream concurrently executes shared expert forward (gate_up → SiLU → down) and sigmoid gating
3. Both streams synchronize via events before combining results

This overlap hides shared expert computation latency, reducing single MoE layer time in production deployments.

**GDN Input Projection Dual-Stream Optimization**

The GatedDeltaNet layer's input projection contains two independent linear layers (`in_proj_qkvz` and `in_proj_ba`), also executed in parallel across streams.

This optimization is only activated during CUDA Graph capture, where the smaller `in_proj_ba` projection is fully hidden behind the larger `in_proj_qkvz` on the alternate stream.

#### 3.2.2 The More You Fuse, The Less Latency You Get

**Gemma AllReduce Fusion**

GemmaRMSNorm uses `x * (1 + weight)` instead of standard RMSNorm's `x * weight`, which previously prevented use of TRT-LLM's fused AllReduce + Residual + RMSNorm kernel.

TokenSpeed pre-computes `gemma_weight = weight + 1.0` and passes it as the gamma parameter to the standard fused kernel — to enable GemmaRMSNorm communication fusion. After fusion, AllReduce + residual addition + RMSNorm per layer is merged from three separate kernel launches into one:
This fusion covers all Qwen3.5 decoder layers and auto-enables on SM90+ single-node TP deployments.

**Fused QK-RMSNorm + Partial RoPE + Gate Split in Attention**

In the original attention path, after the QKV GEMM projection, 5 separate kernels are launched sequentially to normalize, rotate, and split the Q/K/gate vectors:

| Step | Operation | Read from HBM | Write to HBM |
|------|-----------|---------------|--------------|
| 1 | Q RMSNorm | q | q_normed |
| 2 | K RMSNorm | k | k_normed |
| 3 | Q RoPE | q_normed | q_rotated |
| 4 | K RoPE | k_normed | k_rotated |
| 5 | Gate split + contiguous copy | q_gate | gate |

Each intermediate tensor (`q_normed`, `k_normed`, etc.) is written to global memory only to be immediately read by the next kernel — pure bandwidth waste. `fused_qk_rmsnorm_rope_gate` replaces all 5 launches with a single Triton kernel. All intermediate values stay in registers.

**Fused Gate-Sigmoid-Mul-Add in MoE Shared Expert**

In the MoE block, the shared expert output is gated before merging with routed expert output. The original code path launches 5 separate kernels for what is conceptually one expression:

| Step | Kernel | Note |
|------|--------|------|
| 1 | Elementwise multiply | `h[i] * w[i]` — per-element products |
| 2 | Reduce | Sum partial products → 1 scalar per token |
| 3 | Sigmoid | `σ(gate_val)` — elementwise on the scalar |
| 4 | Multiply | `σ(gate_val) * shared_output` — broadcast scalar × full vector |
| 5 | Add | `final_hidden_states += scaled` |

The key inefficiency: `gate_val` is a scalar per token, yet the unfused path materializes both the per-element products and the reduced scalar to HBM between launches. The intermediate `scaled` tensor (full `[num_tokens, hidden_dim]`) is also written and immediately re-read. `fused_gate_sigmoid_mul_add` computes the full expression `final += σ(x·w) * shared` in one Triton kernel, in-place. The dot-product reduction, sigmoid, broadcast multiply, and accumulate all happen within a single thread block per token — intermediates never leave registers.

#### 3.2.3 Death by a Thousand Syncs

TokenSpeed's decode loop captures the core forward pass — target model, sampler, and draft model — into a single CUDA graph. Once captured, thousands of GPU kernels replay with one launch, eliminating per-kernel dispatch overhead entirely.

But CUDA graphs are static by design. Between graph replays, the runtime must still perform dynamic work on the host: preparing inputs, resolving scheduling indices, updating Mamba state pointers after speculative verification, and coordinating transfer state. These "gaps" between graphs are where CPU overhead hides — and where a careless .item() or an unnecessary D2H copy can stall the entire pipeline.

TokenSpeed treats this inter-graph CPU overhead as a first-class optimization target: keep the host out of the critical path, even outside the graph.

**Eliminating Device-to-Host Round-Trips**

The most insidious sync pattern is the "innocent query" — reading a single scalar from GPU to make a branching decision on the host. TokenSpeed replaces these with pre-computed worst-case bounds known at initialization, or captures CPU-side maximums before H2D transfer so both the GPU tensor and its bound are available simultaneously. For speculative decoding state management, boundary detection and slot selection use GPU-side sentinel values — downstream kernels skip invalid entries via bounds checks rather than CPU-side filtering. The entire decision tree stays on device.

**Compile-Fused Index Arithmetic**

Runtime scheduling in hybrid models involves heavy index manipulation: computing slot mappings, draft-token layouts, and pointer updates after verification. In eager PyTorch, each step becomes a separate kernel launch with intermediates written to HBM. TokenSpeed annotates these routines with torch.compile, allowing Inductor to fuse 10–14 individual launches into one or two elementwise kernels where all values flow through registers. The GPU stays busy, and the CPU submits one launch instead of fourteen.

**Asynchronous Everything**

H2D transfers use pinned memory with non-blocking copies throughout. The transfer system polls pinned-host counters instead of calling synchronize(), and layer-wise loading uses event-based barriers that wake only the specific layer that needs data. The CPU prepares the next batch while the current one is still in flight.

The cumulative effect: TokenSpeed's decode loop maintains near-zero CPU overhead — the host thread spends its time submitting work, not waiting for results.

### 3.3 FA4 Support

Flash Attention 4 (FA4) is the next-generation attention kernel targeting NVIDIA Blackwell architecture. Qwen3.5 uses head_dim=256 by default, placing substantial demands on attention compute backends — a configuration that not all kernels support efficiently out of the box.

Support for FA4 with head_dim=256 has been contributed and merged into the upstream community repository. In TokenSpeed, native FA4 support for Qwen3.5 is currently under active development and will be available in an upcoming release, further unlocking the full compute potential of Blackwell GPUs for Qwen3.5 inference.

## 4. Benchmark

Taking Qwen3.5-397B-A17B as a representative example, we present a systematic performance evaluation of Qwen3.5 models on NVIDIA Blackwell GPUs. We would like to thank the EvalScope team for providing the benchmarking tool; all performance results reported below are obtained using EvalScope Benchmark.

**Test Environment**: All benchmarks were conducted using the TokenSpeed latest Docker image (lightseekorg/tokenspeed-runner:latest), based on the recent version. The benchmark scripts and reproduction instructions are available at TokenSpeed's GitHub repository.

### 4.1 Basic Benchmark

We use fixed input/output lengths and measure decode throughput (output token/s). We evaluated performance across varying batch sizes under different parallelism configurations (TP/EP). Two primary test configurations were used:

- Config 1: Attn TP + MoE TP
- Config 2: Attn TP + MoE EP

We benchmarked Qwen3.5-397B-A17B-NVFP4 decode throughput on B200 with MTP enabled and disabled.

Across all input/output length configurations on Attn TP8 + MoE TP8 / Attn TP8 + MoE EP8, MTP delivers +100%~+159% throughput gains at bs=1, where latency is the primary bottleneck. At higher concurrency, the gain is strongly correlated with output length: long-output workloads (e.g., output length >4096) sustain substantial speedups of +38%~+90% at bs=32 / 64, while short-output workloads (e.g., 1024 tokens) at bs=64 see gains diminish to near-zero or turn slightly negative, as speculation overhead begins to outweigh acceptance benefits when decoding is already throughput-bound.

### 4.2 Agentic Workload Benchmark

The rapid proliferation of Agent applications — encompassing tool call histories and multi-turn dialogue context — has fundamentally reshaped the characteristics of production workloads. To reflect real-world agent behavior, we use the Agentic Workload test suite that simulates realistic agent call patterns (50K first-turn context, 800 tokens appended per subsequent turn, 10-15 turns total).

On B200 with NVFP4, TokenSpeed delivers exceptional single-user throughput for Qwen3.5-397B-A17B under agentic workloads. All four parallelism configurations — TP4, TP4EP4, TP8, and TP8EP8 — sustain 500+ tok/s at bs=1, with TP8 achieving a peak of ~580 tok/s.

At concurrent=16, the TP4 family scales to ~2K tok/min/GPU system throughput while the TP8 family reaches ~1K tok/min/GPU. Pure-TP and TP+EP configurations within the same GPU count exhibit comparable throughput-latency tradeoffs, giving users deployment flexibility without sacrificing performance. Notably, the multi-turn agentic workload achieves an average KV cache hit rate exceeding 90%, significantly reducing prefill overhead and contributing to the overall throughput gains.

### 4.3 Up-to-1M Long Context Benchmark

Long-context handling is another key challenge in Agent workloads. While Prefix Cache can hit large amounts of repeated prefixes across multi-turn conversations and significantly reduce Prefill overhead, the Decode stage still has to read and attend to the full historical KV at every step, which the cache cannot bypass — the longer the context, the higher the per-step Decode memory-access cost.

Based on the NIAH (Needle-in-a-Haystack) 1M sample, we sliced four prompt lengths — 128K / 256K / 512K / 1M — for evaluation. On Qwen3.5-397B-A17B, decode throughput remains at ~530 tok/s/user within 128K, ~495 at 256K, and ~445 at 1M (measured on TP8), giving an end-to-end degradation of only ~16% from 128K to 1M — long-context throughput decay is kept well under control.

## 5. Conclusion

Through the optimizations and architectural designs described above, TokenSpeed delivers outstanding performance for the Qwen3.5 models — particularly in agentic workloads, achieving ultra-low latency generation and high inference throughput. TokenSpeed will continue to push the boundaries of Qwen inference optimization, pursuing ever more extreme performance at every level of the stack.

We invite you to follow the TokenSpeed project and experience speed-of-light inference throughput for yourself. A complete installation guide is available on GitHub, making it straightforward to deploy and benchmark on supported hardware. We also warmly welcome performance-oriented pull requests from the community — every contribution helps the Qwen model series run faster and smarter.

## 6. Acknowledgements

This work was made possible through close collaboration across the open-source ecosystem. We would like to thank Alibaba Tongyi Team, NVIDIA DevTech, the Mooncake Team, and the LightSeek Foundation for their engineering collaboration and implementation support. We also thank NVIDIA and Verda for providing Blackwell GPU infrastructure and compute support.

---
Original Link: https://pytorch.org/blog/up-to-580tps-new-speed-record-of-qwen3-5-397b-a17b-on-gpu-for-agentic-workloads-with-tokenspeed/
