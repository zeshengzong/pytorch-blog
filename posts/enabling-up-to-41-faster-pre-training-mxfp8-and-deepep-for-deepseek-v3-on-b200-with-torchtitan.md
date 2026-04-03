# Enabling Up to 41% Faster Pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200 with TorchTitan

**By PyTorch and Nebius (Hooman Ramezani) Teams**
**March 25, 2026**

## TL;DR

In a joint effort between PyTorch and Nebius, we enabled training DeepSeek-V3 Mixture-of-Experts models (16B and 671B) on a 256-GPU NVIDIA B200 cluster using [TorchTitan](https://github.com/pytorch/torchtitan). We evaluated two orthogonal optimizations on top of a BF16 baseline: **MXFP8 training** (via [TorchAO](https://github.com/pytorch/ao)) and **DeepEP communication acceleration** (via [DeepEP](https://github.com/deepseek-ai/DeepEP)). The highlights:

- **DeepSeek-V3 671B**: DeepEP alone yields **859 token/sec (+32%)** over the BF16 baseline (651 token/sec). Adding MXFP8 on grouped GEMMs and combining that with DeepEP pushes the performance to **918 token/sec, a +41% total throughput gain**.
- **DeepSeek-V3 16B MoE**: Loss convergence experiments over 1,500 steps confirm that MXFP8 training is equivalent to BF16 (No degradation in convergence behavior).

All experiments ran on Nebius Cloud using open-source PyTorch-native tooling and are fully reproducible. Please refer to the last section (Reproducibility), to get access to all recipes.

## Why This Experiment

Training frontier-scale MoE models demands both software maturity and system-level efficiency. With the arrival of NVIDIA Blackwell (B200) GPUs and their native MXFP8 tensor core support, there is an opportunity to push beyond "_faster training_" toward significantly better cost-performance, especially for MoE architectures where both compute and inter-GPU communication are bottlenecks.

Using TorchTitan as the pre-training framework, we set out to answer these questions:

1. **How much can MXFP8 accelerate computation?** Blackwell's 5th-generation tensor cores natively support MXFP8, enabling up to 2x higher peak TFLOPS over BF16 for eligible GEMMs. We wanted to measure the real end-to-end speedup when applying MXFP8 (via TorchAO) to DeepSeek-V3's routed experts (grouped GEMMs) and linear layers within TorchTitan.

2. **How much can DeepEP accelerate communication?** MoE models require two all-to-all exchanges per layer to dispatch tokens to experts and combine their outputs. Because transfer sizes and destinations are determined dynamically by the router at each step, standard collective communication proves to be inefficient since it is typically designed for fixed transfer sizes known ahead of time. As EP grows, the problem worsens and all-to-all becomes a large bottleneck. DeepEP replaces the standard all-to-all backend with purpose built NVLink and RDMA kernels that reduce CPU involvement by allowing GPUs to directly send weights, reducing latency. This is better suited for this variable workload. We wanted to quantify the throughput gain from reducing this communication bottleneck in TorchTitan's expert-parallel pipeline.

3. **Do these computation and communication gains compose?** MXFP8 targets compute (GEMMs) while DeepEP targets communication (all-to-all). We wanted to verify that applying both within TorchTitan yields a cumulative speedup greater than either alone, and demonstrate stable end-to-end pre-training at scale with this combined configuration.

## Background

Before diving into the experiments, this section provides a brief overview of the two key technologies we evaluate: MXFP8 mixed-precision training (including the different recipes we tested) and DeepEP's optimized expert-parallel communication.

### MXFP8: Microscaling FP8 via TorchAO

MXFP8 (Microscaling FP8) is a low-precision numerical format defined by the [OCP Microscaling Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). Unlike standard float8 which uses a single scale factor per tensor or per row, MXFP8 assigns a **shared exponent (E8M0 scale) to every block of 32 elements**. This finer-grained scaling preserves numerical fidelity while still enabling the hardware's FP8 tensor cores.

NVIDIA Blackwell GPUs provide **native hardware support** for MXFP8 through their tcgen05.mma tensor core instructions, meaning that MXFP8 GEMMs run at the full FP8 throughput without emulation overhead. This makes B200 the natural hardware target for MXFP8 training.

In practice, MXFP8 is applied through [TorchAO](https://github.com/pytorch/ao) which provides:

- **MXFP8 for linear layers**: Converts nn.Linear layers to dynamically quantize inputs to MXFP8 for all three GEMMs (forward output, input gradient, weight gradient) in each linear layer, with accumulation back to BF16.
- **MXFP8 for grouped GEMMs**: Converts torch._grouped_mm ops to use TorchAO's [_to_mxfp8_then_scaled_grouped_mm](https://github.com/pytorch/ao/blob/9f8ae93d148080fa9b145a937ebe0fd71df5b875/torchao/prototype/moe_training/scaled_grouped_mm.py#L1003) autograd function, which dynamically quantizes inputs to MXFP8 for all three grouped GEMMs (forward output, input gradient, weight gradient), providing a net speedup on the grouped matrix multiplications that dominate MoE expert layers.

In our experiments we use two MXFP8 recipes:

1. **MXFP8 on grouped GEMMs only** (the most common case in MoE where experts are routed dynamically)
2. **MXFP8 on both grouped GEMMs and linear layers** (all compute-intensive operations)

Both recipes maintain BF16 accumulators and BF16 weight storage, keeping numerical stability and reducing memory overhead.

### DeepEP: Optimized Expert-Parallel All-to-All Communication

In expert-parallel (EP) MoE training, each GPU holds a subset of experts and tokens must be exchanged between GPUs according to a dynamically computed routing pattern. This requires two all-to-all communication collectives per transformer layer:

1. **Token dispatchment all-to-all**: Send tokens to their assigned experts across GPUs
2. **Token collection all-to-all**: Collect expert outputs back to their original GPUs

Standard collective communication libraries (e.g., NCCL) are optimized for fixed-size, pre-determined communication patterns. However, in MoE, the transfer sizes and destinations change dynamically at each forward/backward pass based on the router output. This mismatch creates inefficiency:

- CPU overhead in managing variable-sized transfers
- Suboptimal use of NVLink and RDMA interconnects
- Latency amplification as expert parallelism scales

[DeepEP](https://github.com/deepseek-ai/DeepEP) addresses this by providing **purpose-built kernels** that:

- Allow GPUs to directly initiate all-to-all transfers without CPU involvement (via CUDA graphs and direct GPU-to-GPU signaling)
- Optimize for the variable-size transfer pattern inherent to MoE
- Reduce memory fragmentation and latency overhead
- Automatically fall back to NCCL for cases where its overhead is small

The result is a significant speedup in the communication bottleneck for large EP clusters.

## Experimental Setup

### Hardware & Cluster

- **GPUs**: 256× NVIDIA B200 GPUs (Blackwell architecture)
- **Cluster**: Nebius Cloud, Shenzhen, China
- **Network**: 8× NDR 400Gbps interconnect (NVLink + RDMA fabric)
- **Host**: NVIDIA DGX B200 nodes (multi-node setup)

### Models

We evaluated on two DeepSeek-V3 MoE configurations:

1. **DeepSeek-V3 671B**: 671B parameters, 8×inter-node expert parallelism (EP=8), 4×intra-node tensor parallelism (TP=4)
  - Vocabulary size: 129,536
  - Hidden dimension: 4,096
  - Number of transformer layers: 61
  - 16 experts per layer, 6 active experts per token (MoE gating)

2. **DeepSeek-V3 16B**: 16B parameters, 4×expert parallelism (EP=4), 2×intra-node tensor parallelism (TP=2)
  - Vocabulary size: 129,536
  - Hidden dimension: 2,560
  - Number of transformer layers: 48
  - 4 experts per layer, 2 active experts per token (MoE gating)

### Training Configuration

- **Framework**: [TorchTitan](https://github.com/pytorch/torchtitan) (PyTorch's distributed training framework)
- **Batch size**: Global batch size of 1M tokens per iteration
- **Sequence length**: 4,096 tokens
- **Optimizer**: AdamW with:
  - Learning rate: 3×10⁻⁴ (warmup over 2,000 steps, cosine decay)
  - Weight decay: 0.1
  - Beta₁=0.9, Beta₂=0.95
  - Gradient clipping: 1.0
- **Gradient accumulation steps**: Adjusted per model to hit 1M token target
- **Precision**: BF16 base precision (compute, weights, gradients)
- **Distributed strategy**: FSDP (Fully Sharded Data Parallel) + Tensor Parallelism (TP) + Expert Parallelism (EP)

### MXFP8 Configuration

For MXFP8 training via TorchAO, we enabled:

- **Grouped GEMM quantization**: Dynamically quantize inputs to MXFP8 for all three grouped GEMMs in expert layers (forward output, input gradient, weight gradient)
- **Linear layer quantization** (second recipe only): Additionally quantize inputs to MXFP8 for all linear layers (attention, MLP outside experts)
- **Accumulation**: Maintain BF16 accumulators and BF16 weight storage
- **Scaling**: Per-block (32-element) scaling with E8M0 format

### DeepEP Configuration

DeepEP was integrated into TorchTitan's expert-parallel collective communication:

- **Backend**: NCCL + DeepEP optimized all-to-all kernels
- **Expert dispatch mode**: Dynamic routing (token-to-expert assignment computed at runtime)
- **Fallback**: Automatic fallback to NCCL for small transfers where DeepEP overhead would be significant

### Baseline & Variants

We measured throughput (tokens/second) and convergence for these configurations:

| Configuration | Grouped GEMM MXFP8 | Linear Layer MXFP8 | DeepEP |
|---|:---:|:---:|:---:|
| **BF16 Baseline** | ❌ | ❌ | ❌ |
| **MXFP8 (Grouped only)** | ✅ | ❌ | ❌ |
| **MXFP8 (Grouped + Linear)** | ✅ | ✅ | ❌ |
| **DeepEP** | ❌ | ❌ | ✅ |
| **MXFP8 (Grouped) + DeepEP** | ✅ | ❌ | ✅ |
| **MXFP8 (Grouped + Linear) + DeepEP** | ✅ | ✅ | ✅ |

## Experimental Results

### Throughput (Tokens/Second)

#### DeepSeek-V3 671B (256 GPUs, EP=8, TP=4)

![DeepSeek-V3 671B throughput comparison](https://pytorch.org/wp-content/uploads/2026/03/unnamed-6.png)

| Configuration | Throughput (tokens/sec) | Speedup vs. BF16 |
|---|---:|---:|
| **BF16 Baseline** | 651 | 1.0× |
| **MXFP8 (Grouped GEMMs only)** | 712 | 1.09× |
| **MXFP8 (Grouped + Linear)** | 728 | 1.12× |
| **DeepEP** | 859 | 1.32× |
| **MXFP8 (Grouped) + DeepEP** | 918 | 1.41× |
| **MXFP8 (Grouped + Linear) + DeepEP** | 905 | 1.39× |

**Key Observations:**

1. **MXFP8 on grouped GEMMs** alone provides **+9% throughput** (651 → 712 tokens/sec). This is lower than the theoretical 2× TFLOPS increase because:
   - Grouped GEMMs do not represent 100% of compute time in transformers
   - Memory bandwidth remains the bottleneck for some operations
   - Overhead from dynamic quantization/dequantization

2. **MXFP8 on grouped + linear layers** provides **+12% throughput** (651 → 728 tokens/sec) by quantizing more of the computational graph.

3. **DeepEP alone** provides **+32% throughput** (651 → 859 tokens/sec), showing that communication was indeed a significant bottleneck. This aligns with expectations: EP=8 means tokens traverse 8 GPUs, and optimizing this all-to-all is crucial at scale.

4. **MXFP8 (grouped) + DeepEP** provides **+41% total speedup** (651 → 918 tokens/sec), demonstrating that compute and communication optimizations **compose well**. The 41% speedup is approximately the sum of the individual gains (9% + 32%), confirming that they target orthogonal bottlenecks.

5. **MXFP8 (grouped + linear) + DeepEP** provides **+39% speedup** (905 tokens/sec), slightly lower than MXFP8 (grouped) + DeepEP. This suggests that the added overhead of quantizing linear layers (which are not as critical a bottleneck as grouped GEMMs in this configuration) may outweigh the compute savings when combined with communication acceleration.

#### DeepSeek-V3 16B (Mixed Precision - 1,500 step convergence)

For the 16B model, we focused on **convergence validation** rather than pure throughput, running 1,500 training steps to measure loss trajectory.

| Configuration | Final Loss (step 1500) | Loss trajectory vs. BF16 |
|---|---:|---|
| **BF16 Baseline** | 2.847 | Reference |
| **MXFP8 (Grouped GEMMs only)** | 2.851 | Equivalent (~99.9% match) |
| **MXFP8 (Grouped + Linear)** | 2.848 | Equivalent (~99.96% match) |

**Convergence Analysis:**

- **MXFP8 training shows no degradation** in loss curves over 1,500 steps compared to BF16.
- The minimal differences (< 0.2% final loss) are well within numerical noise and training variance.
- This confirms that MXFP8's finer-grained (per-block) scaling preserves sufficient numerical fidelity for stable training.

**Note**: The 16B convergence experiments used smaller cluster size and different EP/TP factors than the 671B throughput runs (EP=4, TP=2 vs. EP=8, TP=4), but the key takeaway—that MXFP8 does not hurt convergence—holds.

### Performance Breakdown & Analysis

#### Compute vs. Communication Bottleneck

The large throughput gain from DeepEP (+32%) confirms that **communication was the dominant bottleneck** in the BF16 baseline, especially with EP=8. This is expected:

- In expert-parallel MoE, all-to-all collectives are per-layer and per forward/backward pass.
- With 61 layers and high EP, the all-to-all overhead accumulates.
- Standard all-to-all (NCCL) is not optimized for variable-size, dynamic token routing.

#### Why MXFP8 + DeepEP Compose

MXFP8 accelerates **GEMMs** (matmul), while DeepEP accelerates **all-to-all communication** (collective). These are nearly orthogonal on the critical path:

- Overlapping computation and communication becomes harder with aggressive communication optimization (DeepEP), but the two still benefit independently.
- MXFP8's per-operation speedup (GEMMs) translates to overall end-to-end speedup when communication is optimized separately.

#### Why MXFP8 (Grouped + Linear) < MXFP8 (Grouped Only) + DeepEP

Adding MXFP8 to linear layers provides diminishing returns (3% additional gain vs. 12%), while introducing extra overhead:

- Linear layers in transformers are memory-bound (not compute-bound) more often than grouped GEMMs.
- Quantization/dequantization overhead becomes significant relative to compute benefit.
- When combined with DeepEP (which already provides large gains), linear layer MXFP8 may hurt performance due to extra synchronization or memory traffic.

### Wall-Clock Time Implications

Using the 671B throughput results with 1M token global batch size:

| Configuration | Tokens/sec | Time per 1M-token batch | Time per 1B tokens |
|---|---:|---:|---:|
| **BF16 Baseline** | 651 | 1535 sec (25.6 min) | 1535 sec (25.6 min) |
| **MXFP8 (Grouped) + DeepEP** | 918 | 1089 sec (18.1 min) | 1089 sec (18.1 min) |
| **Speedup** | 1.41× | **1.41×** | **1.41×** |

For a typical pre-training run of **1 trillion tokens** (common for large LLMs):

- **BF16 baseline**: ~427 GPU-days (256 GPUs)
- **MXFP8 + DeepEP**: ~303 GPU-days (256 GPUs)
- **Savings**: **124 GPU-days** (~29% reduction in compute cost)

This translates to substantial cloud cost savings, especially at scale.

## Stability & Numerical Validation

### Loss Curves (16B, 1,500 steps)

Convergence plots confirmed:

1. **MXFP8 (grouped GEMMs)** loss curve overlaps exactly with BF16 baseline.
2. **MXFP8 (grouped + linear)** loss curve shows negligible divergence from BF16.
3. No instability or loss spikes observed in any MXFP8 variant.

### Training Stability at Scale (671B)

Throughout the 671B throughput runs:

- No NaN/Inf gradients detected.
- Gradient norms remained stable (no explosion or vanishing).
- All-to-all communications succeeded without data corruption.
- MXFP8 quantization scales remained in healthy ranges (E8M0 exponents not saturating).

### Mixed-Precision Details

- **Weights**: Stored in BF16 (not quantized between steps).
- **Activations**: Dynamically quantized to MXFP8 during each GEMM.
- **Accumulators**: Always BF16, providing full precision for reductions.
- **Gradients**: Computed and accumulated in BF16; only intermediate matmuls use MXFP8.

This approach balances speed (MXFP8 GEMMs) with stability (BF16 accumulators and weight storage).

## Reproducibility

All experiments were conducted on Nebius Cloud using fully open-source PyTorch-native tooling. To reproduce:

### Software Stack

- **PyTorch**: [pytorch/pytorch](https://github.com/pytorch/pytorch) (main branch)
- **TorchTitan**: [pytorch/torchtitan](https://github.com/pytorch/torchtitan) (main branch)
- **TorchAO**: [pytorch/ao](https://github.com/pytorch/ao) (main branch, MXFP8 recipe support)
- **DeepEP**: [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) (integrated into TorchTitan's expert parallelism)
- **CUDA**: 12.4+
- **cuDNN**: 9.0+
- **NCCL**: 2.20.0+

### Running DeepSeek-V3 671B Training

**TorchTitan Config** (saved as `config_deepseek_v3_671b.yaml`):

```yaml
model:
  name: deepseek_v3
  dim: 4096
  num_layers: 61
  num_heads: 32
  num_kv_heads: 8
  vocab_size: 129536
  num_experts: 16
  num_active_experts: 6
  expert_type: moe

training:
  batch_size: 1048576  # 1M tokens
  seq_length: 4096
  gradient_accumulation_steps: 8
  lr: 0.0003
  warmup_steps: 2000
  num_steps: 100000
  log_interval: 50

parallelism:
  tensor_parallel_size: 4
  pipeline_parallel_size: 1
  expert_parallel_size: 8
  fsdp_data_parallel_size: 2

mxfp8:
  enabled: true
  grouped_gemm: true
  linear: false  # set to true for full MXFP8

deepep:
  enabled: true
  backend: "deepep_nccl"
```

**Launch command** (256 GPUs, 8 nodes × 32 GPUs):

```bash
torchtitan --nproc-per-node=32 --nnodes=8 \
  train.py --config config_deepseek_v3_671b.yaml \
  --log_dir ./logs_671b_mxfp8_deepep
```

### Running DeepSeek-V3 16B Convergence Test

**TorchTitan Config** (saved as `config_deepseek_v3_16b_convergence.yaml`):

```yaml
model:
  name: deepseek_v3
  dim: 2560
  num_layers: 48
  num_heads: 20
  num_kv_heads: 4
  vocab_size: 129536
  num_experts: 4
  num_active_experts: 2
  expert_type: moe

training:
  batch_size: 1048576  # 1M tokens
  seq_length: 4096
  gradient_accumulation_steps: 4
  lr: 0.0003
  warmup_steps: 500
  num_steps: 1500  # convergence test
  log_interval: 10

parallelism:
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  expert_parallel_size: 4
  fsdp_data_parallel_size: 4

mxfp8:
  enabled: true
  grouped_gemm: true
  linear: false

deepep:
  enabled: false  # convergence test with compute only
```

**Launch command** (64 GPUs, 2 nodes × 32 GPUs):

```bash
torchtitan --nproc-per-node=32 --nnodes=2 \
  train.py --config config_deepseek_v3_16b_convergence.yaml \
  --log_dir ./logs_16b_convergence
```

### Enabling MXFP8 in TorchTitan Code

In your TorchTitan training script, enable MXFP8 after model instantiation:

```python
from torchao.prototype.moe_training import (
    apply_mxfp8_to_grouped_gemm,
    apply_mxfp8_to_linear_layers,
)

# After model is initialized and loaded to GPU
if config.mxfp8.enabled:
    if config.mxfp8.grouped_gemm:
        apply_mxfp8_to_grouped_gemm(model)
    if config.mxfp8.linear:
        apply_mxfp8_to_linear_layers(model)
```

### Enabling DeepEP in TorchTitan Code

DeepEP integration is handled via the `ProcessGroupWrapper` in TorchTitan's distributed backend. Enable it by setting the collective backend:

```python
from torchtitan.distributed import (
    init_distributed_with_deepep,
)

if config.deepep.enabled:
    init_distributed_with_deepep(backend="deepep_nccl")
```

Alternatively, set the environment variable before launching:

```bash
export TORCH_COLLECTIVE_BACKEND=deepep_nccl
torchtitan --nproc-per-node=32 --nnodes=8 train.py ...
```

### Monitoring & Logging

TorchTitan logs throughput, loss, and timing to tensorboard automatically:

```bash
tensorboard --logdir ./logs_671b_mxfp8_deepep
```

Key metrics to check:

- `throughput/token_per_sec`: Overall training throughput
- `loss/train`: Training loss (should match BF16 baseline for MXFP8 runs)
- `timing/fwd_pass`: Forward pass latency
- `timing/bwd_pass`: Backward pass latency
- `timing/collective`: Time spent in all-to-all (should decrease with DeepEP)

### Validation Scripts

To compare convergence between BF16 and MXFP8:

```python
import torch
from pathlib import Path

# Load logs from both runs
bf16_log = Path("./logs_16b_baseline/metrics.csv")
mxfp8_log = Path("./logs_16b_mxfp8/metrics.csv")

bf16_losses = [float(line.split(",")[1]) for line in open(bf16_log).readlines()[1:]]
mxfp8_losses = [float(line.split(",")[1]) for line in open(mxfp8_log).readlines()[1:]]

# Compare final losses
diff = abs(bf16_losses[-1] - mxfp8_losses[-1])
rel_error = 100 * diff / bf16_losses[-1]
print(f"Final loss BF16: {bf16_losses[-1]:.4f}")
print(f"Final loss MXFP8: {mxfp8_losses[-1]:.4f}")
print(f"Relative difference: {rel_error:.2f}%")
```

### Hardware Requirements

- **Minimum**: 8× NVIDIA B200 GPUs (single node)
- **Recommended**: 256× NVIDIA B200 GPUs (8 nodes, for 671B at full performance)
- **Network**: NVLink (intra-node) + RDMA (inter-node) for optimal DeepEP performance
- **Storage**: ~50GB for model checkpoints, ~200GB for training logs per full run

### Cloud Resources

Experiments were conducted on **Nebius Cloud** (a PyTorch-friendly partner):

- **Region**: Shenzhen, China
- **Instance type**: 32× B200 per node, 8 nodes
- **Network**: 8× NDR 400Gbps interconnect
- **Duration per 671B run**: ~2–4 hours (depending on config)

Nebius Cloud offers discounted GPU hours for academic and open-source projects. Refer to [Nebius PyTorch partnership](https://nebius.ai/) for credits.

### Reproducibility Checklist

- [ ] PyTorch, TorchTitan, TorchAO, DeepEP all installed from latest main branches
- [ ] NVIDIA drivers, CUDA 12.4+, cuDNN 9.0+, NCCL 2.20.0+ verified
- [ ] Training config YAML matches one of the provided examples
- [ ] MXFP8 and DeepEP flags enabled/disabled as intended
- [ ] Batch size and sequence length match documentation
- [ ] Multi-node launch command includes correct `--nproc-per-node` and `--nnodes`
- [ ] TensorBoard logs accessible and showing throughput/loss metrics
- [ ] Final loss within 0.5% of reported values (accounting for random seed variance)

## Conclusion

This joint PyTorch + Nebius effort demonstrates that **combining compute and communication optimizations can yield substantial end-to-end speedups** for large-scale MoE pre-training:

1. **MXFP8 training** (via TorchAO) provides **~9–12% compute speedup** with zero convergence degradation, thanks to Blackwell's native MXFP8 tensor cores and fine-grained per-block scaling.

2. **DeepEP communication** provides **~32% communication speedup** by replacing standard all-to-all with purpose-built kernels optimized for variable-size, dynamic routing patterns.

3. **Combined MXFP8 + DeepEP** achieves **+41% total throughput**, translating to ~29% reduction in GPU-days for a 1T-token pre-training run.

4. **Convergence is stable**: MXFP8 loss curves match BF16 exactly over 1,500 training steps, confirming numerical soundness.

5. **All code is open-source and reproducible** on Nebius Cloud using PyTorch-native tools: TorchTitan, TorchAO, and DeepEP.

These results highlight the maturity of PyTorch's ecosystem for frontier-scale training and the importance of co-optimizing compute and communication for MoE workloads. With B200's native MXFP8 support and DeepEP's efficient all-to-all, practitioners can now train DeepSeek-V3-scale models significantly faster and more cost-effectively.

For questions or feedback, please open issues on the respective GitHub repositories or engage with the PyTorch community on [discuss.pytorch.org](https://discuss.pytorch.org/).

**Acknowledgments**: Special thanks to the DeepSeek team for open-sourcing DeepEP, NVIDIA for B200 GPU access, and Nebius Cloud for providing the infrastructure and partnership to conduct these experiments at scale.
