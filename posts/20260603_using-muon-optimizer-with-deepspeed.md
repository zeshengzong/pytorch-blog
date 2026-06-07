# Using Muon Optimizer with DeepSpeed

By Zhipeng Wang, Guokai Ma, Peng Du and Chi McIsaac, DeepSpeed team

June 3, 2026

## TL;DR

DeepSpeed now supports Muon Optimizer! Muon Optimizer has gained great momentum with significant adoption from frontier AI Labs. One of those AI Labs is Moonshot AI, which has adopted Muon Optimizer to train its Large Foundation Model like Kimi-K2-Thinking. This post dives into what Muon Optimizer is and how it performs on DeepSpeed.

## What is Muon Optimizer?

Muon is an optimizer designed for hidden 2D weights of a neural network. It takes gradient of the weight, computes its momentum, and applies Newton-Schulz iterations to orthogonalize the momentum matrix, then uses this orthogonalized matrix to update the weight. Because Muon only maintains one momentum buffer (versus Adam's two), it uses less memory for optimizer states.

The orthogonalization step is key to Muon's convergence advantage in pretraining. In practice, gradient updates for 2D weights in transformers tend to have very high condition numbers — they are nearly low-rank, dominated by a few large singular directions. By orthogonalizing the momentum matrix, Muon equalizes all singular values, effectively amplifying rare but important update directions that would otherwise be overshadowed. This leads to better sample efficiency: in NanoGPT speedrunning benchmarks, Muon improved training speed by 35% over AdamW, and at 1.5B parameter scale it reached GPT-2 XL level performance approximately 25% faster than AdamW.

Unlike Adam optimizer that requires two momentum buffers for each parameter, Muon Optimizer only requires one momentum buffer. This means that for parameters using Muon Optimizer, we only need to allocate one buffer for momentum, which can save memory compared to Adam.

Muon is used by Keller Jordan's mod of NanoGPT, Andrej Karpathy's nanochat, and a variant of Muon (MuonClip) is also used by the production-level LLM Kimi-K2 from MoonShot. More recently, Zhipu AI's GLM-5 (744B parameters) confirmed the use of Muon Optimizer in both GLM-4.5 and GLM-5 pretraining, along with a "Muon Split" technique that splits MLA up-projection matrices by attention head and orthogonalizes each head independently, addressing a performance gap between MLA and GQA when using Muon. DeepSeek-V4 (1.6T parameters) also employs the Muon Optimizer for faster convergence and greater training stability.

## Muon Optimizer Support in DeepSpeed

One of the challenges of applying Muon optimizer to DeepSpeed is that previous optimizers (SGD, Adam) look at gradients as flattened buffers. Thus it is hard to swap in Muon Optimizer in the same place because the gradient buffers are already flattened. We move the Muon update to `get_flat_partition` function of stage 1 and 2 `DeepSpeedZeroOptimizer` in which per parameter gradients are still in unflattened stages, thus we can easily apply the Muon updates.

Muon Optimizer works on 2D weight matrices (attention and MLP weights). It applies Newton-Schulz orthogonalization to the momentum matrix, which requires the weight to be 2D. Non-2D parameters (embeddings, layer norms, biases, lm_head) fall back to AdamW. We apply a parse in model engine initializer to tag the model parameter with `use_muon`, if and only if the model parameter is 2D and belongs to hidden layers. When Muon Optimizer is used, any parameter tagged `use_muon` will use Muon Optimizer to update weight.

Note that Muon is a hybrid optimizer: it uses Muon updates only for 2D hidden weights and falls back to Adam for all other parameters (embeddings, layer norms, biases, lm_head). The DeepSpeed config supports separate learning rates via `muon_lr` (for Muon parameters) and `adam_lr` (for Adam parameters).

## Running DeepSpeed Finetune with Muon Optimizer

Deepspeed finetune demo is a demo to use different DeepSpeed training features and compare their performance in a single place. You can use it to test finetune LLM models with Muon Optimizer:

```
git clone https://github.com/delock/deepspeed_finetune_demo
cd deepspeed_finetune_demo
./finetune.sh z2_muon.json
```

## Muon Optimizer Convergence Experiment Result

We tested Muon Optimizer by finetuning Moonlight-16B-A3B (a Mixture-of-Experts model with 16B total and 3B active parameters), and evaluated on code generation (MBPP/MBPP+), general knowledge (MMLU), and mathematical reasoning (GSM8K) benchmarks. Each benchmark uses its own domain-specific training set.

**Training Configuration:**
- Model: Moonlight-16B-A3B (MoE, 16B total / 3B active)
- Training datasets: sahil2801/CodeAlpaca-20k for MBPP/MBPP+, cais/mmlu (auxiliary_train, ~95k examples) for MMLU, meta-math/MetaMathQA (sample_rate=0.1, ~39.5k examples) for GSM8K
- ZeRO Stage 2, bf16, Expert Parallelism (autoep_size=4)
- Batch size: 16, gradient accumulation: 2, 4 GPUs
- 1 epoch, gradient clipping: 1.0

## Evaluation Results

| Optimizer | Learning Rate | adam_lr (for Muon) | MBPP | MBPP+ | MMLU | GSM8K |
|-----------|---------------|-------------------|------|-------|------|-------|
| baseline (pre-finetune) | — | — | 0.495 | 0.431 | 0.401 | 0.526 |
| AdamW | 2e-6 | — | 0.661 | 0.534 | 0.660 | 0.805 |
| Muon | 1e-4 | 2e-6 | 0.646 | 0.548 | 0.678 | 0.810 |

Muon outperforms AdamW on 3 out of 4 metrics: MBPP+ (0.548 vs 0.534, +1.4pp), MMLU (0.678 vs 0.660, +1.8pp), and GSM8K (0.810 vs 0.805, +0.5pp). On MBPP base tests, AdamW edges out Muon (0.661 vs 0.646, -1.5pp), though Muon achieves a higher score on the more rigorous MBPP+ with extra test cases (0.548 vs 0.534), suggesting better generalization.

## Muon Optimizer Memory Savings

Muon Optimizer uses less memory for optimizer states than Adam, because it maintains one momentum buffer per parameter instead of two (first and second moment).

**Memory Usage Comparison**

| Optimizer | State Buffers per Param | Memory per Parameter |
|-----------|------------------------|----------------------|
| Adam | 2 (m, v) | 8 bytes |
| Muon | 1 (momentum) | 4 bytes |

Note that Muon is a hybrid optimizer: 2D hidden weights use Muon (1 buffer), while remaining parameters (embeddings, layer norms, lm_head) still use Adam (2 buffers). The actual memory savings depend on the fraction of parameters that are 2D hidden weights. For typical transformer models, approximately 90% of parameters are 2D hidden weights, so optimizer state memory is reduced by roughly 45%. However, because total GPU memory also includes model weights, gradients, and activations, the end-to-end memory reduction is smaller (see measured results below).

## Measured GPU Memory: Qwen2.5-3B Fine-tuning

We measured peak GPU memory during fine-tuning Qwen2.5-3B on tatsu-lab/alpaca using the same 8xA100 (40GB) configuration described above (batch size 32, ZeRO Stage 2, bf16).

| Optimizer | Peak Memory per GPU | Savings vs AdamW |
|-----------|-------------------|------------------|
| AdamW | 34.5 GiB | — |
| Muon | 31.4 GiB | 9% |

Muon reduces per-GPU memory by approximately 3 GiB (9%) compared to AdamW. The savings come entirely from optimizer states: Muon parameters store one momentum buffer (4 bytes) instead of Adam's two (8 bytes). However, because optimizer states are only one component of total GPU memory (alongside model weights, gradients, and activations), the end-to-end reduction is modest. For larger models or tighter memory budgets, this 9% savings could make the difference between fitting a workload on-device versus requiring CPU offloading.

## What's Next

Muon is rapidly gaining traction in the community, and production-level adoption by Kimi-K2 (1T parameters) and GLM-5 (744B parameters) signals that it is a serious contender to replace Adam as the default optimizer for large-scale training. We are actively building out full Muon support in DeepSpeed, with a series of improvements already in flight:

- ZeRO Stage 2 support — merged
- ZeRO Stage 3 support — merged
- Gram-Schmidt based Newton-Schulz iteration — a faster orthogonalization kernel, in review
- CPU Offloading — in progress
- MuonClip — the variant used by Kimi-K2, planned

We welcome any thoughts, feedback and contributions related to Muon Optimizer support on DeepSpeed – please start an issue for discussion or submit a PR to DeepSpeed. Let's make Muon rock solid and lightning fast in DeepSpeed!

---
Original Link: https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/
