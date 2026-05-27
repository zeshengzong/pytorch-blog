# PyTorch 2.11 Release Blog

**By PyTorch Foundation | March 23, 2026 | [No Comments](https://pytorch.org/blog/pytorch-2-11-release-blog/#respond)**

---

## Overview

We are excited to announce the release of PyTorch® 2.11 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0))!

The PyTorch 2.11 release features the following changes:

- **Differentiable Collectives for Distributed Training**
- **FlexAttention now has a FlashAttention-4 backend on Hopper and Blackwell GPUs.**
- **MPS (Apple Silicon) Comprehensive Operator Expansion**
- **RNN/LSTM GPU Export Support**
- **XPU Graph**

This release is composed of 2723 commits from 432 contributors since PyTorch 2.10. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.11. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/pytorch-2.0/) page.

On Tuesday, March 31st at 10 am, Andrey Talman and Nikita Shulga will host a live session to walk through what's new in 2.11, including Differentiable Collectives for Distributed Training, FlexAttention with a FlashAttention-4 backend on Hopper and Blackwell GPUs, MPS expansion, and more, followed by a live Q&A. [Register to attend.](https://streamyard.com/watch/zHmCTfH6Y3zQ)

---

## API-UNSTABLE Features

### Differentiable Collectives for Distributed Training

Added differentiability support for functional collectives, enabling training workflows that can backpropagate through collective operations. This is a significant advancement for distributed deep learning research and advanced training techniques, which may be implemented without the need for custom autograd functions.

### FlexAttention now has a FlashAttention-4 backend on Hopper and Blackwell GPUs

This backend adds support for automatically generating CuTeDSL score/mask modification functions and JIT-instantiating FlashAttention-4 kernels from PyTorch, enabling 1.2× to 3.2× speedups over the existing Triton implementation on compute-bound workloads. This feature is still under active development and may change as it stabilizes; for setup details and current limitations, see the [FlexAttention + FlashAttention-4 blog post](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/).

### MPS (Apple Silicon) Development Improvements / Operator Expansion

This release includes support for error reporting from MPS backend as well as continuous expansion of operator coverage, that includes new distributions functions (log_normal, cauchy, geometric), operator migration (erfcx, grid_sampler_2d supports for all operation mode), extended baddbmm/addbmm for integer and complex types.

Asynchronous error reporting enables detection of out-of-bounds access attempts that occur during GPU indexing operations, for example:

```python
import torch
x=torch.rand(10, 1, 10, device='mps')
y=x[:, [1]]
torch.mps.synchronize()  # will raise index out of bounds error
```

### RNN/LSTM GPU Export Support

RNN modules (LSTM, GRU, etc.) can now be exported on GPUs, and tracing LSTM with dynamic shapes is now supported. This significantly expands the model types that can be deployed using torch.export for production inference. GRU API is unchanged; the new API is LSTM.

### ROCm Device-Side Assertions & TopK Optimizations

Added support for device-side assertions on ROCm for better debugging, plus significant TopK operator optimizations and radix select improvements by caching data on shared memory. Improves both developer experience and performance on AMD GPUs.

### XPUGraph support to optimize execution on Intel GPUs

XPUGraph allows users to capture a sequence of XPU operations into a runtime execution graph on Intel GPUs and replay it multiple times. This reduces CPU overhead, such as kernel launch and Python runtime overhead, improving workload performance on Intel GPUs. See [API Doc](https://docs.pytorch.org/docs/2.11/xpu.html#graphs) for usage details.

### FP16 Half-Precision GEMM On CPU Via OpenBLAS

Added FP16 half-precision GEMM support via OpenBLAS on CPU, providing faster FP16 inference for CPU-based deployments. This is valuable for edge devices and CPU-only inference scenarios.

---

## Non-Feature Updates

### CUDA version

Starting with this release, CUDA 13 is now the default version installed for both x86_64 and ARM platforms. Users who need an alternative build can still access the CPU-only version as well as CUDA 12.8 builds from the respective https://download.pytorch.org/whl subfolders.

### Torchscript is now Deprecated

Torchscript was deprecated in 2.10, and [torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html) should be used to replace the jit trace and script APIs, and [Executorch](https://docs.pytorch.org/executorch/stable/index.html) should be used to replace the embedded runtime. For more details, see [this talk](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903) from PTC.

### 2026 Release Cadence

For 2026, the release cadence has been increased to 1 per 2 months, from quarterly. See the [published release schedule](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence).

---

## Learning Resources

### Docs

Access comprehensive developer documentation for PyTorch

[View Docs ›](/docs)

### Tutorials

Get in-depth tutorials for beginners and advanced developers

[View Tutorials ›](/tutorials)

### Resources

Find development resources and get your questions answered

[View Resources ›](/resources)
