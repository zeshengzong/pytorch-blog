# PyTorch Compile 为何如此之快：内核融合

作者：Morrison Turnansky | 2026 年 5 月 27 日

当你使用 PyTorch 的编译器时，你的模型运行速度会更快，最高可达 10 倍。但实际发生了什么？在没有编译的情况下，GPU 会为代码中的每个 torch 操作运行一个内核（GPU 上的一个函数）。这会造成两大性能瓶颈：在内存中移动数据所花费的时间，以及启动每个新内核的开销。每次 GPU 启动一个内核，都会付出一笔固定开销，而每个中间结果都意味着需要写入和读取内存。

这就是融合（fusion）发挥作用的地方。PyTorch 的 Inductor 编译器会自动将相互依赖的操作组合成单个高效的 Triton 内核。这样可以让数据保留在靠近寄存器的更快内存中，并减少内核开销。在本文中，我们将通过一个具体的融合示例来说明，然后概述进一步阅读的主题。你将看到 torch.compile 如何将你的 PyTorch 操作转换为优化的 GPU 代码。

为了充分理解本文，你应该对 PyTorch 有基本的了解，并对 GPU 编程概念有大致的认识。

## 什么是垂直融合？

可以将垂直融合看作是一种"链接"步骤的方式，使一个操作的输出直接流入下一个操作。之所以称为"垂直"，是因为如果你将计算图可视化，这些操作会垂直堆叠——每一个都依赖于前一步的结果。

这是深度学习中最常见的融合模式，因为神经网络是一系列操作的链条：归一化、然后是线性层、然后是激活函数，依此类推。最大的收益在于消除了中间结果——那些临时张量永远不需要写入或从全局内存中读取。它们保留在 GPU 可以更快访问的寄存器中。

让我们深入了解垂直融合的一个示例，即逐元素融合（pointwise fusion）。

## 逐元素融合示例

逐元素操作是对每个元素进行简单数学运算的内核：加法、乘法、激活函数等。让我们看一个你可能在神经网络层中遇到的模式：

### 逐元素 PyTorch 示例

```python
import torch

def pointwise_example(x, w, b):
    # 多个逐元素操作
    tmp = x * w        # 乘法
    tmp = tmp + b      # 加法
    tmp = tmp.sigmoid() # sigmoid 激活
    return tmp
```

### 未融合：三个独立内核

没有融合时，Inductor 会创建三个独立的 Triton 内核。不用担心 Triton 语法看起来令人生畏。重要的不是记住语法，而是理解模式：每个内核加载数据，执行一个操作，然后写入结果。

#### 内核 1：乘法

```python
@triton.jit
def mul_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)
```

为了简洁起见，接下来的内核我们只列出函数签名，因为它们几乎相同，完整源代码请参见我们的 [Git 仓库](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209)。

#### 内核 2：加法

```python
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr)
```

#### 内核 3：Sigmoid

```python
@triton.jit
def sigmoid_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr)
```

三个内核共执行了八次内存操作：乘法读取两次输入，加法读取乘法结果和偏置，sigmoid 读取加法结果，以及三次写入结果。这是大量的内存流量。

### 融合：一个内核

使用融合后，torch.compile 创建了一个单一内核：

#### 内核 4：融合版

```python
@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, in_ptr2,
                                        out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # 一次性加载所有输入
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)

    # 融合逐元素操作：乘法 -> 加法 -> sigmoid
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)

    # 仅存储最终结果
    tl.store(out_ptr0 + (x0), tmp5, xmask)
```

注意区别：我们一次性加载所有输入，连续执行三个操作，并仅存储最终结果。中间值（`tmp2` 和 `tmp4`）保留在寄存器中——GPU 上速度最快的内存。它们永远不会接触到较慢的全局内存。

### 收益

* **内核启动次数**：从 3 次减少到 1 次
* **中间缓冲区**：消除了 2 个（乘法结果和加法结果）
* **内存带宽**：从读取 5 个完整张量并写入 3 个完整张量（8 次内存操作）减少到读取 3 个张量并写入 1 个（4 次内存操作）——内存流量减少了 50%

## 其他融合类型

逐元素融合只是垂直融合的一种类型。Inductor 使用其他形式的垂直融合来保持 GPU 的高效运行：

**规约融合（Reduction Fusion）**：将 max、mean 或 sum 等规约操作与其前后的操作结合起来。这对批归一化等操作至关重要。

**GEMM + 尾声融合（GEMM + Epilogue Fusion）**：在繁重的矩阵计算末尾附加简单的数学运算。不再是先做矩阵乘法、将结果写入内存、再读回来添加偏置和应用 ReLU，而是在同一个内核中，紧接乘法之后完成偏置和激活。

**前奏融合（Prologue Fusion）**：与尾声融合相反——预处理在数据加载时发生。例如，矩阵乘法之前的输入归一化可以在数据进入时即时完成。

除了垂直融合（最主要的融合类型），Inductor 还使用水平融合。

**水平融合（Horizontal Fusion）**：在同一个输入上同时运行多个独立操作。例如，在单个内核中计算 `sin(x)` 和 `cos(x)`，只需加载一次 `x` 而不是两次。

## 上手体验：在自己的代码中查看融合

让我们通过一个使用规约模式的完整示例来演示。

### 第一步：创建一个简单的规约示例

创建一个名为 `fusion_example.py` 的文件：

```python
import torch

def reduction_example(x):
    # 逐元素操作后接规约
    tmp = x * 2.0
    result = tmp.sum(dim=-1)
    result = result + 1.0
    return result

# 创建测试输入
x = torch.randn(1024, 1024, device='cuda')

compiled_fn = torch.compile(reduction_example)
result_fused = compiled_fn(x)
```

### 第二步：查看生成的代码

使用 `TORCH_LOGS` 环境变量运行你的脚本，以查看 Inductor 生成的内容：

```
TORCH_LOGS="output_code" python fusion_example.py
```

这会将生成的 Triton 内核输出到终端。查找名称类似 `triton_per_fused_add_mul_sum_0` 的内核。`per` 前缀表示"按规约"内核，名称告诉你 add、mul 和 sum 都被融合在一起了。

## 结论

融合是 torch.compile 最重要的优化之一。通过将相互依赖的操作链接到单个内核中，它减少了内存流量和内核开销——这通常是 GPU 工作中的主要瓶颈。

尝试用 torch compile 加速你自己的代码。无需更改你的实现，只需添加一个 torch 编译器装饰器，让编译器来完成这些工作。

**了解更多**：PyTorch 文档 [pytorch.org/docs/stable/torch.compiler.html](http://pytorch.org/docs/stable/torch.compiler.html) 提供了关于编译和优化策略的完整指南。完整源代码请参见我们的 [Git 仓库](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209)。

---
原文链接: https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/
