# Why Is PyTorch Compile So Fast: Kernel Fusion

By Morrison Turnansky | May 27, 2026

When you use PyTorch's compiler, your model runs faster, up to 10x faster. But what's actually happening? Without compilation, the GPU runs a kernel, a function on the GPU, for each torch operation in your code. This creates two big slowdowns: the time spent moving data in memory, and the overhead of starting each new kernel. Every time the GPU launches a kernel, it pays an overhead cost, and every intermediate result means writing to and reading from memory.

This is where fusion comes in. PyTorch's Inductor compiler automatically groups dependent operations together into single, efficient Triton kernels. This keeps data in faster memory close to the register and cuts down on kernel overhead. In this article, we'll look at a concrete example of fusion, and then outline topics for further reading. You'll see exactly how torch.compile transforms your PyTorch operations into optimized GPU code.

To get the most out of this article, you should have basic familiarity with PyTorch and a general understanding of GPU programming concepts.

## What is Vertical Fusion?

Think of vertical fusion as a way to "link" steps, so the output of one goes straight into the next. It's called "vertical" because if you picture the computation graph, these operations stack vertically – each one depends on the result of the previous step.

This is the most common fusion pattern in deep learning because neural networks are chains of operations: normalization, then linear layers, then activation functions, and so on. The big win is eliminating intermediate results – those temporary tensors never need to be written to or read from global memory. They stay in fast registers where the GPU can reach them more quickly.

Let's dive into an example of vertical fusion, namely pointwise fusion.

## Pointwise Fusion Example

Pointwise operations are simple math kernels that work on each element: addition, multiplication, activation functions, and more. Let's look at a pattern you might see in a neural network layer:

### Pointwise PyTorch Example

```python
import torch

def pointwise_example(x, w, b):
    # Multiple element-wise operations
    tmp = x * w        # multiply
    tmp = tmp + b      # add
    tmp = tmp.sigmoid() # sigmoid activation
    return tmp
```

### Unfused: Three Separate kernels

Without fusion, Inductor creates three separate Triton kernels. Don't worry if the Triton syntax looks intimidating. The important part isn't memorizing the syntax, but understanding the pattern: each kernel loads data, does one operation, and writes the result.

#### Kernel 1: Multiply

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

For succinctness, we include just the signatures of the next kernels as they are nearly identical, see our [Git Repository](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209) for the full source code.

#### Kernel 2: Add

```python
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr)
```

#### Kernel 3: Sigmoid

```python
@triton.jit
def sigmoid_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr)
```

Across the three kernels you're performing eight memory operations: reading inputs twice for multiply, reading multiply's result and the bias for add, reading add's result for sigmoid, and writing all three results. That's a lot of memory traffic.

### Fused: One Kernel

With fusion, torch.compile creates a single kernel:

#### Kernel 4: Fused

```python
@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, in_ptr2,
                                        out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load all inputs once
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)

    # Fused pointwise operations: mul -> add -> sigmoid
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)

    # Store final result only
    tl.store(out_ptr0 + (x0), tmp5, xmask)
```

Notice the difference: we load all inputs once, do all three operations in a row, and store only the final result. The intermediate values (`tmp2` and `tmp4`) stay in registers – the fastest memory on the GPU. They never touch the slower global memory.

### Benefits

* **Kernel launches**: 3 reduced to 1
* **Intermediate buffers**: 2 eliminated (multiply result and add result)
* **Memory bandwidth**: Reading 5 full tensors and writing 3 full tensors (8 memory operations) reduced to reading 3 tensors and writing 1 (4 memory operations) – a 50% reduction in memory traffic

## Other Fusion Types

Pointwise fusion is just one type of vertical fusion. Inductor uses other forms of vertical fusion to keep your GPU efficient:

**Reduction Fusion**: Combines reducing operations like max, mean, or sum, with the operations that happen before and after them. This is critical for operations like batch normalization.

**GEMM + Epilogue Fusion**: Attaches simple math to the end of heavy matrix calculations. Instead of doing a matrix multiply, writing the result to memory, then reading it back to add bias and apply ReLU, the bias and activation happen right after the multiply in the same kernel.

**Prologue Fusion**: The opposite of epilogue – preprocessing happens as data loads. For instance, normalizing input before matrix multiplication can happen on-the-fly as the data comes in.

In addition to vertical fusion, the most prominent type of fusion, Inductor also uses horizontal fusion.

**Horizontal Fusion**: Runs multiple independent operations on the same input at once. For example, computing both `sin(x)` and `cos(x)` in a single kernel, loading `x` only once instead of twice.

## Get Started: See Fusion in Your Own Code

Let's walk through a complete example using a reduction pattern.

### Step 1: Create a Simple Reduction Example

Create a file called `fusion_example.py`:

```python
import torch

def reduction_example(x):
    # Pointwise operation followed by reduction
    tmp = x * 2.0
    result = tmp.sum(dim=-1)
    result = result + 1.0
    return result

# Create test input
x = torch.randn(1024, 1024, device='cuda')

compiled_fn = torch.compile(reduction_example)
result_fused = compiled_fn(x)
```

### Step 2: View the Generated Code

Run your script with the `TORCH_LOGS` environment variable to see what Inductor generated:

```
TORCH_LOGS="output_code" python fusion_example.py
```

This outputs the generated Triton kernels to your terminal. Look for a kernel named something like `triton_per_fused_add_mul_sum_0`. The `per` prefix means "per-reduction" kernel, and the name tells you that add, mul, and sum were all fused together.

## Conclusion

Fusion is one of the most important optimizations that torch.compile does. By linking dependent operations into single kernels, it cuts down memory traffic and kernel overhead – often the main slowdowns in GPU work.

Try accelerating your own code with torch compile. No need to change your implementation, just add a torch compiler decorator and let the compiler do the work.

**Learn more**: PyTorch documentation at [pytorch.org/docs/stable/torch.compiler.html](http://pytorch.org/docs/stable/torch.compiler.html) has complete guides on compilation and optimization strategies. Reference our [Git Repository](https://gist.github.com/morrison-turnansky/0cc51b498c674aa23d4718ae200e6209) for the full source code.

---
Original Link: https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/
