# vLLM and PyTorch Work Together to Improve the Developer Experience on aarch64

By Kaichao You (Inferact) | May 18, 2026

## TLDR

PyTorch 2.11 makes it possible to install CUDA-enabled PyTorch wheels on aarch64 Linux directly from PyPI, eliminating the need for custom package indexes and workarounds that previously complicated deployment on systems such as NVIDIA GH200, GB200, and GB300. This post explains how this packaging change improves the installation experience for vLLM users and highlights how collaboration between vLLM and PyTorch through PyTorch Foundation helped bring the fix to production.

## An issue I first hit at a hackathon

This story actually starts back in October 2024. I was at the CUDA MODE (now GPU MODE) IRL hackathon, trying to get vLLM running on a GH200 box. It should have been a five-minute job. Instead, I spent a frustrating chunk of the day staring at a `pip install` that, on the surface, looked perfectly fine — wheels were resolved, dependencies were satisfied, the install completed without errors — but at runtime `torch.cuda.is_available()` stubbornly returned `False`.

The reason, once I dug in, was almost comically mundane: on `aarch64` Linux, `pip install torch` was pulling the **CPU-only** wheel from PyPI. There simply was no GPU wheel for `aarch64` published to the default PyPI index. To get a CUDA-enabled build, you had to explicitly point pip at the PyTorch download index:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

That, by itself, would be only mildly annoying. The real damage came from how this interacted with transitive dependencies. PyPI does not let a package specify a custom index for its dependencies. So if any package in vLLM's dependency tree declared a requirement of `torch==<some_version>`, and that version doesn't match, pip would happily go back to the default PyPI index, find the CPU wheel, **silently uninstall** the GPU build I had just carefully installed, and replace it with the CPU one. You'd think everything was fine until your model refused to find a GPU.

For anyone trying to bring up vLLM on GH200 — and later on GB200 / GB300 — this turned a one-line install into a maze of `--index-url` flags, pinned versions, and post-install sanity checks.

## The workarounds vLLM carried in the meantime

While we waited for a proper fix upstream, vLLM had to ship its own workarounds so that aarch64 users were not stuck.

The first one was `use_existing_torch.py`, added in [vllm-project/vllm#8713](https://github.com/vllm-project/vllm/pull/8713) back in September 2024 — explicitly framed in the PR title as _"enable existing pytorch (for GH200, aarch64, nightly)"_. The flow is exactly what the name suggests: you install the right `torch` build yourself (from the PyTorch index, or a nightly, or a custom build), then run `python use_existing_torch.py`, which strips every `torch`/`torchvision`/`torchaudio` requirement out of vLLM's `requirements/*.txt`, `requirements/*.in`, and `pyproject.toml`. With those pins gone, the subsequent vLLM install can no longer trigger pip to "helpfully" reach back into the default PyPI index and silently swap your CUDA-enabled `torch` for the CPU wheel. It is ugly — we are literally rewriting our own dependency files at install time — but it kept GH200 users unblocked for over a year.

Later, as `uv` matured, we got a cleaner option. In [vllm-project/vllm#24303](https://github.com/vllm-project/vllm/pull/24303) we added the following to `pyproject.toml`:

```
[tool.uv]
no-build-isolation-package = ["torch"]
```

This tells `uv` not to build `torch` in an isolated environment — which in practice means uv will reuse the torch already present in the current environment instead of trying to resolve and reinstall its own copy. Combined with installing `torch` first from the right index, this gave us a much more ergonomic path than the file-rewriting trick: a single config line in `pyproject.toml`, and `uv pip install vllm` (or a `uv sync`) would respect the pre-installed CUDA-enabled `torch` on aarch64.

The vLLM workaround is the community improvising around a gap in the packaging standard. [Wheel Variants](https://developer.nvidia.com/blog/streamline-cuda-accelerated-python-install-and-packaging-workflows-with-wheel-variants/) is NVIDIA and Astral formalizing the fix so the improvisation is no longer needed.

## From a hackathon headache to a TAC agenda item

Fast forward to 2025. vLLM joined the PyTorch Foundation, and I became one of its representatives on the Technical Advisory Committee (TAC). The aarch64 wheel situation kept coming up — both in my own work and from other vLLM users on Grace Hopper and Grace Blackwell systems. In August 2025, I filed [pytorch/pytorch#160162](https://github.com/pytorch/pytorch/issues/160162) to track the problem formally, and earlier this year, in a January 2026 TAC meeting, I raised it directly on behalf of vLLM users.

The ask was straightforward: publish aarch64 GPU wheels to the default PyPI index so that `pip install torch` "just works" on GB200-class machines, the same way it does on x86. Those wheels would dynamically link to libraries like NCCL and cuBLAS — the same approach already used on x86 — so they don't balloon in size. Such large binary sizes are both hard to download for users and expensive to host by the PyPi project maintainers. Hence it is limited and heavily discouraged by the PyPi maintainer.

The Nvidia engineering team requested that the CUDA SBSA wheels be published to PyPI, and then drove the small wheel approach that links against them.

This is exactly the kind of cross-project, infrastructure-level issue that the PyTorch Foundation is well-positioned to coordinate. vLLM and PyTorch are both Foundation projects, and having a shared forum to surface ecosystem friction — rather than each project working around it independently — turned out to make a real difference.

## The fix has landed

In April 2026, in another TAC meeting, I learned the issue is resolved: starting with **PyTorch 2.11.0**, the default `pip install torch` on aarch64 Linux now pulls a CUDA-enabled wheel rather than the CPU-only one. Piotr Bialecki from NVIDIA confirmed the change is live in the 2.11.0 release.

I verified it on a GB200, and the difference is exactly what you'd want — boring, in the best possible way:

```
$ uv run --no-project --python 3.12 --with 'torch==2.11.0' -- python -c "import torch; print(torch.cuda.is_available())"
True

$ uv run --no-project --python 3.12 --with 'torch==2.10.0' -- python -c "import torch; print(torch.cuda.is_available())"
False
```

One version bump, and the entire workaround stack disappears. No more custom index URLs propagating through requirements files. No more silent CPU-wheel substitutions clobbering a working install. No more "why is my GB200 not finding the GPU" debugging sessions for new users.

For vLLM specifically, this means installation on GB200 / GB300 is now genuinely smooth. New users showing up with a Grace Blackwell system can follow the standard install instructions and have things work the first time — which, when you're trying to get inference up and running on a brand-new platform, matters a lot.

The workarounds in vLLM — both `use_existing_torch.py` and the `[tool.uv] no-build-isolation-package = ["torch"]` setting — will stay. They are still useful for advanced users who run a custom PyTorch build (a nightly, a patched fork, or a from-source build paired with a vLLM source build) and need vLLM's install to leave that `torch` strictly alone. What changes is the _default_ path: ordinary users on aarch64 no longer have to know any of this exists. They can `pip install` and get on with their work, and the workarounds quietly become an advanced-user tool rather than a tax on everyone.

## Why this is worth writing about

It's a small change in the grand scheme of things — a packaging tweak, not a new feature. But I think it's worth taking a moment to appreciate, for a couple of reasons.

First, it's a concrete example of vLLM and PyTorch collaborating productively under the PyTorch Foundation umbrella. The TAC isn't just a governance ritual; it's a venue where pain points from downstream projects can land in front of the people who can actually fix them, and where coordination across projects happens by default rather than by accident. This issue traveled the full path — from a developer cursing at a terminal during a hackathon, to a TAC discussion, to a tracked GitHub issue, to a release — and the Foundation is what made that path short.

Second, developer experience compounds. Every hour someone doesn't spend wrestling with `--index-url` flags is an hour they spend actually building things on top of vLLM and PyTorch. aarch64 GPU systems are only going to get more common, and it's much better to fix this now, in the boring infrastructure layer, than to leave each user to discover and work around it on their own.

The uv-side workaround (build isolation passthrough) is part of the broader [WheelNext effort](https://wheelnext.dev/proposals/pepxxx_build_isolation_passthrough/) — a very welcome push to rethink how Python packaging handles accelerator-bound dependencies in the AI era.

A big shoutout to the people who made this happen: Alban Desmaison, Nikita Shulga, and Andrey Talman from the PyTorch core team, who picked up the original ask and helped move it through; The NVIDIA PyTorch team, who drove the aarch64 build work and confirmed the fix had landed in 2.11.0 with Piotr Bialecki supporting the effort and acting as the steady point of contact across NVIDIA and upstream on these issues; the PyTorch release engineering team for getting the wheels built and published; and the many engineers behind the scenes — across PyTorch, NVIDIA, and Arm — whose work on toolchains, CI infrastructure, and packaging made this possible. Thanks also to everyone in the TAC for keeping the door open for these kinds of conversations.

Onwards.

---
Original Link: https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/
