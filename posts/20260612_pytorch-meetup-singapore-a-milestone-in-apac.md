# PyTorch Meetup Singapore: A milestone in APAC

By Sudhir Dharanendraiah | June 12, 2026 | No Comments

## TL;DR

Eighty engineers, researchers, and community builders gathered for the inaugural [PyTorch Meetup Singapore](https://luma.com/8scaib4z). Hosted at the Red Hat Asia Pacific office and organised by [Sudhir Dharanendraiah](https://www.linkedin.com/in/sudhir-dharanendraiah-80a0867/), [Ayush Satyam](https://www.linkedin.com/in/ayushsatyam146/), [Sumantro Mukherjee](https://www.linkedin.com/in/sumantrom/), and [Daniel Kang (AER Labs)](https://www.linkedin.com/in/aer-dk/), the event brought together AI practitioners from across the Asia-Pacific region for an evening of technical talks, open-source exchange, and candid conversation about what it takes to move AI from research notebooks into production.

In this blog, we take a closer look at the topics discussed: inference, distributed training, continuous integration, community governance, and the broader question of technological sovereignty.

## Sovereign Intelligence: Architecting APAC's AI Future

Sudhir Dharanendraiah (Red Hat) opened the evening with a provocation: the Asia-Pacific region must transition from being a consumer of AI technologies to becoming an architect of them. His talk, "Sovereign Intelligence: Architecting APAC's AI Future," argued that genuine technological independence requires more than policy ambition, it demands investment in the engineering "plumbing" that makes sovereign AI practically achievable. Sudhir walked the audience through several foundational building blocks in the PyTorch ecosystem, including OpenReg for hardware-agnostic device registration, torch.compile for portable performance, and Fully Sharded Data Parallel (FSDP) for scalable training on locally sourced silicon. You can find the presentation [here](https://drive.google.com/file/d/1yTga_plbL-Ay_EwTsLy5jf4DmCrtnJ1t/view).

He highlighted how high-throughput serving tools such as vLLM can be deployed effectively on sovereign hardware, making the case that open-source communities are uniquely positioned to bridge the gap between global AI research and regional production readiness. The talk resonated strongly with an audience drawn from countries actively investing in domestic AI capabilities, and it set a collaborative tone for the rest of the evening: sovereignty is not isolation, but the freedom to build on shared, open foundations.

## Introducing vLLM's New Rust Frontend

[Ziqi Zhao](https://www.linkedin.com/in/bugenzhao/), a member of the technical staff at Inferact; the company funded by vLLM's original creators, presented work on a new Rust-based frontend for the inference engine. Zhao explained that as GPUs grow faster with each generation, the CPU-side overhead of Python's Global Interpreter Lock, dynamic typing, and garbage collection has become an increasingly visible bottleneck. The Rust frontend addresses this by offering higher concurrency within a single process and predictable memory management, without replacing the Python engine itself.

The architecture follows a cleanly layered design—from a low-level engine-core client communicating over ZMQ and MessagePack, through tokenisation and chat-rendering layers, up to an OpenAI-compatible HTTP server. Streaming is the primary design path rather than an afterthought, with each layer functioning as a stream transformation. Benchmark results on a four-GPU GB200 setup with Qwen3-0.6B showed meaningful improvements in both decode-sensitive and preprocess-heavy workloads. Zhao noted that an integration pull request is expected to land in the main vLLM repository soon, initially behind a git submodule and controlled by an environment variable for easy opt-in. The presentation can be found [here](https://drive.google.com/file/d/1f4pGriEaZOTlTapPz3GvhgbC0dONaFRU/view).

## Introduction to vLLM and Project Update

[Pin Siang Tan](https://www.linkedin.com/in/tanpinsiang/), Co-Founder of Embedded LLM and a vLLM contributor, delivered a comprehensive overview of vLLM's core architecture and its trajectory into 2026. He opened by distinguishing vLLM from lighter-weight tools like Ollama: where the latter targets single-user laptop inference, vLLM is built for concurrent, production-scale serving with continuous batching, tensor/pipeline/expert/data parallelism, and prefill-decode disaggregation. The engine now supports over a hundred model architectures, runs on hardware from NVIDIA and AMD GPUs to TPUs and Ascend NPUs, and counts more than seventy-seven thousand GitHub stars with over two thousand contributors from fifty-plus organisations.

Tan then walked through vLLM's "secret sauce"—continuous batching, torch.compile integration, fusion passes, quantisation (FP8, INT4, INT8, MXFP4, NVFP4), speculative decoding, and a novel sleep mode for efficient resource use. Looking ahead to the second quarter of 2026, he outlined plans to harden Model Runner V2 as the default, make auto-tuning a reality, introduce elastic expert parallelism that allows GPUs to be added or removed from a live deployment without restart, and expand into reinforcement learning rollouts and omni-modal serving. His closing slide summed up the project's confidence: "If you are serving LLMs in 2026, you're either using vLLM or explaining why you're not." The slides can be found [here](https://drive.google.com/file/d/1tmOQABWvHGo6UzA_-6K11GQSwRl6Skv9/view).

## vLLM-Omni: A Unified Platform for Omni-Modal LLMs

[Wang Zhipeng](https://www.linkedin.com/in/%E5%BF%97%E9%B9%8F-%E6%B1%AA-537882216/) from Rakuten Asia introduced vLLM-Omni, a standalone framework that evolved from the core vLLM codebase to serve models that operate across modalities(image, video, speech, and text), behind a single, OpenAI-compatible endpoint. The framework provides native support for both autoregressive and diffusion-based models, unified multi-stage pipelines, and modality-aware scheduling that allocates compute according to each modality's distinct latency and throughput profile.

Wang described how vLLM-Omni is already powering workloads ranging from vision-language assistants (VLA) to world models, and shared the project's plans to expand toward embodied AI and multimodal reinforcement learning through an integration with verl-omni. The talk underscored a broader theme of the evening: the inference stack is rapidly evolving beyond text, and the tooling needs to keep pace.

## torch.compile in the Wild

[Ayush Satyam](https://www.linkedin.com/in/ayushsatyam146/) (Red Hat) delivered what amounted to a source-code safari across twelve popular PyTorch ecosystem projects spanning inference, distributed training, reinforcement learning, and domain-specific frameworks. His central thesis was that torch.compile is not a flag one simply switches on; it is an architectural decision that reshapes a codebase, and the projects that benefit most are those willing to meet the compiler halfway.

Ayush organised his findings into four "acts." In inference, he showed how Hugging Face Transformers compiles only the decode step (since prefill runs once with variable length), while Diffusers goes further by compiling only the repeated blocks within a model; and supports LoRA adapter hotswap without recompilation. In distributed training, he contrasted Lightning AI's approach of reordering wrappers so compile can see through the distributed layer with DeepSpeed's audacious DeepCompile sub-framework, which strips all ZeRO-3 hooks, injects custom ops into the compiled graph, and even replaces the autograd metaclass. In reinforcement learning, he demonstrated how TorchRL compiles only the mathematical update function while leaving environment interaction in Eager Python. Finally, in domain-specific work, he highlighted PyG's Jinja2 code generation at import time to avoid graph breaks, MONAI's strip-and-reattach strategy for medical-imaging metadata, and TorchVision's use of unbacked symbolic integers for data-dependent shapes.

The talk concluded with a taxonomy of compilation patterns; graph splitting, compiler monkeypatching, progressive compilation, opaque wrapping, dual execution modes, and shape stabilisation-that attendees could apply to their own projects. The slides can be found [here](https://drive.google.com/file/d/1SbNrWLO-ALMXl52pA6xl2gWixk1XfQQq/view).

## PyTorch Community and CI

Sumantro Mukherjee (Red Hat) closed the technical programme with a talk that paired community governance with engineering rigour. He began with the contributor journey—from filing a GitHub issue through triage, pull request, maintainer review, and the @pytorchbot merge flow-emphasising that the median triage turnaround is under two business days and that an automated bot-driven revert cycle keeps the trunk green.

The bulk of the talk focused on PyTorch's multi-cloud continuous integration infrastructure, which now runs upwards of ninety thousand CI jobs daily across five cloud providers(AWS, Azure, GCP, IBM Cloud, and the Linux Foundation's OSDC-spanning x86_64, aarch64, and ppc64le architectures). He also introduced the Multi-Cloud Working Group, a PyTorch Foundation TAC initiative with participation from Meta, Red Hat, IBM, NVIDIA, AMD, Google, Intel, Huawei, and the Linux Foundation, whose charter is to develop sustainable, community-managed CI/CD infrastructure.

Beyond infrastructure, Mukherjee gave an overview of the PyTorch Foundation's governance structure. The Technical Advisory Council (TAC), the five working groups (CI Infrastructure, Multi-Cloud, Ecosystem, Accelerators, and Security) and the RFC process. He closed with an open invitation: the TAC meets on the second Tuesday of every month, the Multi-Cloud Working Group convenes every Thursday at 5 PM UTC, and both are open to anyone who [wants to contribute](https://lists.pytorch.org/g/tac/messages). The slides can be found [here](https://drive.google.com/file/d/1clFICwZb5zjS_xxzkDxmcx9-rlqn5CRy/view).

## Looking Ahead

As the formal program gave way to networking over food and drinks (we had pizzas!), conversations continued well into the evening about compiler internals and serving architectures, certification paths and contributor workflows, and about what a sovereign AI stack might look like for the region. For many attendees, the meetup was a first opportunity to connect with peers working on the same open-source tools in the same time zone.

The inaugural PyTorch Meetup Singapore demonstrated that the appetite for deep, technically grounded community events in Southeast Asia is very real. With close to a hundred attendees, six speakers from four organisations, and a venue that placed participants quite literally above the city, the evening was both a starting point and a statement of intent. The organisers have signalled that more meetups will follow and if the energy in the room was any indication, the next one will need a bigger space. Details for the upcoming events will be added to [AI SGP](https://www.ai.engineer/singapore/2026#side-events).

Event page: [https://luma.com/8scaib4z](https://luma.com/8scaib4z)

---
Original Link: https://pytorch.org/blog/pytorch-meetup-singapore-a-milestone-in-apac/
