# ExecuTorch Becomes a Part of PyTorch Core to Expand On-Device Inference Capabilities

By PyTorch Foundation

April 7, 2026

Today, we're excited to share that ExecuTorch is becoming a part of PyTorch Core. ExecuTorch extends PyTorch functionality for efficient AI inference on edge devices, from desktop/laptop to mobile phones and embedded systems.

Becoming a PyTorch Core project under the PyTorch Foundation will provide vendor‑neutral governance, clear IP, trademark, and branding, and ensure that business and ecosystem decisions are made transparently by a diverse group of members, while technical decisions remain with individual maintainers and open source contributors, ultimately strengthening ExecuTorch's adoption within the PyTorch ecosystem.

At this moment, we want to reflect briefly on how ExecuTorch started, share why we're becoming a PyTorch Core project, and what's ahead.

## How ExecuTorch started

ExecuTorch began at Meta as part of our effort to make it easier to run state-of-the-art PyTorch models efficiently on edge and on-device environments—from mobile phones and AR/VR headsets and Glasses to embedded devices and custom accelerators.

When we first introduced ExecuTorch publicly at PyTorch Conference 2023, it was designed around a small set of core principles:

- End-to-end developer experience: From authoring in PyTorch to deployment on-device, with a consistent, predictable workflow.
- Portability across hardware: A runtime that could target a wide variety of CPUs, GPUs, NPUs, DSPs, and other accelerators across platforms.
- Small, modular, and efficient: A lean runtime and a composable architecture suitable for constrained environments.
- Open by default: A project positioned to benefit from and contribute to the broader open-source AI ecosystem.

Since then, ExecuTorch has evolved from an internal runtime into an open platform for on-device AI. It underpins model deployment in Meta products and is increasingly being adopted by partners and the broader community as a flexible way to bring PyTorch-based models to production on edge devices.

## Growth and community

ExecuTorch has quickly grown beyond its initial use cases. It is now used as a foundation for on-device inference across a variety of scenarios, including:

- Mobile and AR/VR experiences
- Generative AI and LLM-based assistants on devices
- Computer vision and sensor processing at the edge
- Low-latency, privacy-preserving applications where models run locally

While Meta has been the primary initial contributor to ExecuTorch, a growing set of companies and individual developers have started investing in the project—adding backends, operators, tooling, and integrations, as well as building their products and research efforts on top of ExecuTorch.

We see contributions and ecosystem work emerging around:

- Hardware-specific optimizations and backends
- Tooling to convert, quantize, and package PyTorch models for ExecuTorch
- Integrations with mobile, AR/VR, IoT, and robotics platforms
- Benchmarks, testing, and documentation improvements

ExecuTorch is becoming an important part of how organizations think about portable, hardware-agnostic on-device AI, and it's clear the project is transitioning into a multi-stakeholder ecosystem. That makes this the right time to move to a broader open-source foundation.

## Why Become a PyTorch Core Project?

In its early phase, ExecuTorch's business governance was intentionally lightweight—we operated a lot like a small startup team within a larger organization. Meta helped put in the initial scaffolding: shaping the project's roadmap, setting up basic contribution processes, aligning ExecuTorch with PyTorch's model export and runtime stack, and engaging with early partners.

As ExecuTorch scaled, we realized that:

- Multiple companies want to invest in ExecuTorch as a neutral, shared layer in their on-device AI stack.
- Hardware vendors and platform providers need a clear and transparent way to influence direction and contribute.
- The project needs a governance structure that outlives any single organization and keeps ExecuTorch vendor-neutral and open.

Becoming a PyTorch core project under the PyTorch Foundation gives ExecuTorch:

- Vendor-neutral governance with the Foundation's governing board and charters.
- Clear IP, trademark, and branding stewardship, independent of any single company.
- Proven open source structures for membership, working groups, and strategic initiatives.
- A natural home alongside adjacent projects in the PyTorch ecosystem.

Meta will remain a major contributor and a key community member, but no single company will control ExecuTorch's business governance. The PyTorch Foundation's experience hosting large, multi-stakeholder projects gives ExecuTorch the right blend of structure and flexibility for this next stage as a PyTorch core project.

## Strengthening technical governance

Since its inception, ExecuTorch has operated under a community-driven open source model: maintainers and contributors working across components such as model conversion, runtimes, kernels, backends, and tooling. Responsibilities have been tied to individuals, not just their employers, and we've followed the spirit and many of the practices of the PyTorch ecosystem. As ExecuTorch grows, we need more explicit, transparent technical governance to scale responsibly.

The ExecuTorch Technical Governance will be as follows:

- The project will adhere to the already existing hierarchical technical governance structure of PyTorch. [Core PyTorch maintainers](https://docs.pytorch.org/docs/main/community/persons_of_interest.html#core-maintainers) will oversee larger cross-cutting changes while existing [Module maintainers](https://docs.pytorch.org/docs/main/community/persons_of_interest.html#executorch-edge-mobile) will oversee ExecuTorch specific changes. The maintainer membership will be individual and merit-based.

In the coming weeks, we will:

- Publish clear, documented technical decision-making processes, proposals, and escalation paths
- Alignment with familiar open-source patterns (e.g., RFC / proposal processes, release management, standards for compatibility and deprecation)
- Invest in shared CI/CD infrastructure for hardware partners to test and validate their backends

This does not fundamentally change how contributors build ExecuTorch day to day. Instead, it adds clarity, predictability, and openness, which are essential for a project that aims to be the neutral, shared runtime layer for on-device AI across the industry.

## What's next

As ExecuTorch become a PyTorch core project, our priorities are:

- Growing a diverse contributor and maintainer base across companies, hardware vendors, and independent developers.
- Deepening the integration with PyTorch for model export, quantization, and deployment flows.
- Expanding hardware and platform coverage so ExecuTorch can run efficiently wherever developers need it—on mobile devices, XR headsets, edge boxes, and embedded systems.
- Continuing to invest in documentation, tooling, and examples to make on-device AI development with ExecuTorch as accessible as possible.

Thank you.
