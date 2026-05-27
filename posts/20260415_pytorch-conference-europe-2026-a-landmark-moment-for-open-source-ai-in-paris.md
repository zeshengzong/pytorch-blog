# PyTorch Conference Europe 2026: A Landmark Moment for Open Source AI in Paris

By PyTorch Foundation | April 15, 2026 | April 17th, 2026

The first-ever PyTorch Conference Europe April 7-8, 2026 brought together more than 600 researchers, developers, practitioners, and academics in Paris for two packed days of keynotes, technical deep dives, lightning talks, poster sessions, and community connection. From bare-metal infrastructure to agentic AI, the sessions spanned the full AI stack and made one thing clear: the open source AI ecosystem is accelerating faster than ever.

All sessions recordings will be available on our YouTube channel within the next week. Here is our recap of conference highlights.

## Major Announcements:

During PyTorchCon EU, the PyTorch Foundation announced new projects joining its community alongside PyTorch, vLLM, DeepSpeed, and Ray. Both Helion and Safetensors have now joined as foundation-hosted projects too. ExecuTorch also became a part of PyTorch Core.

* **Helion**, contributed by Meta, is a Python-embedded domain-specific language (DSL) that makes it easy to write fast, scalable ML kernels with minimal boilerplate. By making kernel authoring a first-class part of PyTorch, Helion strengthens custom kernel creation and reduces manual coding effort through autotuning. Read more.

* **Safetensors**, contributed by Hugging Face, is a secure model file format that prevents arbitrary code execution and enhances performance across multi-GPU and multi-node deployments. It has become one of the most widely used metadata formats for model distribution. Read more.

**ExecuTorch** was officially welcomed as a part of PyTorch Core. Originally developed by Meta, ExecuTorch simplifies running PyTorch models on edge and on-device environments, including mobile phones, AR/VR headsets, and microcontrollers. It was designed around four core principles: end-to-end developer experience, portability across hardware, small and modular efficiency, and openness by default. Read more.

These additions reinforce the Foundation's position as the vendor-neutral hub for the open source AI stack, covering the full lifecycle from training through inference.

## Day 1: Tuesday, April 7

### Opening Keynotes

The morning opened on the Master Stage with a series of keynotes that set the tone for the entire event.

**Mark Collier**, Executive Director of the PyTorch Foundation and General Manager of AI and Infrastructure at the Linux Foundation, kicked things off with "Co-Evolution: How the Open Source Intelligence Stack Compounds." His message was clear: the power of the PyTorch ecosystem comes not from any single project, but from how its many components reinforce and build on each other. As the Foundation continues to grow its project portfolio, that compounding effect is only getting stronger.

**Edward Yang**, Research Engineer at Meta, followed with a "PyTorch Updates" keynote, giving the community a look at the latest developments in the core framework.

**Joe Spisak**, VP of Product and Head of Open Source at Reflection AI, took the stage next with "Community Led Open Source RL," highlighting how the reinforcement learning community is pushing boundaries through open collaboration.

**Ramine Roane**, Corporate Vice President of AI Product Management and Ecosystem Development at AMD, delivered a keynote titled "From One Node to Distributed Training and Inference: How the PyTorch Ecosystem Changed AI," tracing the journey from single-node experiments to planet-scale deployments.

**Patrick von Platen**, Research Engineer at Mistral AI, presented "Stream Everything: Moving from Request Input to Streaming Input," offering a forward-looking vision of how AI systems will process continuous streams of data rather than discrete requests.

**Maryam Tahhan**, Principal Engineer, Red Hat and **Nicolo Lucchesi**, Senior Machine Learning Engineer, Red Hat delivered a keynote, "Any [Agent | Model | Accelerator | Cloud]: Open Source AI Unlocks the World's Potential," making the case that open source is the path to truly flexible, portable AI.

The morning keynote block closed with **Besmira Nushi**, Senior Manager of AI Research at NVIDIA, presenting "The Unbearable Lightness of (Agentic) Evaluations," a thought-provoking look at the challenges of evaluating agentic AI systems.

### Meet the Developers

A highlight of the conference format was the "Meet the Developers" sessions, where attendees could sit down face to face with the people behind the projects. On Day 1, **Meet the Developers of PyTorch Module Maintainers** featured Driss Guessous, Mergen Nachin, Natalia Gimelshein, Jason Ansel, Edward Yang, and Alban Desmaison. Later in the afternoon, a dedicated **Meet the Developers of Helion** brought together Jason Ansel, Oguz Ulgen, Will Feng, and Markus Hoehnerbach.

### Frameworks and Compilers

This track was one of the most densely packed at the conference. Standout sessions included:

* **Helion 1.0: A High-Level DSL for Performance Portable Kernels** by Oguz Ulgen from Meta, introducing the newly contributed Helion project as a first-class tool for kernel authoring in PyTorch.
* **Parameterized CUDA Graph Launch in PyTorch: CUDA Graphs Without the Pain** by Daniel Galvez from NVIDIA, tackling one of the trickiest performance optimization challenges.
* **FlexAttention + FlashAttention-4: Fast and Flexible** by Driss Guessous from Meta, showcasing the next generation of attention mechanisms.
* **Combo Kernels: Horizontal Fusion Optimization in Torch.compile** by Karthick Panner Selvam and Elias Ellison from Meta.
* **Model-Changing Transforms with Torch.compile** by Thomas Viehmann from Lightning AI.
* **Brevitas Quantization Library** by Pablo Monteagudo Lago from AMD.

### Inference and Production

The inference track reflected the community's growing focus on getting models into production efficiently:

* **Tour De Force: LLM Inference Optimization From Simple to Sophisticated** by Christin Pohl from Microsoft, a comprehensive walkthrough on the Master Stage.
* **ExecuTorch on Microcontrollers: Deploying PyTorch to the Smallest Edge** by RJ Ascani and Matthias Cremon from Meta, pushing the boundaries of where PyTorch can run.
* **Write Once, Run Everywhere with PyTorch Transformers** by Pedro Cuenca from Hugging Face.
* **Why WideEP Inference Needs Data-Parallel-Aware Scheduling** by Maroon Ayoub (IBM) and Tyler Michael Smith (Red Hat).
* **The Token Slice: Implementing Preemptive Scheduling via Chunked Decoding** by Maroon Ayoub (IBM) and Kellen Swain (Google).
* **Cross-Region Model Serving: PyTorch Inference, Observability, and LLMOps** by Suraj Muraleedharan from Amazon Web Services.
* **Accelerating On-Device ML Inference with ExecuTorch and Arm SME2** by Jason Zhu from Arm.

### GenAI and Multimodal

* **Lights, Camera, Inference! Video Generation as a Service with VLLM-Omni** by Ricardo Noriega and Doug Smith from Red Hat.
* **The Science and Practice of Open and Scalable LLM Evaluations** by Grzegorz Chlebus from NVIDIA.
* **torch.compile and Diffusers: A Hands-On Guide to Peak Performance** by Sayak Paul from Hugging Face.

### Training Systems

* **Training Embedding Model Resiliently for Multimodal Model Inference Routing** by Huamin Chen (Red Hat) and Haichen Zhang (AMD).
* **TorchJD: Jacobian Descent in PyTorch** by Pierre Quinton (EPFL) and Valerian Rey (Simplex Lab).
* **Bringing Google's Colossus to PyTorch: Rapid Storage via fsspec to Keep GPUs Busy** by Ankita Luthra and Trinadh Kotturu from Google.
* **Jigsaw: Domain and Tensor Parallelism for High-Resolution Input Training** by Deifilia Kieckhefen from Karlsruhe Institute of Technology.

### Applications and Case Studies

* **Deep Learning in the Wild: Embedded PyTorch for Real-World Conservation Bioacoustics** by Taraqur Rahman and Owen O'Donnell from OWL Integrations.
* **How DeepInverse Is Solving Imaging in Science and Healthcare with PyTorch** by Andrew Wang (DeepInverse) and Minh Hai Nguyen (Universite de Toulouse).
* **Flexible Deployment of PyTorch Models on MCU-Class Devices Using ExecuTorch** by Robert Kalmar and Martin Pavella from NXP.

### Responsible AI, Security, and Privacy

* **Engineering for the EU AI Act: What Should PyTorch Expose Natively?** – a Birds of a Feather session led by Roy Saurabh from AffectLog, exploring the intersection of regulation and framework design.
* **From Pretrained to Personal: Privacy-First Fine-Tuning on AI PCs** by Daniel Holanda Noronha and Iswarya Alex from AMD.
* **Ethical, Privacy and Sustainability Considerations in PyTorch Systems** by Paula Mesa Macias from Pau&Company.
* **Why Classic IAM Collapses for Agents: Rethinking IAM for Agentic Systems** by Parul Singh from Red Hat.

### Community Events

Day 1 also featured a **Women and Non-Binary in PyTorch Lunch** and a lively **Flare Party** in the evening featuring good food, good drinks, and good company. Both days featured poster presentations across every track, giving researchers and engineers the chance to share work in progress and engage in one-on-one technical conversations. Poster tracks covered Applications and Case Studies, Frameworks and Compilers, GenAI and Multimodal, Inference and Production, and Responsible AI and Compliance.

## Day 2: Wednesday, April 8

### Morning Keynotes

Day 2 opened with another strong keynote lineup on the Master Stage.

**Matt White**, PyTorch CTO and Global CTO of AI at the Linux Foundation, set the stage for the day.

**Tyler Michael Smith**, Chief Architect, Inference Engineering, Red Hat and **Artur Niederfahrenhorst**, Member of Technical Staff, Anyscale delivered a joint keynote talk on **vLLM and Ray Updates**, two projects that have become cornerstones of production AI infrastructure.

**Lysandre Debut**, Chief Open-Source Officer at Hugging Face, presented "The Hub as Infrastructure: From Open PyTorch Models, to a Safe and Performant Distribution Hub," underscoring how the Hugging Face Hub has evolved from a model repository into critical AI infrastructure.

**Jonathan Bryce**, Executive Director of the Cloud Native Computing Foundation (CNCF), gave a keynote on "Open Source Infrastructure for the AI Native Era," connecting the dots between cloud-native and AI-native development.

**Leonard Hussenot**, Research Scientist at Google DeepMind, closed the keynote block with "Gemma 4: Compacting Intelligence for the Edge," a look at how Google DeepMind is making powerful models small enough to run on edge devices.

### Frameworks and Compilers (Day 2)

* **Monarch: An API to Your Supercomputer** by Marius Eriksen from Meta.
* **How to Write C++ Extensions in 2026** by Jane Xu and Mikayla Gawarecki from Meta.
* **Achieving SOTA GEMM Performance: A CuTeDSL Backend for PyTorch Inductor** by Nikhil Patel from Meta.
* **Accelerating PyTorch Models with Torch.compile's C++ Wrapper Mode** by Bin Bao from Meta.
* **Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm** by Liz Li and Zachary Streeter from AMD.
* **PyTorch on RISC-V: From Cross-Compilation to Native CI** by Ludovic Henry from Meta.
* **PyTorch Symmetric Memory + NCCL Device APIs: A New Path Towards Multi-GPU Kernels** by Ke Wen and Sylvain Jeaugey from NVIDIA.
* **Seamless Integration: Custom Kernels in the Torch.compile Stack Without Graphbreaks** by Kshiteej Kalambarkar, Masaki Kozuki, and Pawel Gadzinski from NVIDIA.
* **Faster Than SOTA Kernels in Torch.compile with Subgraph Fusions and Custom Op Autotuning** by Elias Ellison and Paul Zhang from Meta.
* **De-mystifying PyTorch for ASICs: When (and Why) to Move Your Development to AI Accelerators** by Alpha Romer Coma from Kollab Philippines.

### Inference and Production (Day 2)

* **KV-Cache Centric Inference: Building a State-Aware Serving Platform with Llm-d and VLLM** by Martin Hickey and Maroon Ayoub from IBM Research.
* **Optimizing Large MoE Inference on NVIDIA Blackwell: NVFP4, ADP, and DualPipe Strategies** by Julien Demouth from NVIDIA.
* **Portable High-Performance LLM Serving: A Triton Backend for VLLM** by Burkhard Ringlein and Jan van Lunteren from IBM.
* **Deploying PyTorch Models to the Browser and Beyond with Transformers.js** by Joshua Lochner from Hugging Face.
* **Inside VLLM's KV Offloading Connector: Async Memory Transfers for Higher Inference Throughput** by Nicolo Lucchesi from Red Hat.
* **Optimizing CPU LLM Inference in PyTorch: Lessons from VLLM** by Crefeda Rodrigues and Fadi Arafeh from Arm.
* **From Hugging Face to Handheld: Scaling LLM Deployment with LiteRT Generative API** by Cormac Brick and Weiyi Wang from Google.
* **Full-Stack PyTorch Robotics VLA: From Data to Edge via ExecuTorch/OpenVINO** by Samet Akcay and Dmitriy Pastushenkov from Intel.
* **Beyond the Theory: What Actually Breaks When You Scale Your Disaggregated PyTorch Models** by Ekin Karabulut and Ron Kahn from NVIDIA.
* **Slash LLM Cold-Start Times by Pre-distributing GPU Caches** by Billy McFall and Maryam Tahhan from Red Hat.

### Training Systems (Day 2)

* **FP8 Training from Hopper to Blackwell** by Luca Wehrstedt from Meta.
* **Backpropagation-Free Optimization in PyTorch** by Andrii Krutsylo from the Polish Academy of Sciences.
* **Debugging the Undebuggable: Introducing Torch.distributed.debug** by Tristan Rice from Meta.
* **Scaling Recommendation Systems to 2K GPUs and Beyond** by Zain Huda from Meta.
* **From Responses to Trajectories: Multi-Turn and Multi-Environment Reinforcement Learning** by Kashif Rasul and Sergio Paniego Blanco from Hugging Face.
* **Trinity Large: Torchtitan on 2000+ B300s** by Matej Sirovatka from Prime Intellect.
* **DualPipe from Scratch: Implementing DeepSeek's 5D Parallelism in PyTorch** by Dev Jadhav from ING Bank.
* **Fault-Tolerant Training: How We Build Reliable Clusters for Distributed AI Workloads** by Cyril Konkratenko and Maurits de Groot from Nebius.
* **Why Logging Isn't Enough: Making PyTorch Training Regressions Visible in Practice** by Sahana Venkatesh from Wayve.

### Agents and Interop

* **Beyond JSON-RPC: Scaling Model Context Protocols with gRPC in the PyTorch Ecosystem** by Ashesh Vidyut and Madhav Bissa from Google.
* **Coding Agents for Compiler Construction: Beyond the AI Assistant Paradigm** by Reza Rahimi from yasp.ai and Stefan Krassin from yasp.ai.
* **Bridging the Hardware Gap with Code Harnesses on the Hugging Face Kernels Hub** by Ben Burtenshaw from Hugging Face.

### Security and Privacy

* **Live Migration of PyTorch GPU Nodes from Azure to European Clouds** by Mike Krom from Acf Cyber Solutions.
* **Securing Agentic AI with PyTorch: Threat Modeling and LLM Red Teaming in Practice** by Valeri Milke from VamiSec GmbH.

### Responsible AI and Compliance

* **From Gradients to Governance: Making PyTorch Lineage-Aware** by Kateryna Romashko and Clodagh Walsh from Red Hat.
* **Building Trust for Users and Regulators Alike: A Cost-Efficient PyTorch Path to Compliance-as-Code** by Raja Gopal Hari Vijay from Zoho Corporation.
* **Bridging the Gap: Engineering Compliant "Glass Box" Medical AI with PyTorch** by Muhammad Saqib Hussain and Mohaddisa Maryam from Neurosonic.

### Applications and Case Studies (Day 2)

* **Ball Tracking and Detection in Soccer Videos: Comparison of VLMs and Traditional Pipelines** by Maciej Szymkowski from Future Processing.

### Birds of a Feather and Meet the Maintainers

Day 2 featured two compelling Birds of a Feather sessions: **Disaggregated Tokenization: Building Toward Tokens-In-Tokens-Out LLM Inference** with Maroon Ayoub, IBM Research; Hang Yin & Xi Ning Wang, Alibaba Cloud; Nili Guy, IBM; and Hyunkyun Moon, Moreh and **NCCL in the Wild: Scaling Communications to Thousands of GPUs** with Jeff Hammond, Gabrielle Talavera, Ke Wen, and Asma Farjallah, NVIDIA.

Attendees also had the opportunity to **Meet the vLLM Maintainers** (Tyler Michael Smith and Nicolo Lucchesi) and **Meet the Ray Maintainers** (Artur Niederfahrenhorst).

### Community Expo and Networking

The Community Expo ran throughout both days, giving attendees the chance to explore demos, connect with project maintainers, and discover new tools in the ecosystem. Sponsor activities included a session on "Validating AI on CPUs: The vLLM 3-Phase Evaluation Framework" and "Lobster Trap: OpenClaw in Containers."

## Key Themes and Takeaways

**The open source AI stack is maturing.** With Helion and Safetensors joining as hosted-projects and ExecuTorch becoming part of PyTorch Core, the PyTorch ecosystem now covers an even wider range of the AI lifecycle. From secure model distribution to edge deployment to high-performance kernel authoring, these projects fill critical gaps.

**Inference is the new frontier.** A large share of sessions focused on production deployment, from vLLM and LLM serving optimizations to ExecuTorch on microcontrollers and in-browser inference with Transformers.js. The community is clearly moving beyond training and into making models work in the real world.

**Agentic AI demands new infrastructure.** Several sessions tackled the unique challenges of agentic systems, including rethinking Identity and Access Management (IAM), building trust through evaluation, and securing agentic pipelines. This is a space to watch.

**Europe is stepping up.** From EU AI Act compliance sessions to talks on data sovereignty and cloud migration, the conference reflected Europe's growing role in shaping responsible AI development. The inaugural European conference was a milestone for the global community.

**Hardware diversity is expanding.** Sessions covered AMD ROCm, Arm SME2, Google TPUs, NVIDIA Blackwell, RISC-V, microcontrollers, Qualcomm QNN, Intel OpenVINO, and more. PyTorch's hardware portability story has never been stronger.

## Thank You, Paris

PyTorch Conference Europe 2026 was a landmark event for the community. To every sponsor, speaker, poster presenter, and attendee who joined us in Paris: thank you. The energy, the technical depth, and the spirit of open collaboration were truly remarkable. Hats of to the PyTorchCon Europe Program Chairs: Alban Desmaison, Lysandre Debut, and Luca Antiga and the full Program Committee for putting together a successful event!

All session recordings will be available on the PyTorch YouTube Channel. Browse the full photo album from the event on Flickr.

See you at our next PyTorch Conferences including PyTorch Conference China – September 8-9 in Shanghai, China and PyTorch Conference North America – October 20-21 in San Jose, CA later this year.
