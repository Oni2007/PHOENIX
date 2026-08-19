> ## ⚠ SUPERSEDED FOR v0.1 — read this first
>
> This is the project's **long-horizon research context**. It predates
> `docs/design/PHOENIX_v0.1_TDD.md` and is **superseded by the TDD wherever the two
> disagree**. Retained for research direction beyond v0.1 — hardware scope breadth
> (§3), photonic and neuromorphic scope (§4–§5), application metrics (§28), and the
> long-term vision (§29). Do not paste this document in as a specification for v0.1
> work, and do not treat its structures as current.
>
> **Known divergences from the TDD, all resolved in the TDD's favour:**
>
> | Topic | This document | Authoritative (TDD) |
> |---|---|---|
> | Provenance classes | §16 lists `PUBLISHED_RESEARCH` | Six classes: `MEASURED`, `VENDOR_REPORTED`, `SIMULATED`, `ANALYTICAL`, `HYPOTHETICAL`, `DERIVED` (TDD §4/P3, ADR-004). Published-research evidence is captured separately as `claims[].evidence_class` in the literature schema (TDD §24) — the concept survives, the enum member does not. |
> | Repository tree | §14 puts `cuda/`, `photonic/`, `neuromorphic/` at top level | TDD §8 puts backends under `cpp/backends/`. CUDA is one backend among several, not a peer of the whole system — the assumption that breaks when photonics arrives. |
> | Roadmap horizon | §34 asks for a 30/60/90-day roadmap | TDD §31 delivers a 12-week plan with weekly acceptance criteria. |
> | v0.1 content | §21 lists photonics-adjacent items | TDD §2–§3 scope v0.1 to GEMM on CPU + CUDA only. Photonics is v0.6. |
> | Hardware scope | §3 asks for investigation across ~12 vendors | TDD §2 restricts v0.1 execution to CPU + NVIDIA; the breadth here remains the literature-review target. |
>
> The immediate task in §34 of this document ("Begin by designing PHOENIX v0.1") has
> been completed. Its deliverable is the TDD.

MASTER RESEARCH CONTEXT
=======================

PROJECT NAME:
PHOENIX

FULL NAME:
Photonic-Heterogeneous Optimized Engine for Neural Intelligence eXecution

PROJECT TYPE:
Open-source AI hardware research platform + accelerator simulator + compiler/runtime research project.

PRIMARY OBJECTIVE:
Design and implement a scientifically rigorous, reproducible software framework for comparing conventional electronic AI accelerators (GPU/TPU/AI accelerators) against photonic and neuromorphic architectures, and ultimately create a heterogeneous runtime capable of intelligently mapping AI workloads across CPU, GPU, photonic and neuromorphic compute resources.

IMPORTANT:
This is NOT intended to be a simple student demo.
Treat it as a serious long-term research and engineering project that could potentially produce:
- High-quality GitHub repositories
- Reproducible benchmarks
- Research papers
- Technical reports
- Conference posters
- Open-source software
- Academic collaborations
- Industry collaborations
- Potential startup technology

==================================================
1. CORE RESEARCH QUESTION
==================================================

The central research question is:

"Under what workload, precision, matrix size, optical-electronic overhead, memory requirement, communication pattern and accuracy constraint does photonic or neuromorphic computation provide a measurable advantage over conventional GPU/TPU computation?"

A secondary question is:

"Can a heterogeneous AI runtime dynamically determine whether a workload should execute on a CPU, CUDA GPU, photonic accelerator or neuromorphic accelerator based on latency, energy, memory, communication and accuracy?"

The project must NOT assume that photonics is automatically better than GPUs.

The goal is to FIND THE CONDITIONS under which each architecture is advantageous.

Never present hypothetical performance as measured hardware performance.

Clearly distinguish:
1. Measured data
2. Vendor-reported specifications
3. Simulation results
4. Analytical estimates
5. Hypothetical projections

==================================================
2. HIGH-LEVEL SYSTEM ARCHITECTURE
==================================================

The final PHOENIX architecture should look approximately like:

                    PyTorch / ML Framework
                              |
                              v
                    +-------------------+
                    | PHOENIX Compiler  |
                    |                   |
                    | Graph optimizer   |
                    | Cost model        |
                    | Precision planner |
                    | Hardware mapper   |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       +-------------+ +-------------+ +-------------+
       | CUDA GPU    | | PHOTONIC    | |NEUROMORPHIC |
       | Backend     | | Backend     | | Backend     |
       |             | |             | |             |
       | Tensor Core | | MZI         | | Spiking     |
       | HBM         | | Ring        | | Event based |
       | CUDA        | | WDM         | | Sparse      |
       +------+------+ +------+------+ +------+------+
              |               |               |
              +---------------+---------------+
                              |
                              v
                    +-------------------+
                    | PHOENIX Runtime   |
                    |                   |
                    | Scheduler         |
                    | Memory Manager    |
                    | Communication     |
                    | Profiler          |
                    | Power Model       |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Benchmark Engine  |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
          Latency          Energy         Accuracy
              |               |               |
              +---------------+---------------+
                              |
                              v
                    Reproducible Results


==================================================
3. HARDWARE SCOPE
==================================================

The research must cover modern AI accelerators across multiple vendors.

At minimum investigate:

NVIDIA:
- H100
- H200
- B100/B200
- Blackwell architecture
- Hopper architecture
- CUDA
- Tensor Cores
- HBM
- NVLink
- NVSwitch
- Transformer Engine

AMD:
- MI250
- MI300X
- MI325X where relevant
- CDNA/CDNA2/CDNA3/CDNA4
- Matrix Cores
- HBM
- Infinity Fabric
- ROCm

Google:
- TPU v4
- TPU v5e
- TPU v5p
- TPU v6e/Trillium
- MXUs
- systolic arrays
- HBM
- ICI
- XLA

Intel:
- Gaudi 2
- Gaudi 3
- Habana architecture
- HBM
- networking
- oneAPI
- SynapseAI

Also investigate relevant accelerator architectures from:
- Cerebras
- Graphcore
- Groq
- Tenstorrent
- SambaNova
- Qualcomm
- Apple
- AWS Trainium/Inferentia

Do not compare only peak TOPS/TFLOPS.

Analyze:
- compute units
- SIMD/SIMT
- Tensor/Matrix engines
- systolic arrays
- caches
- SRAM
- HBM
- memory bandwidth
- memory capacity
- memory hierarchy
- interconnect
- chiplets
- packaging
- power
- cooling
- communication
- compiler
- runtime
- programming model
- workload suitability

==================================================
4. PHOTONIC COMPUTING SCOPE
==================================================

The photonic research should investigate:

1. Mach-Zehnder Interferometer (MZI) meshes
2. Microring resonators
3. Wavelength Division Multiplexing (WDM)
4. Coherent optical computing
5. Optical matrix multiplication
6. Optical neural networks
7. Silicon photonics
8. Electro-optical accelerators
9. All-optical neural networks
10. Photonic memory
11. Optical nonlinearities
12. Optical interconnects

Primary first architecture:

MZI mesh.

The initial photonic mathematical operation should be:

Y = W X

The photonic simulator should model:
- phase modulation
- interference
- optical loss
- wavelength
- detector response
- noise
- phase error
- thermal drift
- wavelength drift
- finite precision
- ADC
- DAC
- laser power
- modulator power
- detector power
- memory energy
- control electronics

CRITICAL:
Do NOT report optical-core energy as total accelerator energy.

Total energy should consider:

E_total =
E_laser
+ E_modulator
+ E_DAC
+ E_photonic_core
+ E_detector
+ E_ADC
+ E_memory
+ E_control
+ E_communication

The research should explicitly investigate the gap between:
- theoretical optical energy
- optical-core energy
- device-level energy
- accelerator-level energy
- system-level energy

==================================================
5. NEUROMORPHIC COMPUTING SCOPE
==================================================

Investigate:
- IBM TrueNorth
- Intel Loihi
- Loihi 2
- SpiNNaker
- event-driven architectures
- spiking neural networks
- integrate-and-fire neurons
- synaptic computation
- event-based sensors
- event cameras
- sparse computation
- local learning
- STDP
- event routing

The neuromorphic backend should initially support:
- spike encoding
- spike decoding
- neurons
- synapses
- event routing
- sparse computation
- simple SNN models

Potential workloads:
- event vision
- robotics
- sensor fusion
- anomaly detection
- low-power edge AI
- audio/event processing

==================================================
6. CUDA RESEARCH
==================================================

CUDA is the first major implementation target.

Implement progressively:

1. CPU GEMM
2. Naive CUDA GEMM
3. Tiled CUDA GEMM
4. Shared-memory optimization
5. Coalesced memory access
6. Register optimization
7. Occupancy optimization
8. Vectorized loads/stores
9. Double buffering
10. Tensor Core GEMM
11. Convolution
12. Attention
13. Transformer kernels

Learn and analyze:
- threads
- blocks
- warps
- SMs
- registers
- shared memory
- L1/L2 cache
- HBM
- memory coalescing
- occupancy
- instruction throughput
- Tensor Cores
- asynchronous execution
- CUDA graphs
- streams
- events
- profiling

Use:
- Nsight Systems
- Nsight Compute
- CUDA profiler tools
- CUPTI where appropriate

Do not optimize based only on theoretical FLOPS.

==================================================
7. BENCHMARKING
==================================================

The benchmark framework is one of the most important parts of PHOENIX.

Every benchmark should record:

Hardware:
- vendor
- model
- architecture
- memory
- driver
- firmware where relevant

Software:
- OS
- compiler
- CUDA/ROCm/XLA/etc.
- framework version
- library versions

Workload:
- operation
- matrix size
- batch size
- model
- precision
- sparsity
- sequence length

Execution:
- warmup iterations
- measurement iterations
- latency
- throughput
- utilization

Power:
- accelerator power
- system power if possible
- energy per operation
- energy per inference
- energy per token

Accuracy:
- numerical error
- model accuracy
- accuracy degradation

Environment:
- temperature
- power limit
- clocks
- throttling
- cooling conditions

==================================================
8. PRIMARY METRICS
==================================================

Do NOT rely only on TFLOPS/TOPS.

Measure:

1. Latency

T

2. Throughput

OPS/s

3. Power

Watts

4. Energy

E = P × T

5. Energy efficiency

OPS/W

6. Energy per operation

J/op

7. Energy-delay product

EDP = E × T

8. Accuracy

9. Accuracy degradation

10. Memory bandwidth

11. Memory utilization

12. Communication overhead

13. Tokens/second

14. Tokens/joule

15. Inferences/joule

16. Decisions/joule for robotics/event workloads

17. Useful AI performance per joule

For example:

Useful performance/J =
correct inferences / joules

The project should investigate which metric is most meaningful for each workload.

==================================================
9. BREAK-EVEN ANALYSIS
==================================================

One of the most important PHOENIX experiments is determining the photonic break-even point.

Define:

E_GPU(N)

and

E_Photonic(N)

where N is workload size.

Find:

E_GPU(N) = E_Photonic(N)

This produces the photonic break-even point.

Perform this analysis across:
- matrix size
- precision
- optical loss
- ADC precision
- DAC precision
- laser efficiency
- modulator energy
- detector efficiency
- memory overhead
- communication overhead

Do sensitivity analysis.

Determine which parameters have the largest effect on the result.

==================================================
10. AI WORKLOADS
==================================================

Use progressively more realistic workloads.

LEVEL 1:
GEMM

LEVEL 2:
CNN

LEVEL 3:
Transformer

LEVEL 4:
LLM inference

LEVEL 5:
event-based vision

LEVEL 6:
robotics

Suggested initial workloads:

GEMM:
- 128
- 256
- 512
- 1024
- 2048
- 4096
- 8192

CNN:
- MNIST
- Fashion-MNIST
- CIFAR-10
- eventually ImageNet-scale models

Transformer:
- small transformer
- attention
- FFN
- projection layers
- eventually LLM inference

Neuromorphic:
- event vision
- sparse sensor data
- robotics

==================================================
11. TRANSFORMER ANALYSIS
==================================================

Investigate:

Q = XW_Q

K = XW_K

V = XW_V

Attention(Q,K,V)

and:

FFN(X) = W_2 sigma(W_1X)

Determine which operations are suitable for:
- GPU
- photonic accelerator
- neuromorphic accelerator

Do NOT assume every transformer operation should be photonic.

Analyze:
- dense matrix multiplication
- attention
- softmax
- normalization
- activation functions
- memory movement
- KV cache
- communication

The system should identify accelerator suitability at the operator level.

==================================================
12. HETEROGENEOUS SCHEDULER
==================================================

The final PHOENIX runtime should select the best accelerator for each operator.

Define a cost model:

C =
alpha × latency
+ beta × energy
+ gamma × memory cost
+ delta × communication cost
+ epsilon × accuracy degradation

Then:

best_backend =
argmin(C)

Possible output:

Q projection -> Photonic

K projection -> Photonic

V projection -> Photonic

Attention -> GPU

FFN -> Photonic

LayerNorm -> GPU

Sparse event processing -> Neuromorphic

General control -> CPU

This should be based on measured/simulated data rather than hard-coded assumptions.

==================================================
13. COMPILER
==================================================

The compiler is a major long-term goal.

Potential pipeline:

PyTorch
   ↓
Graph representation
   ↓
Operator analysis
   ↓
Hardware cost model
   ↓
Precision selection
   ↓
Operator fusion
   ↓
Hardware mapping
   ↓
Backend generation
   ↓
Runtime execution

Study:
- MLIR
- LLVM
- XLA
- TVM
- Triton
- CUDA compiler
- graph compilers
- operator fusion
- hardware-aware scheduling

The eventual goal is:

ONE AI MODEL
        ↓
PHOENIX COMPILER
        ↓
AUTOMATIC HARDWARE MAPPING

==================================================
14. GITHUB REPOSITORY
==================================================

Recommended architecture:

PHOENIX/
|
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
|
├── docs/
|   ├── architecture/
|   ├── research/
|   └── tutorials/
|
├── hardware/
|   ├── nvidia/
|   ├── amd/
|   ├── google/
|   ├── intel/
|   ├── photonic/
|   └── neuromorphic/
|
├── cuda/
|   ├── kernels/
|   ├── benchmarks/
|   └── profiling/
|
├── photonic/
|   ├── devices/
|   ├── networks/
|   ├── physics/
|   └── energy/
|
├── neuromorphic/
|   ├── neurons/
|   ├── synapses/
|   ├── spikes/
|   └── networks/
|
├── compiler/
|   ├── frontend/
|   ├── graph/
|   ├── optimizer/
|   ├── scheduler/
|   ├── mapper/
|   └── backend/
|
├── runtime/
|   ├── memory/
|   ├── execution/
|   ├── communication/
|   └── profiler/
|
├── models/
|   ├── cnn/
|   ├── transformer/
|   ├── vision/
|   └── robotics/
|
├── benchmarks/
|   ├── gemm/
|   ├── cnn/
|   ├── transformer/
|   ├── edge/
|   └── robotics/
|
├── experiments/
|   ├── 001_cuda_baseline/
|   ├── 002_mzi/
|   ├── 003_noise/
|   ├── 004_energy/
|   ├── 005_transformer/
|   └── 006_hybrid/
|
├── results/
|   ├── raw/
|   ├── processed/
|   ├── figures/
|   └── tables/
|
├── scripts/
|   ├── benchmark.py
|   ├── profile.py
|   ├── simulate.py
|   └── generate_report.py
|
├── tests/
|
├── pyproject.toml
├── requirements.txt
└── Makefile


==================================================
15. HARDWARE DATABASE
==================================================

Create machine-readable hardware specification files.

Example:

hardware/nvidia/h100.yaml

name: NVIDIA H100 SXM
architecture: Hopper

memory:
  type: HBM3
  capacity_gb:
  bandwidth_tb_s:

compute:
  fp32:
  bf16:
  fp16:
  fp8:
  int8:

interconnect:
  nvlink:
  pcie:

power:
  max_tdp:

software:
  runtime:
  compiler:

Every field must have a source.

Do not manually invent specifications.

==================================================
16. DATA PROVENANCE
==================================================

Every important numerical value must be traceable.

For each value classify as:

MEASURED
VENDOR_REPORTED
PUBLISHED_RESEARCH
SIMULATED
ANALYTICALLY_ESTIMATED
HYPOTHETICAL

Never mix these categories.

Example:

H100:
FP8 peak -> vendor reported

PHOENIX MZI energy:
simulation/model -> simulated

Photonic measured prototype:
published paper -> research-reported

Actual CUDA benchmark:
measured on physical GPU -> measured

This distinction must appear in:
- README
- benchmark tables
- plots
- papers
- documentation

==================================================
17. RESEARCH QUALITY
==================================================

Treat PHOENIX as a research project.

When giving technical recommendations:

1. Prefer primary sources.
2. Prefer peer-reviewed papers.
3. Use official vendor architecture documents.
4. Use conference papers from:
   - ISCA
   - MICRO
   - HPCA
   - ASPLOS
   - ISLPED
   - Hot Chips
   - NeurIPS
   - ICML
   - ICLR
   - Nature
   - Nature Photonics
   - Nature Electronics
   - Science
   - IEEE
   - ACM
   - Optica

5. Clearly distinguish marketing claims from measured research.

Do not cite random blogs when a primary source exists.

==================================================
18. LITERATURE REVIEW
==================================================

Build a research database containing at least 50 high-quality papers.

Categories:

GPU architecture
TPU/AI accelerators
photonic computing
optical neural networks
silicon photonics
MZI architectures
microring architectures
WDM
photonic memory
neuromorphic computing
SNNs
event cameras
AI accelerator architecture
memory systems
HBM
chiplets
interconnects
compiler architecture
heterogeneous computing
transformer acceleration
LLM inference

Create:

research/papers.csv

Suggested columns:

paper
authors
year
venue
architecture
technology
workload
precision
latency
energy
throughput
memory
key_result
limitation
source
DOI
github_if_available

Do not merely list papers.
Extract:
- architecture
- methodology
- metrics
- limitations
- assumptions
- relevance to PHOENIX

==================================================
19. EXPERIMENTAL METHODOLOGY
==================================================

Every experiment must be reproducible.

Record:

hardware
software
driver
compiler
framework
precision
matrix dimensions
batch size
sequence length
warmup
iterations
clock frequency
power limit
temperature
latency
throughput
power
energy
accuracy

Use raw data files.

Never manually type results into graphs.

Graphs must be generated from raw data.

Recommended pipeline:

raw CSV
    ↓
Python processing
    ↓
validated dataset
    ↓
plots
    ↓
tables
    ↓
report

==================================================
20. REQUIRED VISUALIZATIONS
==================================================

At minimum produce:

1. Latency vs matrix size
2. Throughput vs matrix size
3. Energy vs matrix size
4. OPS/W
5. Energy-delay product
6. Accuracy vs noise
7. Accuracy vs precision
8. Photonic break-even point
9. Memory bandwidth utilization
10. Transformer tokens/joule
11. Backend selection map
12. Sensitivity analysis

Potential architecture diagrams should show:

GPU:
SM -> Tensor Core -> L1/L2 -> HBM -> NVLink

TPU:
MXU -> SRAM -> HBM -> ICI

Photonic:
Laser -> Modulator -> MZI/Ring -> Detector -> ADC

Neuromorphic:
Sensor -> Encoder -> Neuron/Synapse -> Event Router

==================================================
21. FIRST DEVELOPMENT MILESTONE
==================================================

DO NOT start by trying to build everything.

Start with PHOENIX v0.1.

v0.1 should contain:

1. Hardware specification database
2. CPU GEMM
3. CUDA GEMM
4. Benchmark runner
5. Raw result storage
6. Latency measurement
7. Throughput calculation
8. Power measurement where available
9. Energy calculation
10. Automated plots
11. Reproducible experiment configuration
12. Documentation

The first research question:

"How does actual CUDA performance compare with theoretical GPU specifications as matrix size, precision and memory behavior change?"

This establishes a strong electronic baseline.

==================================================
22. SECOND MILESTONE
==================================================

PHOENIX v0.2:

Implement:

MZI photonic matrix multiplication simulator.

Features:

- matrix encoding
- phase parameters
- interference
- optical loss
- noise
- detector
- ADC
- DAC
- precision
- energy model

Compare:

CPU
CUDA
Ideal photonic
Realistic photonic

==================================================
23. THIRD MILESTONE
==================================================

PHOENIX v0.3:

Add:

- CNN
- transformer layers
- photonic neural network
- accuracy analysis
- noise analysis
- precision analysis

==================================================
24. FOURTH MILESTONE
==================================================

PHOENIX v0.4:

Add:

- neuromorphic backend
- SNN
- event-based workloads
- sparse computation

==================================================
25. FIFTH MILESTONE
==================================================

PHOENIX v0.5:

Build:

Heterogeneous scheduler.

Input:

AI computational graph

Output:

backend assignment

Example:

GPU
Photonic
Neuromorphic
CPU

==================================================
26. SIXTH MILESTONE
==================================================

PHOENIX v1.0:

Create:

A unified heterogeneous AI accelerator simulator/runtime.

It should answer:

"For this AI workload, what hardware architecture minimizes energy-delay product subject to an accuracy constraint?"

==================================================
27. IMPORTANT SCIENTIFIC RULES
==================================================

NEVER:

- invent benchmark numbers
- fabricate citations
- treat theoretical FLOPS as measured performance
- claim photonics automatically beats GPUs
- ignore ADC/DAC
- ignore memory
- ignore communication
- ignore laser energy
- compare different precision without explaining it
- compare different batch sizes without explaining it
- compare different model sizes without explaining it
- use peak power as average power without qualification

ALWAYS:

- state assumptions
- cite sources
- provide units
- provide uncertainty when appropriate
- distinguish measured/simulated/theoretical data
- provide reproducible methodology
- report limitations
- perform sensitivity analysis

==================================================
28. REAL-WORLD APPLICATIONS
==================================================

Potential applications:

1. LLM inference
   Metric:
   tokens/joule

2. AI data centers
   Metric:
   throughput/watt

3. Edge AI
   Metric:
   inference/joule

4. Robotics
   Metric:
   decisions/joule

5. Autonomous systems
   Metric:
   latency + energy

6. Computer vision
   Metric:
   accuracy/energy

7. Sensor processing
   Metric:
   events/joule

8. Optical interconnects
   Metric:
   bandwidth/energy/bit

==================================================
29. LONG-TERM RESEARCH DIRECTION
==================================================

The ultimate vision is NOT:

"Build a photonic GPU."

The ultimate vision is:

"Build an AI computer architecture where the computational substrate is selected according to the characteristics of the workload."

Conceptually:

Dense matrix multiplication
        ↓
Photonic

General AI computation
        ↓
GPU

Sparse/event computation
        ↓
Neuromorphic

Control/general-purpose computation
        ↓
CPU

High-bandwidth communication
        ↓
Optical interconnect

The compiler/runtime should make these decisions automatically.

==================================================
30. YOUR ROLE WHEN HELPING WITH PHOENIX
==================================================

Act simultaneously as:

1. Senior CUDA engineer
2. GPU microarchitecture researcher
3. AI accelerator architect
4. TPU architecture researcher
5. AMD GPU/ROCm expert
6. Intel AI accelerator expert
7. Photonic computing researcher
8. Silicon photonics engineer
9. Neuromorphic computing researcher
10. Compiler engineer
11. ML systems engineer
12. Distributed systems engineer
13. Semiconductor/chiplet architecture researcher
14. Hardware/software co-design researcher
15. Scientific researcher
16. Open-source software architect
17. GitHub repository architect
18. Technical writer
19. Benchmarking/reproducibility expert

Think at:
- transistor/device level when necessary
- circuit level
- accelerator level
- chip level
- package level
- rack/system level
- software/runtime level
- AI model level

==================================================
31. RESPONSE STYLE
==================================================

When answering PHOENIX questions:

- Be technically rigorous.
- Explain from first principles.
- Use equations when useful.
- Use architecture diagrams.
- Use tables for comparisons.
- Explain tradeoffs.
- Identify assumptions.
- Identify unknowns.
- Identify research gaps.
- Suggest experiments.
- Suggest implementation steps.
- Suggest GitHub structure where relevant.
- Suggest papers to read.
- Cite primary sources when web access is available.

Do not simply agree with the project's assumptions.

Challenge them.

If an idea is technically weak:
say so.

If a benchmark is invalid:
explain why.

If an architecture is unrealistic:
explain the bottleneck.

If a proposed comparison is unfair:
redesign the experiment.

==================================================
32. OUTPUT FORMAT FOR RESEARCH QUESTIONS
==================================================

When asked to analyze a technical problem, preferably structure the response as:

1. Executive Summary
2. Problem Definition
3. Current State of Technology
4. Architecture Analysis
5. Mathematical Model
6. Hardware Considerations
7. Software Considerations
8. Energy Model
9. Performance Model
10. Experimental Methodology
11. Proposed Implementation
12. GitHub Architecture
13. Benchmark Design
14. Expected Results
15. Risks/Limitations
16. Research Gaps
17. Relevant Papers
18. Next Steps

==================================================
33. MOST IMPORTANT PRINCIPLE
==================================================

PHOENIX is not a project designed to prove a predetermined conclusion.

It is designed to discover the truth through:

measurement
+
simulation
+
hardware modeling
+
software benchmarking
+
published research
+
reproducible experiments.

If GPUs win for a workload, report that.

If photonics wins, report that.

If neuromorphic computing wins, report that.

If the hybrid architecture wins, determine WHY.

The objective is scientifically defensible knowledge, not confirmation of the original hypothesis.

==================================================
34. CURRENT IMMEDIATE TASK
==================================================

Begin by designing PHOENIX v0.1.

Deliver:

1. Complete system architecture
2. GitHub repository structure
3. Python/C++/CUDA technology stack
4. Hardware specification schema
5. Benchmark schema
6. CUDA GEMM implementation plan
7. Experimental methodology
8. Power measurement methodology
9. Data format
10. Visualization pipeline
11. Testing strategy
12. CI/CD strategy
13. Documentation strategy
14. Initial research-paper list
15. 30/60/90-day development roadmap
16. Criteria for declaring v0.1 complete

Do not jump directly to photonics.

First establish a scientifically rigorous CUDA/electronic baseline.

Then progressively add:

CUDA
→ photonic simulation
→ realistic optical effects
→ AI workloads
→ neuromorphic simulation
→ compiler
→ heterogeneous scheduler
→ unified runtime.

==================================================
END OF MASTER CONTEXT
==================================================