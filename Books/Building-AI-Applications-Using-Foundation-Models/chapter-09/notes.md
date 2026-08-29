# Chapter 9: Inference Optimization

**Inference** = running the model on hardware to serve actual user requests. This chapter covers techniques to optimize a model's inference behavior for specific hardware and workloads.

---

## Topic 1: Understanding Inference Optimization

AI has two core phases: **training** and **inference**. Inference is the process of generating output in response to a user request.

1. The system that runs inference is called the **inference server** — it hosts the available model and has access to the necessary hardware.
2. Inference workloads face **two main computational bottlenecks:**

| Bottleneck | Definition |
|---|---|
| **Compute-bound** | Execution time is limited by the hardware's raw **processing power** (how many operations it can perform). |
| **Memory bandwidth-bound** | Execution time is limited by the **data transfer rate** within the system — e.g., how fast data can move between the processor and memory. |

3. For LLMs, the balance of **prefilling** vs. **decoding** work is affected by: **context length**, **output length**, and **request batching strategy** — and this in turn determines which bottleneck (compute vs. memory bandwidth) dominates for a given workload.

> **Note:** Many providers offer two kinds of APIs — **online** (real-time, low-latency) and **batch** (higher latency, often cheaper, processed in bulk). Which one to use depends on the application and how sensitive it is to latency.

---

## Topic 2: Inference Performance Metrics

| Metric | What it measures |
|---|---|
| **Latency** | Overall response time. |
| **TTFT** (Time to First Token) | How long before the *first* output token appears. |
| **TPOT** (Time Per Output Token) | Average time to generate each subsequent token. |
| **Time Between Tokens / Inter-token Latency** | Gaps between consecutive generated tokens — affects perceived "smoothness" of streaming output. |
| **Throughput** | Total volume of output the system can generate across all requests in a given time. |
| **Goodput** | Throughput that actually meets a target latency/quality requirement (i.e., "useful" throughput, not just raw volume). |
| **Utilization** | How efficiently the hardware's available capacity is being used. |
| **MFU** (Model FLOP/s Utilization) | Ratio of actual compute achieved vs. the hardware's theoretical peak compute. |
| **MBU** (Model Bandwidth Utilization) | Ratio of actual memory bandwidth achieved vs. theoretical peak bandwidth. |

> **Reading MFU/MBU together:** A **compute-bound** workload tends to show **higher MFU, lower MBU** (it's maxing out the chip's math throughput, not its memory bus). A **memory-bandwidth-bound** workload tends to show **lower MFU, higher MBU** (it's maxing out data movement, while the compute units sit relatively idle waiting for data).

---

### AI Accelerators

An **accelerator** is a chip designed to speed up a specific type of computational workload. An **AI accelerator** is purpose-built for AI workloads — examples: **GPUs** and **TPUs**.

### Computational Capabilities

- Commonly measured in **FLOP/s** (floating point operations per second) — how many operations the chip can perform.
- **Utilization** = ratio of **actual achieved FLOP/s** to the chip's **theoretical peak FLOP/s**.
- **Numeric precision affects throughput:** higher precision (e.g., FP32) → fewer operations per second possible; lower precision (e.g., FP16, INT8) → more operations per second, since each operation is cheaper/simpler for the hardware.

### Memory Size and Bandwidth

GPU memory needs **higher bandwidth and lower latency** than CPU memory, since fast data transfer within the system is critical for keeping compute units fed.

- **CPU memory** typically uses **DDR (Double Data Rate Synchronous Dynamic RAM)**, which has a relatively simple **2D structure**.
- **GPU memory** uses **HBM (High Bandwidth Memory)**, built with a **3D structure** — stacking memory layers vertically to dramatically increase bandwidth.

**GPUs interact with three levels of memory:**

| Level | Description |
|---|---|
| **CPU memory** | Accelerators are deployed physically close to the CPU for fast access to CPU-side memory when needed. |
| **GPU HBM** | Memory dedicated to the GPU itself — the GPU's main working memory. |
| **GPU on-chip RAM** | Integrated directly into the chip; used to store the most **frequently accessed** data for the fastest possible access (smallest but fastest tier). |

### Power Consumption

Every computational operation requires switching transistors on and off, and this switching activity **consumes power** — the more operations performed per second (and the more transistors involved), the higher the power draw and resulting **heat** that needs to be dissipated. This is a key reason accelerator design has to balance raw compute power against power efficiency and cooling requirements, especially at data-center scale where thousands of chips run simultaneously.

Accelerators typically report power consumption as either:
- **Power draw** (actual measured power usage), or
- **TDP** (Thermal Design Power) — a proxy metric representing the maximum heat the chip is expected to generate under sustained load, used to design adequate cooling.

---

## Topic 3: Inference Optimization

Inference optimization can be applied at **three levels**: **model**, **hardware**, and **service**. This chapter focuses mainly on **model-level** and **service-level** optimization.

---

### Model-Level Optimization

**Goal:** make the model itself more efficient — this may change model behavior, so **finetuning afterward** is often needed to recover any lost quality.

#### A) Model Size

Reduced via:
- **Model compression techniques** — **quantization** and **model distillation** (see Chapters 7 & 8).
- **Pruning** — removing less-important parameters/connections.

#### B) Autoregressive Decoding

Autoregressive models generate tokens **one at a time**, which is a fundamental bottleneck. Three approaches to address this:

**1. Speculative Decoding**
- A **faster, smaller "draft" model** generates a candidate sequence of tokens.
- The **target model** (the actual model you want to use for your application) then **verifies** these tokens in a single pass, accepting the ones that match what it would have generated itself.
- This works because verifying a sequence is cheaper than generating it token-by-token from scratch.

**2. Parallel Decoding**
- Given an existing sequence $x_1, x_2, ..., x_t$, the aim is to generate $x_{t+1}, x_{t+2}, ..., x_{t+k}$ **simultaneously**, rather than one at a time.
- This works because the existing sequence often already carries enough information to predict several tokens ahead, especially in natural text.
- **Medusa** is an example technique — it trains **multiple decoding heads**, each dedicated to predicting a different future token position.
- **Critical requirement:** since these tokens aren't generated sequentially, robust **evaluation/verification** of the parallel predictions is essential to catch cases where the shortcut assumption breaks down.

**3. Inference with Reference**
- Responses often need to **reuse tokens directly from the input** (e.g., quoting back part of a document).
- Instead of regenerating these tokens from scratch, the system can **copy them directly** from the context.
- **Challenge:** developing an algorithm that can correctly identify the most relevant text span from the context to copy, at each decoding step.
- **Limitation:** only useful when there's substantial **overlap** between the context and the expected output (e.g., summarization, extraction tasks) — less useful for generative tasks with little verbatim overlap.

#### C) Attention Mechanism Optimization

Generating the next token requires the **key-value (KV) vectors** of all previous tokens.

- **KV Cache:** instead of recomputing KV vectors for the *entire* sequence at every step, store (cache) the KV vectors computed so far, and only compute the KV vector for the **newest token** each step.
- **Redesigning the attention mechanism itself** — a more invasive change, only feasible during **training or finetuning** (not something you can just bolt onto a deployed model):
  - **Windowed/Local attention:** when computing attention for the current token, only consider KV vectors of tokens within a fixed **window size $k$**, rather than the entire sequence — trading some long-range context for speed.
  - This local attention can be **interleaved with global attention layers**, so local layers capture nearby context efficiently while global layers still capture task-relevant information across the *entire* document/context.

**Other attention variants:**
- **Cross-layer attention**
- **Multi-query attention (MQA)**
- **Grouped-query attention (GQA)**

**KV cache optimization frameworks:**
- **vLLM** introduced **PagedAttention** — manages the KV cache more like OS-style virtual memory paging, reducing wasted memory.

**Hardware-specific kernels:**
- Engineers often write **custom kernels** (e.g., in **CUDA**) specifically optimized for how a given chip computes attention scores.
- A **kernel** = code optimized for a specific chip's architecture.
- **FlashAttention** is one of the most well-known attention kernels — it works by **fusing multiple operations into one**, reducing memory reads/writes and speeding up computation significantly.

---

### Inference Service Optimization

**Goal:** efficient **resource management** — allocating compute/memory resources to workloads efficiently.

> **Key distinction from model-level optimization:** service-level techniques do **not modify the model** and do **not change output quality** — they only change *how efficiently* the existing model is served.

#### 1. Batching

Instead of processing one request at a time, requests are grouped and processed together.

| Type | How it works | Downside |
|---|---|---|
| **Static Batching** | Wait until a **fixed number** of requests arrive, process them all together, return results together. | Requests that finish early must **wait** for the whole batch; waiting for the batch to fill can also be slow if traffic is light. |
| **Dynamic Batching** | Wait for a **fixed time window** (e.g., 100ms); process whatever requests arrived in that window. If enough requests arrive *before* the window ends, process early without waiting the full duration. | Still doesn't fully solve uneven per-request processing times within the batch. |
| **Continuous Batching** | As soon as any request in the batch **finishes**, its response is returned immediately and a **new request takes its slot** — batch composition keeps changing continuously rather than processing in discrete rounds. | Most efficient of the three; the modern standard approach (used by frameworks like vLLM). |

#### 2. Decoupling Prefill and Decode

- **Prefill** (processing the input prompt) is **compute-bound**.
- **Decode** (generating output tokens one by one) is **memory-bandwidth-bound**.

Because they have fundamentally different resource profiles, these two phases are often **disaggregated** — run on separate, differently-optimized instances/hardware. The ratio of prefill instances to decode instances is tuned based on workload characteristics and latency requirements.

#### 3. Prompt Caching

Prompts often share **overlapping text segments** across requests — the most common example being a shared **system prompt**.

- A **prompt cache** stores this overlapping portion, so the model doesn't have to reprocess the same system prompt (or other repeated context) on **every single query**.
- Especially valuable for applications involving **long documents** repeated across many queries (e.g., "answer questions about this same 50-page document" scenarios).

#### 4. Parallelism

The more work that can run **in parallel**, the lower the overall latency. Two broad families apply across AI generally, with two more specific to LLMs:

| Strategy | Description |
|---|---|
| **Replica Parallelism** | Create multiple **replicas** of the entire model, and distribute incoming requests across them to be processed in parallel. |
| **Model Parallelism** | Split a **single model** across multiple machines. Sub-approaches include: **Tensor Parallelism** (splitting individual weight matrices/computations across devices) and **Pipeline Parallelism** (splitting the model's *layers* across devices, with data flowing through them like a pipeline). |
| **Context Parallelism** *(LLM-specific)* | The **input sequence itself** is split across different devices to be processed separately. |
| **Sequence Parallelism** *(LLM-specific)* | The operations needed for the **entire input** are split across machines (splitting the computation, rather than splitting the input data itself). |