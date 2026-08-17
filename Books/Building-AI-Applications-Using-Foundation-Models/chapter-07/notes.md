# Chapter 7: Finetuning

**Finetuning** is the process of adapting a model to a specific task — much like **transfer learning** in traditional deep learning/ML.

It's mostly done to improve a model's **efficiency and accuracy** in following instructions for a target task.

> Because finetuning is so **memory-intensive**, memory-efficient methods have become important — most notably **Parameter-Efficient Finetuning (PEFT)** ⭐ *(important concept — covered in depth in Topic 4)*.

> **Note:** Finetuning isn't the only way to do transfer learning — **feature-based transfer** is another approach (using a pretrained model's learned features/embeddings as input to a separate downstream model, without updating the original model's weights).

---

## Topic 1: Overview

1. Finetuning is part of a model's overall training pipeline — essentially an **extension of pretraining**.
2. Before jumping straight to expensive **task-specific** labeled data, it's often worth first finetuning via **self-supervision** using cheaper, task-*related* (but unlabeled) data.

   > **Example:** Suppose you want a model to get better at answering **legal contract questions**. Instead of immediately collecting expensive (question, answer) pairs written by lawyers, you could first self-supervise the model on a large corpus of **unlabeled legal documents** (contracts, case law) — teaching it legal vocabulary and phrasing patterns cheaply — *before* spending money on the smaller, expensive labeled Q&A dataset for the final supervised finetuning step.

3. **Supervised Finetuning (SFT):** the model is trained on **(input, output)** pairs — e.g., input = an instruction, output = the desired response.
4. Beyond self-supervision, a model can also be finetuned via **reinforcement learning** — often called **preference finetuning**. This requires **comparative data** shaped like: `(instruction, winning response, losing response)`.
5. **Long-context finetuning** — finetuning a model to extend its context length. This requires modifying the model's **architecture** itself, making it far more complex. Intuitively, this should be a **last resort**, not a first move.

---

## Topic 2: When to Finetune

### Reasons to Finetune

1. To improve model **quality** — both general capability and task-specific capability.
2. If a model is observed to be **particularly biased**, finetuning can (perhaps surprisingly) be an effective fix.

> **Note:** It's often more beneficial to finetune a **smaller** model rather than attempting to finetune a larger one.
> - One way to do this: have the smaller model **imitate** the behavior of a larger model, training it on the larger model's outputs. This is known as **distillation**.

### Reasons *Not* to Finetune

1. **Catastrophic forgetting risk** — finetuning on one task can degrade performance on other tasks.
   > If a model needs to perform well across multiple distinct tasks, consider using **different specialized models per task**, then combining them, instead of one model finetuned to try to do everything.
2. Finetuning is **not easy** — it demands real ML skill and domain expertise.
3. Post-finetuning, you must decide how to **host and serve** the model — self-hosted vs. API-based, infrastructure costs, etc.

> **Always try prompt engineering first** before resorting to finetuning. Prompt experimentation naturally produces useful byproducts — **evaluation pipelines** and **data annotation guidelines** — which directly carry over and help *if* you do eventually decide to finetune.

---

## Topic 3: Finetuning vs. RAG

The choice between RAG and finetuning depends on **what you're trying to improve**:

| Problem | Likely Fix |
|---|---|
| Model gives **factually wrong** answers | **RAG** — the model likely just lacks access to the right information. |
| Model's **behavior/style/format** is wrong (e.g., can't write correct SQL or a domain-specific query syntax) | **Finetuning** — the model likely hasn't seen enough examples of that pattern during training. |

**Recommended approach:**
1. If answers are factually wrong, try **RAG first** — it's simpler to iterate on than finetuning.
   - Start with the **simplest retrieval method** (e.g., term-based search like BM25) before reaching for more complex techniques (embeddings, hybrid, re-ranking).
2. If RAG alone doesn't fix it, layer in more **prompt engineering** on top and re-evaluate.
3. If the issue is fundamentally about the model's **behavior** rather than its knowledge, that's when you move to **finetuning**.

> **RAG and finetuning are not mutually exclusive** — combining both is a common way to maximize an application's overall performance (RAG for up-to-date facts, finetuning for behavior/format).

---

## Topic 4: Memory Bottleneck

As noted earlier, finetuning is highly **memory-intensive**.

1. At scale, **memory** becomes the key bottleneck — making both finetuning *and even inference* difficult for large models.
2. The main contributors to a model's memory footprint during finetuning:
   - **Number of parameters**
   - **Number of *trainable* parameters**
   - Their **numerical representation** (precision)
3. More trainable parameters → larger memory footprint. **Reducing the number of trainable parameters is the core motivation behind PEFT** (Parameter-Efficient Finetuning).

### Quantization

**Quantization** = converting a model's parameters from a higher-bit numeric representation (e.g., 32-bit float) to a lower-bit one (e.g., 8-bit or 4-bit integer), shrinking memory usage.

- **QLoRA** — a quantization technique mostly used to reduce memory **during inference** (combined with LoRA-style efficient finetuning).
- Quantization is **less common during training** itself.
  - **PTQ (Post-Training Quantization)** — quantize *after* training is complete.
  - **Quantization-Aware Training (QAT)** — trains the model to simulate low-precision behavior *during* training, so it learns to still produce high-quality output once actually quantized for inference. **Note:** this does *not* reduce training time — it may well increase it, since it's optimizing for a different (later) constraint.
- **Directly training in low precision** is difficult because **backpropagation is sensitive to precision** — small gradient values can get lost/rounded to zero at low bit-widths.
  - This is why **mixed precision** training is common: a copy of the weights is kept at **higher precision**, while other values (gradients, activations) are computed/stored at **lower precision**, balancing memory savings against training stability.

**Summary — Inference vs. Training precision:**

| | Precision used |
|---|---|
| **Inference** | Typically the **smallest** bit representation available (fastest, cheapest, no gradient stability concerns). |
| **Training** | Requires **higher precision** overall, though **mixed precision** lets simpler operations run in lower precision while sensitive ones stay high-precision. |

---

### Backpropagation and Trainable Parameters

The number of **trainable parameters** is what ultimately drives a model's memory footprint during finetuning.

| Phase | Parameters Updated? | Passes Executed |
|---|---|---|
| **Pretraining** | All parameters updated | Forward + Backward |
| **Finetuning** | Some *or* all parameters updated (depends on method — e.g., full finetuning vs. PEFT) | Forward + Backward |
| **Inference** | No parameters updated | Forward only |

**A quick primer on backpropagation:**
- The **forward pass** runs input through the model to produce an output and compute the **loss** (how wrong the output was).
- The **backward pass (backpropagation)** works backward through the network, using calculus (the chain rule) to compute the **gradient** for each parameter — essentially, *"how much did this specific weight contribute to the loss, and in which direction should it change?"*
- The **optimizer** then uses these gradients to actually decide **how much to update each weight** (its job is to translate "this weight contributed X to the loss" into "change this weight by Y amount"). Different optimizers (SGD, Adam, etc.) do this differently, and many optimizers (like Adam) also maintain their own extra **optimizer state** (e.g., running averages of past gradients) — adding yet another memory cost on top of the weights and gradients themselves.

So during training, memory has to hold: **weights + gradients + optimizer state + activations** — which is exactly why training needs so much more memory than inference (which only needs the weights, plus a small cache for generation).

---

### Memory Needed for Inference

$$
\text{Memory}_{\text{inference}} \approx N \times M
$$

where:
- $N$ = number of parameters
- $M$ = memory required per parameter (depends on precision — e.g., 2 bytes for FP16, 4 bytes for FP32)

This covers loading the parameters themselves. On top of that, transformer models need memory to store the **key-value (KV) vectors** used in the attention mechanism during generation. If this KV cache is estimated at roughly **30% of the model's weight memory**, the formula becomes:

$$
\text{Memory}_{\text{inference}} \approx N \times M \times 1.3
$$

> **Example:** A 7-billion-parameter model in FP16 (2 bytes/param):
> $$7{,}000{,}000{,}000 \times 2\ \text{bytes} \times 1.3 \approx 18.2\ \text{GB}$$

---

### Memory Needed for Training

$$
\text{Memory}_{\text{training}} \approx N \times (M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer state}} + M_{\text{activations}})
$$

In other words, training memory scales with the number of parameters **times** the combined per-parameter cost of: the weights themselves, their gradients, whatever state the optimizer keeps per parameter, and the activations cached for the backward pass.

> **Simplified numerical example:** Take a 1-billion-parameter model, finetuned fully (all params trainable) in FP16 (2 bytes/param), using **Adam** as the optimizer (Adam typically keeps 2 extra state values per parameter, often stored in FP32 = 4 bytes each → 8 bytes of optimizer state per parameter):
>
> | Component | Bytes per parameter | Total (1B params) |
> |---|---|---|
> | Weights (FP16) | 2 | 2 GB |
> | Gradients (FP16) | 2 | 2 GB |
> | Optimizer state (Adam, FP32) | 8 | 8 GB |
> | **Subtotal (excl. activations)** | **12** | **12 GB** |
>
> Activations add further memory on top of this, scaling with batch size and sequence length — which is exactly why techniques that reduce activation memory (below) matter so much in practice.

**Reducing activation memory:**
One way to cut memory used for storing activations is to simply **not store them** during the forward pass, and instead **recompute** them on the fly when needed during the backward pass.
- This is known as **gradient checkpointing** (a.k.a. **activation recomputation**).
- Trade-off: this **saves memory** but **increases training time**, since parts of the forward pass effectively get run twice.

---
## Topic 5: Finetuning Techniques

### Parameter-Efficient Finetuning (PEFT)

**Full finetuning** = the number of trainable parameters is **exactly equal** to the model's total number of parameters (everything gets updated).

**Partial finetuning** = only a **fraction** of the total parameters are trainable, while still aiming for performance comparable to full finetuning.

- However, naive partial finetuning is itself **parameter-inefficient** — it typically still requires updating roughly **~25% of parameters** to match full-finetuning performance on the GLUE benchmark.

This is the motivation for **PEFT (Parameter-Efficient Finetuning)**, introduced by **Houlsby et al. (2019)**. The paper showed that adding a small number of extra parameters at the *right locations* in a model can achieve strong finetuning performance with far fewer trainable parameters.
- The authors introduced **two adapter modules** into each transformer block of a BERT model.
- The model's original parameters stayed **frozen** — only the adapters were updated.
- **Trade-off:** this approach often **increases inference latency**, since extra layers must now be computed at inference time.

> PEFT methods generally deliver strong performance using **less memory** *and* **fewer training examples** compared to the full dataset typically needed for full finetuning.

---

### PEFT Technique Families

PEFT techniques fall into **two broad classes**:

**1. Adapter-Based Methods** — techniques that add **extra trainable weights** into the model.
- Most common: **LoRA** (see below).
- Others: **BitFit**, **IA3** (particularly efficient for **multi-task** finetuning).

**2. Soft Prompt Methods** — insert **soft prompts** (trainable *vector embeddings*, not actual words) alongside the input tokens.
- Called "soft" prompts because, just like original **hard prompts** (real input text), they also guide the model's behavior — but they're continuous, learned vectors rather than discrete tokens.

**Hard prompt vs. soft prompt — simplified flow:**

```
Hard Prompt (normal input):
  ["Translate", "this", "sentence", ":", "Hello"]  →  [Token Embeddings]  →  Transformer  →  Output
        (real words, human-readable)

Soft Prompt (PEFT):
  [ v1 ][ v2 ][ v3 ]  +  ["Hello"]  →  [Learned Vectors + Token Embeddings]  →  Transformer  →  Output
   (trainable vectors,        (real word)
    NOT actual words —
    frozen base model,
    only v1,v2,v3 are trained)
```

Related techniques that differ mainly in **where** the soft prompt vectors get inserted relative to the input: **Prefix-Tuning**, **P-Tuning**, and **Prompt Tuning**.

---

### LoRA (Low-Rank Adaptation)

> **Full form, for the record: LoRA = Low-Rank Adaptation.** (You'll see it written as "Lora," "LORA," "lora" etc. in raw notes — the correct notation is **LoRA**, and that's what's used consistently from here on.)

**How it works:**

Given a weight matrix $W$ of dimension $(n \times m)$, LoRA decomposes the *update* to $W$ into the product of two much smaller matrices:

1. Choose a **rank** $r$ (a small number, far smaller than $n$ or $m$). Create two matrices:
   - $A$ of dimension $(n \times r)$
   - $B$ of dimension $(r \times m)$
2. Their product $A \times B$ gives a matrix $\Delta W$ of the **same dimensions as $W$** — this is the low-rank approximation of the "ideal" weight update.
3. This is added to the original frozen weight matrix, scaled by a factor:

$$
W' = W + \frac{\alpha}{r} \cdot (A \times B)
$$

where:
- $W$ = original (frozen) weight matrix
- $\alpha$ = a hyperparameter controlling how much the new weights influence the final matrix
- $r$ = the chosen rank
- $W'$ = the effective weight matrix used from then on

4. **During finetuning, only $A$ and $B$ are updated** — the original $W$ stays frozen throughout.

LoRA is built on **low-rank factorization**, a technique long used for dimensionality reduction.

**Practical guidance:**
- LoRA is most commonly applied to the **attention weight matrices** — Query ($Q$), Key ($K$), and Value ($V$).
- LoRA is typically applied **uniformly** to all matrices of the same type — e.g., applying it to one query matrix means applying it to *all* query matrices across the model.
- If you can only afford to target **two** attention matrices, prioritize **Q and V**.
- Applying LoRA to the **feedforward layers** (not just attention) tends to give even better results.
- **Rank ($r$):** values between **4 and 64** are usually sufficient for most use cases.
- **Ratio $r:\alpha$** is most commonly set to **8:1** or **1:8** in practice.

**Why does LoRA work?**
It's believed that LLMs have a low **intrinsic dimension** — pretraining tends to minimize a model's intrinsic dimensionality, and **larger models tend to have even lower intrinsic dimension**. This means the *useful* update needed for a new task can often be captured well by a much lower-rank matrix than the full weight matrix would suggest.

---

### Serving LoRA Adapters

LoRA's modularity makes serving multiple finetuned variants much simpler. Two serving strategies:

| Strategy | Description | Best for |
|---|---|---|
| **1. Merge** | Combine $A$ and $B$ into the original weights to form $W'$ **before** serving. | Serving a **single** LoRA-finetuned model — no extra runtime overhead. |
| **2. Keep Separate** | Keep $W$, $A$, and $B$ separate, combining them **at inference time**. Adds some latency. | **Multi-LoRA serving** — multiple adapters sharing the same frozen base model. |

- Option 2 makes it easy to **switch between tasks on the fly** (just swap which adapter is applied).
- This also enables combining **multiple specialized models** instead of maintaining one giant model for every task — you can keep **one LoRA adapter per task**, all riding on the same base model.
- Publicly available LoRA adapters exist and can be used off-the-shelf, similar to pretrained models.

---

### QLoRA (Quantized LoRA)

An interesting variation: reduce memory usage further by **quantizing** the model's weights, activations, or gradients *during* finetuning.

- QLoRA uses a 4-bit format called **NF4 (Normal Float 4)**, which quantizes values based on the insight that pretrained weights typically follow a **normal distribution centered around zero (median 0)** — so the quantization levels are optimized for that distribution rather than spread out uniformly.
- Alongside NF4, QLoRA also uses an **efficient paging algorithm** that automates data transfer between GPU and CPU memory.
- **Main limitation:** the NF4 conversion process itself is **computationally costly**.

---

## Model Merging and Multi-Task Finetuning

You can take two (or more) foundation models and **combine** them to create a single, better-performing model. Any or all of the models being merged may have been individually finetuned beforehand.

This is especially relevant for **adapter-based** models: given two models finetuned from the *same base*, their adapters can be merged into a **single combined adapter**.

Model merging is one approach to **multi-task finetuning**, alongside:
- **Simultaneous finetuning** — training on multiple tasks at once.
- **Sequential finetuning** — training on tasks one after another.
- **Model merging** — finetune separately (often in parallel) on each task, then merge afterward. Finetuning each task in isolation lets the model learn that task more thoroughly before combining.

> Model merging is also a way to enable a simple form of **federated learning** — since each model can be finetuned independently (e.g., on different, siloed data sources) and only the resulting weights/adapters need to be shared and merged centrally, rather than pooling the raw data itself.

---

### Model Merging Approaches

Merging approaches differ in **how** the constituent parameters are combined. Three key approaches: **Summing**, **Layer Stacking**, and **Concatenation**.

#### 1. Summing

Involves adding the weight values of the constituent models together. Two methods:

**a) Linear Combination**
Includes both a simple **average** and a **weighted average**:

$$
\text{merge}(A, B) = \frac{w_A \cdot A + w_B \cdot B}{w_A + w_B}
$$

- If the parameters of the two models are on **very different scales**, one should be scaled to bring both into the same range before merging.
- Most effective for models that were **finetuned on top of the same base model**.
- Commonly used in **federated learning**.
  > **Federated learning, briefly:** multiple parties each train a model locally on their own private data (which never leaves their device/server), and only the resulting model updates/weights are sent to a central server to be combined — preserving data privacy while still benefiting from everyone's data collectively.
- Models are also often merged linearly at the **component level** — e.g., combining just their adapters rather than the full model.

**Task Vectors**
Once a model has been finetuned for a specific task, subtracting the **base model's weights** from the finetuned model's weights yields a vector that captures *just the change* — the "task vector" (also called **delta parameters**).

- Task vectors enable **task arithmetic**: you can **add** two task vectors together to *combine* their capabilities, or **subtract** one to *reduce/remove* a capability from a model.

**b) Spherical Linear Interpolation (SLERP)**
In simple terms: imagine each model's parameter vector as a point on the surface of a **sphere**. SLERP draws the **shortest path along the sphere's surface** between two such points, and the merged model is a point somewhere along that arc.
- How close the merged point sits to either original vector is controlled by an **interpolation factor**, typically ranging from **0 to 1**.

#### 2. Layer Stacking
*(Brief — combines models by stacking their layers rather than averaging weights directly.)*

#### 3. Concatenation
*(Brief — combines models by concatenating their parameters/components rather than blending them.)*

---

### Pruning Redundant Task-Specific Parameters

During finetuning, many parameters get adjusted — but **not all of them meaningfully contribute** to the performance difference. Parameters that don't contribute much are considered **redundant** to the finetuning outcome, and it's often beneficial to **prune** them before merging.

- Techniques like **TIES** and **DARE** first **prune redundant parameters from each task vector** *before* merging multiple task vectors together — reducing interference between tasks and producing cleaner merged models.