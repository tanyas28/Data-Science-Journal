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

*This chapter will likely get expanded further as we go — flagging it as a living doc.*