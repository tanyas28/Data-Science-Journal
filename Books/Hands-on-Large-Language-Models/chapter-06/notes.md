# Chapter 6: Prompt Engineering

## Controlling Model Output

Before writing good prompts, it helps to understand the **decoding parameters** that control how "creative" or "predictable" a model's output is (these connect back to the decoding strategies in Chapter 3 — greedy vs. sampling).

### Temperature

Controls the **randomness** of the model's output by reshaping the probability distribution over the next token before a token is sampled.

- **Low temperature (close to 0)** → the model becomes more **confident/deterministic**, almost always picking the highest-probability token. This is essentially greedy decoding.
- **High temperature** → the probability distribution gets "flattened," giving lower-probability tokens a better chance of being picked — output becomes more **random and creative**, but also more likely to go off-track.

> **Example:** Prompt: *"The weather today is"* →
> - **Temperature = 0.1:** almost always generates *"sunny"* (the single most likely word), every time you run it.
> - **Temperature = 1.2:** might generate *"unpredictable," "glorious," "a bit odd"* — more varied, more "creative," but also occasionally stranger or less coherent.

### Top K

Restricts the model to only sample from the **top K most likely next tokens**, discarding everything outside that shortlist before sampling happens.

> **Example:** With `top_k = 3`, if the model's top 3 candidates for the next word are *"sunny" (45%), "cloudy" (30%), "rainy" (15%)*, then only these three are considered — even if "snowy" had a small 2% chance, it's excluded entirely from being picked, no matter how high the temperature is set.

### Top P (Nucleus Sampling)

Instead of a fixed *number* of tokens (like top-k), top-p selects the **smallest set of tokens whose cumulative probability adds up to at least P**, and samples only from that set.

> **Example:** With `top_p = 0.9`, the model keeps adding candidate tokens — highest probability first — until their combined probability crosses 90%, then stops. This set might be just 2 tokens on a very "obvious" next word, or 20 tokens on a very open-ended one — **top-p adapts the size of the candidate pool to the situation**, unlike top-k's fixed size.

> **Rule of thumb:** These three parameters are usually combined. A common practical setup: a moderate temperature (e.g., 0.7) + top-p (e.g., 0.9) to get natural variation without wandering off into incoherence.

---

## Introduction to Prompt Engineering

### 1. Instruction-Based Prompting

The most basic form of prompting: simply telling the model **what to do**, in plain instructions — similar to how you might give directions to a person.

> **Example:** *"Summarize the following paragraph in two sentences: [paragraph]"* or *"Translate this sentence into French: [sentence]"* — the instruction itself is the entire prompt; no examples of correct output are provided.

Good instruction prompts are typically **clear, specific, and unambiguous** about the format and content expected — vague instructions ("make this better") tend to produce inconsistent results compared to specific ones ("rewrite this sentence to be more formal, keeping it under 20 words").

### 2. In-Context Learning

Instead of (or in addition to) just instructing the model, you give it **examples directly inside the prompt** to demonstrate the task — the model "learns" the pattern from these examples within that single prompt, without any actual training/finetuning happening.

| Type | Description |
|---|---|
| **Zero-shot** | No examples given — just the instruction (same as instruction-based prompting above). |
| **One-shot** | Exactly **one** example of the task is shown before asking the model to do it for real. |
| **Few-shot** | **Multiple** examples are shown, helping the model pick up the pattern (format, tone, style) more reliably. |

> **Example (few-shot sentiment classification):**
> ```
> Review: "This movie was fantastic!" → Positive
> Review: "Total waste of time." → Negative
> Review: "It was okay, nothing special." → Neutral
> Review: "I loved every minute of it!" → ?
> ```
> By seeing a few labeled examples first, the model infers both the **task** and the **expected output format** ("Positive"/"Negative"/"Neutral"), without ever being finetuned on labeled data.

### 3. Chain Prompting

Breaking a complex task into **multiple smaller prompts**, where the output of one prompt becomes the input to the next — instead of trying to get everything done in a single giant prompt.

> **Example:** To write a blog post, instead of one prompt asking for the whole finished article, you might chain:
> 1. Prompt 1: *"Generate 5 possible titles for a blog post about remote work productivity."*
> 2. Prompt 2 (using the chosen title): *"Write an outline for a blog post titled '[chosen title]'."*
> 3. Prompt 3 (using the outline): *"Write the full blog post based on this outline: [outline]."*
>
> Each step is simpler and more controllable than asking for everything at once, and you can inspect/correct the output at each stage before moving to the next.

### 4. Chain of Thought (CoT)

Instead of asking the model to jump straight to a final answer, you prompt it to **reason step by step** before giving the answer — this often significantly improves accuracy on tasks involving logic, math, or multi-step reasoning.

> **Example:**
> - **Without CoT:** *"If a train travels 60 km in 1.5 hours, what is its speed? Answer:"* → model may just guess an answer directly, sometimes incorrectly.
> - **With CoT:** *"If a train travels 60 km in 1.5 hours, what is its speed? Let's think step by step."* → the model first writes out: *"Speed = distance ÷ time = 60 ÷ 1.5 = 40 km/h"* — working through the reasoning tends to produce the correct final answer more reliably than jumping straight to it.

A simple and surprisingly effective trick: just appending the phrase **"Let's think step by step"** to a prompt is often enough to trigger this reasoning behavior.

### 5. Tree of Thought (ToT)

An extension of Chain of Thought — instead of generating **one single reasoning path**, the model explores **multiple possible reasoning paths/branches** (like a tree), evaluates which branches look most promising, and can backtrack from dead ends — closer to how a person might explore several possible approaches to a hard problem before settling on one.

> **Example:** For a tricky puzzle, instead of committing to one line of reasoning immediately, the model might generate 3 different possible first steps, evaluate how promising each one looks, continue expanding only the most promising one(s), and discard the others — rather than being locked into whichever single reasoning path it happened to start with (as in plain Chain of Thought).

This generally costs more compute (since multiple reasoning paths get explored), but can produce better results on genuinely hard, multi-step problems where a single reasoning path is prone to going wrong.

---

## Output Verification

We can also control/improve the reliability of an LLM's output in a couple of other ways:

### 1. Providing Ample Examples

This connects directly back to **few-shot prompting** (above) — by giving the model enough well-chosen examples of exactly the kind of output you want (format, tone, structure), you reduce the model's freedom to drift into an unwanted format, making output far more consistent and predictable without needing to finetune anything.

> **Example:** If you want a model to always output valid JSON in a specific schema, showing it 3–4 examples of correctly-formatted JSON output for similar inputs dramatically increases the odds it sticks to that exact schema for a new input, compared to just instructing it to "output JSON."

### 2. Grammar / Constrained Sampling

Instead of *hoping* the model follows the format through good prompting alone, this approach **technically restricts** what tokens the model is even allowed to generate at each step — enforcing a required structure (like valid JSON, or a fixed set of category labels) directly at the decoding level, rather than relying on the model's own judgment.

> **Example:** If the output must always be exactly one of `"positive"`, `"negative"`, or `"neutral"`, constrained sampling can make it **structurally impossible** for the model to output anything else — at each decoding step, only tokens that keep the output on track toward one of these three valid completions are allowed to be sampled, no matter what probability the model internally assigns to other tokens. This gives a much stronger guarantee than prompting alone, which can still occasionally be ignored by the model.