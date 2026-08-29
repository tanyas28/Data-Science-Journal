# Chapter 8: Dataset Engineering

The purpose of this chapter is to learn how to **curate a dataset** that is **affordable, high quality, and well-suited** to your application.

Data is needed to train/finetune a foundation model toward the desired behavior. This field has grown large enough to have its own name — **Data-Centric AI** — which focuses on the best ways to curate data for a given benchmark, application, or use case (as opposed to *model*-centric AI, which focuses on improving architectures/training methods while holding data fixed).

---

## Topic 1: Data Curation

Data curation means **collecting the right data** for your application. To do this well, you need to understand how the model actually **learns** from data: the training data should directly reflect the **behavior** you want the final application to exhibit.

> **Example:** If you want your application to always respond in a polite, concise tone and refuse out-of-scope questions, your training examples need to actually *demonstrate* that — polite/concise responses, and clear refusals on out-of-scope prompts. If your dataset never shows a refusal example, the model has no way to learn when *not* to answer.

Three major criteria to keep in mind while curating data:
1. **Data quality**
2. **Data quantity**
3. **Data coverage**

---

### 1. Data Quality

As stressed in Chapter 7: **high-quality small data** can improve performance far more than **low-quality huge data**.

> Data is considered "high quality" if it makes the finetuning process **efficient and reliable**.

**Six characteristics of high-quality data:**

| # | Characteristic | Meaning |
|---|---|---|
| 1 | **Relevant** | Data matches the task and use case you're actually targeting. |
| 2 | **Aligned** | Data matches the requirements/goals of the application. |
| 3 | **Consistent** | Similar examples should be labeled/scored using the **same criteria**. |
| 4 | *(quality bar continued)* | *(see consistency example below)* |
| 5 | **Correctly formatted** | Output examples strictly match the exact format the model needs to produce. |
| 6 | **Sufficiently unique** | Some duplication is fine, but *too much* duplication biases the model. |

> **Example — why consistency matters:** Say you're training a model to write short stories, and each story in your dataset is rated on a 1–5 quality scale (used, e.g., for preference finetuning). If two stories of similar quality get different scores — one rater gives a 5, another gives the same quality story a 4 — that's an **inconsistent** labeling signal. This gets worse if multiple AI models were used to generate/rate the dataset, since different models may apply different implicit standards. Without a **fixed rating guideline** applied consistently across the whole dataset, the model receives contradictory signals and finetuning quality suffers.

> **Example — correctly formatted:** If your application generates SQL queries, every example in your dataset must contain **syntactically valid, correctly formatted SQL** — not just "close enough" queries — since the model will learn to imitate the format shown, mistakes included.

Beyond these six, data must also be strictly **compliant with all applicable laws and policies** (e.g., privacy, copyright, data licensing).

---

### 2. Data Quantity

How much data do you actually need? Most often, this comes down to **budget** — but it also depends on:
- The **finetuning technique** chosen (e.g., full finetuning vs. PEFT — PEFT can work well with less data).
- The **complexity of the task** the application needs to perform.
- The **quality of the base model** you're starting from (a stronger base model often needs less task-specific data).

**Practical way to experiment with data quantity:**
1. Start with a **small, high-quality dataset** and finetune.
2. Check if performance improves. If there's a clear improvement trend, **add more data** and repeat.
3. To estimate how much more data might help: finetune on **subsets** of your current dataset (e.g., 10%, 25%, 50%, 100%) and **plot performance vs. dataset size**.
   - A **steep upward slope** on this curve suggests more data would likely keep helping.
   - A **flattening curve** suggests you're near diminishing returns — more data won't buy much more performance.

---

### 3. Data Coverage

Your dataset needs **diversity** — but diversity *within the relevant domain*, not diversity for its own sake.

> **Example:** An application built as a financial advisor doesn't need to "know" cult literature or unrelated trivia — diversity here means covering a **wide range of scenarios within finance** (different account types, market conditions, user question phrasings, edge cases), not going outside the domain entirely.

- More diverse examples **within the target domain** generally lead to better generalization.
- Aim for a **data mix that closely resembles the real-world distribution** of problems your application will actually face in production.

---

## Topic 2: Data Acquisition and Annotation

### Acquisition

One powerful approach: build a **data flywheel** for your application — a system that captures data **in real time** as the application runs in production. This could include:
- The prompt given
- The model's response
- Any mistakes made
- What correction/feedback followed
- **User feedback** (thumbs up/down, corrections, etc.)

Over time, this continuously growing, real-world dataset becomes one of the most valuable assets for improving the application.

### Annotation

Annotating a dataset is genuinely **challenging** — mainly because of a **lack of clear, consistent annotation guidelines**.

> In practice, annotation guidelines usually end up being **the same as your evaluation guidelines** (see Chapter 4). This is exactly why investing time upfront in designing solid **evaluation criteria and rubrics** pays off twice — once for evaluation, and again for annotation.

---

## Topic 3: Data Augmentation and Synthesis

### Data Augmentation

Not a new idea — it's long been used in traditional deep learning, especially for **vision models**: rotating images, flipping them, adjusting colors, etc., to create more training examples from existing ones.

**Why it matters:** augmentation makes an application more **robust** — small, superficial changes to the input shouldn't change the model's output, and augmentation trains the model to be invariant to exactly those kinds of changes.

**For text models**, common augmentation techniques include:
- Replacing words with **synonyms** (similar meaning).
- **Rephrasing** sentences while preserving meaning.
- **Round-trip translation** (translate to another language and back) to get natural paraphrases.
- Introducing **typos** or minor noise, so the model stays robust to imperfect real-world input.

### Data Synthesis

Synthesis is about **generating new data from scratch** — especially useful when real data isn't accessible (e.g., due to privacy, cost, or rarity of certain scenarios).

Two traditional approaches:

**1. Rule-Based Synthesis**
The simplest synthesis method. Works well when your target data has a **fixed template/structure** with defined fields that just need to be filled in.

> **Example:** Generating synthetic customer support tickets using a template like: `"My {product} stopped working after {event}. I need help with {issue_type}."` — a script can programmatically fill in `{product}`, `{event}`, and `{issue_type}` from predefined lists to generate thousands of varied-but-structurally-consistent examples very cheaply. This works great for structured formats but produces less natural/varied language than real user data.

**2. Simulation**
Used when there are many different possible **scenarios**, and you want to observe/capture how decisions get made, what mistakes occur, etc. — by literally **running a simulated environment** rather than collecting real-world data.

> **Example:** Training a self-driving car model — you can simulate an environment containing reckless drivers, heavy trucks, sudden pedestrian crossings, and adverse weather, then capture the resulting sensor data + correct/incorrect decisions as training data. This lets you generate rare, dangerous, or expensive-to-collect scenarios (like near-accidents) safely and repeatedly, which would be impractical or unsafe to gather from real driving alone.

> Both methods share a key advantage: they let you generate **rare or hard-to-collect edge cases** at scale, and a key risk: synthetic data can fail to capture the full messiness/unpredictability of real-world data, so it's often best used to **supplement**, not fully replace, real data.# Chapter 8: Dataset Engineering

The purpose of this chapter is to learn how to **curate a dataset** that is **affordable, high quality, and well-suited** to your application.

Data is needed to train/finetune a foundation model toward the desired behavior. This field has grown large enough to have its own name — **Data-Centric AI** — which focuses on the best ways to curate data for a given benchmark, application, or use case (as opposed to *model*-centric AI, which focuses on improving architectures/training methods while holding data fixed).

---

## Topic 1: Data Curation

Data curation means **collecting the right data** for your application. To do this well, you need to understand how the model actually **learns** from data: the training data should directly reflect the **behavior** you want the final application to exhibit.

> **Example:** If you want your application to always respond in a polite, concise tone and refuse out-of-scope questions, your training examples need to actually *demonstrate* that — polite/concise responses, and clear refusals on out-of-scope prompts. If your dataset never shows a refusal example, the model has no way to learn when *not* to answer.

Three major criteria to keep in mind while curating data:
1. **Data quality**
2. **Data quantity**
3. **Data coverage**

---

### 1. Data Quality

As stressed in Chapter 7: **high-quality small data** can improve performance far more than **low-quality huge data**.

> Data is considered "high quality" if it makes the finetuning process **efficient and reliable**.

**Six characteristics of high-quality data:**

| # | Characteristic | Meaning |
|---|---|---|
| 1 | **Relevant** | Data matches the task and use case you're actually targeting. |
| 2 | **Aligned** | Data matches the requirements/goals of the application. |
| 3 | **Consistent** | Similar examples should be labeled/scored using the **same criteria**. |
| 4 | *(quality bar continued)* | *(see consistency example below)* |
| 5 | **Correctly formatted** | Output examples strictly match the exact format the model needs to produce. |
| 6 | **Sufficiently unique** | Some duplication is fine, but *too much* duplication biases the model. |

> **Example — why consistency matters:** Say you're training a model to write short stories, and each story in your dataset is rated on a 1–5 quality scale (used, e.g., for preference finetuning). If two stories of similar quality get different scores — one rater gives a 5, another gives the same quality story a 4 — that's an **inconsistent** labeling signal. This gets worse if multiple AI models were used to generate/rate the dataset, since different models may apply different implicit standards. Without a **fixed rating guideline** applied consistently across the whole dataset, the model receives contradictory signals and finetuning quality suffers.

> **Example — correctly formatted:** If your application generates SQL queries, every example in your dataset must contain **syntactically valid, correctly formatted SQL** — not just "close enough" queries — since the model will learn to imitate the format shown, mistakes included.

Beyond these six, data must also be strictly **compliant with all applicable laws and policies** (e.g., privacy, copyright, data licensing).

---

### 2. Data Quantity

How much data do you actually need? Most often, this comes down to **budget** — but it also depends on:
- The **finetuning technique** chosen (e.g., full finetuning vs. PEFT — PEFT can work well with less data).
- The **complexity of the task** the application needs to perform.
- The **quality of the base model** you're starting from (a stronger base model often needs less task-specific data).

**Practical way to experiment with data quantity:**
1. Start with a **small, high-quality dataset** and finetune.
2. Check if performance improves. If there's a clear improvement trend, **add more data** and repeat.
3. To estimate how much more data might help: finetune on **subsets** of your current dataset (e.g., 10%, 25%, 50%, 100%) and **plot performance vs. dataset size**.
   - A **steep upward slope** on this curve suggests more data would likely keep helping.
   - A **flattening curve** suggests you're near diminishing returns — more data won't buy much more performance.

---

### 3. Data Coverage

Your dataset needs **diversity** — but diversity *within the relevant domain*, not diversity for its own sake.

> **Example:** An application built as a financial advisor doesn't need to "know" cult literature or unrelated trivia — diversity here means covering a **wide range of scenarios within finance** (different account types, market conditions, user question phrasings, edge cases), not going outside the domain entirely.

- More diverse examples **within the target domain** generally lead to better generalization.
- Aim for a **data mix that closely resembles the real-world distribution** of problems your application will actually face in production.

---

## Topic 2: Data Acquisition and Annotation

### Acquisition

One powerful approach: build a **data flywheel** for your application — a system that captures data **in real time** as the application runs in production. This could include:
- The prompt given
- The model's response
- Any mistakes made
- What correction/feedback followed
- **User feedback** (thumbs up/down, corrections, etc.)

Over time, this continuously growing, real-world dataset becomes one of the most valuable assets for improving the application.

### Annotation

Annotating a dataset is genuinely **challenging** — mainly because of a **lack of clear, consistent annotation guidelines**.

> In practice, annotation guidelines usually end up being **the same as your evaluation guidelines** (see Chapter 4). This is exactly why investing time upfront in designing solid **evaluation criteria and rubrics** pays off twice — once for evaluation, and again for annotation.

---

## Topic 3: Data Augmentation and Synthesis

### Data Augmentation

Not a new idea — it's long been used in traditional deep learning, especially for **vision models**: rotating images, flipping them, adjusting colors, etc., to create more training examples from existing ones.

**Why it matters:** augmentation makes an application more **robust** — small, superficial changes to the input shouldn't change the model's output, and augmentation trains the model to be invariant to exactly those kinds of changes.

**For text models**, common augmentation techniques include:
- Replacing words with **synonyms** (similar meaning).
- **Rephrasing** sentences while preserving meaning.
- **Round-trip translation** (translate to another language and back) to get natural paraphrases.
- Introducing **typos** or minor noise, so the model stays robust to imperfect real-world input.

### Data Synthesis

Synthesis is about **generating new data from scratch** — especially useful when real data isn't accessible (e.g., due to privacy, cost, or rarity of certain scenarios).

Two traditional approaches:

**1. Rule-Based Synthesis**
The simplest synthesis method. Works well when your target data has a **fixed template/structure** with defined fields that just need to be filled in.

> **Example:** Generating synthetic customer support tickets using a template like: `"My {product} stopped working after {event}. I need help with {issue_type}."` — a script can programmatically fill in `{product}`, `{event}`, and `{issue_type}` from predefined lists to generate thousands of varied-but-structurally-consistent examples very cheaply. This works great for structured formats but produces less natural/varied language than real user data.

**2. Simulation**
Used when there are many different possible **scenarios**, and you want to observe/capture how decisions get made, what mistakes occur, etc. — by literally **running a simulated environment** rather than collecting real-world data.

> **Example:** Training a self-driving car model — you can simulate an environment containing reckless drivers, heavy trucks, sudden pedestrian crossings, and adverse weather, then capture the resulting sensor data + correct/incorrect decisions as training data. This lets you generate rare, dangerous, or expensive-to-collect scenarios (like near-accidents) safely and repeatedly, which would be impractical or unsafe to gather from real driving alone.

> Both methods share a key advantage: they let you generate **rare or hard-to-collect edge cases** at scale, and a key risk: synthetic data can fail to capture the full messiness/unpredictability of real-world data, so it's often best used to **supplement**, not fully replace, real data.
 
---
 
### AI-Powered Data Synthesis
 
AI models themselves can now be used to generate synthetic training data in several ways:
 
1. **Simulating behavior/interactions** — AI can simulate a wide range of scenarios: how an API might respond to a given input, playing out a **chess game**, simulating natural **conversation/talking patterns**, and more. AI agents can even be set up to **negotiate with each other**, with the resulting back-and-forth used to surface better strategies.
2. **Paraphrasing and translation** — AI can paraphrase sentences or generate translations, replacing original sentences with their translated equivalents to add linguistic diversity.
   > **Verifying translation quality:** translate the sentence *back* into the original language (back-translation). If the round-tripped sentence is very close in meaning to the original, the translation was likely accurate. If it diverges significantly, the translation step probably introduced errors.
3. **Watch for bias** — a key challenge (and one to always keep in mind) is that the AI model used to generate data can itself be **biased**, and that bias will propagate directly into your synthetic dataset. This needs to be actively accounted for, not assumed away just because the data is machine-generated.
4. **Instruction data generation** — AI can generate **instruction data**: examples for supervised finetuning, each consisting of an **instruction** paired with its **response**.
5. **Combined code framework** — combining **code translation**, **code back-translation**, and **code generation** together is a notable synthesis framework, and was used in **Llama 3's** data synthesis pipeline.
6. **Evaluating generated data** — a solid pipeline for evaluating synthetic data is essential. Common heuristic filters include removing:
   - **Repetitive examples**
   - Instructions that are **too long or too short**
   - Examples with the **same instruction but different responses** (contradictory signal)
   - Examples where the **output is identical to the input** (no useful learning signal)
7. **Model collapse** — a real and important risk. If synthetic data (generated by AI) increasingly makes up the training data for future models, and those models are then used to generate *more* synthetic data, quality/diversity can progressively degrade generation after generation — this phenomenon is known as **model collapse**.
---
 
## Topic 4: Model Distillation
 
**Distillation** is the process of training a **small ("student") model** to learn to perform as well as a **larger, more capable "teacher" model**.
 
- The student model can either be trained **from scratch**, or can be an existing model that gets **finetuned** to mimic the teacher.
- **Synthetic instruction data** (generated by the teacher model) combined with efficient finetuning techniques like **LoRA** are commonly used together in distillation pipelines — the teacher generates large volumes of high-quality (instruction, response) pairs, and the student is then LoRA-finetuned on this synthetic dataset rather than requiring the student to be fully retrained.
> **How it typically works, step by step:**
> 1. Take a large, capable **teacher model** (e.g., a top-tier foundation model).
> 2. Use it to generate a large **synthetic instruction dataset** — prompting it with diverse instructions and capturing its responses.
> 3. Optionally filter/clean this dataset using the evaluation heuristics from Topic 3 above (remove repetitive, contradictory, or low-quality examples).
> 4. **Finetune** a much smaller **student model** (via full finetuning or PEFT/LoRA) on this dataset, effectively teaching it to imitate the teacher's behavior on the tasks that matter for your application.
>
> The result is a smaller, cheaper, faster model that captures a meaningful slice of the teacher's capability — directly connecting back to the Chapter 7 point that it's often more effective to **finetune a smaller model using a larger model's outputs** than to try to finetune the large model itself.
 
---
 
## Topic 5: Data Processing
 
> Reading papers on data collection, preprocessing, and synthesis (and the statistics researchers used to describe their datasets) is genuinely valuable — it teaches a lot about the practical realities of this whole process.
 
Four main steps in processing a dataset:
 
### 1. Inspect
 
Before doing anything else, **manually inspect** the data:
- Look for **duplicates**.
- Understand the **format** of the data.
- Understand **where and how** it was collected.
- Check for **special tokens** and whether unnecessary information is present.
- See if data can be meaningfully **grouped** — by topic, language, source, etc.
### 2. De-duplicate
 
Excess duplication in a dataset biases the model (as covered in Topic 1). Steps to de-duplicate:
- Compute a **similarity score** between examples.
- Use a **hashing approach** to group similar examples into **buckets**, then examine the distribution of bucket sizes.
- For large datasets, first apply **dimensionality reduction** to make comparisons tractable, then run **pairwise comparisons** within each reduced-dimension bucket rather than across the entire dataset.
### 3. Clean and Filter
 
- Remove **garbage/irrelevant** data — anything not useful for your specific application.
- Use various **statistics** (e.g., length distributions, language detection, token frequency) to systematically identify what to filter out, rather than relying purely on manual review.
- Check for and **filter out data that violates any law or policy** (e.g., copyrighted content, PII, licensing violations).
### 4. Format
 
- Convert the cleaned data into the **exact format** your application/finetuning pipeline requires (e.g., instruction-response JSON pairs, specific delimiter/token conventions expected by the model).
 