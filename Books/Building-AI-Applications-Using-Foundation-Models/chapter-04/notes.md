# Chapter 4: Evaluate AI Systems

> Evaluating a **model** is different from evaluating a model for a **particular application**. For an application, you pick and choose the metrics most relevant to your use case, and exclude the ones that are irrelevant.

---

## Topic 1: Evaluation Criteria

Before building an application using a foundation model, the first thing to decide is **how the model will be evaluated**. This is known as the **evaluation-driven approach**.

### Buckets of Criteria

| # | Criterion |
|---|---|
| A | Domain-Specific Capability |
| B | Generation Capability |
| C | Instruction-Following Capability |
| D | Cost and Latency |

---

### A) Domain-Specific Capability

A model's domain-specific capability is constrained by things like its **architecture** and **training data**. Certain domain-specific benchmarks exist to evaluate this.

**Coding capabilities:**
- **Functional correctness** — as covered in Chapter 3 (e.g., Pass@K).
- **Efficiency** of the generated code — measured against benchmarks.
- **Readability** of the code — evaluated using AI-as-judge.

**Non-coding capabilities:**
- Evaluated using **close-ended tasks**, e.g., MCQs — checking how many the model answers correctly.
- MCQs reliably test both **knowledge** (what the model knows) and **reasoning capability**.

> **Example:** Evaluating a coding assistant might combine Pass@K (does the code run and pass tests?) with an AI judge scoring how readable and idiomatic the code is.

---

### B) Generation Capability

Key metrics: **faithfulness, factual correctness, safety, relevance, fluency, and coherence.**

#### 1. Fluency and Coherence
Tested using **AI-as-judge** or **perplexity**.

#### 2. Factual Consistency
Two types: **local** and **global**.

To evaluate factual correctness against reference text, we can use AI-as-judge via two techniques:

- **Self-evaluation**: The model checks its own response. Based on the assumption that if the model produces multiple different responses that disagree with each other, the response is likely **hallucinated**.
- **Knowledge-augmented verification**: An AI model decomposes the response into small, self-contained statements.
  - *Example:* In "...it is a good book," **"it"** gets replaced with the actual subject → **"Harry Potter is a good book."**
  - A fact-checking API (e.g., Google Search) is then called to verify each statement.
  - An AI judge scores how closely the response matches the search results.

**Checking consistency with context** is essentially a **Textual Entailment** problem (NLP):

| Relation | Meaning |
|---|---|
| **Entailment** | Hypothesis can be inferred from the context |
| **Contradiction** | Hypothesis contradicts the context |
| **Neutral** | Neither entails nor contradicts |

Instead of a general-purpose AI judge, we can use a model **trained specifically for factual correctness**. It takes `(context, hypothesis)` as input and outputs one of: `entailment`, `contradiction`, or `neutral`.

> Especially useful for **RAG applications**, where responses must stay grounded in retrieved context.

#### 3. Safety
- General-purpose AI-as-judge can be used to check the safety of generated text.
- Model producers should also build dedicated **moderation tools** to keep models safe.

---

### C) Instruction-Following Capability

Essential for applications requiring **structured outputs** (JSON, CSV, etc.) — but not limited to structured output alone.

Two key benchmarks:

| Benchmark | Focus |
|---|---|
| **IFEval** | Format compliance — keyword inclusion, length constraints, number of bullet points, etc. |
| **INFOBench** | Format **+** content constraints |

**IFEval example:** *"Write a response under 100 words, using exactly 3 bullet points."* — checks if the model followed the structural rules.

**INFOBench** goes further — checking that the *content* itself respects the instruction's constraints.
> **Example:** If the prompt asks to discuss "the best fiction works of the 90s," the generated text should stay strictly factual to that scope — not drift into unrelated decades or genres.

- INFOBench works by constructing a **checklist of yes/no questions** for each instruction.
- Each question can be answered by either an **AI or a human**.

---

### D) Cost and Latency

Models must also be filtered based on **latency** and **cost** requirements — which can shift over time, so teams should revisit this periodically.

Common latency metrics:

| Metric | Measures |
|---|---|
| **TTFT** (Time to First Token) | Delay before the first token appears |
| **TPT** (Time per Token) | Speed of token generation |
| **TBT** (Time Between Tokens) | Gaps between successive tokens |
| **Time per Query** | Total time for a full response |

The right metric to prioritize depends on the application — e.g., a chat UI cares more about **TTFT** (feels responsive), while a batch-processing pipeline cares more about **total time per query**.