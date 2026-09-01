# Chapter 10: AI Engineering Architecture and User Feedback

## Topic 1: AI Engineering Architecture

**Simplest possible architecture:** the application receives a query, sends it to the model, and the model returns a response.

From there, components get added incrementally to handle real-world needs:

| Step | Addition | Purpose |
|---|---|---|
| a | External data/tool access | Enhance the **context** fed into the model. |
| b | Guardrails | Sanitize both **input** and **output**. |
| c | Model router + gateway | Support complex pipelines and improve **security**. |
| d | Caching | Optimize for **latency and cost**. |
| e | Agent patterns / write actions | Maximize system **capability**. |

---

### Step 1: Enhance Context (RAG + Tool Access)

- Basic context construction is often supported directly by **model APIs**.
- **Custom RAG pipelines** offer more flexibility — e.g., supporting a larger number of uploaded documents, custom chunking/retrieval logic, etc.

**Flow:**
```
User → Query → DB (RAG / context construction) → Model API → Response
```

---

### Step 2: Add Guardrails

- Guardrails **sanitize** both the input going into the model and the output coming out of it.
- **Input guardrails** protect against: leaking personal/sensitive information, and execution of malicious/adversarial prompts.
- **Output guardrails** protect against: the model leaking sensitive information (e.g., about itself or training data), or producing harmful responses to the user.
- **Trade-off:** guardrails add a **reliability vs. latency** trade-off — every check adds processing time.
  - Output guardrails can be tricky with **streaming completions**, since the full output isn't available to check until generation is complete (or requires checking partial chunks).
  - The right guardrail approach also depends on whether you're using a **hosted API** or a **self-hosted model** (self-hosting gives more control over where/how checks run).

---

### Step 3: Add Model Router and Gateway

**Router:**
- Essentially an **intent classifier** — predicts user intent and routes the query to the most appropriate solution/model/pipeline.
  > **Example:** A query about resetting a password gets routed to an **FAQ/help-center flow** rather than the full LLM pipeline.
- Also useful for **agents** with multiple possible actions — helping decide the *next step*.
- For models with a **memory system**, a router can predict *which part* of the memory hierarchy (short-term vs. long-term) to pull information from.
- Routers should be **fast and cheap** — since multiple routers might be chained/used together, they shouldn't add significant extra latency or cost.
- When routing across models with **different context limits**, the query's context may need to be **adjusted/truncated** to fit the target model.

**Gateway:**
- A layer between the application and model APIs that acts as a **unified wrapper**, making the codebase easier to maintain (swap/add models without changing application code everywhere).
- Provides **access control** and **cost management** in one central place.

**Full flow so far:**
```
User → Context Construction → Input Guardrail → Model Gateway → Output Guardrail → Response
```

---

### Step 4: Reduce Latency with Caching

Two major caching mechanisms:

**1. Exact Caching**
- Cached results are reused **only** when the exact same item/query is requested again.
- If no exact match is found, the query is processed and the new result is **cached** for future reuse.
- Also applies to **embedding-based retrieval**: if a query already exists in the **vector search cache**, return the cached result directly; otherwise, perform the vector search and cache the result.
- Can be implemented **in-memory**, or via a database like **Postgres** or **Redis**.

**2. Semantic Caching**
- Cached results are returned if a **new query is semantically similar** to a previously cached one — not necessarily identical.
- Uses a **similarity threshold**: if the semantic similarity score between the new query and a cached query exceeds threshold $x$, return the cached response; otherwise, treat it as new and cache it separately.
- Setting the right threshold is largely **trial and error**, tuned to maximize cache hits without returning wrong/irrelevant cached answers.

---

### Step 5: Add Agent Patterns

Introducing full **agentic workflows** — loops, iterative reasoning, and **write actions** (see Chapter 6). This significantly increases system complexity, but also capability.

---

## Topic 2: Monitoring and Observability

- **Observability** is a well-established best practice, with many ready-to-use **proprietary and open-source** solutions available.
- Monitoring should help reduce key risks: **application failure**, **security attacks**, and **drift**.

### Key Observability Metrics

| Metric | Meaning |
|---|---|
| **MTTD** (Mean Time to Detection) | How long it takes to **detect** that something went wrong. |
| **MTTR** (Mean Time to Response) | How long it takes to **resolve** the issue once detected. |
| **CFR** (Change Failure Rate) | The percentage of deployments that **result in failure**. |

> In general, a model that performs well during **evaluation** tends to also perform well under **production monitoring** — though monitoring is what catches the cases evaluation didn't anticipate.

### What Metrics to Track

- Depends heavily on the **type of application**. A RAG-heavy application should track **RAG-specific metrics** (context precision/recall — see Chapter 6).
- **Length-related metrics** matter for tracking **latency and cost**, since longer context/responses generally increase both.

### Logs and Traces

- Keep thorough logs of everything happening in the application: every **decision**, its **outcome**, **actions taken**, and the model's **chain of thought**.
- At scale, manually reviewing such large logs becomes hard — many **AI-powered log analysis tools** exist to help scan through them.
- Still, periodically reviewing **raw production logs** directly is valuable — it reveals patterns and issues that aggregated dashboards can miss.

### Drift Detection

The more components a system has, the more things can silently **change/drift** over time, including:
1. **System prompt** changes
2. **User behavior** changes (e.g., shifting query patterns)
3. **Underlying model changes** — especially relevant when using a **third-party API**, where the provider may update the model without your explicit control.

---

## Topic 3: AI Pipeline Orchestration

An **AI orchestrator** works in two steps:

**1. Component Definition**
The orchestrator needs to know about all available components: tools, different models, **evaluation tools**, and **monitoring tools**.

**2. Chaining**
This is essentially **function composition** — the orchestrator plans and executes all components **together**, defining the full sequence of steps from receiving a query to producing the final completion.

- The orchestrator is responsible for managing **data flow between components**.

**How to evaluate an orchestrator:**
1. **Integration and extensibility** — how easily new tools/models can be added.
2. **Support for complex pipelines** — branching, loops, conditional logic.
3. **Ease of use, performance, and scalability**.

---

## Topic 4: User Feedback

### Extracting Conversational Feedback

Two types of user feedback:

| Type | Description |
|---|---|
| **Explicit** | Direct feedback deliberately given by the user (ratings, thumbs up/down, written comments). |
| **Implicit** | Feedback *inferred* from user behavior — e.g., time spent, negative follow-ups, abandoning the conversation midway, having to explain their intent multiple times, or opening a message with *"No, what I meant was..."* |

> Feedback extracted from conversations can be used for **evaluation**, **development**, and **personalization**.

---

### Feedback Design

**When to collect feedback:**

1. **At the beginning** — early feedback helps understand how a user intends to use the application, and can calibrate the experience for them. This should usually be **optional**, since mandatory upfront feedback adds friction. Absent an explicit preference, the system should default to a **neutral** setting.
2. **When something goes wrong** — users should have an easy way to report failures or unexpected behavior.
3. **When the model has low confidence** — if the model is uncertain between multiple possible actions/responses, asking the user to choose directly improves both the immediate result and future confidence.
   > **Example:** A model presenting **two candidate responses** and letting the user pick their preferred one.
4. **Positive feedback matters too** — options like a simple **thumbs up** are important; it's valuable to know what's working well, not just what's failing.
5. **Feedback collection should be seamless**, ideally woven into the natural workflow rather than a separate interruption.
   > **Example:** Code editors that show a **suggested completion** — the user either **accepts** it or keeps typing their own version. Both actions implicitly generate a useful feedback signal, with zero extra friction.

---

### Feedback Limitations

Designing a feedback system well requires accounting for its inherent limitations:

**1. Bias**
Users often rate generously just to avoid the effort of explaining what went wrong, or out of social pressure to "be nice" rather than give critical feedback.
> **Workaround:** Instead of an open-ended rating scale, offer a **small set of concrete options** — e.g., *"Food was okay," "Didn't like the food," "Liked the food"* — which lowers the effort needed to give an honest, specific signal.
> For scale-based systems, examine the **overall distribution** of ratings/reviews to detect skew, and calibrate thresholds accordingly (e.g., if 95% of ratings are 5-stars, a 4-star might actually signal dissatisfaction relative to the norm).

**2. Randomness**
Feedback can be essentially **noise** when users don't want to put real thought into it.

**3. Position Bias**
Users may favor a certain option based on **where it appears** (e.g., first or last), independent of its actual quality.

**4. Preference Bias**
Users may prefer a **longer** response even when it isn't the most accurate one, simply due to a stylistic preference for length/detail.

**5. Degenerate Feedback Loops**
Feedback is only ever collected on what the application actually **shows** the user — which can skew the system's behavior over time in a self-reinforcing way.
> **Example:** In a recommendation system, if a popular item (e.g., a well-known franchise) gets ranked highly early on, it keeps getting shown, keeps getting engagement from fans, and the system learns to show it even more — while other equally good but less-initially-visible content gets buried and never gets a fair chance to be evaluated. The same feedback-loop mechanism can reinforce biased outcomes more broadly (e.g., along lines of gender or race) if the underlying exposure patterns are skewed.
> Acting too literally on user feedback can also push a conversational agent toward **telling users what they want to hear** rather than what's factually accurate — optimizing for "preferred" responses at the cost of truthfulness.

---

*— This concludes the notes for "Building AI Applications."* 🎓