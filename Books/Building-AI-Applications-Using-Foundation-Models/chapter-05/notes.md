# Chapter 5: Prompt Engineering

## Topic 1: Introduction to Prompt Engineering

### System Prompt vs. User Prompt

**System prompt** — the task description. This is where you define:
- The **role** you want the model to play.
- The **type of tasks** it will perform.
- The **kind of output** expected.

**User prompt** — the actual task given by the user. This might include some **context**, followed by a **question** the model needs to answer based on that context.

> **Example:** System prompt → *"You are a customer support agent for a software company. Only answer billing-related questions."* User prompt → *"Here's my invoice: [context]. Why was I charged twice?"*

---

### Evaluating Prompt Engineering Tools

Tools like **OpenPrompt** and **DSPy** automate the prompt engineering process. At a high level, you specify:
- Input format
- Output format
- Evaluation metric
- Evaluation data

This is conceptually similar to how **AutoML** searches for the best hyperparameters for an ML model — except here it's searching over prompts.

**Before adopting these tools, check:**
1. **How they actually work** — how many API calls do they make, what's the search strategy, etc.
2. **Tool developer mistakes** — templates can have bugs, typos in the prompt template, etc. Don't trust the tool blindly.

---

## Topic 2: Defensive Prompt Engineering 🛡️

AI models are susceptible to several types of **prompt attacks**:

| Attack Type | Description |
|---|---|
| **Prompt Extraction** | Users craft questions to trick the model into revealing its system prompt. |
| **Jailbreaking & Prompt Injection** | Malicious content inserted into the prompt causes the model to break its safety guardrails. |
| **Information Extraction** | Attackers prompt the model to divulge details about its training data. |

**Risks from prompt attacks:**
1. Remote code / tool execution
2. Data leaks
3. Social harm
4. Misinformation
5. Service interruption and subversion
6. Brand risk

---

### 🪄 Defence Against the Dark Prompts

Defense mechanisms can be placed at **three layers**: model, prompt, and system.

Before anything else, you need to understand what your application is susceptible to. **Public benchmarks** exist to evaluate how robust a system is against adversarial attacks, typically measured using two key metrics:
- **Violation Rate** — how often the model gets successfully attacked.
- **False Refusal Rate** — how often the model refuses *legitimate* requests by mistake.

#### 1. Model-Level Defence 🧠
Sometimes the model struggles to distinguish system instructions from "dark" (malicious) instructions. The model should follow a strict **instruction hierarchy**:

1. System prompt
2. User prompt
3. Model output
4. Tool output

If there's a conflict, the model should always **prioritize the system prompt**.

#### 2. Prompt-Level Defence 📜
Write prompts that are more resistant to the dark arts:
- Be explicit: *"You can never disclose an employee's ID."* or *"Only answer questions about [domain]; otherwise say it's out of scope."*
- **Trick:** Repeat the system prompt at **both the beginning and the end** of the context — this reinforces the instruction and nudges the model toward compliance. Keep the **context window trade-off** in mind, since this doubles the token cost of your system prompt.

#### 3. System-Level Defence 🏰
Best practice: **isolate** the blast radius of your application.
- Run the application in a **sandboxed/virtual environment**, so any action it takes stays contained.
- Use **human-in-the-loop** approval for sensitive actions — e.g., any query that modifies the database (create, edit, delete) should require explicit human sign-off before execution.
- Define **out-of-scope topics** clearly. *Example: an app built to write C++ code should never answer questions about the weather.* ☔