# Chapter 3: Looking Inside Large Language Models

> 💻 **Implementation:** see `simple-text-pipeline.ipynb` for basic loading of a language model and a text generation pipeline.

---

## Overview of Transformer Models

### Inputs and Outputs of a Trained Transformer LLM

1. A transformer model does **not** generate a full response in one shot — it generates **token by token**. Each token generated corresponds to **one forward pass** through the model.
2. After a new token is generated, it's **appended** to the input prompt, and the model processes this extended sequence again to generate the *next* token.

> **Example:** Prompt: *"The capital of France is"* → forward pass → generates **"Paris"** → new input becomes *"The capital of France is Paris"* → next forward pass generates the next token (e.g., a period) → and so on.

3. This class of models is known as **autoregressive** — each output depends on all previous outputs.
   - **BERT is not autoregressive** — it's a **text representation** model, not a **text generation** model (see Chapter 1's encoder-only discussion).

---

### Components of a Forward Pass

1. Two key components: the **tokenizer** and the **LM head** (Language Modeling head).

2. **Overall structure:**
```
Tokenizer → Neural Network (stack of Transformer blocks) → LM Head
```

3. The **LM head** decides the final output — it translates the transformer stack's output into a **probability score for every possible next token** in the vocabulary.

4. The LM head itself is just a **simple neural network layer**. Depending on the model's objective, this final layer can be swapped out — e.g., a **classification head** instead of an LM head, if the goal is classification rather than generation.

5. **Decoding — choosing the actual output token:**
   After the forward pass, you have a probability score for **every word in the vocabulary**. How to pick the actual next token is called **decoding**, with a few common strategies:
   - **Greedy decoding** — always pick the token with the **highest probability**. (This is effectively what happens when **temperature = 0**.)
   - **Sampling** — introduce some randomness by sampling from the probability distribution rather than always taking the top choice — e.g., a token with 40% probability has a **40% chance** of actually being selected, even if it isn't the single highest-scoring option.
   > **Example:** For the prompt *"The weather today is"*, the model might assign: "sunny" → 45%, "cloudy" → 30%, "cold" → 15%, others → 10%. Greedy decoding always picks **"sunny."** Sampling might occasionally pick **"cloudy"** instead — introducing natural variation across multiple generations of the same prompt.

6. **Parallel token processing:** the tokenizer breaks a sentence into tokens, and all of these **input tokens are processed in parallel** through the transformer stack (this is what makes transformers so much faster to train than the old sequential RNN approach). There's a limit to how many tokens can be processed at once — this is the model's **context window** (see Chapter 1).

7. For a text generation model, only the **output vector of the very last token position** is actually used to predict the next token — that single vector is the only input fed into the LM head, since it's the one responsible for calculating the probability of what comes *next*.

8. **KV Caching:** when generating the next token, the new input is technically the entire sequence so far (previous tokens + newly generated one). Instead of **recomputing everything from scratch** each time, we can **cache the key and value calculations** from previous tokens and simply compute values for the **newest token only**, then append. This is the origin of the **KV cache** mechanism (see Chapter 9 for optimization details).

---

### Inside a Transformer Block

Each transformer block has **two core components**:

**a) Feedforward Layer**
The "information/knowledge" layer. During training, this is the part of the network that **learns patterns** in the data well enough to generalize ("interpolate") to unseen inputs.

> **Example:** Given the input *"You are a wizard"*, the feedforward layers have effectively learned (from training data patterns like Harry Potter text) that **"Harry"** is a strong candidate for what comes next — this kind of learned association lives largely in the feedforward layers.

**b) Attention Layer**
Since tokens are processed **in parallel** (not sequentially like RNNs), we need a separate mechanism to make sure each token still has access to relevant **context** from the rest of the sequence — that's the attention layer's job.

> **Example:** *"Tanya is a good student. She always..."* — here, the model needs to know that **"She" refers to "Tanya."** This kind of connection is exactly what the attention mechanism learns to capture. It produces a matrix indicating **which tokens are most related to which other tokens** — i.e., how important token A is when interpreting token B.

- Attention is usually **duplicated and run multiple times in parallel**, giving the transformer richer, multi-angle attention capability. Each of these parallel copies is called an **attention head** — different heads can end up specializing in different types of relationships (e.g., one head might focus on subject-pronoun links like the example above, another on adjacent word order, etc.).

---

### More Efficient Attention

Attention is the most **computationally expensive** part of the forward pass, so a lot of optimization work focuses here.

**1. Local / Sparse Attention**
- Techniques like **sparse attention** and **sliding window attention** reduce how many previous tokens each token is allowed to "look at," cutting computation significantly.
- **Sparse attention** limits the number of previous tokens visible to the model at each step (rather than attending to the *entire* sequence every time).
- **Interleaving:** sparse/local attention layers can be mixed with occasional **full attention** layers, balancing computational efficiency against the ability to still capture long-range context when it matters (see Chapter 9's windowed + global attention discussion).

> **Example:** With a sliding window of size 4, when processing the 100th token, the model only attends to tokens 97–100 in that particular layer, rather than all 100 previous tokens — dramatically cutting the computation for long sequences, at the cost of some long-range context in that layer.

**2. Multi-Query / Grouped-Query Attention**
- Traditionally, **every** query matrix has its **own** dedicated key and value matrix.
- **Grouped-query attention** instead divides queries into **groups**, where each **group shares a single key-value matrix** for its calculations — reducing memory and compute versus giving every single query its own K/V pair, while retaining more flexibility than giving *all* queries just one shared K/V pair (multi-query attention, the more extreme version of this idea).

> **Example:** With 8 query heads and grouped-query attention using 2 groups, heads 1–4 all share one K/V matrix, and heads 5–8 share another — instead of needing 8 separate K/V matrices (traditional multi-head attention) or squeezing all 8 down to just 1 shared K/V matrix (multi-query attention).