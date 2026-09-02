# Chapter 1: An Introduction to Large Language Models

## What is Language AI?

**Language AI** is the subfield focused on developing technology to **understand, process, and generate** human language.

- Closely related to — and often used interchangeably with — **NLP (Natural Language Processing)**.

---

## A Recent History of Language AI

### 1. Bag-of-Words

The earliest approach: represent text as numbers a machine can work with by simply counting word occurrences.

- **Limitation:** completely ignores the **semantic meaning** of text — "the cat sat" and "sat the cat" would look identical, and unrelated words are treated as equally "different" as related ones.

### 2. Dense Vector Embeddings (Word2Vec)

A better representation, built using **neural networks**, that captures semantic meaning far more effectively.

- **Word2Vec** is a landmark algorithm that creates **dense vector embeddings** for tokens.
- These embeddings allow us to compute **semantic distance** between words — e.g., "king" and "queen" end up closer together in vector space than "king" and "banana."
- **Limitation:** Word2Vec embeddings are **static/context-independent** — a word gets the *same* embedding regardless of the sentence it appears in (e.g., "bank" in "river bank" vs. "bank account" gets identical treatment).

### 3. Encoders, Decoders, and RNNs

Since Word2Vec embeddings ignore context, the next step was to process sentences **sequentially**, generating embeddings that account for surrounding words.

- This introduced **RNNs (Recurrent Neural Networks)** used within **encoder** and **decoder** components.
- Typically, an **encoder** takes in embeddings (e.g., from Word2Vec) and refines them into a **better, more context-aware representation**.

### 4. Attention (with RNNs)

Attention was initially introduced **alongside RNNs**.

- **Job of attention:** determine **which word depends on which other word the most**.
- In a **transformer architecture** (commonly used for translation tasks):
  - An **attention matrix** first computes how much attention each input token should pay to every other input token.
  - During **decoding**, **masked attention** is used — a token being generated can only attend to **itself and previous tokens**, while also referencing the input sequence to decide what to focus on.

### 5. "Attention Is All You Need" — Removing RNNs Entirely

The landmark **Transformer** paper showed that tokens could be processed **in parallel**, removing the need for RNNs altogether.

- **Trade-off:** without RNNs' inherently sequential processing, the model loses any built-in sense of word **order** — this is solved by adding explicit **positional encoding**, which injects position information directly into the token representations.
- With pure attention, each **encoder** block contains: **self-attention** + a **feed-forward layer**.
- Each **decoder** block additionally uses **masked self-attention**, since it must not "see" future tokens during generation.

---

## Encoder-Only vs. Decoder-Only Models

### Encoder-Only Models (e.g., BERT)

**Goal:** generate a rich representation of **each word/token** in the input.

- A special **`[CLS]`** token is prepended to the start of the input sequence.
- Since self-attention lets every token attend to every other token, by the end of the attention layers, the `[CLS]` token's representation ends up encoding a **summary of the entire input** — because it has effectively gathered information from every other token.
- This `[CLS]` representation can then be used for **classification tasks** — e.g., adding a simple classification layer on top to classify an email as **spam or not spam**.

### Decoder-Only Models (e.g., GPT)

**Goal:** generate the **next token** based only on the tokens that came before it (since it cannot look ahead).

- In its most basic form, it simply **completes sentences**.
- With **finetuning**, it can be turned into a full **conversational chatbot**.
- If needed, a `[CLS]`-like token can be appended at the **end** of the sequence in a decoder — since it's the *last* token, it's the only one with a **full view of everything before it**, so it naturally accumulates a summary of the whole text.
  > **Example:** A sentence ending in a question mark signals to the model that a question is being asked — the final token effectively "knows" the whole sentence and its intent by that point.
- Further finetuning makes decoder-only models progressively more capable at nuanced conversation and reasoning.

### Context Length

A vital property of these models: **context length** — the number of tokens a model can process in a single pass. This bounds how much text (conversation history, documents, instructions) the model can "see" at once.

---

## The Training Paradigm of LLMs

Building an LLM consists of **two major stages**:

| Stage | Description |
|---|---|
| **1. Pretraining** | The most **compute-intensive** step. The model learns grammar, world knowledge, and general language patterns — also known as **language modeling**. The resulting model at the end of this stage is called a **foundation model**. |
| **2. Finetuning** | The foundation model is further adapted for **specific downstream tasks** (e.g., instruction-following, chat, coding) 