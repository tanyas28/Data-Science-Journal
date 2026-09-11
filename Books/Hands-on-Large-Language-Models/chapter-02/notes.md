# Chapter 2: Tokens and Embeddings

## Topic 1: LLM Tokenization

> 💻 **Implementation:** see `tokenization_implementation.ipynb` for code around tokenization, input tokens, and output tokens.

### How Does a Tokenizer Break Down Text?

Three major factors determine this:

1. **Model design** — the model's creator chooses a tokenization method, e.g.:
   - **Byte Pair Encoding (BPE)** — used in **GPT** models.
   - **WordPiece** — used in **BERT**.
2. **Configuration decisions** — once a method is chosen, decisions like **vocabulary size** and which **special characters/tokens** to include need to be made.
3. **Training the tokenizer** — the tokenizer itself is then **trained on a specific dataset** to establish the best vocabulary for representing that dataset's text.

### Types of Tokenizers

| Type | Description |
|---|---|
| **Word tokenizer** | Splits text into whole words. |
| **Subword tokenizer** | Splits words into a mix of **complete and partial** word-pieces — the most common approach in modern LLMs. |
| **Character tokenizer** | Falls back to **individual letters** — can represent *any* word (huge vocabulary flexibility), but produces very long token sequences since it tokenizes one character at a time. |
| **Byte tokenizer** | Breaks tokens down into **individual bytes** — the most granular fallback. |

> Many subword tokenizers use **byte-level tokens** as their final fallback block — this way, even a completely out-of-vocabulary word can still be represented (just less efficiently), instead of being dropped or replaced with an "unknown" token.

---

## Topic 2: Token Embeddings

The next step: find the best **numerical representation** for tokens — ideally a **dense representation** that clearly captures patterns in the text.

- Once a tokenizer is **trained**, it's used throughout the **training process** of the language model itself.
- Because of this, when using a pretrained base model, you must **always use the exact same tokenizer** it was trained with — a mismatched tokenizer would produce meaningless token IDs.

### Creating Contextualized Word Embeddings

> 💻 **Implementation:** see `tokenization_implementation.ipynb`.

A language model produces **contextualized word embeddings** — meaning the *same* word can get a **different embedding** depending on the context it appears in (unlike static embeddings like Word2Vec).

**Basic pipeline:**

```
Input string
   → Tokenization (break into tokens)
   → Token embedding vectors (initial numerical representation)
   → Language model (processes text, adds contextual information)
   → Contextual token vectors (final, context-aware representation)
```

---

## Topic 3: Text Embeddings

Most real applications need to process **entire sentences or passages**, not just individual words. This led to specialized language models that produce **text embeddings** — a **single vector** representing an entire piece of text.

- **Sentence Transformers** are commonly used for this.
- Sentence Transformer models often output a vector of dimension **768** — this corresponds to the model's **hidden size**.
- For **long documents**, rather than squeezing the entire text into one embedding vector, it's often better to **chunk** the text into smaller parts and embed each chunk separately.

---

## Topic 4: Word Embeddings Beyond LLMs

Embeddings aren't limited to language generation tasks. Converting words (or items) into meaningful mathematical representations is broadly useful — e.g., in **recommender systems**, **robotics**, and other domains entirely outside of text generation.

> 💻 **Implementation:** see `tokenization_implementation.ipynb` for using pretrained word embeddings.

### Word2Vec Algorithm and Contrastive Training

Word2Vec is trained on examples **generated directly from text**.

**Generating training examples — the sliding window:**
- The algorithm uses a **sliding window** over the text to generate training pairs.
- **Example:** With a window size of 3, for any given **center word**, the training examples include the **3 words before it** and the **3 words after it** as its "neighbors."

**Training as a classification task:**
- The embeddings are learned via a **binary classification task**: train a neural network to predict whether **two words commonly appear together** in the same context (i.e., co-occur within sentences in the training corpus).
- The network takes **two words** as input and outputs **1** if they frequently appear together, or **0** if they don't.
- The **center word** pairs with **each word in its sliding window** to form positive training examples.

**Handling bias — negative sampling:**
- If the training set only ever contains **true co-occurring pairs**, the network becomes biased (it would learn to just predict "1" for everything, since it never sees negative examples).
- To fix this, **negative sampling** is introduced — deliberately adding pairs of words that **don't** typically co-occur, labeled as **0**. This is conceptually similar to how bias is managed in other deep learning classification setups.

**The two core components of Word2Vec, summarized:**

| Component | Role |
|---|---|
| **Skip-gram** | Selecting neighboring words (within the sliding window) as positive training pairs. |
| **Negative Sampling** | Adding non-co-occurring word pairs as negative training examples, to prevent bias. |

**Putting it together:**
1. Decide on a tokenization scheme.
2. Generate an embedding vector for each token, **randomly initialized** — shape: `vocab_size × embedding_dim`.
3. Train a model on each (word pair) example: take the embedding vectors of two words, and **predict whether they're related** (co-occurring) or not.

```
Tokens → Embeddings → Neural Network → Prediction (e.g. 0.90)
```

4. Based on whether the prediction was correct, the neural network **adjusts the embeddings** so it classifies similar pairs more accurately next time.
5. By the end of training, each word has a meaningfully **better embedding representation** — words that co-occur often end up closer together in the embedding space.

> 🎵 **Implementation:** see `music_recomm_using_word2vec.ipynb` for a music recommendation system built using Word2Vec.