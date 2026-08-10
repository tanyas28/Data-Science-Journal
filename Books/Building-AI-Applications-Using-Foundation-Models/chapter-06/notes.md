# Chapter 6: RAG and Agents

This chapter focuses on how to create **correct and relevant context** for each query. Applications share the same instruction (system prompt), but the **context differs for every query**.

This can be achieved two ways:
- **RAG** — gives context from **external sources** (documents/data).
- **Agents** — gives context using **external tools** (APIs, code execution, etc.).

---

## Topic 1: RAG → Retrieval Augmented Generation

RAG helps fetch information **relevant to a particular query only**, instead of dumping the entire dataset into context. This:
- Reduces hallucination.
- Removes the need to feed complete information for every task.

### RAG Architecture

1. RAG has two main parts: **Retriever** and **Generator**.
   - In the original paper, the retriever and generator were trained *together*. Now they're typically trained **separately**.
2. The **retriever** largely determines how good the RAG system actually is. It consists of:
   - **Indexing** — how the data is stored.
   - **Querying** — sending a query to the external database to fetch relevant information.

---

### Retrieval Algorithms

#### A) Term-Based Retrieval (Sparse Retrieval)

The most intuitive way to fetch a relevant document for a query is to search for documents containing the **same words** as the query.

**Problem 1: Too many documents share the same words.**
> **Solution:** Use **Term Frequency (TF)** — the number of times a term appears in a document. A document containing a query term more often is assumed to be more relevant to that term.

**Problem 2: Not all words are equally informative.**
> A word that appears in *almost every* document (like "the" or "is") carries very little distinguishing information. So a term's importance is **inversely proportional** to the number of documents it appears in — this is **Inverse Document Frequency (IDF)**. The higher the IDF, the more important the term.

**TF-IDF** combines both signals:

$$
IDF(t) = \log\left(\frac{N}{c(t)}\right)
$$

where:
- $N$ = total number of documents
- $c(t)$ = number of documents containing term $t$

$$
TF\text{-}IDF(D, q) = \sum_{t \in q} IDF(t) \cdot f(t, D)
$$

where $f(t, D)$ is the frequency of term $t$ in document $D$.

> **Example:** For the query *"vector database indexing"*, a document that repeats "indexing" many times but where "indexing" is also a rare term across the whole corpus gets a high TF-IDF score — it's both frequent *here* and distinctive *overall*.

**Two common term-based retrieval solutions:**

| Method | How it works |
|---|---|
| **Elasticsearch** | Uses an **inverted index** — maps terms → documents containing them, enabling fast lookup. It also stores metadata (document count, term frequency per document) used to compute TF-IDF. |
| **BM25** | A refinement of TF-IDF. It **normalizes TF scores by document length**, since longer documents are naturally more likely to contain a given word and to have higher raw term frequency — without normalization, long documents would be unfairly favored. |

---

#### B) Embedding-Based Retrieval (Dense Retrieval)

This retrieves documents based on how close the query is in **meaning** to the context — not just shared words.

**Process:**
1. Store documents as **embeddings** in a vector database.
2. Retrieve the top chunks most **similar** to the query embedding (e.g., via cosine similarity — see Chapter 3).

**Choosing a vector database matters** — it determines how efficiently embeddings are indexed and searched.
- A naive approach is **exact k-NN**, but it's too slow at scale.
- For large datasets, **Approximate Nearest Neighbor (ANN)** algorithms are used instead.
- Popular vector DBs: **FAISS, ScaNN**, etc.

**Vector search algorithms differ by heuristic:**

| Algorithm | How it works |
|---|---|
| **LSH** (Locality-Sensitive Hashing) | Hashes similar vectors into the same bucket, speeding up similarity search. Used in FAISS, Annoy. |
| **HNSW** (Hierarchical Navigable Small World) | Builds a multi-layer graph where each node is a vector and edges connect similar vectors. Search = traversing the graph toward the query. |
| **Product Quantization** | Compresses vectors into simpler, lower-dimensional representations, making distance calculations much faster and cheaper. |
| **Inverted File Index (IVF)** | Uses **K-means clustering** to group similar vectors into clusters (each with ~100–10,000 vectors). Finds the cluster centroid(s) closest to the query embedding as candidates. Often combined with Product Quantization — this pairing is used in FAISS. |
| **Annoy** (Approximate Nearest Neighbors Oh Yeah) | Tree-based approach — builds multiple binary trees, each splitting vectors into clusters using random criteria. Search = traversing the trees to find neighbors. Open-sourced by **Spotify**. |

---

### Comparing Retrieval Algorithms

| | Term-based | Embedding-based |
|---|---|---|
| **Speed** | Faster, simpler, works well out of the box | Slower, more moving parts |
| **Improvability** | Little room to improve further | Can be improved over time — embedding generation, retrieval algorithm, and generation can all be fine-tuned |
| **Weakness** | Misses semantic meaning (synonyms, paraphrasing) | Converting words to embeddings can lose exact-match signals — e.g., specific keywords, product names, IDs |

> **Example:** A term-based search for `"iPhone 15 Pro"` will reliably find documents containing that exact string. An embedding-based search might rank a document about *"Samsung Galaxy S24"* nearly as relevant, since it's semantically close ("a flagship smartphone"), even though it's not what the user wanted.

---

### Evaluating Retrieval Quality

Two key metrics:

- **Context Precision**: Out of all documents retrieved, what percentage was actually relevant?

$$
\text{Context Precision} = \frac{\text{Relevant documents retrieved}}{\text{Total documents retrieved}}
$$

- **Context Recall**: Out of all relevant documents that exist, how many were actually fetched?

$$
\text{Context Recall} = \frac{\text{Relevant documents retrieved}}{\text{Total relevant documents in the corpus}}
$$

To compute either, you need an **annotated dataset** — a list of queries paired with their known relevant documents.

> **Why production favors precision over recall:** Recall requires knowing the *complete* set of relevant documents in your entire corpus for a given query — which is often impossible to fully label at production scale (the corpus keeps growing, and exhaustively annotating "every relevant doc" for every possible query isn't feasible). Precision, on the other hand, only requires judging the documents that were *actually retrieved* — a much smaller, bounded set that can be scored on the fly (e.g., via a human or an AI judge). This makes precision far cheaper and more practical to monitor continuously in production.

---

### RAG System Evaluation Levels

The quality of a RAG system should be evaluated at multiple levels:

1. **Retrieval quality** — precision/recall of the retriever.
2. **Final RAG output** — quality of the generated answer (faithfulness, relevance, correctness — see Chapter 4).
3. **Embedding quality** — how well the embedding model captures semantic similarity for your domain.

---

### Combining Retrieval Algorithms

1. In production, retrieval algorithms are mostly used **in combination**. Example: a cheaper retrieval algorithm first fetches a broad set of candidate chunks, then a more expensive algorithm re-ranks/refines these to get the correct context for the query.
   - This is known as the **hybrid approach**.
2. Different algorithms can also run **in parallel** — known as the **ensemble technique**. Multiple retrievers each fetch documents, and results are ranked by how many retrievers agreed on the same document, building a more robust context. This adds latency and cost, so the trade-off must be managed.

#### Reciprocal Rank Fusion (RRF)

The algorithm used to combine rankings from an ensemble of retrievers.

> **Simplified intuition:** If Doc 1 is ranked 1st by Retriever A and 2nd by Retriever B, a naive combined score would be $1 + \frac{1}{2} = 1.5$.

The actual RRF formula:

$$
RRF(D) = \sum_{i=1}^{n} \frac{1}{k + r_i(D)}
$$

where:
- $n$ = number of ranked lists (i.e., number of retrievers)
- $r_i(D)$ = rank of document $D$ in retriever $i$'s ranked list
- $k$ = a constant to avoid division by zero and to dampen the influence of very high ranks (commonly **k = 60**)

> **Example:** Doc 1 is ranked #1 by Retriever A and #2 by Retriever B, with $k=60$:
> $$RRF(D_1) = \frac{1}{60+1} + \frac{1}{60+2} = 0.0164 + 0.0161 = 0.0325$$
> The document with the highest total RRF score across all retrievers wins the final ranking.

---

### Retrieval Optimization

Four ways to optimize retrieval:

**1. Chunking**
- **Fixed-size chunking** — simplest approach, split text into equal-sized pieces.
- **Overlapping chunking** — adjacent chunks share some overlap, reducing the chance of splitting relevant context awkwardly.
- **Recursive chunking** — splits text hierarchically (e.g., by section → paragraph → sentence) until chunks fit the target size.
- **Token-based chunking** — chunk by token count (using the *same tokenizer* as the model that will consume the chunks) rather than by word count, since token count is what actually determines context window usage.

**2. Re-Ranking**
Useful when you need to reduce the number of retrieved documents — typically to fit the foundation model's context window.
- Common pattern: cheap retrieval fetches a wide set (e.g., 10 candidates), then an expensive re-ranker shortlists the best few.
- Heuristics can be layered in — e.g., documents accessed less frequently can be ranked lower than commonly-fetched ones.
- **Contextual re-ranking** (different from search re-ranking): instead of just re-scoring by query-document similarity, it factors in broader context — e.g., conversation history, user profile, or session state — to re-order results in a way that fits the *current situation*, not just the query text.

**3. Query Rewriting**
A well-written, self-sufficient query yields better retrieval results, since it increases the chance of fetching the correct chunks.

> **Example (real-world):** In an agentic AI application, a **query-rewriter node** can take the raw user query and transform it into more domain-appropriate language that the retrieval system and LLM can better work with — improving the odds of fetching the correct records for most queries.

**4. Contextual Retrieval**
The intuition: augment chunks with metadata that quickly tells the LLM what the chunk is about *before* it's even retrieved.
- **Example:** E-commerce product chunks can be augmented with tags/keywords.
- Special keywords within documents can be captured as metadata.
- Chunks can be augmented with the **questions they can answer** — especially powerful for customer support chatbots (e.g., a chunk about a return policy gets tagged with *"How do I return an item?"*).

---

### Evaluating a Retrieval System

Key questions to ask:
1. What retrieval mechanisms does it support? (e.g., hybrid search)
2. Can it **scale**?
3. **Indexing performance** — how long does it take to index data, and how much data can be bulk-processed (added/deleted) at once?
4. What is the **query latency**?

---

## Topic 2: RAG Beyond Text

RAG isn't limited to plain text — it can be **multimodal**, incorporating images, and can also handle large **tables**.

**Images:**
- Each image can be augmented with **metadata** (captions, tags, source).
- **Multimodal embedding models** (e.g., CLIP-style models) embed both images and text into the **same vector space**, so a text query can directly retrieve semantically relevant images without needing a separate text description.

**Tables:**
- Tables can be handled via **metadata augmentation** — e.g., carrying the **header row** as metadata alongside each chunk of table data, so the LLM knows what each column represents even if it only sees a fragment of the table.

**Case: Structured tables + SQL querying**

When the underlying data lives in **relational tables** rather than free text, a pure embedding/chunking approach doesn't work well — you can't meaningfully "chunk" a database table the way you chunk a document, and similarity search isn't the right tool for precise aggregation questions (e.g., *"What was total revenue last quarter?"*).

Instead, the common pattern is **Text-to-SQL retrieval**:
1. The LLM is given the **table schema** (table names, column names, types, relationships) as context — not the actual row data.
2. The user's natural language question is translated by the LLM into a **SQL query**.
3. The SQL query is **executed** against the actual database.
4. The **query result** (a small, precise set of rows) is fed back to the LLM as context to generate the final natural-language answer.

> **Example:** User asks *"How many employees claimed dental benefits last year?"* → LLM generates `SELECT COUNT(*) FROM claims WHERE type='dental' AND year=2025;` → query runs against the actual database → the LLM turns the returned number into a natural sentence.

This keeps answers **precise and grounded** (no hallucinated numbers, since the answer comes directly from the database) but introduces new risks — the LLM must generate *valid and safe* SQL, which is why validation layers (checking the query is syntactically correct, only touches allowed tables, and is read-only where required) become important before execution.

---

## Topic 3: Agents

An **AI agent** is built to execute a task given by the user, **end to end**.

### Tools

Tools help an agent actually **act**, not just generate text.
- **Read-only actions** — don't modify anything (e.g., fetching data).
- **Write actions** — modify state (e.g., updating a database, sending an email).

The full set of tools an agent can use is its **tool inventory**. Three broad categories:

**1. Knowledge Augmentation**
Tools like search/web search that expand the agent's knowledge base beyond its training data.

**2. Capability Extension**
Tools that give the model abilities it doesn't natively have:
- A **calculator** greatly improves math accuracy.
- Access to a **code interpreter** (e.g., C++, Python) lets the model run code, check if it's correct, and iterate on failures — producing better final code.
- Tools can turn a text-only model **multimodal** — e.g., ChatGPT using **DALL-E** as an image generation tool.
- A text-only model can use an **image captioning tool** to process images, a **transcription tool** to process audio, or **OCR tools** to read PDFs.

**3. Write Actions**
Tools that let the agent actually take action, not just retrieve/generate:
- A SQL executor can retrieve data — but can also modify the database.
- An email API can read emails — but can also send them.
- A banking API can retrieve account info — but can also initiate transactions.

> Because write actions can cause real-world side effects, they need a dedicated **safety layer** (e.g., human approval, permission scoping).

Many foundation models support **function calling** — this is essentially the mechanism underlying tool use.

---

### Planning

Planning is the **heart** of how well an agent executes a task.

#### Overview

1. The model should **decompose** a complex task into smaller sub-tasks. Not every decomposition is efficient, so a common pattern is:
   - Model generates a plan (broken into sub-tasks).
   - A **human** or an **AI-as-judge** evaluates the plan's feasibility *before* execution.
2. **Heuristics for rejecting a plan:**
   - A sub-task requires a tool the agent doesn't have access to → reject, agent re-plans.
   - Plan exceeds a maximum number of steps (e.g., more than $Y$ steps) → reject.
3. After execution, the **plan's output** also needs to be evaluated.

This creates a system with **three parts** — essentially a tiny multi-agent system:
1. **Generate** plans
2. **Validate** plans
3. **Execute** plans

- Multiple plans can be generated **in parallel**, with an evaluator picking the most accurate one — again trading off latency/cost for quality.
- **Intent classification** is a common precursor step: figure out the user's intent *before* generating a plan. An intent classifier can determine which tools are likely needed, what data needs to be accessed, and route the query accordingly.

> **Example (real-world):** An intent classifier can determine whether a user query needs **structured data**, **unstructured data**, or a **hybrid** of both — and which specific agent(s) the user is trying to reach (a single agent, or a combination of several).

---

#### Foundation Models as Planners

Autoregressive LLMs **cannot truly plan** on their own. However, an LLM can still be *part of* a planning system — e.g., giving it access to a **search tool** and a **state-tracking system** to support planning externally.

---

#### Plan Generation

The simplest way to get a model to generate good plans is through **strong prompting**:
1. Write better **system prompts**.
2. Give better **descriptions of tools and their parameters**.
3. **Simplify tools** — fewer, cleaner parameters reduce planning errors.
4. **Fine-tune** the model specifically for better plan generation.