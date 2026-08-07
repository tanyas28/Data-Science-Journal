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