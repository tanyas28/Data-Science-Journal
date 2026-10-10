# Chapter 8: Semantic Search and Retrieval-Augmented Generation

## Overview

There are three broad categories of models/techniques used for search:

1. **Dense Retrieval** — which is basically semantic search, in the way that we embed the document and the query, and then retrieve the **nearest neighbor** of the query vector.
2. **Reranking** — it is more like a **second step** to the search pipeline, where we first fetch the top chunks, and then we **rerank** them based on which chunk is most relevant to the query, and send only the subset to the LLM.
3. **RAG** — which is basically adding a **ground truth**, extra documents, to the model for a specific application, to reduce hallucination and increase the accuracy of the response (see Chapter 6 for the foundational RAG concepts — this chapter builds on that).

---

## Dense Retrieval

**What it is:** Instead of matching on exact keywords (sparse/term-based retrieval, see Chapter 6), dense retrieval represents both the **query** and every **document** as dense vector embeddings in the same vector space, and retrieves whichever document vectors sit **closest** to the query vector (typically via cosine similarity).

> **Example:** A user searches *"how do I get my money back for a cancelled order?"* A term-based search might fail to match a document titled *"Refund Policy"* since there's no literal word overlap. Dense retrieval, however, embeds both the query and the document, and recognizes they're **semantically close** — "money back" and "refund" map to similar regions of the embedding space — successfully retrieving the right document even with zero shared keywords.

### Caveats of Dense Retrieval

- **Domain shift problem:** Embedding models are trained on a particular mix of data, and they find it genuinely **challenging to generalize to domains they weren't trained on**. A model trained mostly on general web text/news may perform noticeably worse on, say, legal contracts or medical records, since the specialized vocabulary and relationships in that domain weren't well-represented during the model's own training.
- This is precisely why **finetuning the embedding model** (below) becomes important for specialized applications.

### Chunking Long Text

Embedding models have a **limited input length** (their own context window), so long documents cannot simply be embedded whole — they need to be broken into smaller **chunks** before embedding (see Chapter 6's chunking strategies — fixed-size, overlapping, recursive, token-based — all apply directly here, since chunking is a shared concern between RAG and dense retrieval in general).

- The chunk size matters: **too large**, and the embedding becomes a diluted "average" of many different ideas, hurting retrieval precision. **Too small**, and individual chunks may lose the surrounding context needed to be meaningful on their own.

### Finetuning the Embedding Model for Dense Retrieval

Since a general-purpose embedding model may not capture domain-specific semantic relationships well, it can be **finetuned** specifically for retrieval on your target domain/data.

- This typically uses **contrastive learning** — training the model so that embeddings of a query and its *actually relevant* document are pulled **closer together**, while embeddings of the query and *irrelevant* documents are pushed **further apart**.
- This requires training data in the form of **(query, relevant document, irrelevant document)** triples — similar in spirit to the negative-sampling idea from Word2Vec (Chapter 2), but applied here to query-document relevance instead of word co-occurrence.
- A well-finetuned embedding model directly fixes the domain-shift caveat above, since it's no longer relying on generic semantic associations — it's learned what "relevant" actually looks like for your specific documents and queries.

---

## Reranking

**What it is:** A **second-stage refinement** step in the retrieval pipeline. The flow looks like:

```
Query → Dense/Sparse Retrieval (fetch top N candidates, e.g. top 50) → Reranker (re-score and reorder) → Top K sent to LLM (e.g. top 5)
```

**Why it's needed:** The first-stage retriever (dense or sparse) is optimized to be **fast** across a huge corpus — it has to compare the query against potentially millions of documents, so it can't afford to do deep, expensive analysis per document. A reranker, by contrast, only has to look at a **small shortlist** (the top N candidates already fetched), so it can afford to use a much more powerful — and much slower — model to carefully judge relevance, before handing only the best few chunks to the LLM.

### How Rerankers Work

Most rerankers use a **cross-encoder** architecture, as opposed to the **bi-encoder** architecture typically used for first-stage dense retrieval:

| Architecture | How it scores relevance | Speed |
|---|---|---|
| **Bi-encoder** (used for initial retrieval) | Query and document are embedded **separately**, into independent vectors, then compared via cosine similarity. | Fast — document embeddings can be precomputed and indexed once, ahead of time. |
| **Cross-encoder** (used for reranking) | Query and document are fed **together, concatenated, into a single model pass**, which directly outputs a relevance score. | Slow — must be run fresh for every (query, document) pair, since nothing can be precomputed ahead of time. |

> **Why cross-encoders score better:** because the model sees the query and document **together** in one pass, it can model fine-grained interactions between specific words/phrases in the query and specific words/phrases in the document — something a bi-encoder's independently-computed vectors can't capture nearly as precisely. This is exactly why cross-encoders are too slow to use for the *first* pass over an entire corpus, but work great as a second-pass filter over a small shortlist.

**Common reranker models:** cross-encoder models from the `sentence-transformers` family, and commercial rerankers like **Cohere Rerank**.

### Mean Average Precision (MAP)

A common metric for evaluating **ranking quality** — how good is the overall *order* of retrieved results, not just whether relevant items were retrieved at all.

- Builds on **Precision@K** — precision calculated after looking at just the top K results.
- **Average Precision (AP)** for a single query: calculate Precision@K at **each rank position where a relevant document appears**, then average those precision values.
- **Mean Average Precision (MAP)**: the **mean of Average Precision scores across multiple queries** — giving one overall number for how well the system ranks results across a whole test set, not just a single query.

> **Example:** For a single query, suppose the top 5 retrieved results are: Relevant, Not Relevant, Relevant, Relevant, Not Relevant.
> - Precision@1 = 1/1 = 1.0 (relevant doc at position 1)
> - Precision@3 = 2/3 ≈ 0.67 (relevant doc at position 3)
> - Precision@4 = 3/4 = 0.75 (relevant doc at position 4)
> - **Average Precision** = average of these three values (only computed at positions where a relevant doc appeared) = (1.0 + 0.67 + 0.75) / 3 ≈ **0.81**
>
> Do this for every query in your evaluation set, then average all those AP scores together to get the final **MAP** score. A higher MAP means relevant documents are consistently showing up **near the top** of the ranking, across many different queries — not just being retrieved somewhere in the list.

---

## RAG (Retrieval-Augmented Generation)

As covered in Chapter 6: RAG grounds an LLM's response in **retrieved external documents**, rather than relying purely on what the model memorized during training — reducing hallucination and keeping answers up to date without needing to retrain/finetune the model itself.

### Advanced RAG Techniques

**1. Query Rewriting**
The user's raw query is often not phrased in a way that retrieves well (too short, ambiguous, or missing context from earlier turns in a conversation). An LLM is used to **rewrite the query** into a clearer, more self-contained, retrieval-friendly form before it's actually sent to the retriever (see also Chapter 6's query-rewriting discussion).

> **Example:** In a multi-turn chat, a user asks *"What about the East region?"* after previously asking about Q3 sales. Query rewriting expands this into a self-contained query like *"What were the Q3 sales figures for the East region?"* — since the retriever has no memory of the earlier turn and would otherwise get a nearly contentless query to search with.

**2. Multi-Query RAG**
Instead of generating just **one** rewritten query, the LLM generates **several different variations/phrasings** of the same underlying question. Each variant is run through retrieval separately, and the retrieved results are combined (e.g., via RRF — see Chapter 6) into a single, broader context.

> **Example:** For the question *"What causes inflation?"*, the system might generate variants like *"What drives rising prices in an economy?"*, *"Causes of inflation"*, and *"Why does the cost of living increase?"* — each phrasing might surface slightly different relevant documents due to differences in wording, so combining all their results gives broader, more robust coverage than relying on a single phrasing of the query.

**3. Multi-Hop RAG**
Used for questions that can't be answered from a **single** retrieval step, because the answer requires **chaining together facts from multiple documents** — the output of one retrieval step informs what to search for next.

> **Example:** *"What is the net worth of the company that acquired Instagram's main competitor?"* This requires: (1) retrieve which company was Instagram's main competitor, (2) retrieve who acquired that company, (3) retrieve that acquirer's net worth. Each step's answer determines the next retrieval query — a single-pass RAG system would fail here, since no single document contains the full chain of reasoning.

**4. Agentic RAG**
Instead of a fixed, hardcoded retrieval pipeline, an **agent** (see Chapter 6, Topic 3) decides dynamically — at each step — whether to retrieve, what to retrieve, which tool/data source to use, and when it has gathered enough information to actually answer. This allows the system to combine retrieval with other agentic capabilities (tool use, multi-step reasoning, planning) rather than retrieval always being a single fixed step in the pipeline.

**5. Query Routing**
Rather than sending every query to the **same** retrieval source, a router (see Chapter 10's router discussion) first classifies the query and directs it to the **most appropriate** data source or retrieval strategy.

> **Example:** A customer support RAG system might route a billing question to a structured database lookup (Text-to-SQL, see Chapter 6), a product-usage question to a vector search over documentation, and a policy question to a separate, smaller knowledge base of legal/policy documents — rather than searching everything everywhere for every query, which would be slower and noisier.

---

### RAG Evaluation

Beyond the context precision/recall metrics from Chapter 6, RAG systems — especially those that must **cite sources** — are also evaluated on:

**Perceived Utility**
Measures whether the retrieved/generated response is actually judged as **useful** to the end user for their underlying goal — not just factually grounded, but genuinely helpful in context. This is typically measured via human judgment or AI-as-judge (see Chapter 4), since "useful" is a more holistic, subjective quality than pure factual correctness.

**Citation Recall**
Out of all the **source documents that genuinely support** the generated response, what percentage did the model actually **cite**? Low citation recall means the model is leaving out citations for claims that *are* actually backed by retrieved sources — making the response look less verifiable than it really is.

**Citation Precision**
Out of all the citations the model **did** include, what percentage of them **actually support** the corresponding claim in the response? Low citation precision means the model is citing sources that don't really back up what it's claiming — a serious trust issue, since a user checking the citation would find it doesn't actually support the statement.

> **Example:** A model generates a response with 4 claims and cites sources for 3 of them.
> - If only 2 of those 3 citations actually support their claims → **citation precision** = 2/3 ≈ 0.67.
> - If the 4th (uncited) claim was actually supported by a retrieved document the model simply failed to cite → **citation recall** suffers, since a genuinely supportable claim went uncited.
>
> Together, these two metrics catch two different failure modes: **precision** catches *fabricated/wrong* citations, while **recall** catches *missing* citations for claims that were actually retrievable.