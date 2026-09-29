# Chapter 5: Text Clustering and Topic Modeling

1. Text clustering helps in categorizing large volumes of unstructured data efficiently, and also helps in quick data exploratory analysis.
2. Text clustering is also used in **topic modeling**, wherein we want to get the topic, or title, of a given unstructured text.

---

## A Common Pipeline for Text Clustering

This basically involves **three steps**:

1. **Convert text into embeddings** — using some transformer-based embedding model (e.g., `thenlper/gte-small`).
2. **Reduce the dimensions** of these embeddings using a dimensionality reduction model — using algorithms like **PCA**, or **UMAP** (better for non-linear data).
3. **Group semantically similar documents together** with a cluster model — using centroid-based algorithms like **K-Means**, or, better yet, density-based algorithms like **DBSCAN** or **HDBSCAN**.

```
Documents → Embedding Model → Dimensionality Reduction (PCA / UMAP) → Clustering (K-Means / DBSCAN / HDBSCAN) → Clusters
```

---

## From Text Clustering to Topic Modeling

The traditional topic modeling models or algorithms, like **Latent Dirichlet Allocation (LDA)**, involve finding keywords (multiple) that best represent the clusters of text. But these models are basically based on **bag-of-words**, and thus do not take into account the **semantic similarity** or **contextual info** of words.

---

## BERTopic: A Modular Topic Modeling Framework

### Algorithm

The algorithm has **two parts**:

**Part 1**
We do the same thing we did for clustering: **embed the documents**, **reduce dimensionality**, and then **cluster them**.

**Part 2**
Next, we want to know **what word appears how many times at the cluster level**. One way is obviously bag-of-words, which will create a frequency map for each word in the documents. But we don't want document-level word frequency — we want **cluster-level** — and for that we use something called **c-TF**, or **cluster-term frequency**.

Now, we do not want **stop words** to influence our topic modeling, so we want a technique that helps us with this — for that we already have **TF-IDF**. Here we just kind of use its variation: **c-TF-IDF**.

Together, Part 1 and Part 2 give us various **keywords for clusters**. The best part about it is that **everything in this pipeline can be changed** — we can use any model for embedding, for clustering, for dimension reduction, and any algorithm in Part 2 as well.

**Disadvantage:** it's still based on **bag-of-words** at the keyword-extraction stage, depending on the **frequency of words used** — so it still inherits some of the same semantic blind spots as older methods, just applied at the cluster level instead of the document level.

---

## Adding a Special Lego Block

We can use the power of bag-of-words to get the **keyword representation of clusters**, and then we can further **fine-tune** it using some model to **rerank the keywords**.

Such reranker models are often known as **representation models** in BERTopic. We can even **stack them multiple times** to fine-tune the result.

There are multiple types of strategies that can be used:

### 1. KeyBERTInspired

What it does is: after we have keywords from the initial c-TF-IDF algorithm, this method **compares the cosine similarity between the topic embedding and the document embeddings** — and reranks the keyword candidates based on this.

**In simpler terms:** c-TF-IDF gives you a first rough list of candidate keywords for a cluster purely by **counting/weighting word frequency**. The problem is, frequency alone doesn't guarantee a word is *semantically* representative of what the cluster is actually about — a frequent word could still be a weak descriptor. KeyBERTInspired fixes this by:
1. Taking each candidate keyword from c-TF-IDF and **embedding it** (turning it into a vector, just like a document).
2. Taking the **average embedding of the documents** in that cluster (or the cluster's overall topic embedding).
3. Computing the **cosine similarity** between each candidate keyword's embedding and this cluster embedding.
4. **Re-ranking** the keywords by this similarity score — keywords that are *semantically* closest to what the cluster is actually about get pushed to the top, even if a different word technically appeared more often.

> **Example:** Suppose a cluster of documents is all about **electric vehicles**. c-TF-IDF might rank the word *"car"* highly just because it's frequent — but *"car"* is a fairly generic word that could describe many unrelated clusters too. KeyBERTInspired might instead push a word like *"battery"* or *"charging"* higher up the list, because their embeddings are **semantically closer** to the overall meaning of the cluster's documents, even if they appeared slightly less often than "car." The result is a keyword list that better captures the *actual theme*, not just the *most repeated word*.

### 2. Maximal Marginal Relevance (MMR)

If we want our topic to be **diversified** (because "summary" and "summaries" are treated differently — i.e., near-duplicate/redundant keywords can otherwise flood the list), we can use this instead. It helps us set **how diverse** we want our topics to be, and based on that, gives the keywords for clusters.

**In simpler terms:** without MMR, a cluster's top keywords can end up being near-duplicates of each other — e.g., different inflections or close synonyms of essentially the same word — which wastes "keyword slots" without adding new information about the topic. MMR balances two competing goals when picking each next keyword:
- **Relevance** — how well does this keyword represent the cluster (same idea as before — similarity to the cluster's embedding)?
- **Diversity** — how *different* is this keyword from the keywords **already selected** so far?

It picks keywords one at a time, each time favoring a candidate that's still reasonably relevant, but **not too similar** to what's already been picked — controlled by a tunable parameter for how much to prioritize diversity vs. pure relevance.

> **Example:** For a cluster about **electric vehicles**, a plain relevance-ranked list might return: *"battery," "batteries," "battery life," "charging," "charge."* Very relevant, but repetitive — it's really just circling "battery" and "charge" in different forms. MMR, tuned toward more diversity, might instead return: *"battery," "charging station," "range anxiety," "Tesla," "subsidy"* — still all clearly relevant to the topic, but each keyword now covers a **distinct angle** of the topic (technology, infrastructure, consumer concern, brand, policy) rather than five variations of the same idea.

### 3. Text Generation (Generative Model)

We can use text generation (generative model) as well. Once we have keywords, we can give the keywords and a subset of documents (say, the best 4) to a generative model and ask it to find a **label or one title** for the documents or cluster.

We can rank documents based on the **highest cosine similarity** between the c-TF-IDF value of the keyword and the documents.