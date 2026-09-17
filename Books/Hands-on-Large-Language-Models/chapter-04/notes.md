# Chapter 4: Text Classification

Text classification can be approached using either a **representation model** or a **generative model**. This chapter starts with representation models.

---

## Text Classification Using Representation Models

When using a **pretrained representation model**, there are two broad routes:

1. **Task-specific model** — already trained (finetuned) for the exact end goal you need.
2. **General embedding model** — not trained for any specific end goal; it just represents text as vectors in embedding space, and needs an additional step to actually classify.

### Route 1: Task-Specific Model

If the model was already finetuned for a task matching your goal, the flow is simple:

```
Input → Task-specific Model → Output category (binary or multi-class)
```

### Route 2: General-Purpose Embedding Model

If using a general embedding model (not finetuned for classification), the embeddings alone can't classify anything on their own — you need to pair them with a separate classifier:

```
Input → Embedding Model → Classification Model (e.g., Logistic Regression) → Output category
```

### Selecting a Model

- **BERT** is a popular choice for both task-specific *and* general embedding use.
- **Generative models** perform well at classification too, but **encoder-only models** (like BERT) achieve **similar performance at a much smaller size** — making them more efficient for this task.
- Many refined BERT variants exist, each trained on different data/objectives to specialize further: **RoBERTa, DistilBERT, ALBERT**, and others.

> 💻 **Implementation:** see `movie-review.ipynb` — movie review classification using the task-specific **Twitter-RoBERTa-base-for-Sentiment-Analysis** model.
>
> **Basic flow:** load the model + tokenizer → run predictions on the test dataset → evaluate using a small helper function that reports **accuracy, precision, recall**, etc.

---

## Text Classification Using Embeddings (No Task-Specific Model)

**The question:** what if no model exists that's already finetuned for your specific context — do you always need to finetune one yourself?

**Answer:** only when it's **genuinely important**, since finetuning carries significant **computational cost**. Often, a lighter-weight approach works just as well.

**The lighter approach:**
1. Use a general embedding model — e.g., from the **`sentence-transformers`** package — to generate embeddings for your dataset.
2. Treat these embeddings as **features**, and train a **classical ML model** on top of them (e.g., **Logistic Regression**) to perform the actual classification.

> Since these downstream classifiers are small and simple, they can be trained easily on a **CPU** — no expensive GPU training required.

> 💻 **Implementation:** see `movie-review.ipynb`.

This approach works well when you have **labeled data** (supervised training). But what about unlabeled data?

---

## What If the Data Is Unlabeled?

With unlabeled data, **clustering** or other unsupervised algorithms can group similar examples together — but they can't tell you *what each cluster actually means* (i.e., no ground-truth label attached).

### Zero-Shot Classification

**The idea:** classify data into categories the model was **never explicitly trained on** — the model only knows the **name/description** of each category, not labeled examples of it.

**How it works:**
1. Write a **descriptive label** for each target category.
   > **Example:** For movie review sentiment, instead of a bare label like "positive," use a descriptive phrase like *"this is a positive review"* and *"this is a negative review."*
2. **Encode the labels** into embeddings, the same way you'd encode any text (e.g., `model.encode()`).
3. You now have **document embeddings** *and* **label embeddings** in the same vector space.
4. Compute the **similarity** between each document's embedding and each label's embedding — typically using **cosine similarity** (see Chapter 3).
5. Assign each document to whichever label embedding it's **most similar to**.

> **Result:** a working classifier — able to sort reviews into "good" or "bad" — built entirely by giving categories a **descriptive name**, with **zero labeled training examples** needed.