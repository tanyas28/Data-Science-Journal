# Chapter 2: Understanding Foundation Models
## Topic 1: Sampling:
**Sampling**  is how model chooses an output from all possible options . Choosing the right sampling startegy can significantly boost a model's performance with relatively little effort.

## Topic 2: Transformer Architecture
This topic introduces **seq2seq(sequence to sequence architecture)**. Transformer is popular on the heels of this architecture.
seq2seq uses **RNN**, and has **encoder and decoder** unit. In its most basic form , encoder processes the input token sequentially, outputing the final hidden state that represents the input. The Decoder then generates output tokens sequentially , conditioned on both the final hidden state of the input and previously generated token.
**Bottleneck** : sequential generation , and processing plus output only looks at final hiddent state output of encoder.
Solution: The **attention mechanism** in **transformer architecture**.
Now how does attention mechanism work?
it has three vectors: **Key, Value and Query**
| Vector | Purpose |
|---------|----------|
| Query (Q) | What information am I looking for? |
| Key (K) | What information does this token contain? |
| Value (V) | The actual information passed forward |


it decides how much attention to give to particular token by performing a **dot product** between query vector and its key vector.

## Self Attention

The self-attention mechanism allows every word in a sentence to look at every other word and decide **which words are important for understanding its own meaning**.

The complete self-attention equation is:

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

## How does the attention function work?

### Algorithm

### 1. Create embeddings

The first step is to convert every word into an embedding vector.

Suppose the sentence has **n words**, and each embedding has **d dimensions**.

The embedding matrix is:

$$
X \in \mathbb{R}^{n \times d}
$$

- **Each row** represents the embedding of one word.
- **Each column** represents one learned embedding feature (its meaning is learned by the model and is not human interpretable).

---

### 2. Calculate Query (Q), Key (K), and Value (V)

These are calculated using three learned weight matrices.

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

where:

- \(W_Q\), \(W_K\), and \(W_V\) are **learned during training** through backpropagation, just like weights in a neural network.
- Initially these matrices are random.
- During training they learn how to produce useful Query, Key, and Value vectors.

After this step:

- Each **row of Q** is the Query vector of one word.
- Each **row of K** is the Key vector of one word.
- Each **row of V** is the Value vector of one word.

---

### 3. Calculate the Score Matrix

Next we calculate

$$
Score = QK^T
$$

This is simply a matrix multiplication between **Q** and the **transpose of K**.

The purpose of this step is to compare **every Query against every Key**.

---

### 4. Why do we transpose K?

Each row of **K** represents the Key vector of one word.

If there are three words, conceptually we want to compute something like:

```
q1·k1
q1·k2
q1·k3

q2·k1
q2·k2
q2·k3

q3·k1
q3·k2
q3·k3
```

Since matrix multiplication performs **row × column**, we transpose **K** so that every Query vector is compared with every Key vector in one matrix multiplication.

After this step:

- **Each row** of the Score matrix corresponds to one Query word.
- **Each column** corresponds to one Key word.
- **Each element** represents how strongly those two words are related.

---

### 5. Scale the scores

Now divide every score by

$$
\sqrt{d_k}
$$

where \(d_k\) is the dimension of the Key vectors.

This scaling is important because larger vectors produce larger dot products.

Without scaling, the scores can become very large, causing the Softmax function to produce extremely peaked probabilities where one word dominates and the model struggles to learn effectively.

---

### 6. Apply Softmax

Now apply Softmax **row-wise**.

This converts the raw scores into probabilities.

Properties:

- Every value lies between **0 and 1**.
- Every row sums to **1**.

Each row now represents the **attention distribution** for one word.

For example,

```
[0.2, 0.5, 0.3]
```

means:

- Give 20% importance to Word 1
- Give 50% importance to Word 2
- Give 30% importance to Word 3

while updating the current word.

---

### 7. Multiply Attention Weights with the Value Matrix

The next step is

$$
Output = AttentionWeights \times V
$$

Notice that **we do NOT transpose V**.

Why?

Each row of **V** already represents the Value vector of one word.

Suppose one row of the attention matrix is:

```
[w1, w2, w3]
```

This means:

- Use **w1** amount of information from **Word 1**
- Use **w2** amount of information from **Word 2**
- Use **w3** amount of information from **Word 3**

Mathematically, this is equivalent to:

$$
w_1V_1 + w_2V_2 + w_3V_3
$$

where \(V_1\), \(V_2\), and \(V_3\) are the Value vectors of the three words.

In simple words:

> We create a **weighted average** of all the Value vectors.

We use a weighted average because not every word contributes equally to the meaning of the current word.

For example, when understanding the word **bank** in:

> *I deposited money in the bank.*

the words **money** and **deposited** should influence the representation of **bank** much more than the word **I**.

Matrix multiplication simply performs this weighted sum efficiently for every word in the sentence.

---

### 8. Final Output

The result is the Output matrix.

- **Each row** is the new contextual representation of one word.
- The original embedding is not modified.
- Instead, a new embedding is created that contains information gathered from the other words in the sentence according to the learned attention weights.

These contextual embeddings are then passed to the next layers of the Transformer.
