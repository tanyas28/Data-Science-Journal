# Chapter 2: Understanding Foundation Models
## Topic 1: Sampling

**Sampling** is how a model chooses an output from all possible options. Choosing the right sampling strategy can significantly improve a model's performance with relatively little effort.

### Sampling Fundamentals

1. The model generates the next token by computing a probability for every token in the vocabulary.
2. The simplest approach is **Greedy Sampling**, where the model always picks the token with the highest probability.
3. **Drawback:** Greedy sampling often produces repetitive, predictable, and less creative responses.
4. Instead of always selecting the highest-probability token, we can **sample from the entire probability distribution**.

To obtain this probability distribution:

- The model first produces **logits** (raw, unnormalized scores) for every token in the vocabulary.
- The logits form a vector of size equal to the vocabulary size.
- These logits are converted into probabilities using the **Softmax** function.

Softmax formula:

\[
P(x_i)=\frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
\]

where:

- \(z_i\) = logit of token *i*
- \(V\) = vocabulary size

**Drawback:** Softmax requires computation over the entire vocabulary (typically two passes), making it computationally expensive for very large vocabularies.

---

### Sampling Variables

The most common sampling variables are:

- Temperature
- Top-K
- Top-P (Nucleus Sampling)

Two important things to learn:

1. How to sample tokens.
2. How to sample outputs in a way that produces structured responses.

---

### Temperature

1. Higher **temperature** → more creative but less coherent responses.
2. Lower **temperature** → more deterministic and focused responses.
3. Temperature is applied **before Softmax** by dividing the logits:

\[
z'_i=\frac{z_i}{T}
\]

where \(T\) is the temperature.

The modified logits are then passed through Softmax.

- **High temperature** flattens the probability distribution, increasing the chance of selecting rarer tokens.
- **Low temperature** sharpens the distribution, making high-probability tokens even more likely.

**Note:** We do **not** set temperature to exactly **0** because dividing by zero is undefined. In practice, a temperature very close to zero behaves almost like greedy sampling.

Example:

- Temperature = **0.2** → factual, deterministic answers.
- Temperature = **1.2** → more diverse and creative answers.

**Log Probability (LogProb):**

Since vocabulary is huge, probabilities become very small. Instead of working directly with probabilities, we often use **log probabilities (logprobs)** because they are numerically more stable and easier to combine across tokens.

---

### Top-K Sampling

1. Top-K reduces the amount of computation.
2. Instead of considering the entire vocabulary, we:
   - Select the **K tokens with the highest logits**.
   - Apply Softmax only over these K tokens.
   - Sample from this reduced set.
3. Typical values range from **50–500**.

Example:

If **K = 50**, only the 50 most likely tokens are considered.

---

### Top-P (Nucleus Sampling)

1. Top-K is sometimes too rigid. Some prompts may only need a few candidate tokens, while others may require many.
2. **Top-P** dynamically chooses the candidate set based on cumulative probability, making it more context-aware.

How it works:

- Sort tokens by probability (highest to lowest).
- Compute the cumulative probability.
- Keep only the smallest set of tokens whose cumulative probability is at least **P**.

Example:

If **Top-P = 0.9**, keep adding tokens until their cumulative probability reaches **90%**, and discard the remaining low-probability tokens.

---

### Test-Time Compute

Instead of generating only one response, we can generate multiple responses and select the best one.

1. Generate multiple candidate responses.
2. Rather than generating them independently, **Beam Search** keeps a fixed number of the most promising partial sequences at every decoding step.
3. To improve effectiveness, we want the generated candidates to be diverse.
4. The best response can be selected by:
   - Asking the user.
   - Using a **Reward Model**.
   - Choosing the response with the highest probability.
     - Compute the product of token probabilities.
     - More commonly, sum the **log probabilities**.
     - Often use the **average log probability** to avoid bias toward shorter responses.
5. To reduce latency, multiple responses can be generated **in parallel**, and the first completed ones can be shown.

---

### Structured Outputs

Goal: Generate outputs in a predefined structure (e.g., JSON, XML, SQL, etc.).

Frameworks supporting structured outputs include:

- Guidance
- Outlines
- Instructor
- llama.cpp

Approaches:

#### 1. Prompting

Use prompt engineering to explicitly instruct the model to follow a required format.

#### 2. Post-processing

If the model repeatedly makes similar formatting mistakes, manually write scripts to detect and fix those errors after generation.

#### 3. Constrained Sampling

During generation, the model is allowed to sample **only from tokens that satisfy predefined constraints or grammar rules**.

**Drawback:** Rarely used in practice because defining complete grammars for real-world tasks is difficult.

#### 4. Fine-tuning

Similar to transfer learning.

Instead of relying only on prompting, we fine-tune the model (or add a task-specific neural network head) for a particular task.

Example:

- Add a **classification head** after the output embeddings.
- The model is then forced to output one of the predefined classes rather than arbitrary text.

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


---
## Topic 2: Positional Encoding :

We know that position of words (order in which they appear) in a sentence is really important, especially if we are trying to build a model for conversation or even generation of the next token.

Suppose a sentence:

> "dog bites man"

and another:

> "man bites dog"

The meaning is completely different, yet the embeddings of these words are the same, and the attention mechanism does not inherently know the position/order of the words.

The attention mechanism only calculates relationships between words based on their Query and Key vectors. It knows **which words are related**, but without additional information, it does not know:

- which word came first,
- which word came after another word,
- the distance between two words.

Now RNNs have built-in capability to deal with the order of words because they process words sequentially. The hidden state carries information from previous words.

However, Transformers do not use RNNs. They use self-attention and process all words in parallel. Therefore, we need some additional mechanism to provide information about the position of each word.

---

### The Solution: Positional Encoding

What is positional encoding?

It is a vector that we add to the embedding vector, so obviously it has the same dimension as that of the embedding vector.

Suppose:

- Sentence length = `n`
- Embedding dimension = `d`

Then:

Embedding matrix:

$$
X \in \mathbb{R}^{n \times d}
$$

Positional Encoding matrix:

$$
PE \in \mathbb{R}^{n \times d}
$$

The final input to the Transformer is:

$$
X + PE
$$

where each word embedding gets combined with its corresponding positional information.

---

### How does adding numbers tell the Transformer about position?

Now what happens essentially is some number (let's say for now) is added to each token element, but how does Transformer know that it actually represents position?

The answer is:

**It does not know initially. It learns this during training.**

The positional values are just additional signals added to the word embeddings.

Because the Transformer is trained on thousands and thousands of examples, it learns patterns between:

- word meaning information from embeddings
- positional information from positional encoding

For example:

The word "cat" will have the same embedding value wherever it appears.

The only difference will be the positional vector added to it.

Example:
Sentence 1:
The cat sleeps

cat embedding + position 2 vector

Sentence 2:
cat is cute

cat embedding + position 0 vector



The word embedding remains the same, but the final input representation changes because of position.

Gradually, through backpropagation, the Transformer learns the association that this difference corresponds to the position of the word.

Every time a sentence appears with "cat" at a certain position, the model learns which patterns help it predict the next token correctly.

When the prediction is wrong, the weights are adjusted, and over millions of examples, the model learns to use positional information.

---

# Mathematical Representation

The original Transformer paper uses **sinusoidal positional encoding**.

The formula is:

$$
PE(pos,2i)=sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos,2i+1)=cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

---

## Understanding the Variables

### `pos`

`pos` represents the position of the word.

It is also the row number of the positional encoding matrix.

Example:
Sentence:

I love AI

Position:

- I → pos = 0
- love → pos = 1
- AI → pos = 2


Each row of the positional encoding matrix corresponds to one position in the sentence.

---

### `i`

`i` represents the column index used to generate positional encoding.

It helps decide which dimension of the positional encoding matrix we are calculating.

The formula uses:

- `2i` → even columns
- `2i + 1` → odd columns

because:

- sine values are used for even dimensions
- cosine values are used for odd dimensions

---

### `d`

`d` represents the dimension of the embedding.

The positional encoding matrix has the same dimension as the embedding matrix because we need to add them together.

For example:

If embedding dimension:

d = 512
then positional encoding dimension: 512


---

### `10000`

The value `10000` controls how quickly the sine and cosine functions change.

The reason we divide by:

$$
10000^{2i/d}
$$

is because we want different dimensions to have different frequencies.

Some dimensions should change quickly with position, while other dimensions should change slowly.

This creates a unique positional pattern for every position.

Think of each dimension as a different clock:

- some clocks tick quickly,
- some clocks tick slowly.

Together, they create a unique fingerprint for every position.

---

# Sine and Cosine Usage

Both sine and cosine are used.

- Sine fills even columns.
- Cosine fills odd columns.

Example:

If embedding dimension = 6:

| Column | Function |
|---|---|
|0|sin|
|1|cos|
|2|sin|
|3|cos|
|4|sin|
|5|cos|

The choice of alternating sine and cosine gives the model two different periodic signals to represent position.

---

# Final Step

We add the positional encoding matrix to the embedding matrix:

$$
Input = Embedding + Positional\ Encoding
$$

After this addition, the Transformer receives embeddings that contain:

- information about the word itself
- information about where the word appears in the sentence

After this step, the remaining attention mechanism follows:

1. Calculate Q, K, V
2. Calculate attention scores
3. Apply scaling
4. Apply softmax
5. Multiply attention weights with V
6. Generate contextualized word representations

---

## Topic 3: Scaling Laws – Building Compute-Optimal Models

Scaling laws study the relationship between:

- Model size (number of parameters)
- Training dataset size (number of tokens)
- Compute budget
- Model performance

The goal is to answer questions such as:

- How big should my model be?
- How much training data do I need?
- Am I wasting compute by using too many parameters or too little data?

---

### Chinchilla Scaling Law

The Chinchilla paper showed that many earlier language models were **under-trained**. They had a huge number of parameters but were not trained on enough data.

For **compute-optimal** training:

- The number of **training tokens** should be approximately **20× the number of model parameters**.

Example:

- 1B parameter model → ~20B training tokens
- 10B parameter model → ~200B training tokens

Another important observation:

- If the model size doubles, the amount of training data should also roughly double to maintain compute-optimal training.

The idea is to balance:

- model capacity
- amount of training data
- available compute

instead of making only the model larger.

> Bigger models are not always better. They also need proportionally more data to fully utilize their capacity.

---

### Scaling Extrapolation (Hyperparameter Transfer)

Hyperparameters play a huge role in determining the performance and efficiency of an LLM.

Examples include:

- Learning rate
- Batch size
- Weight decay
- Optimizer settings
- Warm-up steps
- Learning rate schedule

Finding the best combination is extremely expensive because training large LLMs requires enormous computational resources.

Instead, researchers train **small models** using many different hyperparameter combinations.

The best-performing hyperparameters are then transferred to much larger models.

This idea is called:

- **Scaling Extrapolation**
- **Hyperparameter Transfer**

The intuition is:

> If a hyperparameter configuration performs well on a small model, it will likely perform well on a larger model too.

This saves a tremendous amount of computation.

---

## Topic 4: Post Training

After deciding:

- the Transformer architecture
- model size
- training data

the next step is **post-training**.

During pretraining, the model mainly learns:

> "How to predict the next token."

It becomes very good at completing text.

However, that does **not** automatically make it:

- conversational
- helpful
- safe
- aligned with human preferences

Therefore, post-training is performed to better align the model with human expectations.

There are two major stages:

- Supervised Fine-Tuning (SFT)
- Preference Fine-Tuning

---

### Supervised Fine-Tuning (SFT)

The model is fine-tuned on **high-quality instruction-following datasets**.

These datasets contain examples like:

```
Prompt:
Explain Newton's First Law.

Assistant:
Newton's First Law states...
```

The model simply learns:

> "Given this prompt, generate this response."

This training data is often called:

- Demonstration Data
- Instruction Data

The objective is to make the model:

- follow instructions
- answer questions
- have natural conversations

instead of merely completing text.

---

### Preference Fine-Tuning

After teaching the model **how to communicate**, we now teach it **what humans prefer**.

Different algorithms are used for this purpose, including:

- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Reinforcement Learning from AI Feedback (RLAIF)

The objective is to make the model produce responses that humans find:

- more helpful
- more accurate
- safer
- more natural

---

### Reinforcement Learning from Human Feedback (RLHF)

RLHF mainly consists of two stages:

1. Train a Reward Model
2. Optimize the language model using the Reward Model

---

### 1. Reward Model

The Reward Model is another neural network.

Its job is to assign a score to the responses generated by the language model.

Example:

```
Prompt:
Explain gravity.

Response A

↓

Reward = 3

Response B

↓

Reward = 9
```

Instead of asking humans every time, we train a model to imitate human preferences.

---

### Building the Dataset

Humans compare two responses:

```
Prompt

↓

Response A

Response B

↓

Human chooses better response
```

This produces **comparison data** of the form:

```
(prompt,
better response,
worse response)
```

Instead of absolute scores, humans simply choose which response is better.

This makes collecting data much easier.

---

### Training the Reward Model

The Reward Model is trained so that:

```
Reward(Better Response)

>

Reward(Worse Response)
```

To achieve this, objective functions (typically based on logistic loss) are used.

The objective is to maximize the difference between the reward assigned to the preferred response and the rejected response.

Eventually, the Reward Model learns to approximate human preferences.

---

### 2. Fine-Tuning using the Reward Model

Now we have:

- a Supervised Fine-Tuned model
- a trained Reward Model

The next goal is:

> Fine-tune the language model so that it generates responses with higher Reward Model scores.

This is where reinforcement learning comes in.

---

### Proximal Policy Optimization (PPO)

PPO is a **reinforcement learning algorithm** used during RLHF.

---

### What is a Policy?

In Reinforcement Learning,

**Policy** simply means:

> Given the current state, what action should the model take?

For LLMs,

the "state" is the prompt (and previously generated tokens),

and the "action" is selecting the next token.

So the policy is essentially:

> Given this prompt, what probability should be assigned to every possible next token?

---

### Why not simply maximize the Reward?

Suppose the Reward Model gives a very high score to one particular response.

If we aggressively update the model, the probabilities of certain tokens might change drastically.

This can cause:

- unstable training
- forgetting previously learned knowledge
- reduced language quality

---

### Why "Proximal"?

The word **Proximal** means:

> Stay close.

PPO updates the policy **gradually** instead of allowing huge jumps.

Conceptually:

```
Old Policy

↓

Small Improvement

↓

New Policy
```

instead of

```
Old Policy

↓

Huge Change

↓

Unstable Model
```

The PPO algorithm compares the **new policy** with the **old policy** and limits how much the policy is allowed to change in a single update.

This makes reinforcement learning much more stable.

---

### Where is Gradient Descent used?

Gradient descent is **still the optimization algorithm** used to update the model parameters.

The difference is:

During Supervised Fine-Tuning:

```
Cross Entropy Loss

↓

Backpropagation

↓

Gradient Descent
```

During RLHF:

```
Reward Model

↓

PPO Objective (Loss)

↓

Backpropagation

↓

Gradient Descent
```

So PPO does **not replace gradient descent**.

Instead:

- PPO defines **what objective should be optimized**
- Gradient descent performs the actual weight updates

---

### RLHF Pipeline

```
Pretrained LLM
        │
        ▼
Supervised Fine-Tuning (SFT)
        │
        ▼
Generate Responses
        │
        ▼
Humans Compare Responses
        │
        ▼
Comparison Dataset
        │
        ▼
Train Reward Model
        │
        ▼
Reward Model predicts scores
        │
        ▼
PPO optimizes the language model
        │
        ▼
Better aligned LLM
```

---

