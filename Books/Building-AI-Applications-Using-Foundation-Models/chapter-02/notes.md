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
$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
**How does attention function works?**
