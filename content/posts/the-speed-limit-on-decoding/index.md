---
title: The speed limit on generating a token
summary: Before writing any CUDA for my inference engine, I worked out how fast a 1B model can possibly decode and where its bytes actually go.
tags: Inference, Systems
order: 0
---

I'm building an inference engine for Llama 3.x from scratch in C++ and CUDA. No PyTorch, no cuBLAS, every kernel written by hand, with a plain CPU implementation kept around as the reference every kernel gets checked against. It's early. Right now it can open a checkpoint and tell you what's inside. But the most useful thing I've done so far is arithmetic, and it decides what the rest of the project has to optimize.

## Decoding is a memory problem

When a model generates text one token at a time, each new token needs a pass through every weight in the model. At batch size one that's a long series of matrix vector products, and a matrix vector product does about two floating point operations for every weight it reads. GPUs can do far more math than that per byte of memory they move. So the arithmetic units mostly wait, and the real limit is how fast you can stream the weights out of memory.

That gives a simple ceiling.

$$
\text{max tokens per second} \approx \frac{\text{memory bandwidth}}{\text{bytes of weights}}
$$

No kernel can beat it. The engineering question is just how close you can get.

## What's actually in a 1B model

Pointing the engine's checkpoint inspector at Llama 3.2 1B Instruct gives 146 tensors, 1.236 billion parameters, and 2.30 GiB of weights stored in BF16. The shapes pin down the architecture. It has 16 layers, a hidden size of 2048, 32 query heads sharing 8 key and value heads of size 64, an MLP width of 8192, and a vocabulary of 128,256 tokens.

Splitting the bytes by where they live is where it gets interesting.

| Part | Parameters | Bytes in BF16 | Share |
|---|---|---|---|
| MLP blocks | 805M | 1.61 GB | 65% |
| Embedding and output layer | 263M | 0.53 GB | 21% |
| Attention projections | 168M | 0.34 GB | 14% |

The embedding matrix is a fifth of the model. That's because the vocabulary is huge relative to such a small hidden size, and this checkpoint ties the input embedding to the output layer, so there's no separate output matrix in the file at all. Looking up a token's embedding only touches one row, but producing the next token's logits reads the whole matrix every step. So a fifth of each token's memory traffic goes into scoring all 128,256 possible next tokens. In Llama 3.1 70B the output layer is about 1.5% of the weights.

On an RTX 4090, which has about 1008 GB/s of memory bandwidth, reading 2.47 GB per token caps decoding at roughly 408 tokens per second in BF16. Whatever I build gets measured against that number.

## The KV cache catches up

Weights aren't the only thing read per token. Attention also reads the cached keys and values for every earlier token. Thanks to grouped query attention, with 8 key and value heads instead of 32, each token of context costs 32 KiB across all 16 layers.

That's small until the context gets long. At 8,192 tokens the cache is 256 MiB, about 11% on top of the weights. At 32,768 tokens it's 1 GiB, or 43%. At the full 128K context it's 4 GiB, which is more than the model itself. So for long chats, the attention kernel becomes as much a memory streaming problem as the matrix multiplies.

## Small things that would have bitten me

A few details from loading the checkpoint that I'm glad I caught before writing GPU code.

The output layer is tied to the embedding, as above. If you go looking for an `lm_head` tensor you won't find one.

The model uses Llama 3's own RoPE scaling, which rescales rotary frequencies differently in different bands. Plain RoPE would still run, but its outputs wouldn't match the reference, and the mismatch grows with context length.

The safetensors format doesn't promise any particular alignment for tensor data. Casting a pointer into the mapped file straight to `float*` is a bug waiting to happen. The loader copies or converts into its own buffers instead.

## What's next

The next pieces are a byte level BPE tokenizer that matches Hugging Face's token IDs exactly, then a full CPU forward pass checked layer by layer against reference activations. Only after that does CUDA start. The goal for the 8B model is around 80% of the bandwidth ceiling with 4 bit weights, and I'll write up how close I get, including what doesn't work.
