---
title: LLM Serving Pipeline & Interview Checklist
---

[← ML Systems index](index.md)

# LLM Serving Pipeline & Interview Checklist

> **Core idea.** A request passes through a chain of stages, and the GPU engine in the middle is where the two phases (compute-bound prefill, memory-bound decode) and all the memory-management tricks live. This page is the end-to-end map. The deep dives are in [LLM Serving](llm-serving.md) and [Metrics](metrics.md).

---

## 1. End-to-end pipeline

```
[ Client request ]
       │
       ▼
[ API gateway / load balancer ]
       │
       ▼
[ Tokenizer + request queue ]
       │
       ▼
[ Serving engine: continuous batching + PagedAttention ]
       ├── Prefill phase  (compute-bound)
       └── Decode phase   (memory-bandwidth-bound)
       │        ▲
       │        └── [ KV cache manager ]
       ▼
[ Sampler: temperature, top-k / top-p, penalties ]
       │
       ▼
[ Detokenizer + streamer ] ──► client
```

Engines that implement this: vLLM, TensorRT-LLM, SGLang.

| Stage | Job | Things to know |
|---|---|---|
| **API gateway / load balancer** | Accepts HTTP/gRPC, authenticates, rate-limits, and routes across serving nodes | Route by load and by prefix-cache locality, not round-robin. Apply admission control and backpressure. |
| **Tokenizer + queue** | Converts text to token IDs. Requests arrive asynchronously with different prompt and output lengths, so they wait in a central queue. | Queue wait counts toward TTFT. Queue depth is a good autoscaling signal. |
| **Serving engine** | Schedules the GPU, allocates memory, and runs kernels. Splits work into the prefill and decode profiles. | See sections 2–4 below. |
| **KV cache manager** | Allocates, shares, evicts, and (optionally) swaps KV blocks | PagedAttention block tables, prefix caching, preemption. |
| **Sampler** | Turns the final-layer logits into the next token ID | Order: penalties → temperature scaling → top-k / top-p filtering → sample. Greedy is the temperature → 0 limit. |
| **Detokenizer + streamer** | Converts token IDs back to text and streams chunks to the client over SSE or gRPC streams | Incremental detokenization has to handle multi-token characters (partial UTF-8). |

### The two phases

- **Prefill** processes the whole prompt in parallel. It is **compute-bound**: large matmuls with high arithmetic intensity. It sets TTFT.
- **Decode** generates one token per step, autoregressively. It is **memory-bandwidth-bound**: every step loads all the weights and the KV cache from HBM into on-chip SRAM for a tiny amount of math. It sets TPOT.

---

## 2. KV cache and its memory footprint

- **What it is.** The keys (K) and values (V) of all past tokens, in every layer, are cached so attention over previous tokens is not recomputed.
- **The problem.** It grows linearly with batch size × sequence length × layers × KV dimensions. With long contexts (32K–128K) or large batches it can use up the GPU memory, which forces lower concurrency or an out-of-memory crash.
- **Interview maths.** Per token:
  `2 (K and V) × layers × kv_heads × head_dim × bytes_per_value`.
  Equivalently, per token per layer it is `2 × kv_dim × bytes`, and GQA/MQA shrink `kv_heads` below the number of query heads. A worked table is in [LLM Serving, section 2](llm-serving.md).

---

## 3. Scheduling and memory management

### Continuous (iteration-level) batching

| | Static batching | Continuous batching |
|---|---|---|
| When a sequence finishes | Its slot sits idle until the whole batch finishes | The slot is freed at once and a queued request joins on the **next forward pass** |
| Effect | Short requests wait on the longest one | High GPU utilisation and much higher throughput |

Example: request A finishes quickly while B has 500 tokens left. In static batching A's compute slot idles for those 500 steps.

### PagedAttention (vLLM)

- **Problem.** Frameworks that pre-allocate a contiguous KV buffer per request at max sequence length waste a lot of memory (often 60–80% of what was allocated) through internal and external fragmentation.
- **Fix.** Split the KV cache into fixed-size blocks that can sit anywhere in physical memory, and keep a **block table** that maps each sequence's logical blocks to physical ones, as an OS page table does. Waste falls to under ~4%, so far more sequences fit in a batch.
- **RadixAttention (SGLang).** Keeps a radix tree over token prefixes so KV blocks are shared automatically across requests with a common prefix: system prompts, multi-turn chat history, RAG context.

---

## 4. Distributed inference

When a model such as Llama-3-70B doesn't fit on one GPU:

| Strategy | What is split | Communication | Where to use it |
|---|---|---|---|
| **Tensor parallelism (TP)**, Megatron style | Each weight matrix: **column-parallel** layers split by columns, **row-parallel** layers split by rows | All-reduce in **every layer**, on the critical path | Inside a node over **NVLink** |
| **Pipeline parallelism (PP)** | Consecutive layers (for example layers 1–20 on GPU 0 and 21–40 on GPU 1) | Point-to-point activation transfers | Slower interconnects or across nodes. Cost: pipeline bubbles and idle waiting. |

More on the column/row split and the bubble formula is in [Distributed Training](distributed-training.md).

---

## 5. FlashAttention: why standard attention is slow and how it is fixed

### The problem: standard attention round-trips through HBM

1. Compute `S = Q·Kᵀ`, the raw score matrix of size N × N.
2. **Write S to HBM.**
3. **Read S back** to compute the row-wise softmax `exp(S) / Σ exp(S)`.
4. **Write the probabilities P to HBM.**
5. **Read P back** and multiply by V to get the output.

Why was it done this way? Softmax looks globally dependent: to normalise one element you need the sum over its entire row. With N in the thousands, it seemed necessary to materialise the whole N × N matrix in global memory. That means O(N²) memory and O(N²) HBM traffic, and on modern GPUs the HBM traffic, not the math, is what takes the time.

### The fix: tiling + online softmax

Load small blocks of Q, K, and V into on-chip **SRAM** and process the row incrementally, keeping two running statistics per row:

- `m`: the running maximum of the scores seen so far
- `ℓ`: the running sum of `exp(score − m)`

When a new block of scores arrives, with block max `m̃` and block sum `ℓ̃`:

```
m_new = max(m, m̃)
ℓ_new = ℓ · exp(m − m_new) + ℓ̃ · exp(m̃ − m_new)
O_new = ( O · ℓ · exp(m − m_new) + (exp(S_block − m_new) · V_block) ) / ℓ_new
```

The earlier partial output is **rescaled** whenever the max changes, so the result at the end is exactly the true softmax-weighted sum.

What this buys you:

- The N × N matrix **never exists in full** and is never written to HBM. Only the final output (plus a small per-row statistic for the backward pass) is written.
- One **fused kernel** does it all, so there are no intermediate round-trips.
- It is **exact**, not an approximation.
- Memory drops from O(N²) to O(N), and runtime falls because the bottleneck was HBM traffic.

| Version | Main improvement |
|---|---|
| **FlashAttention 1** | Tiling + online softmax + recomputation in backward |
| **FlashAttention 2** | Better work partitioning and parallelism across the sequence, fewer non-matmul FLOPs |
| **FlashAttention 3** | Hopper-specific: asynchronous data movement (TMA), warp specialisation, FP8 |

---

## 6. Trade-offs and metrics

- **Throughput vs latency.** Large batches keep the compute units busy and raise throughput. They also lengthen queueing and per-step time, which hurts TTFT and TPOT. Real-time chat favours low latency. Offline analytics and batch jobs favour throughput.
- **TTFT** (time to first token): dominated by queueing + prefill. Affected by prompt length and compute availability.
- **TPOT** (time per output token): dominated by decode. Affected by memory bandwidth and batch size.
- **Goodput:** requests per second that meet the SLOs (for example TPOT < 50 ms). The honest objective, since raw throughput can be bought by sacrificing latency.

Definitions and formulas for all of these are in [Metrics](metrics.md).

---

## 7. Rapid-fire interview checklist

**"P99 TTFT spiked on our endpoint. How do you diagnose it?"**
Go through the stages in order:

1. **Input.** Did prompt lengths jump? Prefill cost scales with prompt tokens.
2. **Queue.** Is queue depth or wait time rising? Check for arrival-rate spikes and whether the fleet is under-scaled.
3. **Interference.** Are long prefills stalling decode batches (chunked prefill off)? Is there a noisy tenant?
4. **Memory.** Is KV cache utilisation near 100%? Check the preemption, swap, and recompute rates, and a drop in the prefix-cache hit rate.
5. **Hardware / infra.** A bad node, throttled GPU, or a load balancer routing unevenly, away from prefix-cache locality.

**"How would you serve a 70B model on limited hardware?"**

- **Quantize**: INT8/INT4 weights (AWQ, GPTQ) or FP8. 140 GB in bf16 becomes ~35 GB in int4.
- **Tensor parallelism** across the GPUs you have, over NVLink.
- **KV cache**: paging, KV quantization (FP8/int8), GQA/MLA-style models, and a cap on context length or concurrency.
- Accept lower throughput; use continuous batching to recover what you can.

**"Speculative decoding vs standard autoregressive decoding?"**
Standard decoding runs one large-model pass per token. Speculative decoding has a small, fast **draft** model propose several tokens, then the large **target** model verifies them all in a **single parallel forward pass**. Accepted tokens are free, and rejection sampling keeps the output distribution identical to the target's. It helps most at low batch sizes, where decode is memory-bound. See [LLM Serving, section 6](llm-serving.md).

**Others to rehearse:**

- Why is decode memory-bound and prefill compute-bound? (arithmetic intensity, see [Metrics](metrics.md))
- Static vs continuous batching, and what PagedAttention adds on top.
- Why is TP kept inside a node?
- What does FlashAttention avoid writing to HBM, and why is it still exact?
