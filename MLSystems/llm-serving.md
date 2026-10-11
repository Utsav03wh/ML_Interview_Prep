---
title: LLM Serving
---

[← ML Systems index](index.md)

# LLM Serving

> **Core idea.** Generation has two phases. **Prefill** processes the whole prompt in one forward pass and is **compute-bound**. **Decode** produces one token at a time and is **memory-bandwidth-bound**, because every step reads all the weights and the whole KV cache to do very little math. Almost every serving technique targets one of these phases.

---

## 1. Why decode is memory-bound (arithmetic intensity)

For a dense model with `P` parameters in bf16 (2 bytes each), processing a batch of `B` tokens takes:

- FLOPs ≈ `2 · P · B`
- Bytes read for weights ≈ `2 · P`
- **Arithmetic intensity ≈ B FLOPs/byte**

The H100 does ~990 TFLOP/s on ~3.35 TB/s, so it needs about **300 FLOPs per byte** to be compute-bound.

- **Prefill** of a 2K-token prompt has B = 2048, so it is compute-bound.
- **Decode** with batch size 8 has B = 8, so it is heavily memory-bound. The GPU spends its time streaming weights.

**Lower bound on decode latency per step** = bytes read / bandwidth:

- 8B model, bf16, 1×H100: 16 GB / 3.35 TB/s ≈ **4.8 ms/token**, about 200 tok/s for a single stream.
- 70B model, bf16, 8×H100 with TP=8: 140 GB / (8 × 3.35 TB/s) ≈ **5.2 ms/token**, before communication overhead.

The consequences:

- Batching more sequences during decode is almost free until the KV cache reads dominate or you reach the ridge point. **Throughput comes from batch size, and batch size is limited by KV cache memory.**
- Shrinking bytes (quantization, GQA/MQA, KV compression) speeds up decode directly.
- Making more tokens per weight read (speculative decoding) also speeds up decode.

---

## 2. KV cache

Attention at step `t` needs the keys and values of all previous tokens. The KV cache stores them so they are not recomputed.

**Memory per token:**

```
bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_value
```

| Model | Layers | KV heads | head_dim | Bytes/token (bf16) | 8K-context sequence |
|---|---|---|---|---|---|
| Llama-2-7B (MHA) | 32 | 32 | 128 | 512 KiB | 4 GiB |
| Llama-3-8B (GQA) | 32 | 8 | 128 | 128 KiB | 1 GiB |
| Llama-3-70B (GQA) | 80 | 8 | 128 | 320 KiB | 2.5 GiB |

**Worked example.** Llama-3-8B on one 80 GB H100 has 16 GB of weights, which leaves roughly 60 GB for the KV cache. At 128 KiB/token that is about 490K tokens: roughly 60 concurrent 8K sequences, or about 240 at 2K.

Ways to shrink the KV cache:

- **GQA / MQA**: several query heads share one KV head. Llama-3 uses 8 KV heads instead of 32 or 64.
- **MLA** (DeepSeek-V2/V3): caches a low-rank latent per token instead of full K/V.
- **KV quantization** to FP8 or int8 halves the cache.
- **Sliding-window or local attention layers** cap the cache length for those layers.

---

## 3. Batching

**Static batching.** The server collects B requests, runs them together, and returns when the *longest* one finishes. Finished sequences leave their slots idle (they are padding), and new requests wait for the whole batch to drain. Utilisation is poor when output lengths vary, which they always do.

**Continuous (iteration-level) batching**, introduced by Orca. The scheduler runs **every decode step**:

- Finished sequences are evicted immediately.
- Waiting requests are admitted as soon as KV memory allows, and their prefill is merged into the running batch.

Anyscale measured up to ~**23×** throughput over naive static batching. This is the default in vLLM, SGLang, TensorRT-LLM, and TGI.

**Chunked prefill** (Sarathi-Serve). A long prompt's prefill can stall every in-flight decode for hundreds of milliseconds. Chunked prefill splits prompts into chunks (for example 512 tokens) and piggybacks them onto decode steps. That bounds the per-step latency and evens out utilisation. It is the main *colocated* alternative to disaggregation.

---

## 4. PagedAttention (vLLM)

**Problem.** Earlier systems pre-allocated one contiguous KV buffer per request at max sequence length. That wastes memory three ways:

- **Reserved** space that a short output never uses.
- **Internal fragmentation** inside each buffer.
- **External fragmentation** between buffers.

The vLLM paper found that only 20–40% of KV memory held actual tokens.

**Idea.** Borrow virtual memory from operating systems:

- The KV cache is split into fixed-size **blocks** (pages), for example 16 tokens each.
- Each sequence has a **block table** that maps logical blocks to physical blocks, which can sit anywhere in GPU memory.
- Blocks are allocated on demand as tokens are generated, so waste is under one block per sequence (<4% overall).
- The attention kernel is rewritten to gather K/V through the block table.

**Sharing.** Physical blocks are reference-counted, with **copy-on-write**:

- Parallel sampling (n > 1) and beam search share the prompt's blocks.
- **Prefix caching** lets requests with a common system prompt or few-shot prefix reuse the cached blocks across requests. SGLang's **RadixAttention** generalises this with a radix tree over token prefixes.

**Preemption.** When memory runs out, the scheduler evicts a sequence and either swaps its blocks to CPU or drops them and recomputes later.

**Result.** 2–4× throughput over FasterTransformer and Orca at the same latency, mostly because many more sequences fit in a batch.

---

## 5. Prefill–decode disaggregation

**Problem with colocation.** Prefill and decode share GPUs and batches, so they interfere:

- A big prefill arriving inflates the **TPOT** of every running decode.
- Prioritising decodes inflates **TTFT** for new requests.
- One parallelism config has to serve two phases with opposite needs. Prefill wants compute and TP to cut latency. Decode wants memory capacity and large batches.

**Idea (DistServe, Splitwise, Mooncake).** Run prefill and decode on **separate GPU pools**:

1. A request goes to a **prefill instance**, which computes the prompt's KV cache and the first token. That gives TTFT.
2. The KV cache is **transferred** to a **decode instance** over NVLink or RDMA (InfiniBand/RoCE).
3. The decode instance adds the sequence to its continuous batch.

**Benefits:**

- No interference, so TTFT and TPOT can be tuned **independently**.
- Each pool gets its own parallelism (TP/PP), batch size, and even hardware. For example, prefill on compute-heavy GPUs and decode on GPUs with more HBM capacity.
- The prefill:decode ratio of instances is set from the workload (prompt and output length distributions) to **maximise goodput** per GPU. DistServe reported large gains in requests served within SLO.

**Costs and design issues:**

- **KV transfer size.** For 70B with a 4K prompt: 4096 × 320 KiB ≈ 1.25 GiB. Over one 50 GB/s IB NIC that takes ≈ 25 ms. Over NVLink it is a few ms. Mitigations:
  - Stream it **layer by layer** while prefill is still running, to overlap compute and transfer.
  - Place prefill/decode pairs within a node when possible.
  - Use multiple NICs.
- **Scheduling and routing.** Choose a decode instance by KV memory headroom and load. Prefix-cache-aware routing on the prefill side.
- **Memory pool.** Mooncake keeps a disaggregated KV cache pool (CPU DRAM/SSD) so prefixes can be reused across instances.
- **When it is not worth it.** With short prompts, low load, or no strict SLOs, chunked prefill on colocated instances is simpler and similar in throughput.

**Talking points (be ready to go deep):**

- The goodput definition, and how you pick the P:D ratio.
- KV transfer bandwidth maths and how you overlap the transfer.
- What happens on decode-side memory pressure: backpressure on prefill, or preemption.
- Fault handling when a decode instance dies mid-generation: re-prefill elsewhere.
- How prefix caching interacts with disaggregation.

---

## 6. Speculative decoding

Decode is memory-bound, so verifying **k + 1 tokens costs about the same as generating 1**: the weights are read once either way.

**Algorithm:**

1. A cheap **draft** model q proposes k tokens autoregressively: x₁ … x_k.
2. The **target** model p scores all k + 1 positions in **one forward pass**.
3. For each draft token in order, accept it with probability `min(1, p(x)/q(x))`.
4. At the first rejection, resample from the residual `norm(max(0, p − q))` and stop.
5. If all k tokens are accepted, sample one bonus token from p.

This rejection scheme makes the output distribution **exactly** the target's. It is lossless, not an approximation.

**Expected tokens per target pass** with per-token acceptance rate α: `(1 − α^(k+1)) / (1 − α)`. For α = 0.8 and k = 4 that is ≈ 3.4 tokens per pass.

**Variants:**

- **Separate small draft model** from the same family, so it shares the tokenizer.
- **Self-drafting heads**: Medusa (multiple extra heads), EAGLE (a feature-level draft head; EAGLE-2/3 use dynamic draft trees).
- **Multi-token prediction (MTP) heads** trained with the model, as in DeepSeek-V3.
- **N-gram / prompt-lookup decoding**: copy candidate spans from the prompt. This works well for code editing and RAG.
- **Tree verification**: verify a tree of candidates in one pass using a tree attention mask.

**When it helps:** low batch sizes, where decode is most memory-bound and there is spare compute. At high batch sizes the GPU is closer to compute-bound and verifying rejected tokens wastes FLOPs, so gains shrink or go negative.

---

## 7. Quantization

| Scheme | What it does | Helps |
|---|---|---|
| **Weight-only (W4A16, W8A16)** | Store weights in int4/int8 and dequantize to bf16 inside the matmul kernel | **Decode**, which is memory-bound: fewer bytes to stream |
| **Weight + activation (W8A8 int8, FP8)** | Do the matmul itself in low precision | **Prefill** and large batches, which are compute-bound: 2× tensor-core throughput |
| **KV cache (FP8/int8)** | Store K/V in low precision | Larger batch and longer context; fewer bytes per decode step |

Key methods:

- **GPTQ.** Layer-by-layer post-training quantization. It quantizes the weight columns one at a time and uses approximate second-order (Hessian from calibration activations) information to update the remaining columns to compensate for the error.
- **AWQ.** Observes that ~1% of weight channels are salient, meaning they meet large activations. It scales those channels up before quantization (and folds the inverse scale into the preceding op) so they lose less precision. No backprop and no reordering.
- **SmoothQuant.** Migrates activation outliers into the weights with a per-channel scale, making W8A8 feasible.
- **Granularity.** Per-tensor, per-channel, or per-group scales (for example a group size of 128 for int4). Smaller groups are more accurate but carry more scale overhead.

---

## 8. FlashAttention

Standard attention materialises the `N × N` score matrix S and the probability matrix P in **HBM**: write S, read S, write P, read P. For long sequences that is O(N²) HBM traffic, and it dominates runtime.

**FlashAttention:**

- **Tiles** Q, K, and V into blocks that fit in on-chip **SRAM** (shared memory and registers).
- Computes softmax with the **online softmax** trick: keep a running max `m` and running sum `ℓ` per row, and rescale the partial output when the max changes.
- Never writes the N × N matrix to HBM. HBM traffic falls from Θ(Nd + N²) to Θ(N²d²/M), where M is the SRAM size.
- Is **exact**, not an approximation.
- In the backward pass, **recomputes** the attention blocks from Q, K, V and the saved softmax statistics instead of storing P. That is extra FLOPs in exchange for much less memory.

Versions:

- **FA2** improves parallelism across the sequence dimension and reduces non-matmul FLOPs.
- **FA3** uses Hopper features: asynchronous TMA, warp specialisation, and FP8.
- **FlashDecoding** splits the KV length across thread blocks during decode. A single query token otherwise can't fill the GPU.

---

## 9. Metrics

| Metric | Definition | Dominated by |
|---|---|---|
| **TTFT** (time to first token) | From request arrival to the first output token | Queueing + prefill |
| **TPOT / ITL** (time per output token / inter-token latency) | Average gap between output tokens | Decode step time, which grows with batch size |
| **E2E latency** | `TTFT + TPOT × (n_out − 1)` | Both |
| **Throughput** | Output tokens/s (or requests/s) per GPU or per cluster | Batch size |
| **Goodput** | Requests/s that **meet the SLOs** (for example P90 TTFT < 500 ms *and* TPOT < 50 ms) | The real objective |

Always quote tail percentiles (P50/P90/P99), not just means. There is a basic **throughput–latency trade-off**: a bigger batch gives more tokens/s and a slower TPOT. Goodput is how you pick the operating point.

---

## 10. Design checklist: "Design a serving system for model X at Y QPS"

1. **Memory.** Weights (after quantization) + KV cache per token × the expected tokens in flight. How many GPUs, and what TP degree?
2. **Parallelism.** TP within a node (NVLink) to fit the weights and cut latency. PP or extra replicas across nodes. Expert parallelism for MoE.
3. **Engine.** Continuous batching + paged KV + prefix caching. Chunked prefill or P/D disaggregation depending on SLO strictness and prompt lengths.
4. **Latency levers.** Speculative decoding at low load, FP8/int4 weights, FlashAttention/FlashDecoding kernels, CUDA graphs for decode.
5. **Fleet.** A router that knows about load and prefix-cache locality, autoscaling on queue depth and KV utilisation, admission control and backpressure, and separate pools for long-context traffic.
6. **Observability.** TTFT/TPOT percentiles, KV cache utilisation, batch size, preemption rate, prefix-cache hit rate, goodput.

---

## Reading list

- vLLM blog post, and Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP 2023)
- Anyscale blog: *How continuous batching enables 23x throughput in LLM inference*
- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (OSDI 2024)
- Google DeepMind, *How to Scale Your Model*, the inference chapter
- Further reading: Orca (OSDI 2022), Sarathi-Serve, Splitwise, Mooncake, Leviathan et al. and Chen et al. on speculative decoding, FlashAttention 1/2/3, GPTQ, AWQ, SmoothQuant

## Likely interview questions

- Estimate the KV cache for a 70B GQA model at 32 concurrent 8K-token requests. Does it fit on 8×H100 with bf16 weights?
- Why does increasing batch size improve throughput but hurt TPOT? Where does it stop helping?
- Explain how PagedAttention enables prefix sharing. What is copy-on-write used for?
- When would you choose chunked prefill over disaggregation, and the other way round?
- Prove that speculative decoding preserves the target distribution.
- Would you use W4A16 or W8A8 for a latency-sensitive chatbot at batch size 4? And for an offline batch job?
