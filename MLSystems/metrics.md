---
title: Metrics & Performance Concepts
---

[← ML Systems index](index.md)

# Metrics & Performance Concepts for LLM Infra

> **Core idea.** Every workload uses three resources: **compute** (FLOP/s), **memory bandwidth** (bytes/s), and **communication** (bytes/s between GPUs). Whichever one runs out first sets your speed. Most of the metrics below either measure which resource is the bottleneck or measure how well you are using it.

Reference hardware used below is the **H100 SXM**: ~990 TFLOP/s bf16 dense, ~3.35 TB/s HBM bandwidth, and 80 GB of memory.

---

## Part 1: Hardware and kernel concepts

### FLOPs vs FLOP/s

- **FLOPs** (floating-point operations) is an amount of *work*. One multiply-add counts as **2 FLOPs**.
- **FLOP/s** (FLOPS) is a *rate*, meaning work per second. Hardware specs quote peak FLOP/s.
- **Matmul cost:** `(M×K) · (K×N)` takes `2·M·K·N` FLOPs.
- **Transformer rule of thumb**, for a model with P params processing T tokens:
  - forward ≈ `2·P·T` FLOPs
  - backward ≈ `4·P·T` (twice the forward: gradients with respect to both activations and weights)
  - **training ≈ 6·P·T**, inference ≈ 2·P·T
  - Attention adds about `12·L·d·s` FLOPs per token in training, where L = layers, d = model dim, and s = context length. That term is small for short context and significant at long context.
- **Spec-sheet traps:**
  - Vendors often quote **sparse** FLOP/s, which is 2× the dense number. Use dense.
  - Peak depends on dtype: FP8 is about 2× bf16, and FP32 without tensor cores is far lower.

### Memory bandwidth

This is how many bytes per second can move between **HBM** (the GPU's main memory) and the compute units. On-chip **SRAM** (shared memory and registers) is about 10× faster but tiny (~50 MB total on an H100). Kernel optimisation is mostly about keeping data in SRAM and reading HBM less often (see FlashAttention).

**Memory hierarchy, fastest to slowest:** registers → SRAM/L1 → L2 → HBM (3.35 TB/s) → NVLink to other GPUs (~450 GB/s each way) → PCIe/host DRAM (~64 GB/s) → InfiniBand to other nodes (~50 GB/s per NIC).

### Arithmetic intensity (AI)

```
Arithmetic intensity = FLOPs performed / bytes moved to and from HBM     (units: FLOPs per byte)
```

It measures how much math you get out of each byte you load. Examples in bf16 (2 bytes per value):

| Operation | FLOPs | Bytes | AI |
|---|---|---|---|
| Elementwise add `y = a + b` (n elements) | n | 6n (read 2, write 1) | **~0.17**, very low |
| LayerNorm / softmax / GeLU | ~5–10 per element | ~4 per element | **~1–3**, low |
| Square matmul, n×n by n×n | 2n³ | 6n² | **n/3**. For n = 4096 that is ~1365, high |
| Weight matmul with B tokens (LLM layer, B ≪ d) | 2·B·d² | ≈ 2·d² (weights dominate) | **≈ B** |
| Decode attention (one query per sequence, reads its own KV) | ~2·s·d per head group | ~2·s·d (KV) | **≈ 1 × (query heads per KV head)** |

The last two rows explain LLM serving. **The weight matmuls have intensity equal to the number of tokens in the batch**, so batching helps them. **Decode attention has intensity around 1 no matter the batch size**, because each sequence reads its own KV cache, so batching does not help it.

### Roofline model and the ridge point

```
Attainable FLOP/s = min( Peak FLOP/s ,  AI × Memory bandwidth )
```

```
 FLOP/s
   |                 ______________________  ← compute roof (990 TFLOP/s)
   |               /
   |             /   ← memory roof: slope = bandwidth
   |           /
   |         /
   |_______/_________________________________ AI (FLOPs/byte)
               ^
          ridge point = peak / bandwidth ≈ 990e12 / 3.35e12 ≈ 295
```

- **Ridge point** (also called machine balance or critical intensity) ≈ **~300 FLOPs/byte for bf16 on an H100**. With FP8 the peak doubles, so the ridge point roughly doubles to ~600.
- **AI below the ridge means memory-bound.** Runtime ≈ bytes / bandwidth, so adding FLOP/s won't help. You have to move fewer bytes.
- **AI above the ridge means compute-bound.** Runtime ≈ FLOPs / peak FLOP/s, so moving fewer bytes won't help. You need less math, a lower-precision dtype, or better kernels.

### Compute-bound, memory-bound, communication-bound, overhead-bound

| Regime | Time ≈ | Typical LLM examples | Fixes |
|---|---|---|---|
| **Compute-bound** | FLOPs / peak FLOP/s | Prefill, training with large batches | FP8, better kernels, fewer FLOPs (MoE, pruning) |
| **Memory-bandwidth-bound** | Bytes / HBM bandwidth | Decode, LayerNorm, softmax, optimizer step | Bigger batch, quantization, kernel fusion, GQA/MLA, speculative decoding |
| **Communication-bound** | Bytes / link bandwidth (+ latency) | TP across nodes, FSDP on a slow network, MoE all-to-all | Overlap comms with compute, better parallelism mapping, compression |
| **Overhead-bound** | Python, kernel-launch, and CPU scheduling time | Small-batch decode of small models, dynamic shapes | CUDA graphs, `torch.compile`, fusing ops, C++ schedulers |

**Example: decode with a 7B model in bf16.**

- Batch 1: AI ≈ 1, far below 295. It is memory-bound and reaches about 1/300 of peak FLOP/s.
- Batch 256: AI ≈ 256, close to the ridge. The weight matmuls are now nearly compute-bound, but attention still has to read every sequence's KV cache, so it stays memory-bound.

### Latency vs bandwidth (the α–β model)

Time to transfer n bytes:

```
T(n) = α + n / β          α = fixed latency (µs),  β = bandwidth (bytes/s)
```

- **Small messages** are dominated by latency (α). Batch or fuse them, and use tree algorithms.
- **Large messages** are dominated by bandwidth (β). Use ring algorithms and overlap them with compute.
- The same idea applies to kernels: a tiny kernel is dominated by launch overhead of ~5–10 µs.

---

## Part 2: Utilisation metrics

### MFU (Model FLOPs Utilisation)

```
MFU = (model FLOPs per token × tokens/s) / (num GPUs × peak FLOP/s)
    = (6 · P · tokens/s) / (N · peak)          for training
```

- Count only the FLOPs the model *needs*. Recomputation from activation checkpointing does **not** count.
- **Example.** A 7B model trains at 10,000 tokens/s per H100: 6 × 7e9 × 1e4 = 4.2e14 = 420 TFLOP/s, and 420 / 990 ≈ **42% MFU**.
- Typical good values: **35–55%** for large dense-model training. MoE models and long-context training are usually lower.

### HFU (Hardware FLOPs Utilisation)

This is like MFU but **counts recomputed FLOPs**, so it measures how busy the hardware is. HFU ≥ MFU. A large gap means you spend a lot of compute on recomputation.

### MBU (Model Bandwidth Utilisation)

This is the inference counterpart of MFU, for the memory-bound decode phase:

```
MBU = (weight bytes + KV cache bytes read per step) / TPOT  ÷  peak HBM bandwidth
```

- **Example.** An 8B bf16 model (16 GB) at batch 1 with a 6 ms TPOT: 16 GB / 0.006 s ≈ 2.67 TB/s, and 2.67 / 3.35 ≈ **80% MBU**. That is about as good as it gets.
- A low MBU at small batch points to overhead (kernel launches, Python) or poor kernels.

### "GPU utilisation" from nvidia-smi is misleading

`nvidia-smi` utilisation is the **fraction of time at least one kernel was running**. A tiny kernel on 1 of 132 SMs shows as 100%. Use these instead:

- **SM activity / SM occupancy** (DCGM, Nsight Systems)
- **Tensor-core activity**
- MFU or MBU, computed from your own throughput numbers

### Scaling efficiency

```
Scaling efficiency = throughput on N GPUs / (N × throughput on 1 GPU)      (or relative to a baseline of k GPUs)
```

- **Weak scaling** keeps the per-GPU batch fixed. **Strong scaling** keeps the global batch fixed.
- A drop in efficiency usually comes from communication that isn't overlapped, pipeline bubbles, or stragglers.

### Pipeline bubble fraction

`(p − 1) / (m + p − 1)` for p pipeline stages and m micro-batches. It is the fraction of time GPUs sit idle in pipeline parallelism. See [Distributed Training](distributed-training.md).

### Communication bandwidth: algbw vs busbw (NCCL)

- **algbw** = message size / time.
- **busbw** = algbw × a correction factor that makes it comparable to the link's peak speed. For all-reduce the factor is `2(N−1)/N`. For all-gather and reduce-scatter it is `(N−1)/N`.
- Compare **busbw** against the hardware link bandwidth when checking `nccl-tests` results.

### Communication overlap ratio

This is the fraction of communication time hidden behind compute. Look for exposed communication gaps in an Nsight or PyTorch Profiler trace.

---

## Part 3: Serving metrics

### Latency metrics

| Metric | Definition | Driven by | Typical SLO |
|---|---|---|---|
| **TTFT** (time to first token) | From request arrival to the first output token | Queueing + prefill (compute-bound) | 200 ms – 2 s |
| **TPOT** (time per output token) | `(E2E − TTFT) / (n_out − 1)`, the average per request | Decode step time (memory-bound), which grows with batch size | 20–100 ms |
| **ITL** (inter-token latency) | The gap between each pair of consecutive tokens, as a distribution | Same as TPOT, plus stalls from prefills and preemption | Watch P99 |
| **E2E latency** | `TTFT + TPOT × (n_out − 1)` | Both | Depends on the use case |
| **Normalised latency** | E2E / n_out (s/token) | | Used in vLLM and Orca papers |

**TPOT vs ITL.** TPOT is an average per request. ITL is the full distribution of gaps. A request can have a good TPOT and still show a 500 ms ITL spike when a long prefill gets batched in. Users perceive that spike as stutter.

**Always report percentiles** (P50, P90, P99). Means hide the tail, and the tail is what users notice.

### Throughput metrics

- **Output tokens/s**, per GPU or for the whole cluster. This is the main efficiency number.
- **Total tokens/s**, counting input and output. It looks better because prefill tokens are cheap per token, so say which one you mean.
- **Requests/s (QPS)**, which depends on the length distribution.

### Goodput and SLO attainment

```
Goodput = requests/s that meet ALL SLOs (e.g. P90 TTFT ≤ 500 ms AND P90 TPOT ≤ 50 ms)
SLO attainment = fraction of requests meeting their SLOs
```

Raw throughput can be increased by making latency terrible. Goodput is the honest metric, and it is the objective DistServe optimises. **Goodput per GPU** (or per dollar) is the number to compare systems on.

### The throughput–latency trade-off

Increasing the batch size raises tokens/s, because decode is memory-bound and each weight read serves more tokens. It also raises TPOT, because each step reads more KV cache and does more compute. Plotting **throughput vs TPOT** (or vs QPS) as the load grows traces a curve. The useful operating point is the highest throughput that still meets the SLOs.

### Little's Law (capacity planning)

```
Concurrency (requests in flight) = arrival rate × average latency
```

**Example.** 10 req/s with a 20 s average E2E means 200 concurrent sequences. With 2K tokens each on Llama-3-8B (128 KiB/token), the KV cache needs 200 × 2048 × 128 KiB ≈ 50 GiB. That fits on one H100 alongside the weights, just barely.

### Memory and scheduler metrics

- **KV cache utilisation:** the fraction of KV blocks in use. Sustained values near 100% lead to preemptions.
- **Preemption / swap / recompute rate:** sequences evicted because KV memory ran out. Each one wastes work and adds latency.
- **Running batch size and queue depth:** good inputs for autoscaling.
- **Prefix cache hit rate:** the fraction of prompt tokens served from cache. Each hit skips prefill compute and cuts TTFT.
- **Max concurrent sequences** = free KV memory / (KV bytes per token × average sequence length).

### Speculative decoding metrics

- **Acceptance rate α:** the probability that a single draft token is accepted.
- **Mean accepted length / tokens per step:** `(1 − α^(k+1)) / (1 − α)` for draft length k.
- **Speedup:** tokens per step ÷ (1 + draft cost ÷ target cost per step). A high α is wasted if the draft model is slow.

### Cost

```
$ per 1M output tokens = (GPU $/hour × num GPUs) / (output tokens/s × 3600) × 1e6
```

**Example.** One H100 at $3/hr serving 2,000 output tokens/s costs 3 / 7.2e6 × 1e6 ≈ **$0.42 per 1M tokens**.

---

## Part 4: Training-run metrics

| Metric | What it tells you |
|---|---|
| **Step time / tokens per second** | Raw speed. Break it down into compute, exposed communication, and data loading. |
| **MFU** | How efficiently you use the hardware. The main efficiency number. |
| **Peak memory** (`torch.cuda.max_memory_allocated`) | Headroom for a larger micro-batch or less checkpointing. |
| **Data-loader wait time** | Input pipeline bottleneck. It should be about 0. |
| **Loss and grad norm** | Stability. Spikes in grad norm usually come before loss spikes. |
| **Goodput (training)** | Fraction of wall-clock time spent on useful training, excluding failures, restarts, and checkpoint save/restore. Matters a lot at 1000+ GPUs. |
| **MTBF / time to recover** | Reliability at scale. It sets the checkpointing frequency. |

---

## Quick reference: back-of-envelope formulas

```
Forward FLOPs                ≈ 2 · P · tokens
Training FLOPs               ≈ 6 · P · tokens
Training time                ≈ 6 · P · D / (N_gpus · peak · MFU)
Weights memory               = P × bytes/param            (bf16: 2, int8: 1, int4: 0.5)
Training state (Adam, mixed) ≈ 16 · P bytes
KV cache per token           = 2 · layers · kv_heads · head_dim · bytes
Decode step time (lower bound) ≈ (weight bytes + KV bytes) / HBM bandwidth
Prefill time (lower bound)   ≈ 2 · P · prompt_tokens / peak FLOP/s
Ridge point                  = peak FLOP/s / bandwidth    (H100 bf16 ≈ 300)
Ring all-reduce time         ≈ 2 · (N−1)/N · bytes / link bandwidth
```

**Worked example: training time.** A 7B model on 1T tokens with 64 H100s at 40% MFU: 6 × 7e9 × 1e12 = 4.2e22 FLOPs. Divided by (64 × 990e12 × 0.4) = 2.5e16 FLOP/s, that is ≈ 1.66e6 s ≈ **19 days**.

## Likely interview questions

- What is arithmetic intensity, and why is decode memory-bound while prefill is compute-bound?
- Draw the roofline for an H100. Where do prefill, decode, and LayerNorm sit on it?
- Your training run reaches 20% MFU. How do you find out why?
- Why is `nvidia-smi` utilisation a poor metric?
- Explain the difference between throughput and goodput. Why did DistServe optimise goodput?
- TPOT looks fine but users complain about stutter. What metric do you check, and what is the probable cause?
- Estimate the $/1M tokens for serving a 70B model on 8×H100.
