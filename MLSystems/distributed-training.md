---
title: Distributed Training
---

[← ML Systems index](index.md)

# Distributed Training

> **Core idea.** A large model doesn't fit on one GPU, and training on one GPU is too slow. Each parallelism strategy splits something different across GPUs: data, optimizer state, weight matrices, layers, or sequence. Each one pays for the split with **communication**.

Notation used below: `Ψ` = number of parameters, `N` = number of GPUs in a group.

---

## 1. Memory budget

Mixed-precision training with Adam keeps this much state per parameter:

| State | Precision | Bytes/param |
|---|---|---|
| Weights (used in fwd/bwd) | bf16 | 2 |
| Gradients | bf16 | 2 |
| Master weights | fp32 | 4 |
| Adam first moment `m` | fp32 | 4 |
| Adam second moment `v` | fp32 | 4 |
| **Total** | | **16** |

**7B model → 7e9 × 16 B ≈ 112 GB** of model state before any activations. That is more than one 80 GB H100, so training it needs sharding.

**Activations** scale with batch × sequence × hidden × layers. Per transformer layer, Korthikanti et al. estimate about `s·b·h·(34 + 5·a·s/h)` bytes, where s = sequence length, b = micro-batch size, h = hidden size, and a = number of heads. The `5·a·s/h` term is the attention matrix, which FlashAttention removes. For long sequences, activations can exceed the model state.

---

## 2. Communication primitives

| Op | Before → after | Per-GPU send volume (ring) |
|---|---|---|
| **Broadcast** | One rank's tensor → all ranks | ~Ψ |
| **Reduce** | Sum of all ranks → one rank | ~Ψ |
| **All-reduce** | Every rank gets the sum | `2(N−1)/N · Ψ` |
| **Reduce-scatter** | Rank i gets shard i of the sum | `(N−1)/N · Ψ` |
| **All-gather** | Every rank gets the concatenation of all shards | `(N−1)/N · Ψ` |
| **All-to-all** | Rank i sends chunk j to rank j (MoE token routing) | `(N−1)/N · Ψ` |

The key identity: **all-reduce = reduce-scatter + all-gather.**

**Ring all-reduce.** Arrange the N ranks in a ring and split the tensor into N chunks.

1. **Reduce-scatter phase.** N − 1 steps. In each step every rank sends one chunk to its neighbour and adds the chunk it receives. At the end each rank owns one fully-summed chunk.
2. **All-gather phase.** N − 1 more steps that pass the summed chunks around the ring.

Each rank sends `2(N−1) · Ψ/N` in total. That is **bandwidth-optimal and almost independent of N**. The latency term grows with N (2(N−1) sequential steps), so NCCL switches to tree or other algorithms for small messages or very large N.

**Bandwidth hierarchy.** NVLink inside a node (~900 GB/s on H100) is about 10–20× faster than InfiniBand between nodes (~50 GB/s per NIC). This hierarchy decides which parallelism goes where.

---

## 3. Data parallelism (DDP)

- Each GPU holds a **full replica** of the model and optimizer and processes a different slice of the global batch.
- After backward, gradients are **all-reduced** (averaged) so that every replica takes an identical step.
- PyTorch DDP **buckets** gradients (~25 MB) and launches each bucket's all-reduce as soon as its gradients are ready. That **overlaps communication with the rest of backward**.
- Communication is `2Ψ` per step per GPU, independent of N.
- **Limitation:** memory per GPU is still 16Ψ, so DDP alone can't train a model whose state doesn't fit on one GPU.

---

## 4. ZeRO and FSDP

ZeRO removes DDP's redundancy by **sharding** state across the N data-parallel ranks:

| Stage | Sharded | Memory/GPU (bytes) | Communication vs DDP |
|---|---|---|---|
| DDP | nothing | 16Ψ | 1× (all-reduce) |
| **ZeRO-1** | optimizer states (fp32 master, m, v) | 4Ψ + 12Ψ/N | 1× (reduce-scatter grads + all-gather updated params) |
| **ZeRO-2** | + gradients | 2Ψ + 14Ψ/N | 1× |
| **ZeRO-3** | + parameters | 16Ψ/N | ~1.5× |

**How ZeRO-3 / FSDP runs one step:**

1. **Forward.** Before each layer (FSDP "unit"), **all-gather** its parameter shards to get the full layer. Compute, then **free** the gathered params.
2. **Backward.** All-gather the params again, compute the gradients, then **reduce-scatter** the gradients so each rank keeps only its shard.
3. **Optimizer step** on the local shard only.
4. **Prefetch** the next layer's all-gather during the current layer's compute to hide the communication.

**PyTorch FSDP:**

- `FULL_SHARD` = ZeRO-3, `SHARD_GRAD_OP` = ZeRO-2, `NO_SHARD` = DDP.
- `HYBRID_SHARD` shards within a node and replicates across nodes. That keeps the frequent all-gathers on NVLink.
- **FSDP2** (`fully_shard`) shards per parameter using DTensor. It composes cleanly with TP and handles mixed precision and CPU offload.
- **ZeRO-Offload / ZeRO-Infinity** move optimizer state or params to CPU or NVMe, trading PCIe bandwidth for memory.

**Example:** a 7B model with ZeRO-3 on 8 GPUs needs 112/8 = 14 GB of state per GPU, which leaves room for activations.

---

## 5. Tensor parallelism (Megatron-LM)

TP splits **individual weight matrices** across GPUs. Take an MLP `Y = GeLU(X·A)·B`:

- **A is column-parallel.** Split A by columns, `A = [A₁ | A₂]`. Each GPU computes `GeLU(X·Aᵢ)` locally. GeLU is elementwise, so no communication is needed yet.
- **B is row-parallel.** Split B by rows. Each GPU computes the partial product `Yᵢ = GeLU(X·Aᵢ)·Bᵢ`.
- One **all-reduce** sums the partial products: `Y = Σ Yᵢ`.

**Attention** splits the same way: heads are partitioned across GPUs (column-parallel QKV), then a row-parallel output projection, then one all-reduce.

**Cost per transformer layer:**

- 2 all-reduces in forward (attention + MLP) and 2 in backward.
- They sit on the **critical path**: activation-sized and hard to overlap.
- So TP needs NVLink and stays **inside a node**, typically TP ≤ 8.

**Sequence parallelism** (Megatron-SP). The regions outside TP (LayerNorm, dropout) would otherwise hold replicated activations. SP shards them along the sequence dimension and replaces each all-reduce with **reduce-scatter + all-gather**. Communication volume stays the same, and activation memory drops.

**Context parallelism / ring attention** shards very long sequences across GPUs and passes K/V blocks around a ring.

---

## 6. Pipeline parallelism

PP splits the **layers** into p stages on p GPUs. Activations flow forward between stages and gradients flow back, using point-to-point sends that are cheap and work across nodes.

**The bubble.** With one batch, only one stage works at a time. Splitting the batch into m **micro-batches** fills the pipeline:

```
GPipe bubble fraction = (p − 1) / (m + p − 1)      (≈ (p−1)/m when m ≫ p)
```

Example: with p = 8 and m = 32, the bubble is 7/39 ≈ 18% idle. You want m ≫ p.

Schedules:

- **GPipe.** All forwards, then all backwards. It has to store activations for all m micro-batches.
- **1F1B.** After a warm-up phase, each stage alternates one forward with one backward. The **bubble is the same**, but at most p micro-batches are in flight, so activation memory is much lower.
- **Interleaved 1F1B** (Megatron). Each GPU holds v non-contiguous chunks of layers. The bubble shrinks by a factor of v, at the cost of more communication.
- **Zero-bubble / DualPipe.** Split backward into input-grad and weight-grad parts and schedule them to fill the bubbles. DeepSeek-V3's DualPipe also overlaps the MoE all-to-all.

**Costs:** the bubble, load balancing between stages (embedding and LM-head layers are uneven), and the need for a large global batch to get many micro-batches.

---

## 7. 3D (and 4D/5D) parallelism

Combine the strategies, mapped to the bandwidth hierarchy. Total GPUs = DP × TP × PP (× CP × EP).

| Dimension | Placement | Why |
|---|---|---|
| **TP** | Inside a node (≤ 8) | Frequent, blocking, activation-sized all-reduces need NVLink |
| **PP** | Across nodes | Only point-to-point activation sends; tolerant of slower links |
| **DP / ZeRO-1 / FSDP** | Outermost | Gradient sync once per step, overlappable |
| **CP** (context) | Long sequences | Shards the sequence dimension of attention |
| **EP** (expert) | MoE | Experts on different GPUs, with all-to-all token routing |

**How to choose:**

1. Is the model small enough? Use FSDP/ZeRO with DP only.
2. If one layer's weights and activations don't fit, or FSDP communication dominates, add TP within the node.
3. If it still doesn't fit, or you are scaling to thousands of GPUs, add PP across nodes.
4. Use the remaining GPUs for DP to scale throughput.
5. Check **MFU** (model FLOPs utilisation = achieved 6Ψ·tokens/s ÷ peak FLOPs). 40–55% is good for large dense models.

---

## 8. Memory savers

- **Activation (gradient) checkpointing.** Store only the activations at layer boundaries and recompute the inside of each layer during backward. That costs about one extra forward pass (~33% more compute) and cuts activation memory from O(L) to roughly O(√L) or one layer's worth. **Selective checkpointing** recomputes only the cheap but memory-heavy parts, such as attention softmax and dropout.
- **Mixed precision.** bf16 or fp16 compute with fp32 master weights. **fp16 needs loss scaling** because of its small exponent range, which underflows gradients. **bf16 has fp32's exponent range, so it needs no scaling.** FP8 training (Transformer Engine, DeepSeek-V3) uses per-tensor or per-block scaling.
- **Gradient accumulation.** Run k micro-batches and sum their gradients before the optimizer step, for k× the effective batch at the same activation memory. With DDP, skip the gradient sync on the non-final micro-batches (`no_sync()`).
- **Other options:** CPU offload, 8-bit optimizers, fused kernels (fused Adam, FlashAttention), and sharding the large vocab-embedding layer.

---

## Reading list

- Hugging Face, *The Ultra-Scale Playbook: Training LLMs on GPU Clusters*. The best single resource.
- Lilian Weng, *How to Train Really Large Models on Many GPUs?*
- PyTorch FSDP tutorial (and the FSDP2 `fully_shard` docs)
- Further reading: ZeRO (Rajbhandari et al. 2020), Megatron-LM (Shoeybi et al. 2019; Narayanan et al. 2021), *Reducing Activation Recomputation in Large Transformer Models* (Korthikanti et al. 2022), GPipe, PipeDream

## Likely interview questions

- How much memory does it take to train a 13B model with Adam? How many GPUs do you need with ZeRO-3 vs DDP?
- Walk through ring all-reduce. Why is per-GPU traffic independent of N?
- Why is TP kept inside a node and PP used across nodes?
- Derive the pipeline bubble fraction. How does 1F1B differ from GPipe, and what does it improve?
- What exactly does ZeRO-3 communicate in forward and in backward, and why is it about 1.5× DDP?
- When would you use activation checkpointing instead of adding more TP?
