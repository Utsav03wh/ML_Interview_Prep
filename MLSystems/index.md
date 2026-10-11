---
title: ML Systems Design Notes
---

# ML Systems Design Notes

Interview-prep notes on the systems side of large models. Each page opens with the **core idea**, then goes through the concepts with back-of-envelope numbers, and ends with a reading list and likely interview questions.

| Topic | Core idea in one line |
|---|---|
| [LLM Serving](llm-serving.md) | Prefill is compute-bound and decode is memory-bandwidth-bound. Every serving trick targets one of the two. |
| [Distributed Training](distributed-training.md) | Each parallelism splits something different across GPUs and pays for it in communication. |
| [RL Post-Training Infrastructure](rl-post-training.md) | RL alternates inference-heavy rollouts with training-heavy updates, and the hard part is coordinating the two. |

## Numbers worth memorising

| Quantity | Value (approx.) |
|---|---|
| H100 SXM bf16 dense compute | ~990 TFLOP/s |
| H100 SXM HBM3 bandwidth | ~3.35 TB/s |
| H100 HBM capacity | 80 GB |
| NVLink (H100, per GPU, bidirectional) | ~900 GB/s |
| InfiniBand NDR per NIC | 400 Gb/s ≈ 50 GB/s |
| H100 compute/bandwidth ratio ("ridge point") | ~300 FLOPs per byte |
| Forward pass FLOPs | ≈ 2 × params × tokens |
| Training FLOPs (fwd + bwd) | ≈ 6 × params × tokens |
| Mixed-precision Adam model state | ≈ 16 bytes / param |

## How to answer a systems design question

1. **Clarify the workload.** Model size, context lengths, request rate, latency SLOs, and the hardware available.
2. **Do the arithmetic first.** Memory (weights, KV cache, optimizer state, activations), FLOPs, and bytes moved. Work out which resource is the bottleneck.
3. **Pick techniques that relieve that bottleneck**, and name what each one costs (communication, staleness, complexity).
4. **Name the metrics** you would watch and the knobs you would tune.
5. **Discuss failure modes**: stragglers, OOMs, long tails, numerical instability.
