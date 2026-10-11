---
title: RL Post-Training Infrastructure
---

[← ML Systems index](index.md)

# RL Post-Training Infrastructure

> **Core idea.** RL for LLMs alternates between **generating rollouts**, which is inference-heavy, and **updating weights**, which is training-heavy. The two need different optimizations (paged KV and continuous batching versus sharded optimizer state and activation checkpointing), so the design challenge is coordinating them efficiently.

---

## 1. The training loop

```
repeat:
  1. Rollout:   sample prompts → generate responses with the current policy (inference engine)
  2. Score:     rewards (RM / verifier / sandbox), reference log-probs, old-policy log-probs, values
  3. Advantage: GAE (PPO) or group-relative (GRPO)
  4. Train:     several minibatch gradient steps on the clipped policy objective (+ value loss)
  5. Sync:      push new policy weights to the rollout engine
```

Steps 1 and 3 are where most of the systems work happens. Generation is autoregressive and long (reasoning traces of 10K+ tokens), so it usually takes **the majority of wall-clock time**.

---

## 2. The models involved

| Model | Role | Trained? | PPO | GRPO |
|---|---|---|---|---|
| **Policy (actor)** | Generates responses; the model being optimised | Yes | ✓ | ✓ |
| **Reference** | Frozen initial policy; anchor for the KL penalty | No | ✓ | ✓ (optional) |
| **Reward model** | Scores responses; replaced by verifiers for verifiable tasks | No | ✓ | ✓ / verifier |
| **Value (critic)** | Predicts the expected return per token, used for GAE advantages | Yes | ✓ | ✗ |

**PPO** has two *trained* models (actor and critic), each carrying ~16 bytes/param of optimizer state, plus two frozen models that run inference only. For a 70B policy, that is four 70B-scale models in the system.

**GRPO** (DeepSeekMath) removes the critic:

- For each prompt, sample a **group** of G responses (for example G = 8–64).
- Each response's advantage is its reward normalised against the group: `Âᵢ = (rᵢ − mean(r)) / std(r)`.
- Every token in response i gets the same advantage.

This saves the critic's memory and compute and avoids having to train an accurate value function on long sequences. The group sampling fits naturally with an inference engine's parallel sampling, which shares the prompt's KV blocks. RLOO is a related critic-free method that uses a leave-one-out mean as the baseline.

---

## 3. Rollout generation

Use a real serving engine (**vLLM** or **SGLang**) instead of HuggingFace `generate`:

- Continuous batching and PagedAttention give high throughput on variable-length outputs.
- Prefix sharing across the G samples of each prompt.
- Fast kernels, CUDA graphs, and TP for large policies.

Things that complicate it:

- **Log-prob mismatch.** The inference engine's log-probs differ numerically from the trainer's: different kernels, batch-dependent reductions, sometimes FP8 rollouts. So most frameworks **recompute old log-probs with the training engine**. Recent work also applies **truncated importance sampling** to correct the remaining mismatch.
- **Multi-turn / agentic rollouts.** The engine has to interleave generation with tool calls and environment steps. SGLang/vLLM async APIs plus an environment server handle this.

---

## 4. Weight synchronization

After each policy update, the rollout engine needs the new weights. That is 140 GB for a 70B model in bf16.

### Colocated (time-sharing the same GPUs)

Used by verl's **HybridEngine** and by OpenRLHF's colocate mode.

- Training (FSDP/Megatron) and inference (vLLM/SGLang) live on the **same GPUs** and take turns.
- Switching phases means **resharding** in place, from the training layout (for example FSDP shards or Megatron TP×PP) to the inference layout (for example TP=4). This goes over NVLink or GPU memory, with no network transfer.
- Memory is time-shared too. Offload the optimizer state, gradients, and activations during rollout. Free or "sleep" the inference KV cache during training (vLLM sleep mode, SGLang memory release).
- ✅ No idle GPUs and no cross-node weight transfer. Simple resource accounting.
- ❌ The phases run **serially**, with no overlap. Offload and reload overhead. Both phases have to use the same hardware and GPU count.

### Separated (disaggregated pools)

Used by OpenRLHF with Ray, NeMo-RL, and most async systems.

- Trainer GPUs and rollout GPUs are **different pools**. After an update, weights are **broadcast** over NCCL/RDMA (bucketed, often layer by layer) from the trainer ranks to the inference workers.
- ✅ Each side gets its own optimal parallelism and hardware, the pools can be sized independently, and generation can **overlap** with training (asynchronous RL).
- ❌ Weight transfer over the network every step. Idle time on one side unless the design is asynchronous. More engineering: process groups across frameworks, and resharding during transfer.

A rule of thumb: colocated is simpler and efficient at moderate scale with synchronous on-policy training. Separated wins at large scale and when you want asynchrony.

---

## 5. Synchronous vs. asynchronous RL

**Synchronous (on-policy).** Generate the whole batch, then train, then sync. Each iteration's data comes exactly from the current policy.

- The problem is the **long tail**. The batch finishes only when the *longest* response does, so most rollout GPUs sit idle while a few sequences run to 32K tokens. Trainer GPUs also wait during generation, and the rollout GPUs wait during training.

**Asynchronous (off-policy by k steps).** Rollout workers generate continuously with the latest weights they have. The trainer consumes completed samples from a buffer.

- Generation and training **overlap**, and stragglers no longer block everyone.
- Samples come from a **stale behaviour policy** μ = π_{θ − k}. Correct for that with:
  - **Importance sampling** ratios `π_θ / μ`, usually truncated or clipped (the PPO clip already bounds this somewhat).
  - **A staleness bound**: reject or down-weight samples older than k versions.
- Examples:
  - **AReaL**: fully asynchronous, with staleness-aware PPO.
  - **PipelineRL**: *in-flight weight updates*, swapping weights in the middle of a sequence.
  - **One-step off-policy** pipelines, which generate batch t + 1 while training on batch t.

---

## 6. Rewards

| Type | Use case | Infrastructure concerns |
|---|---|---|
| **Reward model** | Open-ended chat and helpfulness (RLHF) | Serve as a separate inference service (batched forward pass, no generation). Susceptible to reward hacking and over-optimisation. |
| **Rule-based verifier** | Math (answer extraction + symbolic equivalence), format checks | Cheap, CPU-side, deterministic. Robust parsing matters, because false negatives become noisy rewards. |
| **Sandboxed execution** | Code (unit tests), tool use, agents | Isolated containers or micro-VMs, CPU/memory/time limits, no network, high concurrency (thousands of executions per step), timeouts that don't stall the batch. |
| **LLM-as-judge / generative RM** | Rubric grading, hard-to-verify tasks | Itself an LLM serving workload, often as costly as the rollouts. |

Scoring should be **asynchronous and pipelined** with generation: score each response as soon as it finishes instead of waiting for the full batch.

---

## 7. Long-tail generation

A handful of very long responses can stall a synchronous batch. Fixes:

- **Max length limits**, and soft overlong penalties (DAPO) instead of hard truncation, which gives a noisy reward.
- **Partial rollouts** (Kimi k1.5). Cap the tokens generated per iteration. Unfinished responses are saved and **resumed in the next iteration**, so long trajectories are spread over several steps.
- **Over-sampling** prompts and moving on once enough groups finish, or discarding and requeueing stragglers. Be careful: dropping long responses biases training toward short ones.
- **Asynchronous designs** (see §5), so stragglers don't block training.
- **Dynamic sampling** (DAPO). Filter out groups where all G responses get the same reward, since their GRPO advantage is zero, and keep sampling until the batch has enough informative groups.
- Load-balance by **length-aware scheduling** across rollout workers.

---

## 8. Stability practices

- **KL control.** Either put a KL penalty in the reward (`r − β·KL`, classic RLHF with an adaptive β) or add a KL loss term against the reference (GRPO, using the k3 estimator). Many reasoning-RL recipes (DAPO, for example) **drop KL entirely** so the policy can move far from the base model.
- **Advantage normalisation.** Whiten advantages per batch (PPO) or per group (GRPO). Dr. GRPO argues that dividing by the group std, and per-sequence length normalisation, introduce biases toward difficulty and length.
- **Loss aggregation, token-level vs. sequence-level:**
  - *Sequence-level* (original GRPO) averages the loss over each response's tokens, then over responses. Every response gets equal total weight, so each token in a **long** response gets less gradient. That under-penalises long, bad responses and lets length drift.
  - *Token-level* (DAPO) averages over all tokens in the batch. Every token gets equal weight.
- **Clipping.** The PPO ratio clip ε ≈ 0.2. **Clip-higher** (DAPO) raises the upper bound (for example ε_high = 0.28) so low-probability tokens can grow, which counters entropy collapse.
- **Entropy collapse.** Policy entropy falls fast, the policy stops exploring, and rewards plateau. Monitor entropy every step. Remedies include clip-higher, entropy bonuses, higher temperature, and restricting updates to high-entropy "forking" tokens.
- **What to monitor:** reward (train and held-out), response length, entropy, KL to the reference, clip fraction, gradient norm, the importance ratio between the training and inference log-probs, and the fraction of truncated responses.

---

## Reading list

- verl (HybridFlow paper), OpenRLHF, NeMo-RL, and TRL docs
- DeepSeekMath (GRPO), DeepSeek-R1, DAPO, Dr. GRPO (*Understanding R1-Zero-Like Training*), Kimi k1.5 (partial rollouts)
- AReaL and PipelineRL (asynchronous RL)
- Ouyang et al. 2022 (InstructGPT); *The N+ Implementation Details of RLHF with PPO*

## Likely interview questions

- Design an RL training system for a 70B reasoning model on 256 GPUs. Would you colocate or separate, and why?
- Why does GRPO save memory relative to PPO? What does it lose?
- Your synchronous RL run has 40% GPU utilisation. Diagnose it and fix it.
- How do you correct for policy staleness in asynchronous RL? What happens if you don't?
- Token-level vs. sequence-level loss averaging: what bias does each introduce?
- Rewards rise while response length explodes and entropy falls. What is going on and what do you change?
- How would you build a sandbox that scores 10K code rollouts per step?
