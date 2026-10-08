### Easy (warm-ups and screening rounds)

**Tensor fundamentals:**

- Implement matrix multiplication with loops, then vectorize it, and explain broadcasting.
- Write numerically stable softmax and log-softmax (subtract the max).
- Implement cross-entropy loss from logits, with and without label smoothing.
- Compute pairwise Euclidean distances between two sets of vectors without loops.
- Implement one-hot encoding, `gather`based indexing, and masking a padded batch.
- Implement cosine similarity and top-k retrieval over an embedding matrix.

**Classic ML from scratch:**

- Linear regression with gradient descent, plus the closed-form solution.
- Logistic regression with manual gradients.
- K-means clustering, including the initialization choice and convergence check.
- K-nearest neighbors classification.
- Precision, recall, F1, and ROC-AUC computed from predictions.

**Basic neural network pieces:**

- A two-layer MLP in NumPy with a hand-written backward pass.
- A basic PyTorch training loop: dataset, dataloader, optimizer, loss, eval mode, and `no_grad`.
- ReLU, GELU, sigmoid, and tanh with their derivatives.
- Dropout, including the difference between train and eval behavior and the inverted scaling.

### Medium (most common in onsite rounds)

**Core deep learning layers:**

- [ ]  LayerNorm and BatchNorm, including running statistics for BatchNorm at eval time.
- RMSNorm, and why LLMs use it.
- A 2D convolution with loops, then with `unfold`.
- Max pooling and average pooling.
- An embedding layer and a positional encoding (sinusoidal).
- An LSTM or GRU cell from its equations.

**Optimizers and training:**

- SGD with momentum, and Adam and AdamW from scratch (know why AdamW decouples weight decay).
- A learning-rate schedule with linear warmup and cosine decay.
- Gradient clipping by global norm.
- Gradient accumulation to simulate a larger batch.
- Early stopping and checkpointing logic.

**Attention and transformers:**

- Scaled dot-product attention with a causal mask and a padding mask.
- Multi-head attention with the reshape and transpose logic, which is where most people make mistakes.
- A full transformer block (pre-norm, attention, MLP, residuals).
- A tiny GPT with next-token prediction on a character dataset.
- Greedy decoding, temperature sampling, top-k, and top-p (nucleus) sampling.

**RL basics:**

- REINFORCE on CartPole, with a baseline for variance reduction.
- Discounted returns and Generalized Advantage Estimation (GAE).
- Tabular Q-learning and value iteration on a gridworld.
- A DQN update with a target network and replay buffer.

**Contrastive and representation learning:**

- InfoNCE / contrastive loss (as in CLIP or SimCLR), with in-batch negatives and temperature.
- Triplet loss.

### Hard (research engineer and research scientist loops, and senior MLE)

**LLM internals:**

- A KV cache for autoregressive decoding, and explain the memory cost.
- Rotary position embeddings (RoPE).
- Grouped-query attention (GQA) and multi-query attention.
- Sliding-window attention or a block-sparse mask.
- Beam search with length normalization.
- Speculative decoding with a draft model and the acceptance rule.
- A byte-pair encoding (BPE) tokenizer: training and encoding.
- A Mixture-of-Experts layer with top-k routing and a load-balancing loss.
- LoRA: wrap a linear layer with low-rank adapters and merge them for inference.

**Post-training and RL for LLMs** (very relevant for your profile):

- The PPO clipped objective with a value loss and entropy bonus.
- The GRPO loss: group-relative advantages, the clipped ratio, and the KL penalty to a reference model.
- DPO loss from policy and reference log-probabilities.
- Per-token log-probabilities of a response given a prompt, with correct masking of prompt and padding tokens. This comes up constantly and is easy to get subtly wrong.
- A reward model training loss (Bradley–Terry on preference pairs).
- KL divergence estimators between policy and reference (the k1, k2, and k3 estimators).
- SAC or TD3 actor and critic updates for continuous control.

**Systems and efficiency:**

- Mixed-precision training with `autocast` and gradient scaling, and when bf16 doesn't need scaling.
- Activation checkpointing, and explaining the compute–memory trade-off.
- Data-parallel training with `DistributedDataParallel` (understand the gradient all-reduce).
- Conceptual sketches of tensor parallelism for a linear layer (column vs. row splits).
- Online softmax and the tiling idea behind FlashAttention.
- Quantizing a weight matrix to int8 with per-channel scales, then dequantizing.

**Debugging rounds** (increasingly common at labs):

- Given a training script with planted bugs, find them. Typical bugs include a missing `zero_grad`, a wrong softmax dimension, broadcasting mistakes, a leaking causal mask, forgetting `model.eval()`, applying the loss to padding tokens, and a learning rate off by orders of magnitude.
- Diagnose why a loss is NaN or why it plateaus.

### Where to practice

- **deep-ml.com:** ML-specific coding problems graded by difficulty.
- **Sasha Rush's Tensor Puzzles and GPU Puzzles** (on GitHub): excellent for building broadcasting and vectorization fluency.
- **Karpathy's nanoGPT, minbpe, and "Zero to Hero":** for transformers, tokenizers, and training loops.
- **Stanford CS336 assignments:** for the hard LLM problems (tokenizer, transformer, efficient training).
- **CleanRL:** read its single-file PPO, DQN, and SAC implementations after writing your own, to compare details.