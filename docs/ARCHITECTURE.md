# gemma3.c - Architecture Deep Dive

This document provides a comprehensive analysis of the software architecture and the Gemma 3 model architecture as implemented in this project.

## Table of Contents

1. [Software Architecture Overview](#software-architecture-overview)
2. [Component Diagram](#component-diagram)
3. [Data Flow](#data-flow)
4. [Gemma 3 Model Architecture](#gemma-3-model-architecture)
5. [Attention Mechanisms](#attention-mechanisms)
6. [Memory Layout](#memory-layout)
7. [Threading Model](#threading-model)

---

## Software Architecture Overview

The codebase follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Application Layer                                │
│                          (main.c - CLI)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                         Public API Layer                                 │
│                     (gemma3.h, gemma3.c)                                │
│    gemma3_load_dir() | gemma3_generate() | gemma3_chat()                │
├─────────────────────────────────────────────────────────────────────────┤
│                        Core Components                                   │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────────┐ ┌───────────────┐  │
│  │  Transformer │ │   Tokenizer   │ │ SafeTensors │ │  Thread Pool  │  │
│  │              │ │               │ │             │ │               │  │
│  │ - Forward    │ │ - BPE Encode  │ │ - mmap      │ │ - Work Queue  │  │
│  │ - Attention  │ │ - Decode      │ │ - JSON      │ │ - Sync        │  │
│  │ - MLP        │ │ - Chat Format │ │ - Shards    │ │               │  │
│  └──────────────┘ └───────────────┘ └─────────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                       Compute Kernels                                    │
│                    (gemma3_kernels.c)                                   │
│   matmul | matvec | rmsnorm | gelu | softmax | rope | gqa              │
├─────────────────────────────────────────────────────────────────────────┤
│                      Platform Abstraction                                │
│          POSIX (mmap, pthreads) | Optional BLAS | Optional AVX2         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Single Responsibility**: Each source file handles one concern
2. **Dependency Injection**: Transformer receives weights/config at creation
3. **Resource Acquisition Is Initialization (RAII)**: All resources freed via `*_free()` functions
4. **Fail-Fast**: Early validation with descriptive error messages

---

## Component Diagram

### High-Level Components

```
                                    ┌─────────────────┐
                                    │   Application   │
                                    │    (main.c)     │
                                    └────────┬────────┘
                                             │
                                             ▼
                               ┌─────────────────────────┐
                               │     gemma3_ctx          │
                               │  ┌─────────────────────┐│
                               │  │ gemma3_config       ││
                               │  │ - vocab_size        ││
                               │  │ - hidden_size       ││
                               │  │ - num_layers        ││
                               │  │ - num_heads         ││
                               │  └─────────────────────┘│
                               │  ┌─────────────────────┐│
                               │  │ st_context*         ││───┐
                               │  │ (SafeTensors)       ││   │
                               │  └─────────────────────┘│   │
                               │  ┌─────────────────────┐│   │
                               │  │ gemma3_weights_t*   ││───┤
                               │  │ (BF16 pointers)     ││   │
                               │  └─────────────────────┘│   │
                               │  ┌─────────────────────┐│   │
                               │  │ gemma3_tokenizer*   ││   │
                               │  │ (SentencePiece)     ││   │
                               │  └─────────────────────┘│   │
                               │  ┌─────────────────────┐│   │
                               │  │ gemma3_transformer* ││   │
                               │  │ - kv_cache          ││   │
                               │  │ - buffers           ││   │
                               │  │ - rope_freqs        ││   │
                               │  └─────────────────────┘│   │
                               │  ┌─────────────────────┐│   │
                               │  │ logits_buf          ││   │
                               │  │ probs_buf           ││   │
                               │  └─────────────────────┘│   │
                               └─────────────────────────┘   │
                                             │               │
                                             │               │
                                             ▼               ▼
                               ┌──────────────────────────────────────┐
                               │           Memory-Mapped Files         │
                               │  model-00001-of-00002.safetensors    │
                               │  model-00002-of-00002.safetensors    │
                               │  tokenizer.model                      │
                               └──────────────────────────────────────┘
```

### Transformer Internal Structure

```
┌───────────────────────────────────────────────────────────────────┐
│                      gemma3_transformer                            │
├───────────────────────────────────────────────────────────────────┤
│  weights: gemma3_weights_t*                                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ embed_tokens: uint16_t* [262208 × 2560] ──────────────────┐ │  │
│  │ layers[0..33]:                                             │ │  │
│  │   input_layernorm: uint16_t* [2560]                       │ │  │
│  │   q_proj: uint16_t* [2048 × 2560]                         │ │  │
│  │   k_proj: uint16_t* [1024 × 2560]                         │ │  │
│  │   v_proj: uint16_t* [1024 × 2560]                         │ │  │
│  │   o_proj: uint16_t* [2560 × 2048]                         │ │  │
│  │   q_norm, k_norm: uint16_t* [256]                         │ │  │
│  │   post_attention_layernorm: uint16_t* [2560]              │ │  │
│  │   gate_proj: uint16_t* [10240 × 2560]                     │ │  │
│  │   up_proj: uint16_t* [10240 × 2560]                       │ │  │
│  │   down_proj: uint16_t* [2560 × 10240]                     │ │  │
│  │   pre/post_feedforward_layernorm: uint16_t* [2560]        │ │  │
│  │ norm: uint16_t* [2560]                                     │ │  │
│  └────────────────────────────────────────────────────────────│──┘  │
│                                                      (mmap)  │     │
├───────────────────────────────────────────────────────────────┤     │
│  cache: gemma3_kv_cache*                                      │     │
│  ┌─────────────────────────────────────────────────────────┐  │     │
│  │ layers[0..33]:                                          │  │     │
│  │   k: float* [seq_len × 4 × 256] (ring buffer for local) │  │     │
│  │   v: float* [seq_len × 4 × 256]                         │  │     │
│  │   pos: int (current position)                           │  │     │
│  │ current_pos: int (global sequence position)             │  │     │
│  └─────────────────────────────────────────────────────────┘  │     │
├───────────────────────────────────────────────────────────────┤     │
│  buffers: activation_buffers*                                 │     │
│  ┌─────────────────────────────────────────────────────────┐  │     │
│  │ x: float[2560]          (hidden state)                  │  │     │
│  │ x_norm: float[2560]     (normalized hidden)             │  │     │
│  │ q: float[2048]          (queries)                       │  │     │
│  │ k: float[1024]          (keys)                          │  │     │
│  │ v: float[1024]          (values)                        │  │     │
│  │ attn_out: float[2048]   (attention output)              │  │     │
│  │ proj_out: float[2560]   (projection output)             │  │     │
│  │ mlp_gate: float[10240]  (MLP gate)                      │  │     │
│  │ mlp_up: float[10240]    (MLP up)                        │  │     │
│  │ mlp_out: float[2560]    (MLP output)                    │  │     │
│  │ logits: float[262208]   (vocabulary logits)             │  │     │
│  │ mask: float[max_ctx]    (attention mask)                │  │     │
│  │ attn_scores: float[max_ctx] (pre-allocated)             │  │     │
│  └─────────────────────────────────────────────────────────┘  │     │
├───────────────────────────────────────────────────────────────┤     │
│  rope_freqs_local: float* [max_pos × 128 × 2] (cos/sin)       │     │
│  rope_freqs_global: float* [max_pos × 128 × 2] (cos/sin)      │     │
├───────────────────────────────────────────────────────────────┤     │
│  thread_pool: gemma3_thread_pool* (if USE_THREADS)            │     │
└───────────────────────────────────────────────────────────────┘     │
```

---

## Data Flow

### Generation Pipeline

```
User Prompt: "Hello, world!"
         │
         ▼
┌────────────────────────────────────┐
│        gemma3_generate()           │
│  1. Validate inputs                │
│  2. Initialize RNG                 │
│  3. Reset KV cache                 │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│       gemma3_tokenize()            │
│  1. UTF-8 segmentation             │
│  2. Word boundary detection (▁)    │
│  3. BPE merge loop                 │
│  4. Add BOS token                  │
│                                    │
│  Output: [2, 17534, 235269, ...]   │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│    gemma3_forward_batch()          │
│       (Prefill Phase)              │
│  Process all input tokens          │
│  Build up KV cache                 │
│  Return logits for last token      │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│      Generation Loop               │
│  ┌──────────────────────────────┐  │
│  │ 1. sample_token(logits)      │  │
│  │    - Apply temperature       │  │
│  │    - Top-k filtering         │  │
│  │    - Top-p filtering         │  │
│  │    - Softmax                 │  │
│  │    - Random sample           │  │
│  │                              │  │
│  │ 2. Check for EOS/end_turn    │  │
│  │                              │  │
│  │ 3. Store token, call callback│  │
│  │                              │  │
│  │ 4. gemma3_forward()          │  │
│  │    (single token decode)     │  │
│  │                              │  │
│  │ 5. Increment position        │  │
│  └──────────┬───────────────────┘  │
│             │                      │
│             └──── repeat ──────────┤
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│     gemma3_detokenize()            │
│  1. Map token IDs to pieces        │
│  2. Convert ▁ to spaces            │
│  3. Decode byte tokens (<0xNN>)    │
│                                    │
│  Output: "Generated response..."   │
└────────────────────────────────────┘
```

### Single Token Forward Pass

```
Token ID: 17534
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  gemma3_embed_bf16()                                            │
│  x[i] = bf16_to_f32(embed[token_id * hidden + i]) * sqrt(2560)  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  For layer = 0 to 33:                                           │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ SELF-ATTENTION BLOCK                                      │  │
│  │                                                           │  │
│  │ x_norm = rmsnorm(x, input_layernorm)                      │  │
│  │                                                           │  │
│  │ q = x_norm @ q_proj  [2048]                               │  │
│  │ k = x_norm @ k_proj  [1024]                               │  │
│  │ v = x_norm @ v_proj  [1024]                               │  │
│  │                                                           │  │
│  │ q = rmsnorm(q, q_norm) per head                           │  │
│  │ k = rmsnorm(k, k_norm) per head                           │  │
│  │                                                           │  │
│  │ q = rope(q, pos, theta)                                   │  │
│  │ k = rope(k, pos, theta)                                   │  │
│  │                                                           │  │
│  │ cache_kv(k, v)                                            │  │
│  │                                                           │  │
│  │ attn_out = gqa(q, k_cache, v_cache, mask)                 │  │
│  │                                                           │  │
│  │ proj_out = attn_out @ o_proj                              │  │
│  │ proj_out = rmsnorm_inplace(proj_out, post_attn_ln)        │  │
│  │                                                           │  │
│  │ x = x + proj_out  (residual)                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ MLP BLOCK (SwiGLU)                                        │  │
│  │                                                           │  │
│  │ x_norm = rmsnorm(x, pre_feedforward_ln)                   │  │
│  │                                                           │  │
│  │ gate = x_norm @ gate_proj  [10240]                        │  │
│  │ up = x_norm @ up_proj      [10240]                        │  │
│  │                                                           │  │
│  │ gate = gelu(gate)                                         │  │
│  │ hidden = gate * up                                        │  │
│  │                                                           │  │
│  │ mlp_out = hidden @ down_proj  [2560]                      │  │
│  │ mlp_out = rmsnorm_inplace(mlp_out, post_feedforward_ln)   │  │
│  │                                                           │  │
│  │ x = x + mlp_out  (residual)                               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  x_norm = rmsnorm(x, final_norm)                                │
│  logits = x_norm @ embed_tokens.T   [262208]                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Gemma 3 Model Architecture

### Transformer Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gemma 3 4B Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Embedding:                                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ vocab_size: 262,208                                       │  │
│  │ hidden_size: 2,560                                        │  │
│  │ Scaling: x = embed[token] * sqrt(hidden_size)             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Transformer Layers (x34):                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Attention:                                                │  │
│  │   num_heads: 8                                            │  │
│  │   num_kv_heads: 4 (GQA, 2:1 ratio)                        │  │
│  │   head_dim: 256                                           │  │
│  │   Total Q dim: 8 × 256 = 2,048                            │  │
│  │   Total KV dim: 4 × 256 = 1,024                           │  │
│  │                                                           │  │
│  │ Hybrid Attention Pattern:                                 │  │
│  │   Local (5/6 layers): sliding_window=1024, theta=10K      │  │
│  │   Global (1/6 layers): causal, theta=1M                   │  │
│  │                                                           │  │
│  │ MLP (SwiGLU):                                             │  │
│  │   intermediate_size: 10,240 (4x hidden)                   │  │
│  │   Activation: GELU (tanh approximation)                   │  │
│  │                                                           │  │
│  │ Normalization:                                            │  │
│  │   Type: RMSNorm with (1 + weight) formula                 │  │
│  │   Epsilon: 1e-6                                           │  │
│  │   Locations: 6 per layer (input, post-attn, pre/post-ff)  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Output:                                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Tied embeddings: logits = hidden @ embed.T                │  │
│  │ Output shape: [262,208]                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Architecture Detail

```
                         Layer N Input
                              │
                              ▼
                    ┌─────────────────┐
                    │   RMSNorm       │ ←── input_layernorm
                    │  (1 + weight)   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
      ┌─────────┐       ┌─────────┐       ┌─────────┐
      │ Q Proj  │       │ K Proj  │       │ V Proj  │
      │ [2048]  │       │ [1024]  │       │ [1024]  │
      └────┬────┘       └────┬────┘       └────┬────┘
           │                 │                 │
           ▼                 ▼                 │
      ┌─────────┐       ┌─────────┐            │
      │ Q Norm  │       │ K Norm  │            │
      └────┬────┘       └────┬────┘            │
           │                 │                 │
           ▼                 ▼                 │
      ┌─────────┐       ┌─────────┐            │
      │  RoPE   │       │  RoPE   │            │
      │ θ=10K/1M│       │ θ=10K/1M│            │
      └────┬────┘       └────┬────┘            │
           │                 │                 │
           │                 ▼                 ▼
           │            ┌─────────────────────────┐
           │            │      KV Cache           │
           │            │   k: [seq, 4, 256]      │
           │            │   v: [seq, 4, 256]      │
           │            └─────────────────────────┘
           │                      │
           └──────────┬───────────┘
                      ▼
              ┌───────────────┐
              │     GQA       │ ←── 8 Q heads, 4 KV heads
              │  Attention    │
              │               │
              │ scores = Q·K^T│
              │ weights = softmax(scores/√d + mask)
              │ output = weights·V
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │    O Proj     │
              │    [2560]     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   RMSNorm     │ ←── post_attention_layernorm
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
           ┌─▶│      +        │ ←── Residual Connection
           │  └───────┬───────┘
           │          │
           │          ▼
           │  ┌───────────────┐
           │  │   RMSNorm     │ ←── pre_feedforward_layernorm
           │  └───────┬───────┘
           │          │
           │  ┌───────┴───────┐
           │  │               │
           │  ▼               ▼
           │ ┌─────────┐  ┌─────────┐
           │ │Gate Proj│  │ Up Proj │
           │ │ [10240] │  │ [10240] │
           │ └────┬────┘  └────┬────┘
           │      │            │
           │      ▼            │
           │ ┌─────────┐       │
           │ │  GELU   │       │
           │ └────┬────┘       │
           │      │            │
           │      └─────┬──────┘
           │            │
           │            ▼
           │     ┌───────────┐
           │     │     ×     │ ←── Element-wise multiply
           │     └─────┬─────┘
           │           │
           │           ▼
           │     ┌───────────┐
           │     │ Down Proj │
           │     │  [2560]   │
           │     └─────┬─────┘
           │           │
           │           ▼
           │     ┌───────────┐
           │     │  RMSNorm  │ ←── post_feedforward_layernorm
           │     └─────┬─────┘
           │           │
           │           ▼
           └─────┬───────────┐
                 │     +     │ ←── Residual Connection
                 └─────┬─────┘
                       │
                       ▼
                  Layer N Output
```

---

## Attention Mechanisms

### Grouped Query Attention (GQA)

GQA reduces memory and computation by sharing Key-Value heads across multiple Query heads:

```
Standard Multi-Head Attention:
  Q heads: [H0] [H1] [H2] [H3] [H4] [H5] [H6] [H7]
  K heads: [K0] [K1] [K2] [K3] [K4] [K5] [K6] [K7]
  V heads: [V0] [V1] [V2] [V3] [V4] [V5] [V6] [V7]

Grouped Query Attention (Gemma 3):
  Q heads: [H0] [H1] [H2] [H3] [H4] [H5] [H6] [H7]
              \   /       \   /       \   /       \   /
  K heads:    [K0]        [K1]        [K2]        [K3]
  V heads:    [V0]        [V1]        [V2]        [V3]

Benefits:
  - 50% less KV cache memory
  - 50% less KV projection compute
  - Minimal quality loss vs full MHA
```

### Implementation

```c
void gemma3_gqa(float *output, const float *q,
                const float *k_cache, const float *v_cache,
                int n_heads, int n_kv_heads, int seq_len, int head_dim,
                float scale, const float *mask, float *scores_buf) {

    int heads_per_group = n_heads / n_kv_heads;  // = 2 for Gemma 3
    int kv_stride = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / heads_per_group;  // Which KV head to use

        // Q head h uses KV head (h / 2)
        // Heads 0,1 use KV 0
        // Heads 2,3 use KV 1
        // Heads 4,5 use KV 2
        // Heads 6,7 use KV 3

        // Compute attention for this Q head
        for (int i = 0; i < seq_len; i++) {
            scores[i] = dot(q_head, k_cache[i, kv_head]) * scale + mask[i];
        }
        softmax(scores, seq_len);

        // Weighted sum of values
        for (int i = 0; i < seq_len; i++) {
            out_head += scores[i] * v_cache[i, kv_head];
        }
    }
}
```

### Hybrid Attention Pattern

Gemma 3 alternates between local (sliding window) and global (full causal) attention:

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Sequence: [T0] [T1] [T2] [T3] [T4] [T5] [T6] [T7] [T8] [T9] [T10] [T11]   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ LOCAL ATTENTION (layers 0-4, 6-10, ...):                                  │
│ Window size = 1024                                                        │
│                                                                           │
│ Query at T11 can attend to: [T8] [T9] [T10] [T11]                         │
│                              └─────────────────────┘                      │
│                                   window of 4                             │
│                                                                           │
│ Mask pattern for T11:                                                     │
│   T0   T1   T2   T3   T4   T5   T6   T7   T8   T9   T10  T11              │
│  [-∞] [-∞] [-∞] [-∞] [-∞] [-∞] [-∞] [-∞]  [0]  [0]  [0]  [0]             │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ GLOBAL ATTENTION (layers 5, 11, 17, 23, 29):                              │
│ Full causal (can attend to all previous positions)                        │
│                                                                           │
│ Query at T11 can attend to: [T0] [T1] [T2] ... [T10] [T11]                │
│                                                                           │
│ Mask pattern for T11:                                                     │
│   T0   T1   T2   T3   T4   T5   T6   T7   T8   T9   T10  T11              │
│   [0]  [0]  [0]  [0]  [0]  [0]  [0]  [0]  [0]  [0]  [0]  [0]             │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### RoPE (Rotary Position Embeddings)

RoPE encodes position information by rotating query and key vectors:

```
For dimension pair (i, i + d/2):

  [q_i]       [cos(θ)  -sin(θ)] [q_i]
  [       ] = [               ] [       ]
  [q_{i+d/2}] [sin(θ)   cos(θ)] [q_{i+d/2}]

Where:
  θ = position × frequency
  frequency = 1 / (base^(2i/d))

Gemma 3 RoPE parameters:
  Local layers:  base = 10,000
  Global layers: base = 1,000,000 (longer effective context)
```

---

## Memory Layout

### KV Cache Organization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KV Cache Memory Layout                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ GLOBAL LAYERS (5, 11, 17, 23, 29):                                       │
│ Full sequence storage: [max_context × num_kv_heads × head_dim]           │
│                                                                          │
│ Position:    0      1      2      ...    max_context-1                   │
│ K cache: [k0,h0][k0,h1][k0,h2][k0,h3] | [k1,h0]... | ... | [kN,h0]...    │
│ V cache: [v0,h0][v0,h1][v0,h2][v0,h3] | [v1,h0]... | ... | [vN,h0]...    │
│                                                                          │
│ Access pattern: cache[pos * kv_stride + head * head_dim + d]             │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ LOCAL LAYERS (0-4, 6-10, 12-16, ...):                                    │
│ Ring buffer: [sliding_window × num_kv_heads × head_dim]                  │
│                                                                          │
│ Ring position = global_pos % sliding_window                              │
│                                                                          │
│ Example with window=4, global_pos=6:                                     │
│                                                                          │
│   Ring index:   0      1      2      3                                   │
│   Global pos:   4      5      6      7                                   │
│   K cache:    [k4,*] [k5,*] [k6,*] [k7,*]  (next write at index 3)       │
│   V cache:    [v4,*] [v5,*] [v6,*] [v7,*]                                │
│                                                                          │
│   After pos=8:                                                           │
│   Ring index:   0      1      2      3                                   │
│   Global pos:   8      5      6      7                                   │
│   K cache:    [k8,*] [k5,*] [k6,*] [k7,*]  (overwrote old k4)            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Activation Buffer Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Activation Buffer Sizes (Gemma 3 4B)                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Buffer          Formula                   Size (bytes)                   │
│ ──────          ───────                   ────────────                   │
│ x               hidden × 4                10,240                         │
│ x_norm          hidden × 4                10,240                         │
│ q               num_heads × head_dim × 4  8,192                          │
│ k               num_kv_heads × head_dim×4 4,096                          │
│ v               num_kv_heads × head_dim×4 4,096                          │
│ attn_out        num_heads × head_dim × 4  8,192                          │
│ proj_out        hidden × 4                10,240                         │
│ mlp_gate        intermediate × 4          40,960                         │
│ mlp_up          intermediate × 4          40,960                         │
│ mlp_out         hidden × 4                10,240                         │
│ logits          vocab × 4                 1,048,832                      │
│ mask            max_context × 4           variable                       │
│ attn_scores     max_context × 4           variable                       │
│ ────────────────────────────────────────────────────────                 │
│ Total (fixed)                             ~1.2 MB                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Threading Model

### Thread Pool Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Thread Pool Structure                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐                                                         │
│  │   Master    │ ◄─── Main inference thread                              │
│  │   Thread    │                                                         │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         │ gemma3_thread_pool_run(fn, arg)                                │
│         │                                                                │
│         ▼                                                                │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │                    Work Distribution                        │          │
│  │                                                             │          │
│  │  task = { function, argument, generation_counter }          │          │
│  │                                                             │          │
│  │  Signal all workers via condition variable                  │          │
│  │                                                             │          │
│  └─────────────────────────┬──────────────────────────────────┘          │
│                            │                                             │
│         ┌──────────────────┼──────────────────┐                          │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐                    │
│  │  Worker 0  │     │  Worker 1  │     │  Worker N  │                    │
│  │            │     │            │     │            │                    │
│  │ fn(arg,0,N)│     │ fn(arg,1,N)│     │ fn(arg,N,N)│                    │
│  └────────────┘     └────────────┘     └────────────┘                    │
│         │                  │                  │                          │
│         └──────────────────┼──────────────────┘                          │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │                    Synchronization                          │          │
│  │                                                             │          │
│  │  Workers signal completion via condition variable           │          │
│  │  Master waits until all workers finish                      │          │
│  │                                                             │          │
│  └────────────────────────────────────────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Parallelization Strategy

```c
// Matrix-vector multiplication parallelized by rows
static void matvec_bf16_worker(void *arg, int thread_idx, int num_threads) {
    matvec_bf16_task_t *t = (matvec_bf16_task_t *)arg;

    // Each thread processes a contiguous chunk of rows
    int rows_per_thread = (t->M + num_threads - 1) / num_threads;
    int start = thread_idx * rows_per_thread;
    int end = min(start + rows_per_thread, t->M);

    for (int i = start; i < end; i++) {
        t->y[i] = dot_product(t->A[i], t->x, t->K);
    }
}

// Example work distribution for M=2560 rows, 4 threads:
// Thread 0: rows 0-639
// Thread 1: rows 640-1279
// Thread 2: rows 1280-1919
// Thread 3: rows 1920-2559
```

---

## Related Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Main project documentation
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API reference
- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Build system details
- [INTERNALS.md](INTERNALS.md) - Implementation deep dive
