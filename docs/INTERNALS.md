# gemma3.c - Implementation Internals

Deep dive into the implementation details, algorithms, and design decisions.

## Table of Contents

1. [SafeTensors Parser](#safetensors-parser)
2. [SentencePiece Tokenizer](#sentencepiece-tokenizer)
3. [Transformer Forward Pass](#transformer-forward-pass)
4. [Compute Kernels](#compute-kernels)
5. [Memory Mapping Strategy](#memory-mapping-strategy)
6. [BF16 Handling](#bf16-handling)
7. [Sampling Algorithms](#sampling-algorithms)
8. [Thread Pool Implementation](#thread-pool-implementation)
9. [Optimization Techniques](#optimization-techniques)

---

## SafeTensors Parser

### File Format Overview

SafeTensors is a simple, safe tensor serialization format:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Byte 0-7: Header size (uint64, little-endian)                       │
├─────────────────────────────────────────────────────────────────────┤
│ Byte 8 to 8+header_size: JSON header                                │
│                                                                     │
│ {                                                                   │
│   "tensor_name": {                                                  │
│     "dtype": "BF16",                                                │
│     "shape": [262144, 2304],                                        │
│     "data_offsets": [start, end]                                    │
│   },                                                                │
│   "__metadata__": { ... }                                           │
│ }                                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Byte 8+header_size onwards: Binary tensor data                      │
│                                                                     │
│ [tensor1_data][tensor2_data][tensor3_data]...                       │
└─────────────────────────────────────────────────────────────────────┘
```

### JSON Parser Implementation

The codebase includes a minimal JSON parser specifically for SafeTensors headers:

```c
// gemma3_safetensors.c:103-248

typedef struct {
    const char *str;
    int pos;
    int len;
} json_parser;

// Skip whitespace
static void json_skip_ws(json_parser *p) {
    while (p->pos < p->len) {
        char c = p->str[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->pos++;
        } else {
            break;
        }
    }
}

// Parse string with escape handling
static int json_parse_string(json_parser *p, char *out, int max_len) {
    // Handle: \"  \\  \n  \t  \r
    // ...
}

// Parse 64-bit integer
static int json_parse_int64(json_parser *p, int64_t *out) {
    // Handle negative numbers and standard decimal parsing
    // ...
}
```

**Design Decisions:**
- No external JSON library dependency
- Only parses what SafeTensors needs (objects, strings, integers, arrays)
- Recursive descent parser for simplicity

### Memory Mapping

```c
// gemma3_safetensors.c:348-376

static int st_open_file(st_file *f, const char *path) {
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) return 0;

    struct stat st;
    fstat(f->fd, &st);
    f->file_size = st.st_size;

    // Memory map the entire file
    f->mmap_ptr = mmap(NULL, f->file_size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->mmap_ptr == MAP_FAILED) {
        close(f->fd);
        return 0;
    }

    // Parse header size from first 8 bytes
    uint64_t header_size;
    memcpy(&header_size, f->mmap_ptr, 8);
    f->header_size = header_size;

    // Data starts after header
    f->data_start = (char *)f->mmap_ptr + 8 + f->header_size;
    return 1;
}
```

**Benefits:**
- Zero-copy access to tensor data
- OS handles page faulting and caching
- Multiple processes can share mapped pages

### Sharded Model Support

Gemma 3 4B uses two SafeTensors files:

```c
// gemma3_safetensors.c:393-440

st_context *st_load(const char *model_dir) {
    // Find all .safetensors files
    DIR *dir = opendir(model_dir);
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".safetensors")) {
            // Add to file list
        }
    }

    // Sort for consistent ordering (model-00001, model-00002, ...)
    // ...

    // Open each file and parse headers
    for (int i = 0; i < num_files; i++) {
        st_open_file(&ctx->files[i], file_paths[i]);
        st_parse_header(header_str, &ctx->tensors[ctx->num_tensors], ...);
    }
}
```

---

## SentencePiece Tokenizer

### Protobuf Parsing

SentencePiece models use Protocol Buffers format:

```c
// gemma3_tokenizer.c:79-98

// Protobuf wire types
#define PB_VARINT 0   // int32, int64, uint32, uint64, sint32, sint64, bool, enum
#define PB_64BIT 1    // fixed64, sfixed64, double
#define PB_LENDELIM 2 // string, bytes, embedded messages, packed repeated fields
#define PB_32BIT 5    // fixed32, sfixed32, float

// Read variable-length integer
static uint64_t pb_read_varint(const uint8_t **ptr, const uint8_t *end) {
    uint64_t result = 0;
    int shift = 0;
    while (*ptr < end) {
        uint8_t byte = *(*ptr)++;
        result |= (uint64_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;  // MSB clear = last byte
        shift += 7;
    }
    return result;
}
```

### Vocabulary Loading

```c
// gemma3_tokenizer.c:142-304

gemma3_tokenizer *gemma3_tokenizer_load(const char *path) {
    // Read entire file
    FILE *f = fopen(path, "rb");
    uint8_t *data = malloc(file_size);
    fread(data, 1, file_size, f);

    // Parse protobuf messages
    while (ptr < end) {
        uint64_t tag = pb_read_varint(&ptr, end);
        int field = tag >> 3;
        int wire_type = tag & 7;

        if (field == 1 && wire_type == PB_LENDELIM) {
            // SentencePiece entry: piece (string), score (float), type (enum)
            // ...
            tok->vocab[piece_idx].piece = piece;
            tok->vocab[piece_idx].score = score;
            tok->vocab[piece_idx].type = type;

            // Add to hash table for O(1) lookup
            ht_insert(tok, piece, piece_idx);
        }
    }
}
```

### BPE Encoding Algorithm

```c
// gemma3_tokenizer.c:346-524

int gemma3_tokenize(gemma3_tokenizer *tok, const char *text,
                    int *tokens, int max_tokens, int add_bos, int add_eos) {
    // 1. Initialize: each UTF-8 character becomes a symbol
    bpe_symbol *symbols = malloc(text_len * sizeof(bpe_symbol));
    int n_symbols = 0;

    while (pos < text_len) {
        // Handle word boundaries: prepend ▁ (U+2581) at word starts
        if (pos == 0 || text[pos - 1] == ' ') {
            prepend_space = 1;
        }

        // Try to find character (with optional ▁ prefix)
        int id = find_piece(tok, piece, piece_len);

        if (id >= 0) {
            symbols[n_symbols++] = { .id = id, .start = pos, .len = char_len };
        } else {
            // Byte fallback: encode unknown bytes as <0xNN>
            for (int b = 0; b < char_len; b++) {
                symbols[n_symbols++] = { .id = tok->byte_tokens[byte], ... };
            }
        }
    }

    // 2. BPE merge loop: repeatedly merge the highest-scoring pair
    while (1) {
        float best_score = -INFINITY;
        int best_i = -1;

        // Find best merge
        for (int i = 0; i < n_symbols; i++) {
            if (symbols[i].id < 0) continue;  // Deleted
            int j = symbols[i].next;
            if (j < 0) continue;

            // Create merged piece string
            char merged[MAX_TOKEN_LEN];
            sprintf(merged, "%s%s", tok->vocab[symbols[i].id].piece,
                                    tok->vocab[symbols[j].id].piece);

            int merged_id = ht_lookup(tok, merged);
            if (merged_id >= 0) {
                float score = tok->vocab[merged_id].score;
                if (score > best_score) {
                    best_score = score;
                    best_i = i;
                }
            }
        }

        if (best_i < 0) break;  // No more merges possible

        // Apply merge: update symbol i, mark symbol j as deleted
        int j = symbols[best_i].next;
        symbols[best_i].id = best_merged_id;
        symbols[best_i].next = symbols[j].next;
        symbols[j].id = -1;  // Mark deleted
    }

    // 3. Collect output tokens
    // ...
}
```

**Time Complexity:** O(n² × v) where n = input length, v = vocab size
**Space Complexity:** O(n) for symbol array

### Hash Table for Piece Lookup

```c
// gemma3_tokenizer.c:104-136

#define HASH_SIZE 524288  // Must be power of 2, > vocab_size

// DJB2 hash function
static uint32_t hash_string(const char *str) {
    uint32_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + (uint8_t)*str++;
    }
    return hash;
}

// Open addressing with linear probing
static int ht_lookup(gemma3_tokenizer *tok, const char *piece) {
    uint32_t idx = hash_string(piece) & (HASH_SIZE - 1);
    uint32_t start = idx;

    while (tok->piece_to_id[idx] >= 0) {
        int id = tok->piece_to_id[idx];
        if (strcmp(tok->vocab[id].piece, piece) == 0) {
            return id;  // Found
        }
        idx = (idx + 1) & (HASH_SIZE - 1);
        if (idx == start) break;  // Table full
    }
    return -1;  // Not found
}
```

---

## Transformer Forward Pass

### Single Token Forward

```c
// gemma3_transformer.c:385-497

int gemma3_transformer_forward(
    float *logits,
    int token_id,
    int pos,
    const gemma3_weights_t *weights,
    gemma3_kv_cache *cache,
    activation_buffers *buf,
    const gemma3_config *cfg,
    int compute_logits,
    const float *rope_freqs_local,
    const float *rope_freqs_global,
    void *thread_pool
) {
    // 1. Embedding lookup with scaling
    gemma3_embed_bf16(buf->x, weights->embed_tokens, token_id, hidden_size);
    float embed_scale = sqrtf((float)hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        buf->x[i] *= embed_scale;
    }

    // 2. Process each layer
    for (int l = 0; l < num_layers; l++) {
        // Self-Attention Block
        gemma3_rmsnorm_bf16(buf->x_norm, buf->x, layer_weights_input_ln, ...);

        layer_attention(buf->proj_out, buf->x_norm, ...);

        gemma3_rmsnorm_bf16_inplace(buf->proj_out, layer_weights_post_attn_ln, ...);
        gemma3_vec_add(buf->x, buf->x, buf->proj_out, hidden_size);

        // MLP Block
        gemma3_rmsnorm_bf16(buf->x_norm, buf->x, layer_weights_pre_ff_ln, ...);

        layer_mlp(buf->mlp_out, buf->x_norm, ...);

        gemma3_rmsnorm_bf16_inplace(buf->mlp_out, layer_weights_post_ff_ln, ...);
        gemma3_vec_add(buf->x, buf->x, buf->mlp_out, hidden_size);
    }

    // 3. Final norm and output projection (only for last token)
    if (compute_logits) {
        gemma3_rmsnorm_bf16(buf->x_norm, buf->x, weights->norm, ...);
        gemma3_matvec_bf16(logits, weights->embed_tokens, buf->x_norm, ...);
    }
}
```

### Attention Implementation

```c
// gemma3_transformer.c:243-341

static void layer_attention(
    float *output,
    const float *x,
    /* weight pointers... */
    layer_kv_cache *cache,
    /* buffer pointers... */
    const gemma3_config *cfg,
    int layer_idx,
    int pos
) {
    // 1. Project Q, K, V
    matvec_bf16_dispatch(q_buf, q_weight, x, q_size, hidden_size, ...);
    matvec_bf16_dispatch(k_buf, k_weight, x, kv_size, hidden_size, ...);
    matvec_bf16_dispatch(v_buf, v_weight, x, kv_size, hidden_size, ...);

    // 2. Apply QK normalization (Gemma 3 specific)
    for (int h = 0; h < num_heads; h++) {
        gemma3_rmsnorm_bf16(q_buf + h * head_dim, q_buf + h * head_dim,
                            q_norm, head_dim, eps);
    }
    for (int h = 0; h < num_kv_heads; h++) {
        gemma3_rmsnorm_bf16(k_buf + h * head_dim, k_buf + h * head_dim,
                            k_norm, head_dim, eps);
    }

    // 3. Apply RoPE
    for (int h = 0; h < num_heads; h++) {
        gemma3_rope_apply_precomputed(q_buf + h * head_dim, rope_freqs, head_dim, pos);
    }
    for (int h = 0; h < num_kv_heads; h++) {
        gemma3_rope_apply_precomputed(k_buf + h * head_dim, rope_freqs, head_dim, pos);
    }

    // 4. Cache K, V
    cache_kv(cache, k_buf, v_buf, kv_size, is_global, sliding_window, pos);

    // 5. Compute attention
    float scale = 1.0f / sqrtf((float)head_dim);
    gemma3_gqa(attn_buf, q_buf, k_cache, v_cache,
               num_heads, num_kv_heads, seq_len, head_dim,
               scale, mask_buf, scores_buf);

    // 6. Output projection
    matvec_bf16_dispatch(output, o_weight, attn_buf, hidden_size, q_size, ...);
}
```

### MLP Implementation (SwiGLU)

```c
// gemma3_transformer.c:347-378

static void layer_mlp(
    float *output,
    const float *x,
    const uint16_t *gate_weight,
    const uint16_t *up_weight,
    const uint16_t *down_weight,
    float *gate_buf,
    float *up_buf,
    float *matvec_tmp,
    void *thread_pool,
    const gemma3_config *cfg
) {
    // 1. Gate and up projections
    matvec_bf16_dispatch(gate_buf, gate_weight, x, intermediate_size, hidden_size, ...);
    matvec_bf16_dispatch(up_buf, up_weight, x, intermediate_size, hidden_size, ...);

    // 2. SwiGLU activation
    // Gemma 3 uses GELU instead of SiLU in the gate
    gemma3_gelu_tanh_inplace(gate_buf, intermediate_size);
    gemma3_vec_mul(gate_buf, gate_buf, up_buf, intermediate_size);

    // 3. Down projection
    matvec_bf16_dispatch(output, down_weight, gate_buf, hidden_size, intermediate_size, ...);
}
```

---

## Compute Kernels

### BF16 Matrix-Vector Multiplication

```c
// gemma3_kernels.c:195-245

void gemma3_matvec_bf16(float *y, const uint16_t *A, const float *x, int M, int K,
                        float *scratch) {
#ifdef __AVX2__
    // AVX2 optimized path
    for (int i = 0; i < M; i++) {
        y[i] = avx2_bf16_dot(A + i * K, x, K);
    }
#else
    // Scalar fallback
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const uint16_t *row = A + i * K;
        for (int k = 0; k < K; k++) {
            sum += bf16_to_f32(row[k]) * x[k];
        }
        y[i] = sum;
    }
#endif
}
```

### AVX2 BF16 Dot Product

```c
// gemma3_kernels.c:146-192

static float avx2_bf16_dot(const uint16_t *a_bf16, const float *x, int K) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int k = 0;
    // Process 16 elements per iteration (2x8)
    for (; k + 15 < K; k += 16) {
        // Load 8 BF16 values
        __m128i bf16_lo = _mm_loadu_si128((const __m128i *)(a_bf16 + k));

        // Zero-extend uint16 to uint32
        __m256i i32_lo = _mm256_cvtepu16_epi32(bf16_lo);

        // Shift left 16 bits to get F32 bits
        __m256i f32_bits_lo = _mm256_slli_epi32(i32_lo, 16);

        // Reinterpret as float
        __m256 a_lo = _mm256_castsi256_ps(f32_bits_lo);

        // Load 8 F32 values from x
        __m256 x_lo = _mm256_loadu_ps(x + k);

        // Fused multiply-add
        acc0 = _mm256_fmadd_ps(a_lo, x_lo, acc0);

        // Repeat for next 8 elements...
    }

    // Horizontal sum
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 hi128 = _mm256_extractf128_ps(acc0, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc0);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));

    float result = _mm_cvtss_f32(sum1);

    // Scalar tail
    for (; k < K; k++) {
        result += bf16_to_f32(a_bf16[k]) * x[k];
    }
    return result;
}
```

### Grouped Query Attention

```c
// gemma3_kernels.c:527-597

void gemma3_gqa(float *output, const float *q,
                const float *k_cache, const float *v_cache,
                int n_heads, int n_kv_heads, int seq_len, int head_dim,
                float scale, const float *mask, float *scores_buf) {

    int heads_per_group = n_heads / n_kv_heads;  // = 2 for Gemma 3
    int kv_stride = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / heads_per_group;  // Map Q head to KV head

        const float *q_head = q + h * head_dim;
        float *out_head = output + h * head_dim;

        // Compute attention scores: scores[i] = q · k[i] * scale + mask[i]
        for (int i = 0; i < seq_len; i++) {
            const float *k_pos = k_cache + i * kv_stride + kv_head * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_pos[d];
            }
            scores_buf[i] = score * scale;
            if (mask) scores_buf[i] += mask[i];
        }

        // Softmax
        gemma3_softmax_inplace(scores_buf, seq_len);

        // Weighted sum of values
        gemma3_vec_zero(out_head, head_dim);
        for (int i = 0; i < seq_len; i++) {
            const float *v_pos = v_cache + i * kv_stride + kv_head * head_dim;
            float w = scores_buf[i];
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += w * v_pos[d];
            }
        }
    }
}
```

### RMSNorm with Gemma Formula

```c
// gemma3_kernels.c:289-320

void gemma3_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight,
                         int n, float eps) {
    // Compute mean of squares
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / n + eps;
    float rsqrt_ss = 1.0f / sqrtf(ss);

    // Gemma uses (1.0 + weight) formula
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * rsqrt_ss * (1.0f + bf16_to_f32(weight[i]));
    }
}
```

### RoPE with Precomputed Tables

```c
// gemma3_kernels.c:455-484

void gemma3_rope_precompute(float *freqs, int max_pos, int head_dim, float theta) {
    int half_dim = head_dim / 2;

    for (int pos = 0; pos < max_pos; pos++) {
        for (int i = 0; i < half_dim; i++) {
            // Frequency: 1 / (theta^(2i/d))
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;

            // Store cos and sin
            freqs[(pos * half_dim + i) * 2] = cosf(angle);
            freqs[(pos * half_dim + i) * 2 + 1] = sinf(angle);
        }
    }
}

void gemma3_rope_apply_precomputed(float *x, const float *freqs,
                                    int head_dim, int pos) {
    int half_dim = head_dim / 2;
    const float *pos_freqs = freqs + pos * half_dim * 2;

    for (int i = 0; i < half_dim; i++) {
        float cos_val = pos_freqs[i * 2];
        float sin_val = pos_freqs[i * 2 + 1];

        // 2D rotation
        float x0 = x[i];
        float x1 = x[i + half_dim];
        x[i] = x0 * cos_val - x1 * sin_val;
        x[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}
```

---

## Memory Mapping Strategy

### Why mmap?

1. **Zero-copy loading:** Data accessed directly from disk cache
2. **Demand paging:** Only pages actually used are loaded into RAM
3. **Shared memory:** Multiple processes can share the same pages
4. **Automatic cleanup:** OS handles unmapping on process exit

### Implementation Details

```c
// gemma3_safetensors.c:348-376

static int st_open_file(st_file *f, const char *path) {
    f->fd = open(path, O_RDONLY);

    struct stat st;
    fstat(f->fd, &st);
    f->file_size = st.st_size;

    // MAP_PRIVATE: Copy-on-write (we only read, so no copies made)
    // PROT_READ: Read-only access
    f->mmap_ptr = mmap(NULL, f->file_size, PROT_READ, MAP_PRIVATE, f->fd, 0);

    // After mmap, we can close the fd (mapping stays valid)
    // But we keep it open for potential madvise calls
}
```

### Weight Access Pattern

```c
// Weights are accessed via direct pointer into mmap'd region
const uint16_t *embed = weights->embed_tokens;  // Points into mmap

// Access pattern: row-major, sequential within rows
for (int i = 0; i < hidden_size; i++) {
    output[i] = bf16_to_f32(embed[token_id * hidden_size + i]);
}
```

**Memory access pattern:**
- Embedding: Random access (different tokens use different rows)
- Projections: Sequential access during matvec (good cache utilization)

---

## BF16 Handling

### BF16 Format

Brain Float 16 (BF16) is a truncated IEEE 754 single-precision float:

```
FP32: [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
       1    8               23 bits

BF16: [S][EEEEEEEE][MMMMMMM]
       1    8        7 bits

Conversion: BF16 = FP32 >> 16 (just drop lower 16 mantissa bits)
```

### Conversion Functions

```c
// gemma3_kernels.c:135-140

static inline float bf16_to_f32(uint16_t bf16) {
    // Shift left 16 bits to get F32 bit pattern
    uint32_t bits = ((uint32_t)bf16) << 16;

    // Reinterpret as float (use memcpy to avoid strict aliasing issues)
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}
```

### Why BF16?

1. **Same exponent range as F32:** No special handling for large/small values
2. **50% memory reduction:** 2 bytes vs 4 bytes per weight
3. **Simple conversion:** Just bit shift, no complex operations
4. **Hardware support:** Many CPUs have BF16 instructions (not used here, but future-proof)

---

## Sampling Algorithms

### Temperature Scaling

```c
// gemma3_kernels.c:644-651

void gemma3_apply_temperature(float *logits, int vocab_size, float temperature) {
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }
}
```

**Effect:**
- `temperature < 1`: Sharper distribution (more deterministic)
- `temperature > 1`: Flatter distribution (more random)
- `temperature = 0`: Greedy (argmax)

### Top-K Filtering

```c
// gemma3_kernels.c:667-723

void gemma3_topk_filter(float *logits, int vocab_size, int k) {
    // Use min-heap of size k to find threshold in O(n log k)
    float *heap = malloc(k * sizeof(float));

    // Initialize with first k elements
    for (int i = 0; i < k; i++) heap[i] = logits[i];

    // Build min-heap
    // ...

    // Process remaining elements
    for (int i = k; i < vocab_size; i++) {
        if (logits[i] > heap[0]) {
            heap[0] = logits[i];
            // Sift down
            // ...
        }
    }

    // heap[0] is now the k-th largest value
    float threshold = heap[0];

    // Set logits below threshold to -inf
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) {
            logits[i] = -INFINITY;
        }
    }
}
```

**Time Complexity:** O(n log k) instead of O(n log n) for full sort

### Top-P (Nucleus) Filtering

```c
// gemma3_kernels.c:725-782

void gemma3_topp_filter(float *logits, int vocab_size, float p) {
    // 1. Convert logits to probabilities
    float *probs = malloc(vocab_size * sizeof(float));
    gemma3_softmax(probs, logits, vocab_size);

    // 2. Create indexed array for sorting
    IndexedFloat *indexed = malloc(vocab_size * sizeof(IndexedFloat));
    for (int i = 0; i < vocab_size; i++) {
        indexed[i].value = probs[i];
        indexed[i].index = i;
    }

    // 3. Sort by probability descending
    qsort(indexed, vocab_size, sizeof(IndexedFloat), compare_desc);

    // 4. Find cutoff where cumulative prob > p
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += indexed[i].value;
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // 5. Set non-selected logits to -inf
    int *keep = calloc(vocab_size, sizeof(int));
    for (int i = 0; i < cutoff; i++) {
        keep[indexed[i].index] = 1;
    }
    for (int i = 0; i < vocab_size; i++) {
        if (!keep[i]) logits[i] = -INFINITY;
    }
}
```

### Random Sampling

```c
// gemma3_kernels.c:845-858

// xorshift64* PRNG
static uint64_t g_rng_state = 12345678901234567ULL;

float gemma3_random(void) {
    g_rng_state ^= g_rng_state >> 12;
    g_rng_state ^= g_rng_state << 25;
    g_rng_state ^= g_rng_state >> 27;
    uint64_t result = g_rng_state * 0x2545F4914F6CDD1DULL;

    // Convert to float in [0, 1)
    return (float)(result >> 11) * (1.0f / 9007199254740992.0f);
}

int gemma3_sample(const float *probs, int vocab_size) {
    float r = gemma3_random();
    float cumsum = 0.0f;

    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return vocab_size - 1;  // Fallback
}
```

---

## Thread Pool Implementation

### Structure

```c
// gemma3_threads.c

struct gemma3_thread_pool {
    pthread_t *threads;
    int num_threads;

    // Task state
    gemma3_task_fn current_fn;
    void *current_arg;
    int generation;       // Incremented each task

    // Synchronization
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;    // Signal workers
    pthread_cond_t done_cond;    // Signal master
    int active_workers;
    int shutdown;
};
```

### Worker Function

```c
static void *worker_thread(void *arg) {
    gemma3_thread_pool *pool = (gemma3_thread_pool *)arg;
    int thread_idx = /* determined at creation */;

    while (1) {
        pthread_mutex_lock(&pool->mutex);

        // Wait for work or shutdown
        while (pool->generation == local_generation && !pool->shutdown) {
            pthread_cond_wait(&pool->work_cond, &pool->mutex);
        }

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        // Get task
        gemma3_task_fn fn = pool->current_fn;
        void *arg = pool->current_arg;
        local_generation = pool->generation;

        pthread_mutex_unlock(&pool->mutex);

        // Execute task
        fn(arg, thread_idx, pool->num_threads);

        // Signal completion
        pthread_mutex_lock(&pool->mutex);
        pool->active_workers--;
        if (pool->active_workers == 0) {
            pthread_cond_signal(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}
```

### Task Dispatch

```c
void gemma3_thread_pool_run(gemma3_thread_pool *pool,
                            gemma3_task_fn fn, void *arg) {
    pthread_mutex_lock(&pool->mutex);

    // Set up task
    pool->current_fn = fn;
    pool->current_arg = arg;
    pool->generation++;
    pool->active_workers = pool->num_threads;

    // Wake all workers
    pthread_cond_broadcast(&pool->work_cond);

    // Wait for completion
    while (pool->active_workers > 0) {
        pthread_cond_wait(&pool->done_cond, &pool->mutex);
    }

    pthread_mutex_unlock(&pool->mutex);
}
```

---

## Optimization Techniques

### 1. Precomputed Tables

RoPE frequencies are computed once at model load:
```c
gemma3_rope_precompute(rope_freqs_local, max_context, head_dim, 10000.0f);
gemma3_rope_precompute(rope_freqs_global, max_context, head_dim, 1000000.0f);
```

### 2. Buffer Reuse

Activation buffers are pre-allocated and reused:
```c
activation_buffers *buf = alloc_buffers(cfg);
// All forward passes use the same buffers
```

### 3. Skip Logit Computation During Prefill

```c
// Only compute logits for the last token
int is_last = (i == num_tokens - 1);
gemma3_transformer_forward(logits, tokens[i], pos, ..., is_last, ...);
```

### 4. Ring Buffer for Local Attention

```c
// Local layers only store sliding_window entries
if (!is_global) {
    cache_pos = pos % sliding_window;  // Ring buffer
}
```

### 5. Fused BF16 Conversion

AVX2 path converts BF16 to F32 and computes dot product in single pass, avoiding intermediate storage.

### 6. Pre-allocated Attention Scores

```c
// Avoid malloc/free in hot loop
float *scores_buf = buf->attn_scores;  // Pre-allocated
gemma3_gqa(..., scores_buf);
```

---

## Related Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Main project documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture deep dive
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API reference
- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Build system details
