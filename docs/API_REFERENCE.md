# gemma3.c - API Reference

Complete API documentation for the gemma3.c library.

## Table of Contents

1. [Configuration Constants](#configuration-constants)
2. [Error Codes](#error-codes)
3. [Data Types](#data-types)
4. [Model Loading](#model-loading)
5. [Text Generation](#text-generation)
6. [Chat Interface](#chat-interface)
7. [Tokenization](#tokenization)
8. [Low-Level Forward Pass](#low-level-forward-pass)
9. [KV Cache Management](#kv-cache-management)
10. [Utility Functions](#utility-functions)
11. [Compute Kernels](#compute-kernels)
12. [Thread Pool](#thread-pool)

---

## Configuration Constants

Defined in `gemma3.h`:

### Model Parameters

```c
#define GEMMA3_VOCAB_SIZE        262208    // Vocabulary size
#define GEMMA3_HIDDEN_SIZE       2560      // Hidden dimension
#define GEMMA3_INTERMEDIATE_SIZE 10240     // MLP intermediate dimension
#define GEMMA3_NUM_LAYERS        34        // Number of transformer layers
#define GEMMA3_NUM_HEADS         8         // Number of attention heads
#define GEMMA3_NUM_KV_HEADS      4         // Number of key-value heads (GQA)
#define GEMMA3_HEAD_DIM          256       // Dimension per head
#define GEMMA3_MAX_CONTEXT       131072    // Maximum context length (128K)
#define GEMMA3_SLIDING_WINDOW    1024      // Local attention window size
#define GEMMA3_LOCAL_RATIO       5         // Local:Global layer ratio (5:1)
#define GEMMA3_RMSNORM_EPS       1e-6f     // RMSNorm epsilon
#define GEMMA3_DEFAULT_CONTEXT   8192      // Default context allocation
```

### RoPE Parameters

```c
#define GEMMA3_ROPE_THETA_LOCAL  10000.0f    // RoPE theta for local attention
#define GEMMA3_ROPE_THETA_GLOBAL 1000000.0f  // RoPE theta for global attention
```

### Special Token IDs

```c
#define GEMMA3_TOKEN_PAD        0    // <pad>
#define GEMMA3_TOKEN_EOS        1    // <eos>
#define GEMMA3_TOKEN_BOS        2    // <bos>
#define GEMMA3_TOKEN_UNK        3    // <unk>
#define GEMMA3_TOKEN_START_TURN 105  // <start_of_turn>
#define GEMMA3_TOKEN_END_TURN   106  // <end_of_turn>
```

---

## Error Codes

```c
typedef enum {
    GEMMA3_OK                  =  0,  // Success
    GEMMA3_ERR_INVALID_ARG     = -1,  // Invalid argument
    GEMMA3_ERR_FILE_NOT_FOUND  = -2,  // File not found
    GEMMA3_ERR_INVALID_FORMAT  = -3,  // Invalid file format
    GEMMA3_ERR_OUT_OF_MEMORY   = -4,  // Memory allocation failed
    GEMMA3_ERR_MMAP_FAILED     = -5,  // Memory mapping failed
    GEMMA3_ERR_TOKENIZER_FAILED= -6,  // Tokenizer error
    GEMMA3_ERR_GENERATION_FAILED=-7,  // Generation error
    GEMMA3_ERR_CONTEXT_OVERFLOW= -8,  // Context length exceeded
} gemma3_error;
```

---

## Data Types

### gemma3_config

Model configuration structure.

```c
typedef struct {
    int vocab_size;           // Vocabulary size
    int hidden_size;          // Hidden dimension
    int intermediate_size;    // MLP intermediate dimension
    int num_layers;           // Number of layers
    int num_heads;            // Number of attention heads
    int num_kv_heads;         // Number of KV heads
    int head_dim;             // Dimension per head
    int max_context;          // Maximum context length
    int sliding_window;       // Sliding window size
    float rmsnorm_eps;        // RMSNorm epsilon
    float rope_theta_local;   // RoPE theta for local layers
    float rope_theta_global;  // RoPE theta for global layers
} gemma3_config;
```

### gemma3_gen_params

Generation parameters.

```c
typedef struct {
    int max_tokens;           // Maximum tokens to generate
    float temperature;        // Sampling temperature (0 = greedy)
    int top_k;                // Top-k sampling (0 = disabled)
    float top_p;              // Top-p (nucleus) sampling (1.0 = disabled)
    int seed;                 // Random seed (-1 for random)
    int stop_on_eos;          // Stop when EOS token generated
    int greedy;               // Force greedy decoding
    int verbose_tokens;       // Print token IDs during generation
} gemma3_gen_params;
```

**Default values:**
- `max_tokens`: 512
- `temperature`: 0.7
- `top_k`: 50
- `top_p`: 0.9
- `seed`: -1 (random)
- `stop_on_eos`: 1
- `greedy`: 0
- `verbose_tokens`: 0

### gemma3_role

Chat message roles.

```c
typedef enum {
    GEMMA3_ROLE_USER,    // User message
    GEMMA3_ROLE_MODEL,   // Model/assistant message
    GEMMA3_ROLE_SYSTEM,  // System prompt
} gemma3_role;
```

### gemma3_message

Chat message structure.

```c
typedef struct {
    gemma3_role role;     // Message role
    const char *content;  // Message content
} gemma3_message;
```

### gemma3_token_callback

Callback function for streaming output.

```c
typedef int (*gemma3_token_callback)(
    int token_id,           // The generated token ID
    const char *token_str,  // Decoded string (may be partial UTF-8)
    void *user_data         // User-provided context
);
```

**Return value:**
- `0`: Continue generation
- Non-zero: Stop generation early

---

## Model Loading

### gemma3_load_dir

```c
gemma3_ctx *gemma3_load_dir(const char *model_dir);
```

Load a Gemma 3 model from a HuggingFace model directory.

**Parameters:**
- `model_dir`: Path to directory containing model files

**Required files:**
- `model.safetensors` or `model-00001-of-*.safetensors`
- `tokenizer.model` (SentencePiece)

**Returns:** Context pointer on success, `NULL` on failure.

**Example:**
```c
gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");
if (!ctx) {
    fprintf(stderr, "Error: %s\n", gemma3_get_error());
    exit(1);
}
```

### gemma3_load_dir_ex

```c
gemma3_ctx *gemma3_load_dir_ex(const char *model_dir, int max_context);
```

Load model with custom context length.

**Parameters:**
- `model_dir`: Path to model directory
- `max_context`: Maximum context length to support (affects memory usage)

**Example:**
```c
// Load with reduced context for memory savings
gemma3_ctx *ctx = gemma3_load_dir_ex("./gemma-3-4b-it", 2048);
```

### gemma3_free

```c
void gemma3_free(gemma3_ctx *ctx);
```

Free all resources associated with a context.

**Parameters:**
- `ctx`: Context to free (can be `NULL`)

### gemma3_get_error

```c
const char *gemma3_get_error(void);
```

Get the last error message (thread-local).

**Returns:** Error message string.

### gemma3_get_config

```c
const gemma3_config *gemma3_get_config(const gemma3_ctx *ctx);
```

Get the model configuration.

**Parameters:**
- `ctx`: Model context

**Returns:** Pointer to configuration, or `NULL` if `ctx` is `NULL`.

### gemma3_get_tokenizer

```c
gemma3_tokenizer *gemma3_get_tokenizer(gemma3_ctx *ctx);
```

Get the tokenizer from a context.

**Parameters:**
- `ctx`: Model context

**Returns:** Tokenizer pointer.

---

## Text Generation

### gemma3_default_params

```c
gemma3_gen_params gemma3_default_params(void);
```

Get default generation parameters.

**Returns:** Default parameters structure.

**Default values:**
```c
{
    .max_tokens = 512,
    .temperature = 0.7f,
    .top_k = 50,
    .top_p = 0.9f,
    .seed = -1,
    .stop_on_eos = 1,
    .greedy = 0,
    .verbose_tokens = 0,
}
```

### gemma3_generate

```c
char *gemma3_generate(
    gemma3_ctx *ctx,
    const char *prompt,
    gemma3_gen_params *params,
    gemma3_token_callback callback,
    void *user_data
);
```

Generate text from a prompt.

**Parameters:**
- `ctx`: Model context
- `prompt`: Input prompt (raw text)
- `params`: Generation parameters (`NULL` for defaults)
- `callback`: Optional streaming callback
- `user_data`: Data passed to callback

**Returns:** Generated text (caller must `free()`), or `NULL` on error.

**Example:**
```c
gemma3_gen_params params = gemma3_default_params();
params.max_tokens = 256;
params.temperature = 0.8;

char *output = gemma3_generate(ctx, "Hello, world!", &params, NULL, NULL);
if (output) {
    printf("%s\n", output);
    free(output);
}
```

### gemma3_generate_tokens

```c
char *gemma3_generate_tokens(
    gemma3_ctx *ctx,
    const int *tokens,
    int num_tokens,
    gemma3_gen_params *params,
    gemma3_token_callback callback,
    void *user_data
);
```

Generate text with pre-tokenized input.

**Parameters:**
- `ctx`: Model context
- `tokens`: Array of input token IDs
- `num_tokens`: Number of input tokens
- `params`: Generation parameters
- `callback`: Optional streaming callback
- `user_data`: Data passed to callback

**Returns:** Generated text (caller must `free()`).

**Example:**
```c
int tokens[] = {2, 17534, 235269, 2134, 235341};  // <bos>Hello, world!
char *output = gemma3_generate_tokens(ctx, tokens, 5, NULL, NULL, NULL);
```

---

## Chat Interface

### gemma3_chat

```c
char *gemma3_chat(
    gemma3_ctx *ctx,
    const gemma3_message *messages,
    int num_msgs,
    gemma3_gen_params *params,
    gemma3_token_callback callback,
    void *user_data
);
```

Generate chat completion using Gemma 3 chat template.

**Parameters:**
- `ctx`: Model context
- `messages`: Array of chat messages
- `num_msgs`: Number of messages
- `params`: Generation parameters
- `callback`: Optional streaming callback
- `user_data`: Data passed to callback

**Returns:** Generated response (caller must `free()`).

**Example:**
```c
gemma3_message messages[] = {
    { GEMMA3_ROLE_SYSTEM, "You are a helpful assistant." },
    { GEMMA3_ROLE_USER, "What is the capital of France?" },
};

char *response = gemma3_chat(ctx, messages, 2, NULL, NULL, NULL);
printf("Assistant: %s\n", response);
free(response);
```

### gemma3_format_chat

```c
char *gemma3_format_chat(
    gemma3_tokenizer *tok,
    const gemma3_message *messages,
    int num_msgs
);
```

Format messages with Gemma 3 chat template.

**Parameters:**
- `tok`: Tokenizer
- `messages`: Array of chat messages
- `num_msgs`: Number of messages

**Returns:** Formatted prompt string (caller must `free()`).

**Chat template format:**
```
<bos><start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
{model_message}<end_of_turn>
<start_of_turn>model
```

---

## Tokenization

### gemma3_tokenize

```c
int gemma3_tokenize(
    gemma3_tokenizer *tok,
    const char *text,
    int *tokens,
    int max_tokens,
    int add_bos,
    int add_eos
);
```

Encode text to token IDs.

**Parameters:**
- `tok`: Tokenizer
- `text`: Input text (UTF-8)
- `tokens`: Output array for token IDs
- `max_tokens`: Maximum tokens to output
- `add_bos`: Add beginning-of-sequence token
- `add_eos`: Add end-of-sequence token

**Returns:** Number of tokens written, or negative error code.

**Example:**
```c
int tokens[1024];
int n = gemma3_tokenize(tok, "Hello, world!", tokens, 1024, 1, 0);
// tokens now contains: [2, ...] where 2 is BOS
```

### gemma3_detokenize

```c
char *gemma3_detokenize(
    gemma3_tokenizer *tok,
    const int *tokens,
    int num_tokens
);
```

Decode token IDs to text.

**Parameters:**
- `tok`: Tokenizer
- `tokens`: Array of token IDs
- `num_tokens`: Number of tokens

**Returns:** Decoded string (caller must `free()`).

### gemma3_decode_token

```c
const char *gemma3_decode_token(gemma3_tokenizer *tok, int token_id);
```

Decode a single token ID to text.

**Parameters:**
- `tok`: Tokenizer
- `token_id`: Token ID to decode

**Returns:** Token string (internal storage, do not free), or `NULL`.

### Special Token Accessors

```c
int gemma3_bos_token(gemma3_tokenizer *tok);        // Beginning-of-sequence
int gemma3_eos_token(gemma3_tokenizer *tok);        // End-of-sequence
int gemma3_pad_token(gemma3_tokenizer *tok);        // Padding
int gemma3_start_turn_token(gemma3_tokenizer *tok); // <start_of_turn>
int gemma3_end_turn_token(gemma3_tokenizer *tok);   // <end_of_turn>
```

---

## Low-Level Forward Pass

### gemma3_forward

```c
int gemma3_forward(
    gemma3_ctx *ctx,
    int token_id,
    int pos,
    float *logits
);
```

Run forward pass for a single token.

**Parameters:**
- `ctx`: Model context
- `token_id`: Input token ID
- `pos`: Position in sequence
- `logits`: Output logits array (must be pre-allocated `[vocab_size]`)

**Returns:** 0 on success, negative error code on failure.

### gemma3_forward_batch

```c
int gemma3_forward_batch(
    gemma3_ctx *ctx,
    const int *tokens,
    int num_tokens,
    int start_pos,
    float *logits
);
```

Run forward pass for multiple tokens (prefill).

**Parameters:**
- `ctx`: Model context
- `tokens`: Input token IDs
- `num_tokens`: Number of input tokens
- `start_pos`: Starting position in sequence
- `logits`: Output logits for last token `[vocab_size]`

**Returns:** 0 on success, negative error code on failure.

---

## KV Cache Management

### gemma3_reset_cache

```c
void gemma3_reset_cache(gemma3_ctx *ctx);
```

Reset the KV cache (start fresh generation).

### gemma3_get_cache_position

```c
int gemma3_get_cache_position(gemma3_ctx *ctx);
```

Get current cache position (number of tokens processed).

---

## Utility Functions

### gemma3_version

```c
const char *gemma3_version(void);
```

Get library version string.

**Returns:** Version string (e.g., "0.1.0").

### gemma3_is_global_layer

```c
static inline int gemma3_is_global_layer(int layer_idx);
```

Check if a layer uses global attention.

**Parameters:**
- `layer_idx`: Layer index (0-based)

**Returns:** 1 if global attention, 0 if local.

**Pattern:** Global every 6th layer (indices 5, 11, 17, 23, 29).

### gemma3_layer_rope_theta

```c
static inline float gemma3_layer_rope_theta(int layer_idx);
```

Get RoPE theta for a layer.

**Returns:** 10,000.0 for local layers, 1,000,000.0 for global layers.

---

## Compute Kernels

Defined in `gemma3_kernels.h`:

### Matrix Operations

```c
// Matrix multiplication: C = A @ B
void gemma3_matmul(float *C, const float *A, const float *B, int M, int K, int N);

// Matrix-vector multiplication: y = A @ x
void gemma3_matvec(float *y, const float *A, const float *x, int M, int K);

// BF16 matrix-vector: y = A @ x (A is BF16, x and y are F32)
void gemma3_matvec_bf16(float *y, const uint16_t *A, const float *x,
                        int M, int K, float *scratch);

// Threaded BF16 matvec (requires USE_THREADS)
void gemma3_matvec_bf16_mt(float *y, const uint16_t *A, const float *x,
                           int M, int K, float *scratch,
                           gemma3_thread_pool *pool);
```

### Vector Operations

```c
void gemma3_vec_add(float *y, const float *a, const float *b, int n);
void gemma3_vec_mul(float *y, const float *a, const float *b, int n);
void gemma3_vec_scale(float *y, const float *x, float scale, int n);
void gemma3_vec_copy(float *dst, const float *src, int n);
void gemma3_vec_zero(float *x, int n);
float gemma3_vec_sum(const float *x, int n);
float gemma3_vec_max(const float *x, int n);
float gemma3_dot(const float *a, const float *b, int n);
```

### Normalization

```c
// RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight
void gemma3_rmsnorm(float *y, const float *x, const float *weight,
                    int n, float eps);
void gemma3_rmsnorm_inplace(float *x, const float *weight, int n, float eps);

// BF16 weight variants (uses Gemma's (1+weight) formula)
void gemma3_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight,
                         int n, float eps);
void gemma3_rmsnorm_bf16_inplace(float *x, const uint16_t *weight,
                                  int n, float eps);
```

### Activation Functions

```c
// GELU (tanh approximation)
void gemma3_gelu_tanh(float *y, const float *x, int n);
void gemma3_gelu_tanh_inplace(float *x, int n);

// SiLU (Swish): x * sigmoid(x)
void gemma3_silu(float *y, const float *x, int n);
void gemma3_silu_inplace(float *x, int n);

// Softmax (numerically stable)
void gemma3_softmax(float *y, const float *x, int n);
void gemma3_softmax_inplace(float *x, int n);
```

### Positional Encoding

```c
// Apply RoPE to Q and K
void gemma3_rope(float *q, float *k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta);

// Single vector RoPE
void gemma3_rope_single(float *x, int head_dim, int pos, float theta);

// Precompute RoPE frequencies
void gemma3_rope_precompute(float *freqs, int max_pos, int head_dim,
                            float theta);

// Apply precomputed RoPE
void gemma3_rope_apply_precomputed(float *x, const float *freqs,
                                    int head_dim, int pos);
```

### Attention

```c
// Single-head attention
void gemma3_attention_single(float *output, const float *q,
                             const float *k_cache, const float *v_cache,
                             int seq_len, int head_dim, float scale,
                             const float *mask);

// Grouped Query Attention
void gemma3_gqa(float *output, const float *q,
                const float *k_cache, const float *v_cache,
                int n_heads, int n_kv_heads, int seq_len, int head_dim,
                float scale, const float *mask, float *scores_buf);

// Attention masks
void gemma3_sliding_window_mask(float *mask, int query_pos, int window_size);
void gemma3_causal_mask(float *mask, int seq_len, int query_pos);
```

### Sampling

```c
// Temperature scaling
void gemma3_apply_temperature(float *logits, int vocab_size, float temperature);

// Top-k filtering
void gemma3_topk_filter(float *logits, int vocab_size, int k);

// Top-p (nucleus) filtering
void gemma3_topp_filter(float *logits, int vocab_size, float p);

// Sample from distribution
int gemma3_sample(const float *probs, int vocab_size);

// Greedy selection
int gemma3_argmax(const float *x, int n);

// Random number generation
void gemma3_set_seed(uint64_t seed);
float gemma3_random(void);  // Returns [0, 1)
```

### Data Type Conversion

```c
// BF16 <-> F32 conversion
void gemma3_bf16_to_f32(float *f32, const uint16_t *bf16, int n);
void gemma3_f32_to_bf16(uint16_t *bf16, const float *f32, int n);

// Single value conversion
static inline float gemma3_bf16_to_f32_single(uint16_t bf16);
static inline uint16_t gemma3_f32_to_bf16_single(float f32);

// Embedding lookup
void gemma3_embed_bf16(float *output, const uint16_t *embed,
                       int token_id, int hidden_size);
```

---

## Thread Pool

Defined in `gemma3_threads.h`:

### Types

```c
typedef struct gemma3_thread_pool gemma3_thread_pool;

// Task function signature
typedef void (*gemma3_task_fn)(void *arg, int thread_idx, int num_threads);
```

### Functions

```c
// Create thread pool (num_threads <= 0 for auto-detect)
gemma3_thread_pool *gemma3_thread_pool_create(int num_threads);

// Destroy thread pool
void gemma3_thread_pool_destroy(gemma3_thread_pool *pool);

// Get number of threads
int gemma3_thread_pool_size(const gemma3_thread_pool *pool);

// Run parallel task
void gemma3_thread_pool_run(gemma3_thread_pool *pool,
                            gemma3_task_fn fn, void *arg);
```

**Example:**
```c
void my_task(void *arg, int thread_idx, int num_threads) {
    int *data = (int *)arg;
    // Process portion of data based on thread_idx
    int start = thread_idx * (SIZE / num_threads);
    int end = (thread_idx + 1) * (SIZE / num_threads);
    for (int i = start; i < end; i++) {
        data[i] *= 2;
    }
}

gemma3_thread_pool *pool = gemma3_thread_pool_create(0);  // Auto-detect
int data[SIZE];
gemma3_thread_pool_run(pool, my_task, data);
gemma3_thread_pool_destroy(pool);
```

---

## Complete Usage Example

```c
#include <stdio.h>
#include <stdlib.h>
#include "gemma3.h"

// Streaming callback
int print_token(int token_id, const char *token_str, void *user_data) {
    (void)token_id;
    (void)user_data;
    printf("%s", token_str);
    fflush(stdout);
    return 0;  // Continue generation
}

int main(int argc, char **argv) {
    // Load model
    gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", gemma3_get_error());
        return 1;
    }

    // Configure generation
    gemma3_gen_params params = gemma3_default_params();
    params.max_tokens = 256;
    params.temperature = 0.7f;
    params.top_p = 0.9f;

    // Chat completion
    gemma3_message messages[] = {
        { GEMMA3_ROLE_SYSTEM, "You are a helpful coding assistant." },
        { GEMMA3_ROLE_USER, "Write a hello world program in C." },
    };

    printf("Assistant: ");
    char *response = gemma3_chat(ctx, messages, 2, &params, print_token, NULL);
    printf("\n");

    // Cleanup
    free(response);
    gemma3_free(ctx);

    return 0;
}
```

---

## Related Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Main project documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture deep dive
- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Build system details
- [INTERNALS.md](INTERNALS.md) - Implementation deep dive
