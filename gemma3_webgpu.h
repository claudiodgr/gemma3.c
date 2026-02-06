/*
 * gemma3_webgpu.h - WebGPU acceleration for Gemma 3 inference
 *
 * This header provides GPU-accelerated compute kernels using WebGPU.
 * Requires a WebGPU implementation (wgpu-native, Dawn, or browser).
 */

#ifndef GEMMA3_WEBGPU_H
#define GEMMA3_WEBGPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef USE_WEBGPU

/* Forward declaration for WebGPU types if not using full header */
#ifndef WGPU_SKIP_DECLARATIONS
#include <webgpu/webgpu.h>
#endif

/* ============================================================================
 * GPU Context
 * ========================================================================== */

/**
 * GPU buffer handle for managing device memory
 */
typedef struct {
    WGPUBuffer buffer;
    size_t size;
    WGPUBufferUsage usage;
} gemma3_gpu_buffer;

/**
 * GPU context holding device, queue, and compute pipelines
 */
typedef struct gemma3_gpu_context {
    /* WebGPU core objects */
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;

    /* Compute pipelines */
    WGPUComputePipeline matvec_bf16_pipeline;
    WGPUComputePipeline rmsnorm_bf16_pipeline;
    WGPUComputePipeline rmsnorm_bf16_inplace_pipeline;
    WGPUComputePipeline gelu_pipeline;
    WGPUComputePipeline gelu_mul_pipeline;
    WGPUComputePipeline softmax_pipeline;
    WGPUComputePipeline rope_pipeline;
    WGPUComputePipeline gqa_pipeline;
    WGPUComputePipeline vec_add_pipeline;
    WGPUComputePipeline vec_mul_pipeline;
    WGPUComputePipeline embed_bf16_pipeline;

    /* Shader modules */
    WGPUShaderModule shader_module;

    /* Bind group layouts */
    WGPUBindGroupLayout matvec_layout;
    WGPUBindGroupLayout rmsnorm_layout;
    WGPUBindGroupLayout gelu_layout;
    WGPUBindGroupLayout gelu_mul_layout;
    WGPUBindGroupLayout softmax_layout;
    WGPUBindGroupLayout rope_layout;
    WGPUBindGroupLayout gqa_layout;
    WGPUBindGroupLayout vec_op_layout;
    WGPUBindGroupLayout embed_layout;

    /* Staging buffers for CPU-GPU data transfer */
    gemma3_gpu_buffer staging_read;
    gemma3_gpu_buffer staging_write;

    /* Persistent buffers for activation tensors */
    gemma3_gpu_buffer buf_x;              /* [hidden_size] */
    gemma3_gpu_buffer buf_x_norm;         /* [hidden_size] */
    gemma3_gpu_buffer buf_q;              /* [num_heads * head_dim] */
    gemma3_gpu_buffer buf_k;              /* [num_kv_heads * head_dim] */
    gemma3_gpu_buffer buf_v;              /* [num_kv_heads * head_dim] */
    gemma3_gpu_buffer buf_attn_out;       /* [num_heads * head_dim] */
    gemma3_gpu_buffer buf_proj_out;       /* [hidden_size] */
    gemma3_gpu_buffer buf_mlp_gate;       /* [intermediate_size] */
    gemma3_gpu_buffer buf_mlp_up;         /* [intermediate_size] */
    gemma3_gpu_buffer buf_mlp_out;        /* [hidden_size] */
    gemma3_gpu_buffer buf_logits;         /* [vocab_size] */
    gemma3_gpu_buffer buf_attn_scores;    /* [max_context] */
    gemma3_gpu_buffer buf_mask;           /* [max_context] */

    /* Uniform buffer for shader parameters */
    gemma3_gpu_buffer buf_params;

    /* Configuration */
    int max_context;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;

    /* Workgroup sizes */
    uint32_t workgroup_size_1d;
    uint32_t workgroup_size_2d_x;
    uint32_t workgroup_size_2d_y;

    /* Synchronization */
    int pending_commands;

    /* --- Phase 1: Command Batching + Buffer Reuse --- */

    /* Command encoder manager */
    WGPUCommandEncoder active_encoder;
    WGPUComputePassEncoder active_pass;
    int encoder_open;     /* 1 if encoder created, 0 otherwise */
    int pass_open;        /* 1 if compute pass started, 0 otherwise */

    /* Uniform params ring buffer (256KB, 256-byte aligned allocations) */
    gemma3_gpu_buffer buf_params_ring;
    uint32_t params_ring_offset;
    uint32_t params_ring_size;

    /* Reusable per-layer weight buffers */
    gemma3_gpu_buffer buf_weight_large_0;  /* 50MB -- gate/down projections */
    gemma3_gpu_buffer buf_weight_large_1;  /* 50MB -- up projection */
    gemma3_gpu_buffer buf_weight_medium;   /* 10MB -- q/o projections */
    gemma3_gpu_buffer buf_weight_small_0;  /* 5MB -- k projection */
    gemma3_gpu_buffer buf_weight_small_1;  /* 5MB -- v projection */
    gemma3_gpu_buffer buf_weight_norm_0;   /* 20KB -- layernorm weights (slot 0) */
    gemma3_gpu_buffer buf_weight_norm_1;   /* 20KB -- layernorm weights (slot 1) */

    /* RMSNorm scratch buffer (persistent, avoids per-call alloc) */
    gemma3_gpu_buffer buf_rmsnorm_scratch;

    /* --- Phase 2: Full GPU Forward Pass --- */

    /* Additional weight buffers (all 13 weights per layer in separate buffers) */
    gemma3_gpu_buffer buf_weight_large_2;  /* 50MB -- down projection */
    gemma3_gpu_buffer buf_weight_medium_1; /* 10MB -- o projection */
    gemma3_gpu_buffer buf_weight_norm_2;   /* 20KB -- pre-feedforward layernorm */
    gemma3_gpu_buffer buf_weight_norm_3;   /* 20KB -- post-feedforward layernorm */
    gemma3_gpu_buffer buf_weight_qk_norm_q; /* 512B -- QK norm for Q (head_dim BF16) */
    gemma3_gpu_buffer buf_weight_qk_norm_k; /* 512B -- QK norm for K (head_dim BF16) */

    /* GPU-resident KV cache (per-layer) */
    struct {
        gemma3_gpu_buffer k;  /* [layer_max_seq * num_kv_heads * head_dim] F32 */
        gemma3_gpu_buffer v;  /* [layer_max_seq * num_kv_heads * head_dim] F32 */
        int max_seq;          /* max_context for global layers, sliding_window for local */
    } gpu_kv_cache[34];  /* GEMMA3_NUM_LAYERS = 34 */
    int gpu_kv_num_layers;

    /* Precomputed RoPE cos/sin tables */
    gemma3_gpu_buffer buf_rope_local;     /* [max_context * head_dim/2 * 2] F32 */
    gemma3_gpu_buffer buf_rope_global;    /* [max_context * head_dim/2 * 2] F32 */

    /* Phase 2 pipelines */
    WGPUComputePipeline kv_cache_write_pipeline;
    WGPUComputePipeline multi_head_rmsnorm_pipeline;
    WGPUComputePipeline rope_precomputed_pipeline;
    WGPUComputePipeline sliding_window_mask_pipeline;
    WGPUComputePipeline causal_mask_pipeline;

    /* Phase 2 bind group layouts */
    WGPUBindGroupLayout kv_cache_write_layout;
    WGPUBindGroupLayout multi_head_rmsnorm_layout;
    WGPUBindGroupLayout rope_precomputed_layout;
    WGPUBindGroupLayout mask_layout;

    /* Sliding window size for local attention layers */
    int sliding_window;

    /* Phase 3 non-aliasing in-place pipelines and layouts */
    WGPUComputePipeline vec_add_inplace_pipeline;
    WGPUComputePipeline vec_mul_inplace_pipeline;
    WGPUComputePipeline rmsnorm_bf16_inplace_v2_pipeline;
    WGPUBindGroupLayout inplace_vec_op_layout;      /* params, y(rw), b(ro) */
    WGPUBindGroupLayout rmsnorm_inplace_v2_layout;  /* params, data(rw), weight(ro), scratch(rw) */

    /* --- Phase 3: Persistent Weight Residency --- */

    /* Per-layer weight buffers (all resident on GPU, zero PCIe transfers during inference) */
    struct {
        gemma3_gpu_buffer q_proj;     /* [num_heads*head_dim, hidden_size] BF16 */
        gemma3_gpu_buffer k_proj;     /* [num_kv_heads*head_dim, hidden_size] BF16 */
        gemma3_gpu_buffer v_proj;     /* [num_kv_heads*head_dim, hidden_size] BF16 */
        gemma3_gpu_buffer o_proj;     /* [hidden_size, num_heads*head_dim] BF16 */
        gemma3_gpu_buffer gate_proj;  /* [intermediate_size, hidden_size] BF16 */
        gemma3_gpu_buffer up_proj;    /* [intermediate_size, hidden_size] BF16 */
        gemma3_gpu_buffer down_proj;  /* [hidden_size, intermediate_size] BF16 */
        gemma3_gpu_buffer input_layernorm;    /* [hidden_size] BF16 */
        gemma3_gpu_buffer post_attn_ln;       /* [hidden_size] BF16 */
        gemma3_gpu_buffer pre_ff_ln;          /* [hidden_size] BF16 */
        gemma3_gpu_buffer post_ff_ln;         /* [hidden_size] BF16 */
        gemma3_gpu_buffer q_norm;     /* [head_dim] BF16 */
        gemma3_gpu_buffer k_norm;     /* [head_dim] BF16 */
    } *layer_weights;                 /* Dynamically allocated [num_layers] */
    int num_weight_layers;            /* Number of layers with resident weights */

    gemma3_gpu_buffer buf_final_norm; /* [hidden_size] BF16 -- final RMSNorm weight */
    int weights_resident;             /* 1 = all weights on GPU, 0 = Phase 2 fallback */

} gemma3_gpu_context;

/* ============================================================================
 * Shader Parameter Structures (must match WGSL structs)
 * ========================================================================== */

/**
 * Parameters for matrix-vector multiplication
 */
typedef struct {
    uint32_t M;           /* Output dimension (rows) */
    uint32_t K;           /* Input dimension (cols) */
    uint32_t _pad0;
    uint32_t _pad1;
} gemma3_matvec_params;

/**
 * Parameters for RMSNorm
 */
typedef struct {
    uint32_t n;           /* Vector dimension */
    float eps;            /* Epsilon for numerical stability */
    uint32_t _pad0;
    uint32_t _pad1;
} gemma3_rmsnorm_params;

/**
 * Parameters for GQA attention
 */
typedef struct {
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t seq_len;
    uint32_t head_dim;
    float scale;
    uint32_t use_mask;
    uint32_t scores_stride;  /* Per-head stride into scores buffer (seq_len for per-head) */
    uint32_t _pad1;
} gemma3_gqa_params;

/**
 * Parameters for RoPE
 */
typedef struct {
    uint32_t head_dim;
    uint32_t pos;
    float theta;
    uint32_t num_heads;
} gemma3_rope_params;

/**
 * Parameters for softmax
 */
typedef struct {
    uint32_t n;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
} gemma3_softmax_params;

/**
 * Parameters for KV cache write (Phase 2)
 */
typedef struct {
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t cache_pos;       /* Position in cache (ring-buffered for local layers) */
    uint32_t _pad0;
} gemma3_kv_cache_write_params;

/**
 * Parameters for multi-head RMSNorm (Phase 2)
 */
typedef struct {
    uint32_t head_dim;
    uint32_t num_heads;
    float eps;
    uint32_t _pad0;
} gemma3_multi_head_rmsnorm_params;

/**
 * Parameters for precomputed RoPE (Phase 2)
 */
typedef struct {
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t pos;
    uint32_t _pad0;
} gemma3_rope_precomputed_params;

/**
 * Parameters for attention mask generation (Phase 2)
 */
typedef struct {
    uint32_t query_pos;
    uint32_t window_size;
    uint32_t is_causal;
    uint32_t seq_len;
} gemma3_mask_params;

/* ============================================================================
 * Initialization and Cleanup
 * ========================================================================== */

/**
 * Initialize WebGPU context
 *
 * @return GPU context on success, NULL on failure
 */
gemma3_gpu_context *gemma3_gpu_init(void);

/**
 * Initialize GPU buffers for a specific model configuration
 *
 * @param ctx GPU context
 * @param hidden_size Model hidden dimension
 * @param intermediate_size MLP intermediate dimension
 * @param vocab_size Vocabulary size
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of KV heads
 * @param head_dim Dimension per head
 * @param max_context Maximum context length
 * @param num_layers Number of transformer layers (34 for Gemma 3 4B)
 * @param sliding_window Sliding window size for local attention (1024)
 * @return 0 on success, -1 on failure
 */
int gemma3_gpu_init_buffers(gemma3_gpu_context *ctx,
                            int hidden_size,
                            int intermediate_size,
                            int vocab_size,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            int max_context,
                            int num_layers,
                            int sliding_window);

/**
 * Upload precomputed RoPE cos/sin tables to GPU (Phase 2)
 *
 * @param ctx GPU context
 * @param rope_local Local RoPE table [max_context * head_dim/2 * 2] F32
 * @param rope_global Global RoPE table [max_context * head_dim/2 * 2] F32
 */
void gemma3_gpu_upload_rope_tables(gemma3_gpu_context *ctx,
                                    const float *rope_local,
                                    const float *rope_global);

/**
 * Free GPU context and all resources
 */
void gemma3_gpu_free(gemma3_gpu_context *ctx);

/* ============================================================================
 * Buffer Management
 * ========================================================================== */

/**
 * Create a GPU buffer
 *
 * @param ctx GPU context
 * @param size Buffer size in bytes
 * @param usage Buffer usage flags
 * @return Buffer handle
 */
gemma3_gpu_buffer gemma3_gpu_create_buffer(gemma3_gpu_context *ctx,
                                            size_t size,
                                            WGPUBufferUsage usage);

/**
 * Upload data to GPU buffer
 */
void gemma3_gpu_write_buffer(gemma3_gpu_context *ctx,
                             gemma3_gpu_buffer *buf,
                             const void *data,
                             size_t size);

/**
 * Download data from GPU buffer
 */
void gemma3_gpu_read_buffer(gemma3_gpu_context *ctx,
                            gemma3_gpu_buffer *buf,
                            void *data,
                            size_t size);

/**
 * Free a GPU buffer
 */
void gemma3_gpu_destroy_buffer(gemma3_gpu_buffer *buf);

/* ============================================================================
 * Compute Kernels - GPU Accelerated
 * ========================================================================== */

/**
 * GPU matrix-vector multiplication: y = A @ x
 *
 * @param ctx GPU context
 * @param y Output buffer [M] (F32, must be on GPU)
 * @param A Weight matrix [M, K] (BF16, host memory - will be uploaded)
 * @param x Input vector [K] (F32, must be on GPU)
 * @param M Output dimension
 * @param K Input dimension
 */
void gemma3_matvec_bf16_gpu(gemma3_gpu_context *ctx,
                            gemma3_gpu_buffer *y,
                            const uint16_t *A,
                            gemma3_gpu_buffer *x,
                            int M, int K);

/**
 * GPU RMSNorm with BF16 weights: y = x * rsqrt(mean(x^2) + eps) * (1 + weight)
 *
 * @param ctx GPU context
 * @param y Output buffer [n] (F32)
 * @param x Input buffer [n] (F32)
 * @param weight Weight buffer [n] (BF16, host)
 * @param n Vector dimension
 * @param eps Epsilon for numerical stability
 */
void gemma3_rmsnorm_bf16_gpu(gemma3_gpu_context *ctx,
                             gemma3_gpu_buffer *y,
                             gemma3_gpu_buffer *x,
                             const uint16_t *weight,
                             int n, float eps);

/**
 * GPU GELU activation (tanh approximation)
 *
 * @param ctx GPU context
 * @param x Buffer to apply GELU in-place [n] (F32)
 * @param n Vector dimension
 */
void gemma3_gelu_gpu(gemma3_gpu_context *ctx,
                     gemma3_gpu_buffer *x,
                     int n);

/**
 * GPU Softmax
 *
 * @param ctx GPU context
 * @param x Buffer to apply softmax in-place [n] (F32)
 * @param n Vector dimension
 */
void gemma3_softmax_gpu(gemma3_gpu_context *ctx,
                        gemma3_gpu_buffer *x,
                        int n);

/**
 * GPU RoPE application
 *
 * @param ctx GPU context
 * @param x Vector to rotate in-place [num_heads * head_dim] (F32)
 * @param num_heads Number of heads
 * @param head_dim Dimension per head (must be even)
 * @param pos Position in sequence
 * @param theta RoPE theta parameter
 */
void gemma3_rope_gpu(gemma3_gpu_context *ctx,
                     gemma3_gpu_buffer *x,
                     int num_heads,
                     int head_dim,
                     int pos,
                     float theta);

/**
 * GPU Grouped Query Attention
 *
 * @param ctx GPU context
 * @param output Output buffer [n_heads * head_dim] (F32)
 * @param q Query buffer [n_heads * head_dim] (F32)
 * @param k_cache Key cache [seq_len, n_kv_heads, head_dim] (F32)
 * @param v_cache Value cache [seq_len, n_kv_heads, head_dim] (F32)
 * @param n_heads Number of query heads
 * @param n_kv_heads Number of KV heads
 * @param seq_len Current sequence length
 * @param head_dim Dimension per head
 * @param scale Attention scale factor
 * @param mask Attention mask (NULL for no mask)
 */
void gemma3_gqa_gpu(gemma3_gpu_context *ctx,
                    gemma3_gpu_buffer *output,
                    gemma3_gpu_buffer *q,
                    gemma3_gpu_buffer *k_cache,
                    gemma3_gpu_buffer *v_cache,
                    int n_heads, int n_kv_heads,
                    int seq_len, int head_dim,
                    float scale,
                    gemma3_gpu_buffer *mask);

/**
 * GPU vector addition: y = a + b
 */
void gemma3_vec_add_gpu(gemma3_gpu_context *ctx,
                        gemma3_gpu_buffer *y,
                        gemma3_gpu_buffer *a,
                        gemma3_gpu_buffer *b,
                        int n);

/**
 * GPU element-wise multiplication: y = a * b
 */
void gemma3_vec_mul_gpu(gemma3_gpu_context *ctx,
                        gemma3_gpu_buffer *y,
                        gemma3_gpu_buffer *a,
                        gemma3_gpu_buffer *b,
                        int n);

/**
 * GPU embedding lookup with BF16 conversion
 *
 * @param ctx GPU context
 * @param output Output buffer [hidden_size] (F32)
 * @param embed Embedding table (BF16, host)
 * @param token_id Token ID to look up
 * @param hidden_size Embedding dimension
 */
void gemma3_embed_bf16_gpu(gemma3_gpu_context *ctx,
                           gemma3_gpu_buffer *output,
                           const uint16_t *embed,
                           int token_id,
                           int hidden_size);

/* ============================================================================
 * Command Encoder Management (Phase 1: Command Batching)
 * ========================================================================== */

/**
 * Begin recording GPU commands (idempotent -- safe to call if already open)
 */
void gemma3_gpu_begin_commands(gemma3_gpu_context *ctx);

/**
 * Ensure a compute pass is open within the current encoder (internal use).
 * Calls begin_commands if no encoder exists.
 */
void gemma3_gpu_ensure_pass(gemma3_gpu_context *ctx);

/**
 * End the current compute pass, finish the encoder, and submit the command buffer.
 * Resets the params ring buffer offset.
 */
void gemma3_gpu_flush_commands(gemma3_gpu_context *ctx);

/* ============================================================================
 * Params Ring Buffer (Phase 1)
 * ========================================================================== */

/**
 * Allocate space in the params ring buffer and write uniform data.
 *
 * @param ctx GPU context
 * @param data Pointer to parameter struct data
 * @param size Size of the parameter data in bytes
 * @return Byte offset into buf_params_ring where data was written (256-byte aligned)
 */
uint32_t gemma3_gpu_alloc_params(gemma3_gpu_context *ctx, const void *data, uint32_t size);

/* ============================================================================
 * Batched Kernel Dispatch (Phase 1: record into active pass, no sync)
 * ========================================================================== */

/**
 * Batched matvec: records dispatch into active pass.
 * Weight buffer must already be uploaded (reusable weight buffer).
 *
 * @param ctx GPU context
 * @param y Output buffer [M] F32
 * @param A_buf GPU buffer with weight matrix [M,K] BF16 (pre-uploaded)
 * @param A_size Size of weight data in bytes
 * @param x Input buffer [K] F32
 * @param M Output dimension
 * @param K Input dimension
 */
void gemma3_matvec_bf16_dispatch_gpu(gemma3_gpu_context *ctx,
                                      gemma3_gpu_buffer *y,
                                      gemma3_gpu_buffer *A_buf,
                                      size_t A_size,
                                      gemma3_gpu_buffer *x,
                                      int M, int K);

/**
 * Batched RMSNorm: records dispatch into active pass.
 * Weight buffer must already contain norm weights.
 */
void gemma3_rmsnorm_bf16_dispatch_gpu(gemma3_gpu_context *ctx,
                                       gemma3_gpu_buffer *y,
                                       gemma3_gpu_buffer *x,
                                       gemma3_gpu_buffer *weight_buf,
                                       size_t weight_size,
                                       int n, float eps);

/**
 * Batched in-place RMSNorm: records dispatch into active pass.
 * Reads and writes to the same buffer (y). Weight buffer pre-uploaded.
 */
void gemma3_rmsnorm_bf16_inplace_dispatch_gpu(gemma3_gpu_context *ctx,
                                               gemma3_gpu_buffer *y,
                                               gemma3_gpu_buffer *weight_buf,
                                               size_t weight_size,
                                               int n, float eps);

/**
 * Batched GELU: records dispatch into active pass.
 */
void gemma3_gelu_dispatch_gpu(gemma3_gpu_context *ctx,
                               gemma3_gpu_buffer *x,
                               int n);

/**
 * Fused GELU + multiply: gate[i] = gelu(gate[i]) * up[i]
 * Replaces separate GELU + vec_mul dispatches.
 */
void gemma3_gelu_mul_dispatch_gpu(gemma3_gpu_context *ctx,
                                    gemma3_gpu_buffer *gate,
                                    gemma3_gpu_buffer *up,
                                    int n);

/**
 * Batched vec_add: y = a + b, records dispatch into active pass.
 */
void gemma3_vec_add_dispatch_gpu(gemma3_gpu_context *ctx,
                                  gemma3_gpu_buffer *y,
                                  gemma3_gpu_buffer *a,
                                  gemma3_gpu_buffer *b,
                                  int n);

/**
 * Batched vec_mul: y = a * b, records dispatch into active pass.
 */
void gemma3_vec_mul_dispatch_gpu(gemma3_gpu_context *ctx,
                                  gemma3_gpu_buffer *y,
                                  gemma3_gpu_buffer *a,
                                  gemma3_gpu_buffer *b,
                                  int n);

/**
 * Batched RoPE: records dispatch into active pass.
 */
void gemma3_rope_dispatch_gpu(gemma3_gpu_context *ctx,
                               gemma3_gpu_buffer *x,
                               int num_heads,
                               int head_dim,
                               int pos,
                               float theta);

/**
 * Batched GQA: records dispatch into active pass.
 */
void gemma3_gqa_dispatch_gpu(gemma3_gpu_context *ctx,
                              gemma3_gpu_buffer *output,
                              gemma3_gpu_buffer *q,
                              gemma3_gpu_buffer *k_cache,
                              gemma3_gpu_buffer *v_cache,
                              int n_heads, int n_kv_heads,
                              int seq_len, int head_dim,
                              float scale,
                              gemma3_gpu_buffer *mask);

/* ============================================================================
 * Phase 2 Batched Dispatch Functions (GPU-resident KV cache, all ops on GPU)
 * ========================================================================== */

/**
 * Write K/V vectors to GPU-resident KV cache at the given position.
 */
void gemma3_kv_cache_write_dispatch_gpu(gemma3_gpu_context *ctx,
                                         gemma3_gpu_buffer *k_in,
                                         gemma3_gpu_buffer *v_in,
                                         gemma3_gpu_buffer *k_cache,
                                         gemma3_gpu_buffer *v_cache,
                                         int num_kv_heads, int head_dim,
                                         int cache_pos);

/**
 * Per-head RMSNorm (in-place) for QK normalization.
 * Each head is independently normalized using the same weight vector.
 */
void gemma3_multi_head_rmsnorm_dispatch_gpu(gemma3_gpu_context *ctx,
                                             gemma3_gpu_buffer *x,
                                             gemma3_gpu_buffer *weight_buf,
                                             size_t weight_size,
                                             int head_dim, int num_heads,
                                             float eps);

/**
 * RoPE using precomputed cos/sin table (avoids trig computation).
 * @param rope_table buf_rope_local or buf_rope_global
 */
void gemma3_rope_precomputed_dispatch_gpu(gemma3_gpu_context *ctx,
                                           gemma3_gpu_buffer *x,
                                           gemma3_gpu_buffer *rope_table,
                                           int num_heads, int head_dim,
                                           int pos);

/**
 * Generate attention mask on GPU (causal or sliding window).
 */
void gemma3_mask_dispatch_gpu(gemma3_gpu_context *ctx,
                               gemma3_gpu_buffer *mask_out,
                               int query_pos, int seq_len,
                               int window_size, int is_causal);

/* ============================================================================
 * Phase 3 Non-Aliasing In-Place Dispatch Functions
 *
 * These avoid the WebGPU validation error where the same buffer is bound
 * as both STORAGE_READ_ONLY (input) and STORAGE_READ_WRITE (output).
 * ========================================================================== */

/**
 * In-place vector addition: y[i] += b[i]
 * No aliasing -- y is only bound as read-write, b as read-only.
 */
void gemma3_vec_add_inplace_dispatch_gpu(gemma3_gpu_context *ctx,
                                          gemma3_gpu_buffer *y,
                                          gemma3_gpu_buffer *b,
                                          int n);

/**
 * In-place vector multiplication: y[i] *= b[i]
 * No aliasing -- y is only bound as read-write, b as read-only.
 */
void gemma3_vec_mul_inplace_dispatch_gpu(gemma3_gpu_context *ctx,
                                          gemma3_gpu_buffer *y,
                                          gemma3_gpu_buffer *b,
                                          int n);

/**
 * In-place RMSNorm (non-aliasing): data = rmsnorm(data, weight)
 * data is bound once as read-write. No dual-binding aliasing.
 */
void gemma3_rmsnorm_bf16_inplace_v2_dispatch_gpu(gemma3_gpu_context *ctx,
                                                   gemma3_gpu_buffer *data,
                                                   gemma3_gpu_buffer *weight_buf,
                                                   size_t weight_size,
                                                   int n, float eps);

/* ============================================================================
 * Synchronization
 * ========================================================================== */

/**
 * Submit pending GPU commands and wait for completion
 */
void gemma3_gpu_sync(gemma3_gpu_context *ctx);

/**
 * Submit pending GPU commands without waiting
 */
void gemma3_gpu_submit(gemma3_gpu_context *ctx);

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

/**
 * Check if WebGPU is available on this system
 *
 * @return 1 if available, 0 otherwise
 */
int gemma3_gpu_available(void);

/**
 * Get GPU device name
 */
const char *gemma3_gpu_device_name(gemma3_gpu_context *ctx);

/**
 * Get last GPU error message
 */
const char *gemma3_gpu_get_error(void);

#endif /* USE_WEBGPU */

#endif /* GEMMA3_WEBGPU_H */
