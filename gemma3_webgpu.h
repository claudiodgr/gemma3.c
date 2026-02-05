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
    WGPUBufferUsageFlags usage;
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
    WGPUComputePipeline gelu_pipeline;
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
    uint32_t _pad0;
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
 * @return 0 on success, -1 on failure
 */
int gemma3_gpu_init_buffers(gemma3_gpu_context *ctx,
                            int hidden_size,
                            int intermediate_size,
                            int vocab_size,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            int max_context);

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
                                            WGPUBufferUsageFlags usage);

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
