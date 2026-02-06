/*
 * gemma3_transformer.c - Transformer forward pass implementation
 *
 * Implements the Gemma 3 transformer architecture:
 * - Grouped Query Attention (GQA) with 8 Q heads and 4 KV heads
 * - Hybrid local/global attention (5:1 ratio)
 * - SwiGLU MLP
 * - RMSNorm with additional pre/post feedforward norms
 * - RoPE with layer-specific theta
 */

#include "gemma3.h"
#include "gemma3_kernels.h"
#ifdef USE_THREADS
#include "gemma3_threads.h"
#endif
#ifdef USE_WEBGPU
#include "gemma3_webgpu.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Helper to dispatch matvec_bf16 to WebGPU, threaded, or single-threaded path.
 *
 * Priority: WebGPU > Threads > Single-threaded
 *
 * Note: For WebGPU, the 'pool' parameter is reused to pass the GPU context.
 * The caller should set pool to either a gemma3_thread_pool* or gemma3_gpu_context*
 * depending on the build configuration.
 */
static inline void matvec_bf16_dispatch(float *y, const uint16_t *A, const float *x,
                                         int M, int K, float *scratch, void *pool) {
#ifdef USE_THREADS
    if (pool) {
        gemma3_matvec_bf16_mt(y, A, x, M, K, scratch, (gemma3_thread_pool *)pool);
        return;
    }
#endif
    (void)pool;
    gemma3_matvec_bf16(y, A, x, M, K, scratch);
}

/* ============================================================================
 * Internal Structures (shared with gemma3.c)
 * ========================================================================== */

/* Forward declarations from safetensors - BF16 weights */
typedef struct {
    const uint16_t *embed_tokens;
    struct {
        const uint16_t *input_layernorm;
        const uint16_t *q_proj;
        const uint16_t *k_proj;
        const uint16_t *v_proj;
        const uint16_t *o_proj;
        const uint16_t *q_norm;  /* QK normalization */
        const uint16_t *k_norm;  /* QK normalization */
        const uint16_t *post_attention_layernorm;
        const uint16_t *gate_proj;
        const uint16_t *up_proj;
        const uint16_t *down_proj;
        const uint16_t *pre_feedforward_layernorm;
        const uint16_t *post_feedforward_layernorm;
    } layers[GEMMA3_NUM_LAYERS];
    const uint16_t *norm;
} gemma3_weights_t;

/* KV Cache for a single layer */
typedef struct {
    float *k;  /* [max_seq, num_kv_heads, head_dim] */
    float *v;  /* [max_seq, num_kv_heads, head_dim] */
    int pos;   /* Current position (for ring buffer on local layers) */
} layer_kv_cache;

/* Full KV cache */
struct gemma3_kv_cache {
    layer_kv_cache layers[GEMMA3_NUM_LAYERS];
    int max_seq;
    int current_pos;  /* Global sequence position */
};

/* ============================================================================
 * Activation Buffers
 * ========================================================================== */

typedef struct {
    float *x;           /* [hidden_size] - current hidden state */
    float *x_norm;      /* [hidden_size] - normalized hidden state */
    float *q;           /* [num_heads * head_dim] - query */
    float *k;           /* [num_kv_heads * head_dim] - key */
    float *v;           /* [num_kv_heads * head_dim] - value */
    float *attn_out;    /* [num_heads * head_dim] - attention output */
    float *proj_out;    /* [hidden_size] - projection output */
    float *mlp_gate;    /* [intermediate_size] - MLP gate */
    float *mlp_up;      /* [intermediate_size] - MLP up */
    float *mlp_out;     /* [hidden_size] - MLP output */
    float *logits;      /* [vocab_size] - output logits */
    float *mask;        /* [max_seq] - attention mask */
    float *attn_scores; /* [max_context] - pre-allocated attention scores */
    float *matvec_tmp;  /* [vocab_size] - scratch buffer for BF16 matvec BLAS path */
} activation_buffers;

static activation_buffers *alloc_buffers(const gemma3_config *cfg) {
    activation_buffers *buf = (activation_buffers *)calloc(1, sizeof(activation_buffers));
    if (!buf) return NULL;

    buf->x = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->x_norm = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->q = (float *)malloc(cfg->num_heads * cfg->head_dim * sizeof(float));
    buf->k = (float *)malloc(cfg->num_kv_heads * cfg->head_dim * sizeof(float));
    buf->v = (float *)malloc(cfg->num_kv_heads * cfg->head_dim * sizeof(float));
    buf->attn_out = (float *)malloc(cfg->num_heads * cfg->head_dim * sizeof(float));
    buf->proj_out = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->mlp_gate = (float *)malloc(cfg->intermediate_size * sizeof(float));
    buf->mlp_up = (float *)malloc(cfg->intermediate_size * sizeof(float));
    buf->mlp_out = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->logits = (float *)malloc(cfg->vocab_size * sizeof(float));
    buf->mask = (float *)malloc(cfg->max_context * sizeof(float));
    buf->attn_scores = (float *)malloc(cfg->max_context * sizeof(float));
    /* vocab_size is the largest row dimension for matvec; reuse for BF16 conversion */
    int matvec_max = cfg->vocab_size > cfg->intermediate_size ? cfg->vocab_size : cfg->intermediate_size;
    buf->matvec_tmp = (float *)malloc(matvec_max * sizeof(float));

    if (!buf->x || !buf->x_norm || !buf->q || !buf->k || !buf->v ||
        !buf->attn_out || !buf->proj_out || !buf->mlp_gate || !buf->mlp_up ||
        !buf->mlp_out || !buf->logits || !buf->mask || !buf->attn_scores ||
        !buf->matvec_tmp) {
        free(buf->x);
        free(buf->x_norm);
        free(buf->q);
        free(buf->k);
        free(buf->v);
        free(buf->attn_out);
        free(buf->proj_out);
        free(buf->mlp_gate);
        free(buf->mlp_up);
        free(buf->mlp_out);
        free(buf->logits);
        free(buf->mask);
        free(buf->attn_scores);
        free(buf->matvec_tmp);
        free(buf);
        return NULL;
    }

    return buf;
}

static void free_buffers(activation_buffers *buf) {
    if (!buf) return;
    free(buf->x);
    free(buf->x_norm);
    free(buf->q);
    free(buf->k);
    free(buf->v);
    free(buf->attn_out);
    free(buf->proj_out);
    free(buf->mlp_gate);
    free(buf->mlp_up);
    free(buf->mlp_out);
    free(buf->logits);
    free(buf->mask);
    free(buf->attn_scores);
    free(buf->matvec_tmp);
    free(buf);
}

/* ============================================================================
 * KV Cache Management
 * ========================================================================== */

gemma3_kv_cache *gemma3_kv_cache_alloc(const gemma3_config *cfg, int max_seq) {
    gemma3_kv_cache *cache = (gemma3_kv_cache *)calloc(1, sizeof(gemma3_kv_cache));
    if (!cache) return NULL;

    cache->max_seq = max_seq;
    cache->current_pos = 0;

    int kv_size = cfg->num_kv_heads * cfg->head_dim;

    for (int l = 0; l < cfg->num_layers; l++) {
        /* For local layers with sliding window, we only need window_size entries */
        int layer_max_seq;
        if (gemma3_is_global_layer(l)) {
            layer_max_seq = max_seq;  /* Global: full context */
        } else {
            layer_max_seq = cfg->sliding_window;  /* Local: ring buffer */
        }

        cache->layers[l].k = (float *)calloc(layer_max_seq * kv_size, sizeof(float));
        cache->layers[l].v = (float *)calloc(layer_max_seq * kv_size, sizeof(float));
        cache->layers[l].pos = 0;

        if (!cache->layers[l].k || !cache->layers[l].v) {
            /* Cleanup on failure */
            for (int j = 0; j <= l; j++) {
                free(cache->layers[j].k);
                free(cache->layers[j].v);
            }
            free(cache);
            return NULL;
        }
    }

    return cache;
}

void gemma3_kv_cache_free(gemma3_kv_cache *cache) {
    if (!cache) return;

    for (int l = 0; l < GEMMA3_NUM_LAYERS; l++) {
        free(cache->layers[l].k);
        free(cache->layers[l].v);
    }
    free(cache);
}

void gemma3_kv_cache_reset(gemma3_kv_cache *cache) {
    if (!cache) return;

    cache->current_pos = 0;
    for (int l = 0; l < GEMMA3_NUM_LAYERS; l++) {
        cache->layers[l].pos = 0;
    }
}

/* Add KV to cache for a layer */
static void cache_kv(layer_kv_cache *cache, const float *k, const float *v,
                     int kv_size, int is_global, int sliding_window, int pos) {
    int cache_pos;

    if (is_global) {
        /* Global layer: simple append */
        cache_pos = pos;
    } else {
        /* Local layer: ring buffer */
        cache_pos = pos % sliding_window;
    }

    memcpy(cache->k + cache_pos * kv_size, k, kv_size * sizeof(float));
    memcpy(cache->v + cache_pos * kv_size, v, kv_size * sizeof(float));
    cache->pos = pos + 1;
}

/* ============================================================================
 * Attention Implementation
 * ========================================================================== */

/* Compute attention for a single layer */
static void layer_attention(
    float *output,           /* [hidden_size] */
    const float *x,          /* [hidden_size] - input */
    const uint16_t *q_weight,   /* [num_heads * head_dim, hidden_size] BF16 */
    const uint16_t *k_weight,   /* [num_kv_heads * head_dim, hidden_size] BF16 */
    const uint16_t *v_weight,   /* [num_kv_heads * head_dim, hidden_size] BF16 */
    const uint16_t *o_weight,   /* [hidden_size, num_heads * head_dim] BF16 */
    const uint16_t *q_norm,     /* [head_dim] BF16 - QK normalization */
    const uint16_t *k_norm,     /* [head_dim] BF16 - QK normalization */
    layer_kv_cache *cache,
    float *q_buf,            /* [num_heads * head_dim] */
    float *k_buf,            /* [num_kv_heads * head_dim] */
    float *v_buf,            /* [num_kv_heads * head_dim] */
    float *attn_buf,         /* [num_heads * head_dim] */
    float *mask_buf,         /* [max_seq] */
    float *scores_buf,       /* [max_seq] - pre-allocated attention scores */
    float *matvec_tmp,       /* scratch buffer for BF16 matvec */
    const float *rope_freqs, /* precomputed RoPE cos/sin table */
    void *thread_pool,       /* gemma3_thread_pool* or NULL */
    const gemma3_config *cfg,
    int layer_idx,
    int pos
) {
    int num_heads = cfg->num_heads;
    int num_kv_heads = cfg->num_kv_heads;
    int head_dim = cfg->head_dim;
    int hidden_size = cfg->hidden_size;

    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;

    /* Project Q, K, V (BF16 weights) */
    matvec_bf16_dispatch(q_buf, q_weight, x, q_size, hidden_size, matvec_tmp, thread_pool);
    matvec_bf16_dispatch(k_buf, k_weight, x, kv_size, hidden_size, matvec_tmp, thread_pool);
    matvec_bf16_dispatch(v_buf, v_weight, x, kv_size, hidden_size, matvec_tmp, thread_pool);

    /* Apply QK normalization (per-head RMSNorm with BF16 weights) */
    if (q_norm && k_norm) {
        for (int h = 0; h < num_heads; h++) {
            gemma3_rmsnorm_bf16(q_buf + h * head_dim, q_buf + h * head_dim,
                                q_norm, head_dim, cfg->rmsnorm_eps);
        }
        for (int h = 0; h < num_kv_heads; h++) {
            gemma3_rmsnorm_bf16(k_buf + h * head_dim, k_buf + h * head_dim,
                                k_norm, head_dim, cfg->rmsnorm_eps);
        }
    }

    /* Apply RoPE using precomputed cos/sin tables */
    for (int h = 0; h < num_heads; h++) {
        gemma3_rope_apply_precomputed(q_buf + h * head_dim, rope_freqs, head_dim, pos);
    }
    for (int h = 0; h < num_kv_heads; h++) {
        gemma3_rope_apply_precomputed(k_buf + h * head_dim, rope_freqs, head_dim, pos);
    }

    /* Add K, V to cache */
    int is_global = gemma3_is_global_layer(layer_idx);
    cache_kv(cache, k_buf, v_buf, kv_size, is_global, cfg->sliding_window, pos);

    /* Determine attention range */
    int seq_len;
    const float *k_cache, *v_cache;

    if (is_global) {
        /* Global attention: attend to all previous positions */
        seq_len = pos + 1;
        k_cache = cache->k;
        v_cache = cache->v;

        /* Causal mask */
        gemma3_causal_mask(mask_buf, seq_len, pos);
    } else {
        /* Local attention: sliding window */
        int window = cfg->sliding_window;
        int start_pos = (pos >= window) ? pos - window + 1 : 0;
        seq_len = pos - start_pos + 1;

        /* For ring buffer, we need to handle wraparound */
        /* Simplified: just use the cached entries that are valid */
        seq_len = (pos < window) ? pos + 1 : window;
        k_cache = cache->k;
        v_cache = cache->v;

        /* Sliding window mask */
        for (int i = 0; i < seq_len; i++) {
            mask_buf[i] = 0.0f;  /* All positions in window are valid */
        }
    }

    /* Compute scaled dot-product attention with GQA */
    float scale = 1.0f / sqrtf((float)head_dim);
    gemma3_gqa(attn_buf, q_buf, k_cache, v_cache,
               num_heads, num_kv_heads, seq_len, head_dim,
               scale, mask_buf, scores_buf);

    /* Output projection (BF16 weights) */
    matvec_bf16_dispatch(output, o_weight, attn_buf, hidden_size, q_size, matvec_tmp, thread_pool);
}

/* ============================================================================
 * MLP Implementation (SwiGLU)
 * ========================================================================== */

static void layer_mlp(
    float *output,            /* [hidden_size] */
    const float *x,           /* [hidden_size] */
    const uint16_t *gate_weight, /* [intermediate_size, hidden_size] BF16 */
    const uint16_t *up_weight,   /* [intermediate_size, hidden_size] BF16 */
    const uint16_t *down_weight, /* [hidden_size, intermediate_size] BF16 */
    float *gate_buf,          /* [intermediate_size] */
    float *up_buf,            /* [intermediate_size] */
    float *matvec_tmp,        /* scratch buffer for BF16 matvec */
    void *thread_pool,        /* gemma3_thread_pool* or NULL */
    const gemma3_config *cfg,
    int layer_idx,
    int pos
) {
    int hidden_size = cfg->hidden_size;
    int intermediate_size = cfg->intermediate_size;

    /* Gate and up projections (BF16 weights) */
    matvec_bf16_dispatch(gate_buf, gate_weight, x, intermediate_size, hidden_size, matvec_tmp, thread_pool);
    matvec_bf16_dispatch(up_buf, up_weight, x, intermediate_size, hidden_size, matvec_tmp, thread_pool);

    /* SwiGLU: gate = SiLU(gate) * up */
    /* Gemma 3 uses GELU instead of SiLU for the gate */
    gemma3_gelu_tanh_inplace(gate_buf, intermediate_size);
    gemma3_vec_mul(gate_buf, gate_buf, up_buf, intermediate_size);

    /* Down projection (BF16 weights) */
    matvec_bf16_dispatch(output, down_weight, gate_buf, hidden_size, intermediate_size, matvec_tmp, thread_pool);

    (void)layer_idx;
    (void)pos;
}

/* ============================================================================
 * Phase 3: Persistent Weight Upload
 *
 * Upload all model weights to GPU VRAM at init time (~6GB for Gemma 3 4B).
 * 13 buffers per layer x 34 layers = 442 persistent GPU buffers.
 * Falls back to Phase 2 (per-layer upload) if allocation fails.
 * ========================================================================== */

#ifdef USE_WEBGPU
static int upload_weights_to_gpu(gemma3_gpu_context *gpu,
                                  const gemma3_weights_t *weights,
                                  const gemma3_config *cfg) {
    int num_layers = cfg->num_layers;
    int hidden_size = cfg->hidden_size;
    int intermediate_size = cfg->intermediate_size;
    int num_heads = cfg->num_heads;
    int num_kv_heads = cfg->num_kv_heads;
    int head_dim = cfg->head_dim;

    size_t q_weight_bytes = (size_t)(num_heads * head_dim) * hidden_size * sizeof(uint16_t);
    size_t kv_weight_bytes = (size_t)(num_kv_heads * head_dim) * hidden_size * sizeof(uint16_t);
    size_t o_weight_bytes = (size_t)hidden_size * (num_heads * head_dim) * sizeof(uint16_t);
    size_t gate_weight_bytes = (size_t)intermediate_size * hidden_size * sizeof(uint16_t);
    size_t norm_weight_bytes = (size_t)hidden_size * sizeof(uint16_t);
    size_t qk_norm_bytes = (size_t)head_dim * sizeof(uint16_t);

    WGPUBufferUsage storage_read = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

    /* Allocate layer_weights array */
    gpu->layer_weights = (void *)calloc(num_layers, sizeof(gpu->layer_weights[0]));
    if (!gpu->layer_weights) {
        fprintf(stderr, "Phase 3: Failed to allocate layer_weights array\n");
        return -1;
    }

    gpu->num_weight_layers = 0;
    gpu->weights_resident = 0;

    size_t total_uploaded = 0;

    /* Allocate and upload per-layer weights */
    for (int l = 0; l < num_layers; l++) {
        /* Create all 13 buffers */
        gpu->layer_weights[l].q_proj = gemma3_gpu_create_buffer(gpu, q_weight_bytes, storage_read);
        gpu->layer_weights[l].k_proj = gemma3_gpu_create_buffer(gpu, kv_weight_bytes, storage_read);
        gpu->layer_weights[l].v_proj = gemma3_gpu_create_buffer(gpu, kv_weight_bytes, storage_read);
        gpu->layer_weights[l].o_proj = gemma3_gpu_create_buffer(gpu, o_weight_bytes, storage_read);
        gpu->layer_weights[l].gate_proj = gemma3_gpu_create_buffer(gpu, gate_weight_bytes, storage_read);
        gpu->layer_weights[l].up_proj = gemma3_gpu_create_buffer(gpu, gate_weight_bytes, storage_read);
        gpu->layer_weights[l].down_proj = gemma3_gpu_create_buffer(gpu, gate_weight_bytes, storage_read);
        gpu->layer_weights[l].input_layernorm = gemma3_gpu_create_buffer(gpu, norm_weight_bytes, storage_read);
        gpu->layer_weights[l].post_attn_ln = gemma3_gpu_create_buffer(gpu, norm_weight_bytes, storage_read);
        gpu->layer_weights[l].pre_ff_ln = gemma3_gpu_create_buffer(gpu, norm_weight_bytes, storage_read);
        gpu->layer_weights[l].post_ff_ln = gemma3_gpu_create_buffer(gpu, norm_weight_bytes, storage_read);
        gpu->layer_weights[l].q_norm = gemma3_gpu_create_buffer(gpu, qk_norm_bytes, storage_read);
        gpu->layer_weights[l].k_norm = gemma3_gpu_create_buffer(gpu, qk_norm_bytes, storage_read);

        /* Check all buffers were created */
        if (!gpu->layer_weights[l].q_proj.buffer || !gpu->layer_weights[l].k_proj.buffer ||
            !gpu->layer_weights[l].v_proj.buffer || !gpu->layer_weights[l].o_proj.buffer ||
            !gpu->layer_weights[l].gate_proj.buffer || !gpu->layer_weights[l].up_proj.buffer ||
            !gpu->layer_weights[l].down_proj.buffer || !gpu->layer_weights[l].input_layernorm.buffer ||
            !gpu->layer_weights[l].post_attn_ln.buffer || !gpu->layer_weights[l].pre_ff_ln.buffer ||
            !gpu->layer_weights[l].post_ff_ln.buffer || !gpu->layer_weights[l].q_norm.buffer ||
            !gpu->layer_weights[l].k_norm.buffer) {
            fprintf(stderr, "Phase 3: GPU buffer allocation failed at layer %d, falling back to Phase 2\n", l);
            /* Clean up partially allocated layer */
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].q_proj);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].k_proj);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].v_proj);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].o_proj);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].gate_proj);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].up_proj);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].down_proj);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].input_layernorm);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].post_attn_ln);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].pre_ff_ln);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].post_ff_ln);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].q_norm);
            gemma3_gpu_destroy_buffer(&gpu->layer_weights[l].k_norm);
            /* Clean up previously completed layers */
            for (int j = 0; j < l; j++) {
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].q_proj);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].k_proj);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].v_proj);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].o_proj);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].gate_proj);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].up_proj);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].down_proj);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].input_layernorm);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].post_attn_ln);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].pre_ff_ln);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].post_ff_ln);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].q_norm);
                gemma3_gpu_destroy_buffer(&gpu->layer_weights[j].k_norm);
            }
            free(gpu->layer_weights);
            gpu->layer_weights = NULL;
            gpu->num_weight_layers = 0;
            return -1;
        }

        /* Upload weight data */
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].q_proj,
                                weights->layers[l].q_proj, q_weight_bytes);
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].k_proj,
                                weights->layers[l].k_proj, kv_weight_bytes);
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].v_proj,
                                weights->layers[l].v_proj, kv_weight_bytes);
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].o_proj,
                                weights->layers[l].o_proj, o_weight_bytes);
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].gate_proj,
                                weights->layers[l].gate_proj, gate_weight_bytes);
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].up_proj,
                                weights->layers[l].up_proj, gate_weight_bytes);
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].down_proj,
                                weights->layers[l].down_proj, gate_weight_bytes);
        gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].input_layernorm,
                                weights->layers[l].input_layernorm, norm_weight_bytes);
        if (weights->layers[l].post_attention_layernorm)
            gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].post_attn_ln,
                                    weights->layers[l].post_attention_layernorm, norm_weight_bytes);
        if (weights->layers[l].pre_feedforward_layernorm)
            gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].pre_ff_ln,
                                    weights->layers[l].pre_feedforward_layernorm, norm_weight_bytes);
        if (weights->layers[l].post_feedforward_layernorm)
            gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].post_ff_ln,
                                    weights->layers[l].post_feedforward_layernorm, norm_weight_bytes);
        if (weights->layers[l].q_norm)
            gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].q_norm,
                                    weights->layers[l].q_norm, qk_norm_bytes);
        if (weights->layers[l].k_norm)
            gemma3_gpu_write_buffer(gpu, &gpu->layer_weights[l].k_norm,
                                    weights->layers[l].k_norm, qk_norm_bytes);

        gpu->num_weight_layers = l + 1;
        total_uploaded += q_weight_bytes + kv_weight_bytes * 2 + o_weight_bytes +
                          gate_weight_bytes * 3 + norm_weight_bytes * 4 + qk_norm_bytes * 2;
    }

    /* Upload final RMSNorm weight */
    gpu->buf_final_norm = gemma3_gpu_create_buffer(gpu, norm_weight_bytes, storage_read);
    if (!gpu->buf_final_norm.buffer) {
        fprintf(stderr, "Phase 3: Failed to allocate final norm buffer, falling back to Phase 2\n");
        return -1;
    }
    gemma3_gpu_write_buffer(gpu, &gpu->buf_final_norm, weights->norm, norm_weight_bytes);
    total_uploaded += norm_weight_bytes;

    /* Wait for all uploads to complete */
    wgpuDevicePoll(gpu->device, 1, NULL);

    gpu->weights_resident = 1;
    fprintf(stderr, "Phase 3: Uploaded %.1f MB of weights to GPU (%d layers)\n",
            (double)total_uploaded / (1024.0 * 1024.0), num_layers);
    return 0;
}
#endif /* USE_WEBGPU */

/* ============================================================================
 * GPU-Optimized Forward Pass
 *
 * Phase 3 (weights_resident=1): Single-submit, 1 sync per token.
 *   All ~750 dispatches recorded into 1 command encoder. No weight transfers.
 *   Params ring buffer holds all dispatches (749 x 256B = 192KB < 256KB).
 *
 * Phase 2 fallback (weights_resident=0): 1 sync per layer = 35 syncs/token.
 *   Uploads 13 weight buffers per layer, then dispatches ~22 ops.
 *
 * Hidden state stays on GPU between layers. KV cache is GPU-resident.
 * Embedding on CPU (upload once). Final logit projection on CPU (readback once).
 * ========================================================================== */

#ifdef USE_WEBGPU
static int gemma3_transformer_forward_gpu(
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
    gemma3_gpu_context *gpu,
    void *thread_pool
) {
    int hidden_size = cfg->hidden_size;
    int intermediate_size = cfg->intermediate_size;
    int vocab_size = cfg->vocab_size;
    int num_heads = cfg->num_heads;
    int num_kv_heads = cfg->num_kv_heads;
    int head_dim = cfg->head_dim;
    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;

    size_t q_weight_bytes = (size_t)q_size * hidden_size * sizeof(uint16_t);
    size_t kv_weight_bytes = (size_t)kv_size * hidden_size * sizeof(uint16_t);
    size_t o_weight_bytes = (size_t)hidden_size * q_size * sizeof(uint16_t);
    size_t gate_weight_bytes = (size_t)intermediate_size * hidden_size * sizeof(uint16_t);
    size_t norm_weight_bytes = (size_t)hidden_size * sizeof(uint16_t);
    size_t qk_norm_bytes = (size_t)head_dim * sizeof(uint16_t);

    (void)rope_freqs_local;
    (void)rope_freqs_global;

    /* Token embedding lookup (BF16, CPU) then upload once */
    gemma3_embed_bf16(buf->x, weights->embed_tokens, token_id, hidden_size);
    float embed_scale = sqrtf((float)hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        buf->x[i] *= embed_scale;
    }
    gemma3_gpu_write_buffer(gpu, &gpu->buf_x, buf->x, hidden_size * sizeof(float));

    if (gpu->weights_resident) {
        /* ==============================================================
         * Phase 3: Single-submit forward pass (1 sync per token)
         *
         * All weights are persistent on GPU. Record ALL layer dispatches
         * into one command buffer, then submit once.
         * ~749 dispatches, 1 submit, 1 sync, 1 readback.
         * ============================================================== */

        gemma3_gpu_begin_commands(gpu);

        for (int l = 0; l < cfg->num_layers; l++) {
            int is_global = gemma3_is_global_layer(l);
            int cache_pos = is_global ? pos : (pos % cfg->sliding_window);
            int seq_len = is_global ? (pos + 1) : ((pos < cfg->sliding_window) ? pos + 1 : cfg->sliding_window);
            float scale = 1.0f / sqrtf((float)head_dim);

            /* Persistent weight references (no upload needed) */
            gemma3_gpu_buffer *w_input_ln = &gpu->layer_weights[l].input_layernorm;
            gemma3_gpu_buffer *w_q = &gpu->layer_weights[l].q_proj;
            gemma3_gpu_buffer *w_k = &gpu->layer_weights[l].k_proj;
            gemma3_gpu_buffer *w_v = &gpu->layer_weights[l].v_proj;
            gemma3_gpu_buffer *w_o = &gpu->layer_weights[l].o_proj;
            gemma3_gpu_buffer *w_qn = &gpu->layer_weights[l].q_norm;
            gemma3_gpu_buffer *w_kn = &gpu->layer_weights[l].k_norm;
            gemma3_gpu_buffer *w_post_attn = &gpu->layer_weights[l].post_attn_ln;
            gemma3_gpu_buffer *w_pre_ff = &gpu->layer_weights[l].pre_ff_ln;
            gemma3_gpu_buffer *w_gate = &gpu->layer_weights[l].gate_proj;
            gemma3_gpu_buffer *w_up = &gpu->layer_weights[l].up_proj;
            gemma3_gpu_buffer *w_down = &gpu->layer_weights[l].down_proj;
            gemma3_gpu_buffer *w_post_ff = &gpu->layer_weights[l].post_ff_ln;

            /* --- Attention block --- */

            /* 1. Pre-attention RMSNorm */
            gemma3_rmsnorm_bf16_dispatch_gpu(gpu, &gpu->buf_x_norm, &gpu->buf_x,
                                              w_input_ln, norm_weight_bytes,
                                              hidden_size, cfg->rmsnorm_eps);

            /* 2-4. QKV projections */
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_q, w_q,
                                             q_weight_bytes, &gpu->buf_x_norm,
                                             q_size, hidden_size);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_k, w_k,
                                             kv_weight_bytes, &gpu->buf_x_norm,
                                             kv_size, hidden_size);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_v, w_v,
                                             kv_weight_bytes, &gpu->buf_x_norm,
                                             kv_size, hidden_size);

            /* 5-6. QK normalization */
            gemma3_multi_head_rmsnorm_dispatch_gpu(gpu, &gpu->buf_q,
                                                    w_qn, qk_norm_bytes,
                                                    head_dim, num_heads, cfg->rmsnorm_eps);
            gemma3_multi_head_rmsnorm_dispatch_gpu(gpu, &gpu->buf_k,
                                                    w_kn, qk_norm_bytes,
                                                    head_dim, num_kv_heads, cfg->rmsnorm_eps);

            /* 7-8. RoPE */
            {
                gemma3_gpu_buffer *rope_table = is_global ? &gpu->buf_rope_global : &gpu->buf_rope_local;
                gemma3_rope_precomputed_dispatch_gpu(gpu, &gpu->buf_q, rope_table,
                                                      num_heads, head_dim, pos);
                gemma3_rope_precomputed_dispatch_gpu(gpu, &gpu->buf_k, rope_table,
                                                      num_kv_heads, head_dim, pos);
            }

            /* 9. KV cache write */
            gemma3_kv_cache_write_dispatch_gpu(gpu,
                                                &gpu->buf_k, &gpu->buf_v,
                                                &gpu->gpu_kv_cache[l].k, &gpu->gpu_kv_cache[l].v,
                                                num_kv_heads, head_dim, cache_pos);

            /* 10. Attention mask */
            gemma3_mask_dispatch_gpu(gpu, &gpu->buf_mask, pos, seq_len,
                                      cfg->sliding_window, is_global);

            /* 11. GQA */
            gemma3_gqa_dispatch_gpu(gpu, &gpu->buf_attn_out, &gpu->buf_q,
                                     &gpu->gpu_kv_cache[l].k, &gpu->gpu_kv_cache[l].v,
                                     num_heads, num_kv_heads, seq_len, head_dim,
                                     scale, &gpu->buf_mask);

            /* --- Post-attention + MLP block --- */

            /* 12. O projection */
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_proj_out, w_o,
                                             o_weight_bytes, &gpu->buf_attn_out,
                                             hidden_size, q_size);

            /* 13. Post-attention RMSNorm (non-aliasing v2) */
            gemma3_rmsnorm_bf16_inplace_v2_dispatch_gpu(gpu, &gpu->buf_proj_out,
                                                          w_post_attn, norm_weight_bytes,
                                                          hidden_size, cfg->rmsnorm_eps);

            /* 14. Residual: x += proj_out (in-place, non-aliasing) */
            gemma3_vec_add_inplace_dispatch_gpu(gpu, &gpu->buf_x, &gpu->buf_proj_out, hidden_size);

            /* 15. Pre-feedforward RMSNorm */
            gemma3_rmsnorm_bf16_dispatch_gpu(gpu, &gpu->buf_x_norm, &gpu->buf_x,
                                              w_pre_ff, norm_weight_bytes,
                                              hidden_size, cfg->rmsnorm_eps);

            /* 16-17. Gate + Up projections */
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_mlp_gate, w_gate,
                                             gate_weight_bytes, &gpu->buf_x_norm,
                                             intermediate_size, hidden_size);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_mlp_up, w_up,
                                             gate_weight_bytes, &gpu->buf_x_norm,
                                             intermediate_size, hidden_size);

            /* 18. Fused GELU + Gate*Up: gate[i] = gelu(gate[i]) * up[i] */
            gemma3_gelu_mul_dispatch_gpu(gpu, &gpu->buf_mlp_gate,
                                           &gpu->buf_mlp_up, intermediate_size);

            /* 19. Down projection */
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_mlp_out, w_down,
                                             gate_weight_bytes, &gpu->buf_mlp_gate,
                                             hidden_size, intermediate_size);

            /* 21. Post-feedforward RMSNorm (non-aliasing v2) */
            gemma3_rmsnorm_bf16_inplace_v2_dispatch_gpu(gpu, &gpu->buf_mlp_out,
                                                          w_post_ff, norm_weight_bytes,
                                                          hidden_size, cfg->rmsnorm_eps);

            /* 22. Residual: x += mlp_out (in-place, non-aliasing) */
            gemma3_vec_add_inplace_dispatch_gpu(gpu, &gpu->buf_x, &gpu->buf_mlp_out, hidden_size);

            /* Keep CPU-side cache pos in sync */
            cache->layers[l].pos = pos + 1;
        }

        /* Final RMSNorm (using persistent final_norm buffer) */
        if (compute_logits) {
            gemma3_rmsnorm_bf16_dispatch_gpu(gpu, &gpu->buf_x_norm, &gpu->buf_x,
                                              &gpu->buf_final_norm, norm_weight_bytes,
                                              hidden_size, cfg->rmsnorm_eps);
        }

        /* Single flush + sync for ALL 34 layers + final norm */
        gemma3_gpu_flush_commands(gpu);
        gemma3_gpu_sync(gpu);

        /* Logit projection on CPU (embed table ~1.28GB, stays on host) */
        if (compute_logits) {
            gemma3_gpu_read_buffer(gpu, &gpu->buf_x_norm, buf->x_norm, hidden_size * sizeof(float));
            matvec_bf16_dispatch(logits, weights->embed_tokens, buf->x_norm,
                                 vocab_size, hidden_size, buf->matvec_tmp, thread_pool);
        }

    } else {
        /* ==============================================================
         * Phase 2 fallback: Per-layer weight upload (35 syncs/token)
         * ============================================================== */

        for (int l = 0; l < cfg->num_layers; l++) {
            int is_global = gemma3_is_global_layer(l);
            int cache_pos = is_global ? pos : (pos % cfg->sliding_window);
            int seq_len = is_global ? (pos + 1) : ((pos < cfg->sliding_window) ? pos + 1 : cfg->sliding_window);
            float scale = 1.0f / sqrtf((float)head_dim);

            /* Upload all 13 weight buffers for this layer */
            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_norm_0,
                                    weights->layers[l].input_layernorm, norm_weight_bytes);
            if (weights->layers[l].post_attention_layernorm)
                gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_norm_1,
                                        weights->layers[l].post_attention_layernorm, norm_weight_bytes);
            if (weights->layers[l].pre_feedforward_layernorm)
                gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_norm_2,
                                        weights->layers[l].pre_feedforward_layernorm, norm_weight_bytes);
            if (weights->layers[l].post_feedforward_layernorm)
                gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_norm_3,
                                        weights->layers[l].post_feedforward_layernorm, norm_weight_bytes);

            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_medium,
                                    weights->layers[l].q_proj, q_weight_bytes);
            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_small_0,
                                    weights->layers[l].k_proj, kv_weight_bytes);
            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_small_1,
                                    weights->layers[l].v_proj, kv_weight_bytes);

            if (weights->layers[l].q_norm)
                gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_qk_norm_q,
                                        weights->layers[l].q_norm, qk_norm_bytes);
            if (weights->layers[l].k_norm)
                gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_qk_norm_k,
                                        weights->layers[l].k_norm, qk_norm_bytes);

            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_medium_1,
                                    weights->layers[l].o_proj, o_weight_bytes);
            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_large_0,
                                    weights->layers[l].gate_proj, gate_weight_bytes);
            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_large_1,
                                    weights->layers[l].up_proj, gate_weight_bytes);
            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_large_2,
                                    weights->layers[l].down_proj, gate_weight_bytes);

            /* Record all dispatches */
            gemma3_gpu_begin_commands(gpu);

            gemma3_rmsnorm_bf16_dispatch_gpu(gpu, &gpu->buf_x_norm, &gpu->buf_x,
                                              &gpu->buf_weight_norm_0, norm_weight_bytes,
                                              hidden_size, cfg->rmsnorm_eps);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_q, &gpu->buf_weight_medium,
                                             q_weight_bytes, &gpu->buf_x_norm,
                                             q_size, hidden_size);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_k, &gpu->buf_weight_small_0,
                                             kv_weight_bytes, &gpu->buf_x_norm,
                                             kv_size, hidden_size);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_v, &gpu->buf_weight_small_1,
                                             kv_weight_bytes, &gpu->buf_x_norm,
                                             kv_size, hidden_size);

            if (weights->layers[l].q_norm)
                gemma3_multi_head_rmsnorm_dispatch_gpu(gpu, &gpu->buf_q,
                                                        &gpu->buf_weight_qk_norm_q, qk_norm_bytes,
                                                        head_dim, num_heads, cfg->rmsnorm_eps);
            if (weights->layers[l].k_norm)
                gemma3_multi_head_rmsnorm_dispatch_gpu(gpu, &gpu->buf_k,
                                                        &gpu->buf_weight_qk_norm_k, qk_norm_bytes,
                                                        head_dim, num_kv_heads, cfg->rmsnorm_eps);

            {
                gemma3_gpu_buffer *rope_table = is_global ? &gpu->buf_rope_global : &gpu->buf_rope_local;
                gemma3_rope_precomputed_dispatch_gpu(gpu, &gpu->buf_q, rope_table,
                                                      num_heads, head_dim, pos);
                gemma3_rope_precomputed_dispatch_gpu(gpu, &gpu->buf_k, rope_table,
                                                      num_kv_heads, head_dim, pos);
            }

            gemma3_kv_cache_write_dispatch_gpu(gpu,
                                                &gpu->buf_k, &gpu->buf_v,
                                                &gpu->gpu_kv_cache[l].k, &gpu->gpu_kv_cache[l].v,
                                                num_kv_heads, head_dim, cache_pos);
            gemma3_mask_dispatch_gpu(gpu, &gpu->buf_mask, pos, seq_len,
                                      cfg->sliding_window, is_global);
            gemma3_gqa_dispatch_gpu(gpu, &gpu->buf_attn_out, &gpu->buf_q,
                                     &gpu->gpu_kv_cache[l].k, &gpu->gpu_kv_cache[l].v,
                                     num_heads, num_kv_heads, seq_len, head_dim,
                                     scale, &gpu->buf_mask);

            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_proj_out, &gpu->buf_weight_medium_1,
                                             o_weight_bytes, &gpu->buf_attn_out,
                                             hidden_size, q_size);
            if (weights->layers[l].post_attention_layernorm)
                gemma3_rmsnorm_bf16_inplace_v2_dispatch_gpu(gpu, &gpu->buf_proj_out,
                                                              &gpu->buf_weight_norm_1, norm_weight_bytes,
                                                              hidden_size, cfg->rmsnorm_eps);
            gemma3_vec_add_inplace_dispatch_gpu(gpu, &gpu->buf_x, &gpu->buf_proj_out, hidden_size);

            if (weights->layers[l].pre_feedforward_layernorm)
                gemma3_rmsnorm_bf16_dispatch_gpu(gpu, &gpu->buf_x_norm, &gpu->buf_x,
                                                  &gpu->buf_weight_norm_2, norm_weight_bytes,
                                                  hidden_size, cfg->rmsnorm_eps);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_mlp_gate, &gpu->buf_weight_large_0,
                                             gate_weight_bytes, &gpu->buf_x_norm,
                                             intermediate_size, hidden_size);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_mlp_up, &gpu->buf_weight_large_1,
                                             gate_weight_bytes, &gpu->buf_x_norm,
                                             intermediate_size, hidden_size);
            gemma3_gelu_mul_dispatch_gpu(gpu, &gpu->buf_mlp_gate,
                                           &gpu->buf_mlp_up, intermediate_size);
            gemma3_matvec_bf16_dispatch_gpu(gpu, &gpu->buf_mlp_out, &gpu->buf_weight_large_2,
                                             gate_weight_bytes, &gpu->buf_mlp_gate,
                                             hidden_size, intermediate_size);
            if (weights->layers[l].post_feedforward_layernorm)
                gemma3_rmsnorm_bf16_inplace_v2_dispatch_gpu(gpu, &gpu->buf_mlp_out,
                                                              &gpu->buf_weight_norm_3, norm_weight_bytes,
                                                              hidden_size, cfg->rmsnorm_eps);
            gemma3_vec_add_inplace_dispatch_gpu(gpu, &gpu->buf_x, &gpu->buf_mlp_out, hidden_size);

            gemma3_gpu_flush_commands(gpu);
            gemma3_gpu_sync(gpu);

            cache->layers[l].pos = pos + 1;
        }

        /* Final: norm + logits */
        if (compute_logits) {
            gemma3_gpu_write_buffer(gpu, &gpu->buf_weight_norm_0, weights->norm, norm_weight_bytes);
            gemma3_gpu_begin_commands(gpu);
            gemma3_rmsnorm_bf16_dispatch_gpu(gpu, &gpu->buf_x_norm, &gpu->buf_x,
                                              &gpu->buf_weight_norm_0, norm_weight_bytes,
                                              hidden_size, cfg->rmsnorm_eps);
            gemma3_gpu_flush_commands(gpu);
            gemma3_gpu_sync(gpu);

            gemma3_gpu_read_buffer(gpu, &gpu->buf_x_norm, buf->x_norm, hidden_size * sizeof(float));
            matvec_bf16_dispatch(logits, weights->embed_tokens, buf->x_norm,
                                 vocab_size, hidden_size, buf->matvec_tmp, thread_pool);
        }
    }

    cache->current_pos = pos + 1;
    return 0;
}
#endif /* USE_WEBGPU */

/* ============================================================================
 * Full Forward Pass
 * ========================================================================== */

/* Forward pass for a single token */
int gemma3_transformer_forward(
    float *logits,            /* Output: [vocab_size] (only written if compute_logits) */
    int token_id,             /* Input token */
    int pos,                  /* Position in sequence */
    const gemma3_weights_t *weights,
    gemma3_kv_cache *cache,
    activation_buffers *buf,
    const gemma3_config *cfg,
    int compute_logits,       /* If false, skip final norm + vocab projection */
    const float *rope_freqs_local,  /* precomputed RoPE for local layers */
    const float *rope_freqs_global, /* precomputed RoPE for global layers */
    void *thread_pool               /* gemma3_thread_pool* or NULL */
) {
    int hidden_size = cfg->hidden_size;
    int vocab_size = cfg->vocab_size;

    /* Token embedding lookup (BF16) */
    gemma3_embed_bf16(buf->x, weights->embed_tokens, token_id, hidden_size);
    const float *embed = buf->x;  /* buf->x now contains the F32 embedding */

    /* Gemma scales embeddings by sqrt(hidden_size) */
    float embed_scale = sqrtf((float)hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        buf->x[i] = embed[i] * embed_scale;
    }

    /* Process each layer */
    for (int l = 0; l < cfg->num_layers; l++) {
        const uint16_t *layer_weights_input_ln = weights->layers[l].input_layernorm;
        const uint16_t *layer_weights_q = weights->layers[l].q_proj;
        const uint16_t *layer_weights_k = weights->layers[l].k_proj;
        const uint16_t *layer_weights_v = weights->layers[l].v_proj;
        const uint16_t *layer_weights_o = weights->layers[l].o_proj;
        const uint16_t *layer_weights_q_norm = weights->layers[l].q_norm;
        const uint16_t *layer_weights_k_norm = weights->layers[l].k_norm;
        const uint16_t *layer_weights_post_attn_ln = weights->layers[l].post_attention_layernorm;
        const uint16_t *layer_weights_gate = weights->layers[l].gate_proj;
        const uint16_t *layer_weights_up = weights->layers[l].up_proj;
        const uint16_t *layer_weights_down = weights->layers[l].down_proj;
        const uint16_t *layer_weights_pre_ff_ln = weights->layers[l].pre_feedforward_layernorm;
        const uint16_t *layer_weights_post_ff_ln = weights->layers[l].post_feedforward_layernorm;

        /* === Self-Attention Block === */

        /* Pre-attention RMSNorm (BF16 weights) */
        gemma3_rmsnorm_bf16(buf->x_norm, buf->x, layer_weights_input_ln,
                            hidden_size, cfg->rmsnorm_eps);

        /* Attention */
        const float *rope_freqs = gemma3_is_global_layer(l) ? rope_freqs_global : rope_freqs_local;
        layer_attention(
            buf->proj_out,
            buf->x_norm,
            layer_weights_q, layer_weights_k, layer_weights_v, layer_weights_o,
            layer_weights_q_norm, layer_weights_k_norm,
            &cache->layers[l],
            buf->q, buf->k, buf->v, buf->attn_out, buf->mask,
            buf->attn_scores, buf->matvec_tmp,
            rope_freqs, thread_pool,
            cfg, l, pos
        );

        /* Post-attention RMSNorm (Gemma 2/3 specific, BF16 weights with 1+weight) */
        if (layer_weights_post_attn_ln) {
            gemma3_rmsnorm_bf16_inplace(buf->proj_out, layer_weights_post_attn_ln,
                                        hidden_size, cfg->rmsnorm_eps);
        }

        /* Residual connection */
        gemma3_vec_add(buf->x, buf->x, buf->proj_out, hidden_size);

        /* === MLP Block === */

        /* Pre-feedforward RMSNorm (Gemma 3 specific, BF16 weights) */
        if (layer_weights_pre_ff_ln) {
            gemma3_rmsnorm_bf16(buf->x_norm, buf->x, layer_weights_pre_ff_ln,
                                hidden_size, cfg->rmsnorm_eps);
        } else {
            gemma3_vec_copy(buf->x_norm, buf->x, hidden_size);
        }

        /* MLP */
        layer_mlp(
            buf->mlp_out,
            buf->x_norm,
            layer_weights_gate, layer_weights_up, layer_weights_down,
            buf->mlp_gate, buf->mlp_up,
            buf->matvec_tmp, thread_pool,
            cfg, l, pos
        );

        /* Post-feedforward RMSNorm (Gemma 2/3 specific, BF16 weights with 1+weight) */
        if (layer_weights_post_ff_ln) {
            gemma3_rmsnorm_bf16_inplace(buf->mlp_out, layer_weights_post_ff_ln,
                                        hidden_size, cfg->rmsnorm_eps);
        }

        /* Residual connection */
        gemma3_vec_add(buf->x, buf->x, buf->mlp_out, hidden_size);
    }

    /* Skip final norm + vocab projection during prefill (non-last tokens) */
    if (compute_logits) {
        /* Final RMSNorm (BF16 weights) */
        gemma3_rmsnorm_bf16(buf->x_norm, buf->x, weights->norm, hidden_size, cfg->rmsnorm_eps);

        /* Output projection (tied embeddings, BF16) */
        /* logits = x_norm @ embed_tokens.T */
        matvec_bf16_dispatch(logits, weights->embed_tokens, buf->x_norm, vocab_size, hidden_size, buf->matvec_tmp, thread_pool);
    }

    return 0;
}

/* Convert a BF16 weight matrix to F32 into a pre-allocated buffer */
static void bf16_matrix_to_f32(float *dst, const uint16_t *src, int rows, int cols) {
    int n = rows * cols;
    for (int i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
#if defined(_MSC_VER)
        memcpy(&dst[i], &bits, sizeof(float));
#else
        __builtin_memcpy(&dst[i], &bits, sizeof(float));
#endif
    }
}

#ifdef USE_BLAS
/* Batched prefill: process all tokens through each layer using sgemm.
 * Linear projections become matrix-matrix multiplies; attention remains per-token. */
static int gemma3_transformer_prefill_batched(
    float *logits,
    const int *tokens,
    int num_tokens,
    int start_pos,
    const gemma3_weights_t *weights,
    gemma3_kv_cache *cache,
    activation_buffers *buf,
    const gemma3_config *cfg,
    const float *rope_freqs_local,
    const float *rope_freqs_global,
    void *thread_pool
) {
    int hidden_size = cfg->hidden_size;
    int intermediate_size = cfg->intermediate_size;
    int num_heads = cfg->num_heads;
    int num_kv_heads = cfg->num_kv_heads;
    int head_dim = cfg->head_dim;
    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;
    int N = num_tokens;

    /* Allocate batched activation matrices:
     * X: [N, hidden_size], X_norm: [N, hidden_size]
     * Q_all: [N, q_size], K_all: [N, kv_size], V_all: [N, kv_size]
     * attn_out_all: [N, q_size], proj_out_all: [N, hidden_size]
     * gate_all: [N, intermediate_size], up_all: [N, intermediate_size]
     * mlp_out_all: [N, hidden_size] */
    size_t total_size = (size_t)N * (
        hidden_size * 3 +       /* X, X_norm, proj_out */
        q_size * 2 +            /* Q, attn_out */
        kv_size * 2 +           /* K, V */
        hidden_size +           /* mlp_out */
        intermediate_size * 2   /* gate, up */
    );
    /* Weight conversion buffer: largest weight is embed [vocab_size, hidden_size] or
     * intermediate [intermediate_size, hidden_size]. For per-layer we need max of:
     * q_proj: [q_size, hidden_size], gate/up/down: [intermediate_size, hidden_size] */
    int max_weight_elems = intermediate_size * hidden_size;
    if (q_size * hidden_size > max_weight_elems) max_weight_elems = q_size * hidden_size;
    if (hidden_size * q_size > max_weight_elems) max_weight_elems = hidden_size * q_size;
    if (hidden_size * intermediate_size > max_weight_elems) max_weight_elems = hidden_size * intermediate_size;

    float *batch_buf = (float *)malloc(total_size * sizeof(float));
    float *weight_f32 = (float *)malloc(max_weight_elems * sizeof(float));
    if (!batch_buf || !weight_f32) {
        free(batch_buf);
        free(weight_f32);
        /* Fall back to sequential */
        goto sequential_fallback;
    }

    /* Assign sub-buffers */
    float *X       = batch_buf;
    float *X_norm  = X + N * hidden_size;
    float *Q_all   = X_norm + N * hidden_size;
    float *K_all   = Q_all + N * q_size;
    float *V_all   = K_all + N * kv_size;
    float *attn_out_all = V_all + N * kv_size;
    float *proj_out_all = attn_out_all + N * q_size;
    float *gate_all = proj_out_all + N * hidden_size;
    float *up_all  = gate_all + N * intermediate_size;
    float *mlp_out_all = up_all + N * intermediate_size;

    /* Embed all tokens: X[i] = embed(tokens[i]) * sqrt(hidden_size) */
    float embed_scale = sqrtf((float)hidden_size);
    for (int i = 0; i < N; i++) {
        gemma3_embed_bf16(X + i * hidden_size, weights->embed_tokens, tokens[i], hidden_size);
        for (int j = 0; j < hidden_size; j++) {
            X[i * hidden_size + j] *= embed_scale;
        }
    }

    /* Process each layer */
    for (int l = 0; l < cfg->num_layers; l++) {
        int is_global = gemma3_is_global_layer(l);
        const float *rope_freqs = is_global ? rope_freqs_global : rope_freqs_local;

        /* --- Pre-attention RMSNorm (per-token, no good way to batch) --- */
        for (int i = 0; i < N; i++) {
            gemma3_rmsnorm_bf16(X_norm + i * hidden_size,
                                X + i * hidden_size,
                                weights->layers[l].input_layernorm,
                                hidden_size, cfg->rmsnorm_eps);
        }

        /* --- Batched QKV projection ---
         * Q_all = X_norm @ q_proj^T  ->  [N, hidden_size] @ [hidden_size, q_size] = [N, q_size]
         * Weight is stored [q_size, hidden_size] in BF16, so we convert it and
         * compute: Q_all = X_norm * W^T using sgemm */
        bf16_matrix_to_f32(weight_f32, weights->layers[l].q_proj, q_size, hidden_size);
        /* C = alpha * A * B^T + beta * C
         * A = X_norm [N, hidden_size], B = weight_f32 [q_size, hidden_size]
         * C = Q_all [N, q_size]
         * We want C[i,j] = sum_k X_norm[i,k] * W[j,k] = X_norm * W^T
         * sgemm: CblasNoTrans for A, CblasTrans for B */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, q_size, hidden_size,
                    1.0f, X_norm, hidden_size, weight_f32, hidden_size,
                    0.0f, Q_all, q_size);

        bf16_matrix_to_f32(weight_f32, weights->layers[l].k_proj, kv_size, hidden_size);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, kv_size, hidden_size,
                    1.0f, X_norm, hidden_size, weight_f32, hidden_size,
                    0.0f, K_all, kv_size);

        bf16_matrix_to_f32(weight_f32, weights->layers[l].v_proj, kv_size, hidden_size);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, kv_size, hidden_size,
                    1.0f, X_norm, hidden_size, weight_f32, hidden_size,
                    0.0f, V_all, kv_size);

        /* --- Per-token: QK norm, RoPE, cache KV, attention --- */
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int i = 0; i < N; i++) {
            int pos = start_pos + i;
            float *qi = Q_all + i * q_size;
            float *ki = K_all + i * kv_size;
            float *vi = V_all + i * kv_size;

            /* QK normalization */
            if (weights->layers[l].q_norm && weights->layers[l].k_norm) {
                for (int h = 0; h < num_heads; h++) {
                    gemma3_rmsnorm_bf16(qi + h * head_dim, qi + h * head_dim,
                                        weights->layers[l].q_norm, head_dim, cfg->rmsnorm_eps);
                }
                for (int h = 0; h < num_kv_heads; h++) {
                    gemma3_rmsnorm_bf16(ki + h * head_dim, ki + h * head_dim,
                                        weights->layers[l].k_norm, head_dim, cfg->rmsnorm_eps);
                }
            }

            /* RoPE */
            for (int h = 0; h < num_heads; h++) {
                gemma3_rope_apply_precomputed(qi + h * head_dim, rope_freqs, head_dim, pos);
            }
            for (int h = 0; h < num_kv_heads; h++) {
                gemma3_rope_apply_precomputed(ki + h * head_dim, rope_freqs, head_dim, pos);
            }

            /* Cache KV */
            cache_kv(&cache->layers[l], ki, vi, kv_size, is_global,
                     cfg->sliding_window, pos);

            /* Attention (must be sequential due to causal dependency on cache) */
            int seq_len;
            const float *k_cache, *v_cache;
            if (is_global) {
                seq_len = pos + 1;
                k_cache = cache->layers[l].k;
                v_cache = cache->layers[l].v;
                gemma3_causal_mask(buf->mask, seq_len, pos);
            } else {
                int window = cfg->sliding_window;
                seq_len = (pos < window) ? pos + 1 : window;
                k_cache = cache->layers[l].k;
                v_cache = cache->layers[l].v;
                for (int s = 0; s < seq_len; s++) buf->mask[s] = 0.0f;
            }

            gemma3_gqa(attn_out_all + i * q_size, qi, k_cache, v_cache,
                       num_heads, num_kv_heads, seq_len, head_dim,
                       scale, buf->mask, buf->attn_scores);
        }

        /* --- Batched output projection: proj_out = attn_out @ o_proj^T --- */
        bf16_matrix_to_f32(weight_f32, weights->layers[l].o_proj, hidden_size, q_size);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, hidden_size, q_size,
                    1.0f, attn_out_all, q_size, weight_f32, q_size,
                    0.0f, proj_out_all, hidden_size);

        /* Post-attention RMSNorm + residual */
        for (int i = 0; i < N; i++) {
            if (weights->layers[l].post_attention_layernorm) {
                gemma3_rmsnorm_bf16_inplace(proj_out_all + i * hidden_size,
                                            weights->layers[l].post_attention_layernorm,
                                            hidden_size, cfg->rmsnorm_eps);
            }
            gemma3_vec_add(X + i * hidden_size, X + i * hidden_size,
                           proj_out_all + i * hidden_size, hidden_size);
        }

        /* --- MLP Block --- */
        /* Pre-feedforward RMSNorm */
        for (int i = 0; i < N; i++) {
            if (weights->layers[l].pre_feedforward_layernorm) {
                gemma3_rmsnorm_bf16(X_norm + i * hidden_size,
                                    X + i * hidden_size,
                                    weights->layers[l].pre_feedforward_layernorm,
                                    hidden_size, cfg->rmsnorm_eps);
            } else {
                gemma3_vec_copy(X_norm + i * hidden_size, X + i * hidden_size, hidden_size);
            }
        }

        /* Batched gate + up projections */
        bf16_matrix_to_f32(weight_f32, weights->layers[l].gate_proj, intermediate_size, hidden_size);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, intermediate_size, hidden_size,
                    1.0f, X_norm, hidden_size, weight_f32, hidden_size,
                    0.0f, gate_all, intermediate_size);

        bf16_matrix_to_f32(weight_f32, weights->layers[l].up_proj, intermediate_size, hidden_size);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, intermediate_size, hidden_size,
                    1.0f, X_norm, hidden_size, weight_f32, hidden_size,
                    0.0f, up_all, intermediate_size);

        /* SwiGLU activation (per-token) */
        for (int i = 0; i < N; i++) {
            gemma3_gelu_tanh_inplace(gate_all + i * intermediate_size, intermediate_size);
            gemma3_vec_mul(gate_all + i * intermediate_size,
                           gate_all + i * intermediate_size,
                           up_all + i * intermediate_size, intermediate_size);
        }

        /* Batched down projection */
        bf16_matrix_to_f32(weight_f32, weights->layers[l].down_proj, hidden_size, intermediate_size);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, hidden_size, intermediate_size,
                    1.0f, gate_all, intermediate_size, weight_f32, intermediate_size,
                    0.0f, mlp_out_all, hidden_size);

        /* Post-feedforward RMSNorm + residual */
        for (int i = 0; i < N; i++) {
            if (weights->layers[l].post_feedforward_layernorm) {
                gemma3_rmsnorm_bf16_inplace(mlp_out_all + i * hidden_size,
                                            weights->layers[l].post_feedforward_layernorm,
                                            hidden_size, cfg->rmsnorm_eps);
            }
            gemma3_vec_add(X + i * hidden_size, X + i * hidden_size,
                           mlp_out_all + i * hidden_size, hidden_size);
        }
    }

    /* Final: compute logits only for the last token */
    int last = N - 1;
    gemma3_rmsnorm_bf16(buf->x_norm, X + last * hidden_size, weights->norm,
                        hidden_size, cfg->rmsnorm_eps);
    matvec_bf16_dispatch(logits, weights->embed_tokens, buf->x_norm,
                         cfg->vocab_size, hidden_size, buf->matvec_tmp, thread_pool);

    free(batch_buf);
    free(weight_f32);
    cache->current_pos = start_pos + num_tokens;
    return 0;

sequential_fallback:;
    /* Fall through to sequential path below */
    for (int i = 0; i < N; i++) {
        int pos = start_pos + i;
        int is_last = (i == N - 1);
        gemma3_transformer_forward(logits, tokens[i], pos, weights, cache, buf, cfg, is_last,
                                   rope_freqs_local, rope_freqs_global, thread_pool);
    }
    cache->current_pos = start_pos + num_tokens;
    return 0;
}
#endif /* USE_BLAS */

/* Prefill: process multiple tokens at once */
int gemma3_transformer_prefill(
    float *logits,            /* Output: [vocab_size] for last token */
    const int *tokens,        /* Input tokens */
    int num_tokens,           /* Number of tokens */
    int start_pos,            /* Starting position */
    const gemma3_weights_t *weights,
    gemma3_kv_cache *cache,
    activation_buffers *buf,
    const gemma3_config *cfg,
    const float *rope_freqs_local,
    const float *rope_freqs_global,
    void *thread_pool
) {
#ifdef USE_BLAS
    /* Use batched prefill with sgemm when BLAS is available */
    if (num_tokens > 1) {
        return gemma3_transformer_prefill_batched(
            logits, tokens, num_tokens, start_pos,
            weights, cache, buf, cfg,
            rope_freqs_local, rope_freqs_global, thread_pool);
    }
#endif
    /* Sequential fallback for single tokens or non-BLAS builds */
    for (int i = 0; i < num_tokens; i++) {
        int pos = start_pos + i;
        int is_last = (i == num_tokens - 1);

        /* Only compute logits for last token — skips ~40% of work for others */
        gemma3_transformer_forward(logits, tokens[i], pos, weights, cache, buf, cfg, is_last,
                                   rope_freqs_local, rope_freqs_global, thread_pool);
    }

    cache->current_pos = start_pos + num_tokens;
    return 0;
}

/* ============================================================================
 * Transformer Context (combines weights, cache, buffers)
 * ========================================================================== */

typedef struct gemma3_transformer {
    gemma3_weights_t *weights;
    gemma3_kv_cache *cache;
    activation_buffers *buffers;
    gemma3_config config;
    float *rope_freqs_local;   /* [max_context, head_dim/2, 2] cos/sin for theta=10000 */
    float *rope_freqs_global;  /* [max_context, head_dim/2, 2] cos/sin for theta=1000000 */
#ifdef USE_THREADS
    gemma3_thread_pool *thread_pool;
#endif
#ifdef USE_WEBGPU
    gemma3_gpu_context *gpu_ctx;
#endif
} gemma3_transformer;

gemma3_transformer *gemma3_transformer_create(
    gemma3_weights_t *weights,
    const gemma3_config *cfg,
    int max_context
) {
    gemma3_transformer *t = (gemma3_transformer *)calloc(1, sizeof(gemma3_transformer));
    if (!t) return NULL;

    t->weights = weights;
    t->config = *cfg;

    t->cache = gemma3_kv_cache_alloc(cfg, max_context);
    if (!t->cache) {
        free(t);
        return NULL;
    }

    t->buffers = alloc_buffers(cfg);
    if (!t->buffers) {
        gemma3_kv_cache_free(t->cache);
        free(t);
        return NULL;
    }

    /* Precompute RoPE cos/sin tables for local and global theta */
    int rope_table_size = max_context * (cfg->head_dim / 2) * 2;
    t->rope_freqs_local = (float *)malloc(rope_table_size * sizeof(float));
    t->rope_freqs_global = (float *)malloc(rope_table_size * sizeof(float));
    if (!t->rope_freqs_local || !t->rope_freqs_global) {
        free(t->rope_freqs_local);
        free(t->rope_freqs_global);
        free_buffers(t->buffers);
        gemma3_kv_cache_free(t->cache);
        free(t);
        return NULL;
    }
    gemma3_rope_precompute(t->rope_freqs_local, max_context, cfg->head_dim, cfg->rope_theta_local);
    gemma3_rope_precompute(t->rope_freqs_global, max_context, cfg->head_dim, cfg->rope_theta_global);

#ifdef USE_THREADS
    t->thread_pool = gemma3_thread_pool_create(0); /* 0 = auto-detect CPU count */
#endif

#ifdef USE_WEBGPU
    t->gpu_ctx = gemma3_gpu_init();
    if (t->gpu_ctx) {
        gemma3_gpu_init_buffers(t->gpu_ctx,
                                cfg->hidden_size,
                                cfg->intermediate_size,
                                cfg->vocab_size,
                                cfg->num_heads,
                                cfg->num_kv_heads,
                                cfg->head_dim,
                                max_context,
                                cfg->num_layers,
                                cfg->sliding_window);
        /* Upload precomputed RoPE tables to GPU (Phase 2) */
        gemma3_gpu_upload_rope_tables(t->gpu_ctx, t->rope_freqs_local, t->rope_freqs_global);

        /* Upload all model weights to GPU for persistent residency (Phase 3) */
        if (upload_weights_to_gpu(t->gpu_ctx, t->weights, cfg) == 0) {
            fprintf(stderr, "WebGPU: Phase 3 active -- all weights resident on GPU (%s)\n",
                    gemma3_gpu_device_name(t->gpu_ctx));
        } else {
            fprintf(stderr, "WebGPU: Phase 2 fallback -- per-layer weight upload (%s)\n",
                    gemma3_gpu_device_name(t->gpu_ctx));
        }
    } else {
        fprintf(stderr, "WebGPU: Not available, falling back to CPU\n");
    }
#endif

    return t;
}

void gemma3_transformer_destroy(gemma3_transformer *t) {
    if (!t) return;
#ifdef USE_WEBGPU
    if (t->gpu_ctx) gemma3_gpu_free(t->gpu_ctx);
#endif
#ifdef USE_THREADS
    gemma3_thread_pool_destroy(t->thread_pool);
#endif
    gemma3_kv_cache_free(t->cache);
    free_buffers(t->buffers);
    free(t->rope_freqs_local);
    free(t->rope_freqs_global);
    free(t);
}

int gemma3_transformer_forward_token(
    gemma3_transformer *t,
    int token_id,
    int pos,
    float *logits
) {
#ifdef USE_WEBGPU
    /* Use the optimized GPU forward pass unless GEMMA3_NO_GPU is set */
    if (t->gpu_ctx && !getenv("GEMMA3_NO_GPU")) {
        void *tp = NULL;
#ifdef USE_THREADS
        tp = t->thread_pool;
#endif
        return gemma3_transformer_forward_gpu(
            logits, token_id, pos,
            t->weights, t->cache, t->buffers, &t->config, 1,
            t->rope_freqs_local, t->rope_freqs_global, t->gpu_ctx, tp
        );
    }
#endif
    void *pool = NULL;
#ifdef USE_THREADS
    pool = t->thread_pool;
#endif
    return gemma3_transformer_forward(
        logits, token_id, pos,
        t->weights, t->cache, t->buffers, &t->config, 1,
        t->rope_freqs_local, t->rope_freqs_global, pool
    );
}

int gemma3_transformer_prefill_tokens(
    gemma3_transformer *t,
    const int *tokens,
    int num_tokens,
    int start_pos,
    float *logits
) {
    void *pool = NULL;
#ifdef USE_THREADS
    pool = t->thread_pool;   /* CPU threads for prefill */
#endif
    int result = gemma3_transformer_prefill(
        logits, tokens, num_tokens, start_pos,
        t->weights, t->cache, t->buffers, &t->config,
        t->rope_freqs_local, t->rope_freqs_global, pool
    );

#ifdef USE_WEBGPU
    /* After CPU prefill, copy the filled KV cache to GPU so GPU generation
     * reads valid K/V vectors for all prefilled positions. Without this,
     * the GPU GQA kernel reads empty/garbage data for positions 0..num_tokens-1. */
    if (t->gpu_ctx && result == 0) {
        gemma3_gpu_context *gpu = t->gpu_ctx;
        int kv_size = t->config.num_kv_heads * t->config.head_dim;
        int end_pos = start_pos + num_tokens;

        for (int l = 0; l < t->config.num_layers; l++) {
            /* Copy min(filled_positions, layer_cache_size) entries.
             * Global layers: cache_pos == pos (simple append up to max_context).
             * Local layers: ring buffer of sliding_window entries. */
            int layer_max = gpu->gpu_kv_cache[l].max_seq;
            int copy_entries = end_pos < layer_max ? end_pos : layer_max;
            size_t copy_bytes = (size_t)copy_entries * kv_size * sizeof(float);

            gemma3_gpu_write_buffer(gpu, &gpu->gpu_kv_cache[l].k,
                                    t->cache->layers[l].k, copy_bytes);
            gemma3_gpu_write_buffer(gpu, &gpu->gpu_kv_cache[l].v,
                                    t->cache->layers[l].v, copy_bytes);
        }
        fprintf(stderr, "WebGPU: Copied KV cache to GPU after prefill (%d tokens)\n", num_tokens);
    }
#endif

    return result;
}

void gemma3_transformer_reset(gemma3_transformer *t) {
    if (t && t->cache) {
        gemma3_kv_cache_reset(t->cache);
    }
}

int gemma3_transformer_get_pos(gemma3_transformer *t) {
    return t && t->cache ? t->cache->current_pos : 0;
}
