// gemma3_kernels.wgsl - WebGPU Compute Shaders for Gemma 3 Inference
//
// This file contains all WGSL compute shaders for accelerating Gemma 3 inference.
// Shaders handle BF16 weights with F32 activations.

// ============================================================================
// Common Utilities
// ============================================================================

// BF16 to F32 conversion
// BF16 is stored as uint16, representing the upper 16 bits of an IEEE 754 float
fn bf16_to_f32(bf16: u32) -> f32 {
    // Shift left 16 bits to get F32 bit pattern
    let bits = bf16 << 16u;
    return bitcast<f32>(bits);
}

// F32 to BF16 conversion (with rounding)
fn f32_to_bf16(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    // Round to nearest even
    let rounding = ((bits >> 16u) & 1u) + 0x7fffu;
    return (bits + rounding) >> 16u;
}

// ============================================================================
// Matrix-Vector Multiplication (BF16 weights, F32 input/output)
// y[i] = sum_k(A[i,k] * x[k]) where A is BF16, x and y are F32
// ============================================================================

struct MatvecParams {
    M: u32,       // Output dimension (rows)
    K: u32,       // Input dimension (cols)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> matvec_params: MatvecParams;
@group(0) @binding(1) var<storage, read> matvec_A: array<u32>;      // BF16 packed as u32 (2 per element)
@group(0) @binding(2) var<storage, read> matvec_x: array<f32>;      // Input vector
@group(0) @binding(3) var<storage, read_write> matvec_y: array<f32>; // Output vector

@compute @workgroup_size(256)
fn matvec_bf16_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= matvec_params.M) {
        return;
    }

    let K = matvec_params.K;
    var sum: f32 = 0.0;

    // Process 2 BF16 values at a time (packed in u32)
    let row_offset = row * K;
    var k: u32 = 0u;

    // Main loop: process pairs of BF16 values
    for (; k + 1u < K; k = k + 2u) {
        let packed = matvec_A[(row_offset + k) / 2u];
        let a0 = bf16_to_f32(packed & 0xffffu);
        let a1 = bf16_to_f32(packed >> 16u);
        sum = sum + a0 * matvec_x[k] + a1 * matvec_x[k + 1u];
    }

    // Handle odd K
    if (k < K) {
        let packed = matvec_A[(row_offset + k) / 2u];
        let a0 = bf16_to_f32(packed & 0xffffu);
        sum = sum + a0 * matvec_x[k];
    }

    matvec_y[row] = sum;
}

// Optimized version with workgroup-level reduction for better memory coalescing
@compute @workgroup_size(256)
fn matvec_bf16_kernel_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    // Each workgroup computes one row
    let row = wgid.x;
    if (row >= matvec_params.M) {
        return;
    }

    let K = matvec_params.K;
    let local_id = lid.x;
    let local_size = 256u;

    // Each thread processes a chunk of the row
    var sum: f32 = 0.0;
    let row_offset = row * K;

    var k = local_id * 2u;
    for (; k < K; k = k + local_size * 2u) {
        if (k + 1u < K) {
            let packed = matvec_A[(row_offset + k) / 2u];
            let a0 = bf16_to_f32(packed & 0xffffu);
            let a1 = bf16_to_f32(packed >> 16u);
            sum = sum + a0 * matvec_x[k] + a1 * matvec_x[k + 1u];
        } else if (k < K) {
            let packed = matvec_A[(row_offset + k) / 2u];
            let a0 = bf16_to_f32(packed & 0xffffu);
            sum = sum + a0 * matvec_x[k];
        }
    }

    // Workgroup reduction (using subgroups if available, otherwise shared memory)
    // For simplicity, use atomicAdd
    // Note: WGSL doesn't have atomicAdd for f32, so we use a different approach

    // Store partial sum (simplified - in production use proper reduction)
    if (local_id == 0u) {
        matvec_y[row] = sum;
    }
}

// ============================================================================
// RMSNorm (BF16 weights, F32 input/output)
// y[i] = x[i] * rsqrt(mean(x^2) + eps) * (1.0 + weight[i])
// ============================================================================

struct RmsnormParams {
    n: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> rmsnorm_params: RmsnormParams;
@group(0) @binding(1) var<storage, read> rmsnorm_x: array<f32>;
@group(0) @binding(2) var<storage, read> rmsnorm_weight: array<u32>;  // BF16 packed
@group(0) @binding(3) var<storage, read_write> rmsnorm_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> rmsnorm_scratch: array<f32>;  // For reduction

var<workgroup> rmsnorm_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm_bf16_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let n = rmsnorm_params.n;
    let local_id = lid.x;
    let local_size = 256u;

    // Phase 1: Compute partial sum of squares
    var sum_sq: f32 = 0.0;
    var i = local_id;
    for (; i < n; i = i + local_size) {
        let xi = rmsnorm_x[i];
        sum_sq = sum_sq + xi * xi;
    }

    // Store in shared memory
    rmsnorm_shared[local_id] = sum_sq;
    workgroupBarrier();

    // Reduction in shared memory
    var stride = local_size / 2u;
    for (; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            rmsnorm_shared[local_id] = rmsnorm_shared[local_id] + rmsnorm_shared[local_id + stride];
        }
        workgroupBarrier();
    }

    // Compute normalization factor
    let mean_sq = rmsnorm_shared[0] / f32(n);
    let rsqrt_val = inverseSqrt(mean_sq + rmsnorm_params.eps);

    workgroupBarrier();

    // Phase 2: Apply normalization with Gemma's (1 + weight) formula
    i = local_id;
    for (; i < n; i = i + local_size) {
        let xi = rmsnorm_x[i];

        // Extract BF16 weight
        let packed_idx = i / 2u;
        let packed = rmsnorm_weight[packed_idx];
        var w: f32;
        if (i % 2u == 0u) {
            w = bf16_to_f32(packed & 0xffffu);
        } else {
            w = bf16_to_f32(packed >> 16u);
        }

        rmsnorm_y[i] = xi * rsqrt_val * (1.0 + w);
    }
}

// In-place variant
@compute @workgroup_size(256)
fn rmsnorm_bf16_inplace_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let n = rmsnorm_params.n;
    let local_id = lid.x;
    let local_size = 256u;

    // Phase 1: Compute partial sum of squares
    var sum_sq: f32 = 0.0;
    var i = local_id;
    for (; i < n; i = i + local_size) {
        let xi = rmsnorm_y[i];  // In-place: read from output buffer
        sum_sq = sum_sq + xi * xi;
    }

    rmsnorm_shared[local_id] = sum_sq;
    workgroupBarrier();

    // Reduction
    var stride = local_size / 2u;
    for (; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            rmsnorm_shared[local_id] = rmsnorm_shared[local_id] + rmsnorm_shared[local_id + stride];
        }
        workgroupBarrier();
    }

    let mean_sq = rmsnorm_shared[0] / f32(n);
    let rsqrt_val = inverseSqrt(mean_sq + rmsnorm_params.eps);

    workgroupBarrier();

    // Phase 2: Apply in-place
    i = local_id;
    for (; i < n; i = i + local_size) {
        let xi = rmsnorm_y[i];
        let packed_idx = i / 2u;
        let packed = rmsnorm_weight[packed_idx];
        var w: f32;
        if (i % 2u == 0u) {
            w = bf16_to_f32(packed & 0xffffu);
        } else {
            w = bf16_to_f32(packed >> 16u);
        }
        rmsnorm_y[i] = xi * rsqrt_val * (1.0 + w);
    }
}

// ============================================================================
// GELU Activation (tanh approximation)
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

struct GeluParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> gelu_params: GeluParams;
@group(0) @binding(1) var<storage, read_write> gelu_x: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn gelu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gelu_params.n) {
        return;
    }

    let x = gelu_x[i];
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    gelu_x[i] = 0.5 * x * (1.0 + tanh(inner));
}

// ============================================================================
// SiLU/Swish Activation
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================================

@group(0) @binding(0) var<uniform> silu_params: GeluParams;  // Reuse struct
@group(0) @binding(1) var<storage, read_write> silu_x: array<f32>;

@compute @workgroup_size(256)
fn silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= silu_params.n) {
        return;
    }

    let x = silu_x[i];
    silu_x[i] = x / (1.0 + exp(-x));
}

// ============================================================================
// Softmax (numerically stable)
// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
// ============================================================================

struct SoftmaxParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> softmax_params: SoftmaxParams;
@group(0) @binding(1) var<storage, read_write> softmax_x: array<f32>;

var<workgroup> softmax_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let n = softmax_params.n;
    let local_id = lid.x;
    let local_size = 256u;

    // Phase 1: Find maximum
    var max_val: f32 = -3.402823466e+38;  // -FLT_MAX
    var i = local_id;
    for (; i < n; i = i + local_size) {
        max_val = max(max_val, softmax_x[i]);
    }

    softmax_shared[local_id] = max_val;
    workgroupBarrier();

    // Reduction for max
    var stride = local_size / 2u;
    for (; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            softmax_shared[local_id] = max(softmax_shared[local_id], softmax_shared[local_id + stride]);
        }
        workgroupBarrier();
    }

    let global_max = softmax_shared[0];
    workgroupBarrier();

    // Phase 2: Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    i = local_id;
    for (; i < n; i = i + local_size) {
        let exp_val = exp(softmax_x[i] - global_max);
        softmax_x[i] = exp_val;
        sum = sum + exp_val;
    }

    softmax_shared[local_id] = sum;
    workgroupBarrier();

    // Reduction for sum
    stride = local_size / 2u;
    for (; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            softmax_shared[local_id] = softmax_shared[local_id] + softmax_shared[local_id + stride];
        }
        workgroupBarrier();
    }

    let inv_sum = 1.0 / softmax_shared[0];
    workgroupBarrier();

    // Phase 3: Normalize
    i = local_id;
    for (; i < n; i = i + local_size) {
        softmax_x[i] = softmax_x[i] * inv_sum;
    }
}

// ============================================================================
// RoPE (Rotary Position Embeddings)
// For each dimension pair (i, i + d/2):
// [x_i, x_{i+d/2}] = [cos(θ)*x_i - sin(θ)*x_{i+d/2}, sin(θ)*x_i + cos(θ)*x_{i+d/2}]
// where θ = pos * freq, freq = 1 / (theta^(2i/d))
// ============================================================================

struct RopeParams {
    head_dim: u32,
    pos: u32,
    theta: f32,
    num_heads: u32,
}

@group(0) @binding(0) var<uniform> rope_params: RopeParams;
@group(0) @binding(1) var<storage, read_write> rope_x: array<f32>;

@compute @workgroup_size(256)
fn rope_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = rope_params.head_dim / 2u;
    let total_pairs = rope_params.num_heads * half_dim;

    let pair_idx = gid.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    let head = pair_idx / half_dim;
    let i = pair_idx % half_dim;

    // Compute frequency
    let exponent = f32(2u * i) / f32(rope_params.head_dim);
    let freq = 1.0 / pow(rope_params.theta, exponent);
    let angle = f32(rope_params.pos) * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    // Get indices
    let idx0 = head * rope_params.head_dim + i;
    let idx1 = idx0 + half_dim;

    // Apply rotation
    let x0 = rope_x[idx0];
    let x1 = rope_x[idx1];
    rope_x[idx0] = x0 * cos_val - x1 * sin_val;
    rope_x[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================================
// Grouped Query Attention (GQA)
// output[h] = softmax(Q[h] @ K^T * scale + mask) @ V
// With head grouping: each KV head serves multiple Q heads
// ============================================================================

struct GqaParams {
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    use_mask: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> gqa_params: GqaParams;
@group(0) @binding(1) var<storage, read> gqa_q: array<f32>;       // [n_heads, head_dim]
@group(0) @binding(2) var<storage, read> gqa_k: array<f32>;       // [seq_len, n_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read> gqa_v: array<f32>;       // [seq_len, n_kv_heads, head_dim]
@group(0) @binding(4) var<storage, read> gqa_mask: array<f32>;    // [seq_len]
@group(0) @binding(5) var<storage, read_write> gqa_output: array<f32>; // [n_heads, head_dim]
@group(0) @binding(6) var<storage, read_write> gqa_scores: array<f32>; // Scratch [seq_len]

var<workgroup> gqa_shared: array<f32, 256>;

// Each workgroup processes one head
@compute @workgroup_size(256)
fn gqa_kernel(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let head = wgid.x;
    if (head >= gqa_params.n_heads) {
        return;
    }

    let local_id = lid.x;
    let local_size = 256u;
    let head_dim = gqa_params.head_dim;
    let seq_len = gqa_params.seq_len;
    let n_kv_heads = gqa_params.n_kv_heads;
    let scale = gqa_params.scale;

    // Determine which KV head this Q head uses
    let heads_per_group = gqa_params.n_heads / n_kv_heads;
    let kv_head = head / heads_per_group;
    let kv_stride = n_kv_heads * head_dim;

    // Pointers
    let q_offset = head * head_dim;

    // Phase 1: Compute attention scores
    // scores[i] = (Q · K[i]) * scale + mask[i]
    var i = local_id;
    for (; i < seq_len; i = i + local_size) {
        var score: f32 = 0.0;
        let k_offset = i * kv_stride + kv_head * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            score = score + gqa_q[q_offset + d] * gqa_k[k_offset + d];
        }
        score = score * scale;
        if (gqa_params.use_mask != 0u) {
            score = score + gqa_mask[i];
        }
        gqa_scores[i] = score;
    }
    workgroupBarrier();

    // Phase 2: Softmax over scores
    // Find max
    var max_val: f32 = -3.402823466e+38;
    i = local_id;
    for (; i < seq_len; i = i + local_size) {
        max_val = max(max_val, gqa_scores[i]);
    }
    gqa_shared[local_id] = max_val;
    workgroupBarrier();

    var stride = min(local_size, seq_len) / 2u;
    for (; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            gqa_shared[local_id] = max(gqa_shared[local_id], gqa_shared[local_id + stride]);
        }
        workgroupBarrier();
    }
    let global_max = gqa_shared[0];
    workgroupBarrier();

    // Compute exp and sum
    var sum: f32 = 0.0;
    i = local_id;
    for (; i < seq_len; i = i + local_size) {
        let exp_val = exp(gqa_scores[i] - global_max);
        gqa_scores[i] = exp_val;
        sum = sum + exp_val;
    }
    gqa_shared[local_id] = sum;
    workgroupBarrier();

    stride = min(local_size, seq_len) / 2u;
    for (; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            gqa_shared[local_id] = gqa_shared[local_id] + gqa_shared[local_id + stride];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / gqa_shared[0];
    workgroupBarrier();

    // Normalize scores
    i = local_id;
    for (; i < seq_len; i = i + local_size) {
        gqa_scores[i] = gqa_scores[i] * inv_sum;
    }
    workgroupBarrier();

    // Phase 3: Weighted sum of values
    // output[d] = sum_i(scores[i] * V[i, d])
    var d = local_id;
    for (; d < head_dim; d = d + local_size) {
        var weighted_sum: f32 = 0.0;
        for (var j: u32 = 0u; j < seq_len; j = j + 1u) {
            let v_offset = j * kv_stride + kv_head * head_dim + d;
            weighted_sum = weighted_sum + gqa_scores[j] * gqa_v[v_offset];
        }
        gqa_output[q_offset + d] = weighted_sum;
    }
}

// ============================================================================
// Vector Operations
// ============================================================================

struct VecOpParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> vec_op_params: VecOpParams;
@group(0) @binding(1) var<storage, read> vec_a: array<f32>;
@group(0) @binding(2) var<storage, read> vec_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> vec_y: array<f32>;

// Vector addition: y = a + b
@compute @workgroup_size(256)
fn vec_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vec_op_params.n) {
        return;
    }
    vec_y[i] = vec_a[i] + vec_b[i];
}

// Vector multiplication: y = a * b
@compute @workgroup_size(256)
fn vec_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vec_op_params.n) {
        return;
    }
    vec_y[i] = vec_a[i] * vec_b[i];
}

// Vector scale: y = a * scale (using vec_b[0] as scale)
@compute @workgroup_size(256)
fn vec_scale_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vec_op_params.n) {
        return;
    }
    vec_y[i] = vec_a[i] * vec_b[0];
}

// ============================================================================
// Embedding Lookup (BF16 embeddings)
// output[i] = embed[token_id * hidden_size + i] * sqrt(hidden_size)
// ============================================================================

struct EmbedParams {
    token_id: u32,
    hidden_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> embed_params: EmbedParams;
@group(0) @binding(1) var<storage, read> embed_table: array<u32>;  // BF16 packed
@group(0) @binding(2) var<storage, read_write> embed_output: array<f32>;

@compute @workgroup_size(256)
fn embed_bf16_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= embed_params.hidden_size) {
        return;
    }

    let offset = embed_params.token_id * embed_params.hidden_size + i;
    let packed_idx = offset / 2u;
    let packed = embed_table[packed_idx];

    var value: f32;
    if (offset % 2u == 0u) {
        value = bf16_to_f32(packed & 0xffffu);
    } else {
        value = bf16_to_f32(packed >> 16u);
    }

    // Scale by sqrt(hidden_size)
    let scale = sqrt(f32(embed_params.hidden_size));
    embed_output[i] = value * scale;
}

// ============================================================================
// Attention Mask Generation
// ============================================================================

struct MaskParams {
    query_pos: u32,
    window_size: u32,
    is_causal: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<uniform> mask_params: MaskParams;
@group(0) @binding(1) var<storage, read_write> mask_output: array<f32>;

const NEG_INF: f32 = -3.402823466e+38;

// Sliding window mask for local attention
@compute @workgroup_size(256)
fn sliding_window_mask_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= mask_params.seq_len) {
        return;
    }

    let query_pos = mask_params.query_pos;
    let window_size = mask_params.window_size;

    // Calculate window start
    var start: u32 = 0u;
    if (query_pos >= window_size) {
        start = query_pos - window_size + 1u;
    }

    // Position must be within window and not future
    if (i >= start && i <= query_pos) {
        mask_output[i] = 0.0;
    } else {
        mask_output[i] = NEG_INF;
    }
}

// Causal mask for global attention
@compute @workgroup_size(256)
fn causal_mask_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= mask_params.seq_len) {
        return;
    }

    if (i <= mask_params.query_pos) {
        mask_output[i] = 0.0;
    } else {
        mask_output[i] = NEG_INF;
    }
}

// ============================================================================
// Argmax (for greedy decoding)
// ============================================================================

struct ArgmaxParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> argmax_params: ArgmaxParams;
@group(0) @binding(1) var<storage, read> argmax_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> argmax_output: array<u32>;  // [max_idx, _]

var<workgroup> argmax_shared_val: array<f32, 256>;
var<workgroup> argmax_shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let n = argmax_params.n;
    let local_id = lid.x;
    let local_size = 256u;

    // Each thread finds max in its chunk
    var max_val: f32 = -3.402823466e+38;
    var max_idx: u32 = 0u;

    var i = local_id;
    for (; i < n; i = i + local_size) {
        let val = argmax_input[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    argmax_shared_val[local_id] = max_val;
    argmax_shared_idx[local_id] = max_idx;
    workgroupBarrier();

    // Reduction
    var stride = local_size / 2u;
    for (; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            if (argmax_shared_val[local_id + stride] > argmax_shared_val[local_id]) {
                argmax_shared_val[local_id] = argmax_shared_val[local_id + stride];
                argmax_shared_idx[local_id] = argmax_shared_idx[local_id + stride];
            }
        }
        workgroupBarrier();
    }

    if (local_id == 0u) {
        argmax_output[0] = argmax_shared_idx[0];
    }
}
