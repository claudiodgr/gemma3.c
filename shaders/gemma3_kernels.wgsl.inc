// Auto-generated from gemma3_kernels.wgsl - do not edit
// Split into chunks for MSVC compatibility (C2026: max 16380 chars per literal)
"// gemma3_kernels.wgsl - WebGPU Compute Shaders for Gemma 3 Inference\n\
//\n\
// This file contains all WGSL compute shaders for accelerating Gemma 3 inference.\n\
// Shaders handle BF16 weights with F32 activations.\n\
\n\
// ============================================================================\n\
// Common Utilities\n\
// ============================================================================\n\
\n\
// BF16 to F32 conversion\n\
// BF16 is stored as uint16, representing the upper 16 bits of an IEEE 754 float\n\
fn bf16_to_f32(bf16: u32) -> f32 {\n\
    // Shift left 16 bits to get F32 bit pattern\n\
    let bits = bf16 << 16u;\n\
    return bitcast<f32>(bits);\n\
}\n\
\n\
// F32 to BF16 conversion (with rounding)\n\
fn f32_to_bf16(f: f32) -> u32 {\n\
    let bits = bitcast<u32>(f);\n\
    // Round to nearest even\n\
    let rounding = ((bits >> 16u) & 1u) + 0x7fffu;\n\
    return (bits + rounding) >> 16u;\n\
}\n\
\n\
// ============================================================================\n\
// Matrix-Vector Multiplication (BF16 weights, F32 input/output)\n\
// y[i] = sum_k(A[i,k] * x[k]) where A is BF16, x and y are F32\n\
// ============================================================================\n\
\n\
struct MatvecParams {\n\
    M: u32,       // Output dimension (rows)\n\
    K: u32,       // Input dimension (cols)\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> matvec_params: MatvecParams;\n\
@group(0) @binding(1) var<storage, read> matvec_A: array<u32>;      // BF16 packed as u32 (2 per element)\n\
@group(0) @binding(2) var<storage, read> matvec_x: array<f32>;      // Input vector\n\
@group(0) @binding(3) var<storage, read_write> matvec_y: array<f32>; // Output vector\n\
\n\
@compute @workgroup_size(256)\n\
fn matvec_bf16_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let row = gid.x;\n\
    if (row >= matvec_params.M) {\n\
        return;\n\
    }\n\
\n\
    let K = matvec_params.K;\n\
    var sum: f32 = 0.0;\n\
\n\
    // Process 2 BF16 values at a time (packed in u32)\n\
    let row_offset = row * K;\n\
    var k: u32 = 0u;\n\
\n\
    // Main loop: process pairs of BF16 values\n\
    for (; k + 1u < K; k = k + 2u) {\n\
        let packed = matvec_A[(row_offset + k) / 2u];\n\
        let a0 = bf16_to_f32(packed & 0xffffu);\n\
        let a1 = bf16_to_f32(packed >> 16u);\n\
        sum = sum + a0 * matvec_x[k] + a1 * matvec_x[k + 1u];\n\
    }\n\
\n\
    // Handle odd K\n\
    if (k < K) {\n\
        let packed = matvec_A[(row_offset + k) / 2u];\n\
        let a0 = bf16_to_f32(packed & 0xffffu);\n\
        sum = sum + a0 * matvec_x[k];\n\
    }\n\
\n\
    matvec_y[row] = sum;\n\
}\n\
\n\
// Optimized version with workgroup-level reduction for better memory coalescing\n\
@compute @workgroup_size(256)\n\
fn matvec_bf16_kernel_tiled(\n\
    @builtin(global_invocation_id) gid: vec3<u32>,\n\
    @builtin(local_invocation_id) lid: vec3<u32>,\n\
    @builtin(workgroup_id) wgid: vec3<u32>\n\
) {\n\
    // Each workgroup computes one row\n\
    let row = wgid.x;\n\
    if (row >= matvec_params.M) {\n\
        return;\n\
    }\n\
\n\
    let K = matvec_params.K;\n\
    let local_id = lid.x;\n\
    let local_size = 256u;\n\
\n\
    // Each thread processes a chunk of the row\n\
    var sum: f32 = 0.0;\n\
    let row_offset = row * K;\n\
\n\
    var k = local_id * 2u;\n\
    for (; k < K; k = k + local_size * 2u) {\n\
        if (k + 1u < K) {\n\
            let packed = matvec_A[(row_offset + k) / 2u];\n\
            let a0 = bf16_to_f32(packed & 0xffffu);\n\
            let a1 = bf16_to_f32(packed >> 16u);\n\
            sum = sum + a0 * matvec_x[k] + a1 * matvec_x[k + 1u];\n\
        } else if (k < K) {\n\
            let packed = matvec_A[(row_offset + k) / 2u];\n\
            let a0 = bf16_to_f32(packed & 0xffffu);\n\
            sum = sum + a0 * matvec_x[k];\n\
        }\n\
    }\n\
\n\
    // Workgroup reduction (using subgroups if available, otherwise shared memory)\n\
    // For simplicity, use atomicAdd\n\
    // Note: WGSL doesn't have atomicAdd for f32, so we use a different approach\n\
\n\
    // Store partial sum (simplified - in production use proper reduction)\n\
    if (local_id == 0u) {\n\
        matvec_y[row] = sum;\n\
    }\n\
}\n\
\n\
// ============================================================================\n\
// RMSNorm (BF16 weights, F32 input/output)\n\
// y[i] = x[i] * rsqrt(mean(x^2) + eps) * (1.0 + weight[i])\n\
// ============================================================================\n\
\n\
struct RmsnormParams {\n\
    n: u32,\n\
    eps: f32,\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> rmsnorm_params: RmsnormParams;\n\
@group(0) @binding(1) var<storage, read> rmsnorm_x: array<f32>;\n\
@group(0) @binding(2) var<storage, read> rmsnorm_weight: array<u32>;  // BF16 packed\n\
@group(0) @binding(3) var<storage, read_write> rmsnorm_y: array<f32>;\n\
@group(0) @binding(4) var<storage, read_write> rmsnorm_scratch: array<f32>;  // For reduction\n\
\n\
var<workgroup> rmsnorm_shared: array<f32, 256>;\n\
\n\
@compute @workgroup_size(256)\n\
fn rmsnorm_bf16_kernel(\n\
    @builtin(global_invocation_id) gid: vec3<u32>,\n\
    @builtin(local_invocation_id) lid: vec3<u32>\n\
) {\n\
    let n = rmsnorm_params.n;\n\
    let local_id = lid.x;\n\
    let local_size = 256u;\n\
\n\
    // Phase 1: Compute partial sum of squares\n\
    var sum_sq: f32 = 0.0;\n\
    var i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        let xi = rmsnorm_x[i];\n\
        sum_sq = sum_sq + xi * xi;\n\
    }\n\
\n\
    // Store in shared memory\n\
    rmsnorm_shared[local_id] = sum_sq;\n\
    workgroupBarrier();\n\
\n\
    // Reduction in shared memory\n\
    var stride = local_size / 2u;\n\
    for (; stride > 0u; stride = stride / 2u) {\n\
        if (local_id < stride) {\n\
            rmsnorm_shared[local_id] = rmsnorm_shared[local_id] + rmsnorm_shared[local_id + stride];\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
\n\
    // Compute normalization factor\n\
    let mean_sq = rmsnorm_shared[0] / f32(n);\n\
    let rsqrt_val = inverseSqrt(mean_sq + rmsnorm_params.eps);\n\
\n\
    workgroupBarrier();\n\
\n\
    // Phase 2: Apply normalization with Gemma's (1 + weight) formula\n\
    i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        let xi = rmsnorm_x[i];\n\
\n\
        // Extract BF16 weight\n\
        let packed_idx = i / 2u;\n\
        let packed = rmsnorm_weight[packed_idx];\n\
        var w: f32;\n\
        if (i % 2u == 0u) {\n\
            w = bf16_to_f32(packed & 0xffffu);\n\
        } else {\n\
            w = bf16_to_f32(packed >> 16u);\n\
        }\n\
\n\
        rmsnorm_y[i] = xi * rsqrt_val * (1.0 + w);\n\
    }\n\
}\n\
\n\
// In-place variant\n\
@compute @workgroup_size(256)\n\
fn rmsnorm_bf16_inplace_kernel(\n\
    @builtin(global_invocation_id) gid: vec3<u32>,\n\
    @builtin(local_invocation_id) lid: vec3<u32>\n\
) {\n\
    let n = rmsnorm_params.n;\n\
    let local_id = lid.x;\n\
    let local_size = 256u;\n\
\n\
    // Phase 1: Compute partial sum of squares\n\
    var sum_sq: f32 = 0.0;\n\
    var i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        let xi = rmsnorm_y[i];  // In-place: read from output buffer\n\
        sum_sq = sum_sq + xi * xi;\n\
    }\n\
\n\
    rmsnorm_shared[local_id] = sum_sq;\n\
    workgroupBarrier();\n\
\n\
    // Reduction\n\
    var stride = local_size / 2u;\n\
    for (; stride > 0u; stride = stride / 2u) {\n\
        if (local_id < stride) {\n\
            rmsnorm_shared[local_id] = rmsnorm_shared[local_id] + rmsnorm_shared[local_id + stride];\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
\n\
    let mean_sq = rmsnorm_shared[0] / f32(n);\n\
    let rsqrt_val = inverseSqrt(mean_sq + rmsnorm_params.eps);\n\
\n\
    workgroupBarrier();\n\
\n\
    // Phase 2: Apply in-place\n\
    i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        let xi = rmsnorm_y[i];\n\
        let packed_idx = i / 2u;\n\
        let packed = rmsnorm_weight[packed_idx];\n\
        var w: f32;\n\
        if (i % 2u == 0u) {\n\
            w = bf16_to_f32(packed & 0xffffu);\n\
        } else {\n\
            w = bf16_to_f32(packed >> 16u);\n\
        }\n\
        rmsnorm_y[i] = xi * rsqrt_val * (1.0 + w);\n\
    }\n\
}\n\
\n\
// ============================================================================\n\
// GELU Activation (tanh approximation)\n\
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n\
// ============================================================================\n\
\n\
struct GeluParams {\n\
    n: u32,\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
    _pad2: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> gelu_params: GeluParams;\n\
@group(0) @binding(1) var<storage, read_write> gelu_x: array<f32>;\n\
\n\
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;\n\
const GELU_COEFF: f32 = 0.044715;\n\
\n\
@compute @workgroup_size(256)\n\
fn gelu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= gelu_params.n) {\n\
        return;\n\
    }\n\
\n\
    let x = gelu_x[i];\n\
    let x3 = x * x * x;\n\
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);\n\
    gelu_x[i] = 0.5 * x * (1.0 + tanh(inner));\n\
}\n\
\n\
// ============================================================================\n\
// SiLU/Swish Activation\n\
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))\n\
// ============================================================================\n\
\n\
@group(0) @binding(0) var<uniform> silu_params: GeluParams;  // Reuse struct\n\
@group(0) @binding(1) var<storage, read_write> silu_x: array<f32>;\n\
\n\
@compute @workgroup_size(256)\n\
fn silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= silu_params.n) {\n\
        return;\n\
    }\n\
\n\
    let x = silu_x[i];\n\
    silu_x[i] = x / (1.0 + exp(-x));\n\
}\n\
\n\
// ============================================================================\n\
// Softmax (numerically stable)\n\
// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))\n\
// ============================================================================\n\
\n\
struct SoftmaxParams {\n\
    n: u32,\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
    _pad2: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> softmax_params: SoftmaxParams;\n\
@group(0) @binding(1) var<storage, read_write> softmax_x: array<f32>;\n\
\n\
var<workgroup> softmax_shared: array<f32, 256>;\n\
\n\
@compute @workgroup_size(256)\n\
fn softmax_kernel(\n\
    @builtin(global_invocation_id) gid: vec3<u32>,\n\
    @builtin(local_invocation_id) lid: vec3<u32>\n\
) {\n\
    let n = softmax_params.n;\n\
    let local_id = lid.x;\n\
    let local_size = 256u;\n\
\n\
    // Phase 1: Find maximum\n\
    var max_val: f32 = -3.402823466e+38;  // -FLT_MAX\n\
    var i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        max_val = max(max_val, softmax_x[i]);\n\
    }\n\
\n\
    softmax_shared[local_id] = max_val;\n\
    workgroupBarrier();\n\
\n\
    // Reduction for max\n\
    var stride = local_size / 2u;\n\
    for (; stride > 0u; stride = stride / 2u) {\n\
        if (local_id < stride) {\n\
            softmax_shared[local_id] = max(softmax_shared[local_id], softmax_shared[local_id + stride]);\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
\n\
    let global_max = softmax_shared[0];\n\
    workgroupBarrier();\n\
\n\
    // Phase 2: Compute exp(x - max) and sum\n\
    var sum: f32 = 0.0;\n\
    i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        let exp_val = exp(softmax_x[i] - global_max);\n\
        softmax_x[i] = exp_val;\n\
        sum = sum + exp_val;\n\
    }\n\
\n\
    softmax_shared[local_id] = sum;\n\
    workgroupBarrier();\n\
\n\
    // Reduction for sum\n\
    stride = local_size / 2u;\n\
    for (; stride > 0u; stride = stride / 2u) {\n\
        if (local_id < stride) {\n\
            softmax_shared[local_id] = softmax_shared[local_id] + softmax_shared[local_id + stride];\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
\n\
    let inv_sum = 1.0 / softmax_shared[0];\n\
    workgroupBarrier();\n\
\n\
    // Phase 3: Normalize\n\
    i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        softmax_x[i] = softmax_x[i] * inv_sum;\n\
    }\n\
}\n\
\n\
// ============================================================================\n\
// RoPE (Rotary Position Embeddings)\n\
// For each dimension pair (i, i + d/2):\n\
// [x_i, x_{i+d/2}] = [cos(Î¸)*x_i - sin(Î¸)*x_{i+d/2}, sin(Î¸)*x_i + cos(Î¸)*x_{i+d/2}]\n\
// where Î¸ = pos * freq, freq = 1 / (theta^(2i/d))\n\
// ============================================================================\n\
\n\
struct RopeParams {\n\
    head_dim: u32,\n\
    pos: u32,\n\
    theta: f32,\n\
    num_heads: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> rope_params: RopeParams;\n\
@group(0) @binding(1) var<storage, read_write> rope_x: array<f32>;\n\
\n\
@compute @workgroup_size(256)\n\
fn rope_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let half_dim = rope_params.head_dim / 2u;\n\
    let total_pairs = rope_params.num_heads * half_dim;\n\
\n\
    let pair_idx = gid.x;\n\
    if (pair_idx >= total_pairs) {\n\
        return;\n\
    }\n\
\n\
    let head = pair_idx / half_dim;\n\
    let i = pair_idx % half_dim;\n\
\n\
    // Compute frequency\n\
    let exponent = f32(2u * i) / f32(rope_params.head_dim);\n\
    let freq = 1.0 / pow(rope_params.theta, exponent);\n\
    let angle = f32(rope_params.pos) * freq;\n\
    let cos_val = cos(angle);\n\
    let sin_val = sin(angle);\n\
\n\
    // Get indices\n\
    let idx0 = head * rope_params.head_dim + i;\n\
    let idx1 = idx0 + half_dim;\n\
\n\
    // Apply rotation\n\
    let x0 = rope_x[idx0];\n\
    let x1 = rope_x[idx1];\n\
    rope_x[idx0] = x0 * cos_val - x1 * sin_val;\n\
    rope_x[idx1] = x0 * sin_val + x1 * cos_val;\n\
}\n\
\n\
// ============================================================================\n\
// Grouped Query Attention (GQA)\n\
// output[h] = softmax(Q[h] @ K^T * scale + mask) @ V\n\
// With head grouping: each KV head serves multiple Q heads\n\
// ============================================================================\n\
\n\
struct GqaParams {\n"
"    n_heads: u32,\n\
    n_kv_heads: u32,\n\
    seq_len: u32,\n\
    head_dim: u32,\n\
    scale: f32,\n\
    use_mask: u32,\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> gqa_params: GqaParams;\n\
@group(0) @binding(1) var<storage, read> gqa_q: array<f32>;       // [n_heads, head_dim]\n\
@group(0) @binding(2) var<storage, read> gqa_k: array<f32>;       // [seq_len, n_kv_heads, head_dim]\n\
@group(0) @binding(3) var<storage, read> gqa_v: array<f32>;       // [seq_len, n_kv_heads, head_dim]\n\
@group(0) @binding(4) var<storage, read> gqa_mask: array<f32>;    // [seq_len]\n\
@group(0) @binding(5) var<storage, read_write> gqa_output: array<f32>; // [n_heads, head_dim]\n\
@group(0) @binding(6) var<storage, read_write> gqa_scores: array<f32>; // Scratch [seq_len]\n\
\n\
var<workgroup> gqa_shared: array<f32, 256>;\n\
\n\
// Each workgroup processes one head\n\
@compute @workgroup_size(256)\n\
fn gqa_kernel(\n\
    @builtin(workgroup_id) wgid: vec3<u32>,\n\
    @builtin(local_invocation_id) lid: vec3<u32>\n\
) {\n\
    let head = wgid.x;\n\
    if (head >= gqa_params.n_heads) {\n\
        return;\n\
    }\n\
\n\
    let local_id = lid.x;\n\
    let local_size = 256u;\n\
    let head_dim = gqa_params.head_dim;\n\
    let seq_len = gqa_params.seq_len;\n\
    let n_kv_heads = gqa_params.n_kv_heads;\n\
    let scale = gqa_params.scale;\n\
\n\
    // Determine which KV head this Q head uses\n\
    let heads_per_group = gqa_params.n_heads / n_kv_heads;\n\
    let kv_head = head / heads_per_group;\n\
    let kv_stride = n_kv_heads * head_dim;\n\
\n\
    // Pointers\n\
    let q_offset = head * head_dim;\n\
\n\
    // Phase 1: Compute attention scores\n\
    // scores[i] = (Q Â· K[i]) * scale + mask[i]\n\
    var i = local_id;\n\
    for (; i < seq_len; i = i + local_size) {\n\
        var score: f32 = 0.0;\n\
        let k_offset = i * kv_stride + kv_head * head_dim;\n\
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {\n\
            score = score + gqa_q[q_offset + d] * gqa_k[k_offset + d];\n\
        }\n\
        score = score * scale;\n\
        if (gqa_params.use_mask != 0u) {\n\
            score = score + gqa_mask[i];\n\
        }\n\
        gqa_scores[i] = score;\n\
    }\n\
    workgroupBarrier();\n\
\n\
    // Phase 2: Softmax over scores\n\
    // Find max\n\
    var max_val: f32 = -3.402823466e+38;\n\
    i = local_id;\n\
    for (; i < seq_len; i = i + local_size) {\n\
        max_val = max(max_val, gqa_scores[i]);\n\
    }\n\
    gqa_shared[local_id] = max_val;\n\
    workgroupBarrier();\n\
\n\
    var stride = min(local_size, seq_len) / 2u;\n\
    for (; stride > 0u; stride = stride / 2u) {\n\
        if (local_id < stride) {\n\
            gqa_shared[local_id] = max(gqa_shared[local_id], gqa_shared[local_id + stride]);\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
    let global_max = gqa_shared[0];\n\
    workgroupBarrier();\n\
\n\
    // Compute exp and sum\n\
    var sum: f32 = 0.0;\n\
    i = local_id;\n\
    for (; i < seq_len; i = i + local_size) {\n\
        let exp_val = exp(gqa_scores[i] - global_max);\n\
        gqa_scores[i] = exp_val;\n\
        sum = sum + exp_val;\n\
    }\n\
    gqa_shared[local_id] = sum;\n\
    workgroupBarrier();\n\
\n\
    stride = min(local_size, seq_len) / 2u;\n\
    for (; stride > 0u; stride = stride / 2u) {\n\
        if (local_id < stride) {\n\
            gqa_shared[local_id] = gqa_shared[local_id] + gqa_shared[local_id + stride];\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
    let inv_sum = 1.0 / gqa_shared[0];\n\
    workgroupBarrier();\n\
\n\
    // Normalize scores\n\
    i = local_id;\n\
    for (; i < seq_len; i = i + local_size) {\n\
        gqa_scores[i] = gqa_scores[i] * inv_sum;\n\
    }\n\
    workgroupBarrier();\n\
\n\
    // Phase 3: Weighted sum of values\n\
    // output[d] = sum_i(scores[i] * V[i, d])\n\
    var d = local_id;\n\
    for (; d < head_dim; d = d + local_size) {\n\
        var weighted_sum: f32 = 0.0;\n\
        for (var j: u32 = 0u; j < seq_len; j = j + 1u) {\n\
            let v_offset = j * kv_stride + kv_head * head_dim + d;\n\
            weighted_sum = weighted_sum + gqa_scores[j] * gqa_v[v_offset];\n\
        }\n\
        gqa_output[q_offset + d] = weighted_sum;\n\
    }\n\
}\n\
\n\
// ============================================================================\n\
// Vector Operations\n\
// ============================================================================\n\
\n\
struct VecOpParams {\n\
    n: u32,\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
    _pad2: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> vec_op_params: VecOpParams;\n\
@group(0) @binding(1) var<storage, read> vec_a: array<f32>;\n\
@group(0) @binding(2) var<storage, read> vec_b: array<f32>;\n\
@group(0) @binding(3) var<storage, read_write> vec_y: array<f32>;\n\
\n\
// Vector addition: y = a + b\n\
@compute @workgroup_size(256)\n\
fn vec_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= vec_op_params.n) {\n\
        return;\n\
    }\n\
    vec_y[i] = vec_a[i] + vec_b[i];\n\
}\n\
\n\
// Vector multiplication: y = a * b\n\
@compute @workgroup_size(256)\n\
fn vec_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= vec_op_params.n) {\n\
        return;\n\
    }\n\
    vec_y[i] = vec_a[i] * vec_b[i];\n\
}\n\
\n\
// Vector scale: y = a * scale (using vec_b[0] as scale)\n\
@compute @workgroup_size(256)\n\
fn vec_scale_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= vec_op_params.n) {\n\
        return;\n\
    }\n\
    vec_y[i] = vec_a[i] * vec_b[0];\n\
}\n\
\n\
// ============================================================================\n\
// Embedding Lookup (BF16 embeddings)\n\
// output[i] = embed[token_id * hidden_size + i] * sqrt(hidden_size)\n\
// ============================================================================\n\
\n\
struct EmbedParams {\n\
    token_id: u32,\n\
    hidden_size: u32,\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> embed_params: EmbedParams;\n\
@group(0) @binding(1) var<storage, read> embed_table: array<u32>;  // BF16 packed\n\
@group(0) @binding(2) var<storage, read_write> embed_output: array<f32>;\n\
\n\
@compute @workgroup_size(256)\n\
fn embed_bf16_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= embed_params.hidden_size) {\n\
        return;\n\
    }\n\
\n\
    let offset = embed_params.token_id * embed_params.hidden_size + i;\n\
    let packed_idx = offset / 2u;\n\
    let packed = embed_table[packed_idx];\n\
\n\
    var value: f32;\n\
    if (offset % 2u == 0u) {\n\
        value = bf16_to_f32(packed & 0xffffu);\n\
    } else {\n\
        value = bf16_to_f32(packed >> 16u);\n\
    }\n\
\n\
    // Scale by sqrt(hidden_size)\n\
    let scale = sqrt(f32(embed_params.hidden_size));\n\
    embed_output[i] = value * scale;\n\
}\n\
\n\
// ============================================================================\n\
// Attention Mask Generation\n\
// ============================================================================\n\
\n\
struct MaskParams {\n\
    query_pos: u32,\n\
    window_size: u32,\n\
    is_causal: u32,\n\
    seq_len: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> mask_params: MaskParams;\n\
@group(0) @binding(1) var<storage, read_write> mask_output: array<f32>;\n\
\n\
const NEG_INF: f32 = -3.402823466e+38;\n\
\n\
// Sliding window mask for local attention\n\
@compute @workgroup_size(256)\n\
fn sliding_window_mask_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= mask_params.seq_len) {\n\
        return;\n\
    }\n\
\n\
    let query_pos = mask_params.query_pos;\n\
    let window_size = mask_params.window_size;\n\
\n\
    // Calculate window start\n\
    var start: u32 = 0u;\n\
    if (query_pos >= window_size) {\n\
        start = query_pos - window_size + 1u;\n\
    }\n\
\n\
    // Position must be within window and not future\n\
    if (i >= start && i <= query_pos) {\n\
        mask_output[i] = 0.0;\n\
    } else {\n\
        mask_output[i] = NEG_INF;\n\
    }\n\
}\n\
\n\
// Causal mask for global attention\n\
@compute @workgroup_size(256)\n\
fn causal_mask_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= mask_params.seq_len) {\n\
        return;\n\
    }\n\
\n\
    if (i <= mask_params.query_pos) {\n\
        mask_output[i] = 0.0;\n\
    } else {\n\
        mask_output[i] = NEG_INF;\n\
    }\n\
}\n\
\n\
// ============================================================================\n\
// Argmax (for greedy decoding)\n\
// ============================================================================\n\
\n\
struct ArgmaxParams {\n\
    n: u32,\n\
    _pad0: u32,\n\
    _pad1: u32,\n\
    _pad2: u32,\n\
}\n\
\n\
@group(0) @binding(0) var<uniform> argmax_params: ArgmaxParams;\n\
@group(0) @binding(1) var<storage, read> argmax_input: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> argmax_output: array<u32>;  // [max_idx, _]\n\
\n\
var<workgroup> argmax_shared_val: array<f32, 256>;\n\
var<workgroup> argmax_shared_idx: array<u32, 256>;\n\
\n\
@compute @workgroup_size(256)\n\
fn argmax_kernel(\n\
    @builtin(local_invocation_id) lid: vec3<u32>\n\
) {\n\
    let n = argmax_params.n;\n\
    let local_id = lid.x;\n\
    let local_size = 256u;\n\
\n\
    // Each thread finds max in its chunk\n\
    var max_val: f32 = -3.402823466e+38;\n\
    var max_idx: u32 = 0u;\n\
\n\
    var i = local_id;\n\
    for (; i < n; i = i + local_size) {\n\
        let val = argmax_input[i];\n\
        if (val > max_val) {\n\
            max_val = val;\n\
            max_idx = i;\n\
        }\n\
    }\n\
\n\
    argmax_shared_val[local_id] = max_val;\n\
    argmax_shared_idx[local_id] = max_idx;\n\
    workgroupBarrier();\n\
\n\
    // Reduction\n\
    var stride = local_size / 2u;\n\
    for (; stride > 0u; stride = stride / 2u) {\n\
        if (local_id < stride) {\n\
            if (argmax_shared_val[local_id + stride] > argmax_shared_val[local_id]) {\n\
                argmax_shared_val[local_id] = argmax_shared_val[local_id + stride];\n\
                argmax_shared_idx[local_id] = argmax_shared_idx[local_id + stride];\n\
            }\n\
        }\n\
        workgroupBarrier();\n\
    }\n\
\n\
    if (local_id == 0u) {\n\
        argmax_output[0] = argmax_shared_idx[0];\n\
    }\n\
}\n"