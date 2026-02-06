/*
 * gemma3_webgpu.c - WebGPU acceleration implementation for Gemma 3 inference
 *
 * Provides GPU-accelerated compute kernels using WebGPU (wgpu-native or Dawn).
 * Handles device initialization, shader compilation, and kernel dispatch.
 *
 * Compatible with wgpu-native v27.x API.
 */

#ifdef USE_WEBGPU

#include "gemma3_webgpu.h"
#include "gemma3_platform.h"
#include <webgpu/wgpu.h>  /* wgpu-native extensions (wgpuDevicePoll) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

/* ============================================================================
 * WGPUStringView helper
 * ========================================================================== */

static WGPUStringView wgpu_str(const char *s) {
    WGPUStringView sv;
    sv.data = s;
    sv.length = s ? strlen(s) : 0;
    return sv;
}

/* ============================================================================
 * Embedded Shader Source
 * ========================================================================== */

/* Include the WGSL shader source as a string */
static const char *SHADER_SOURCE =
#include "shaders/gemma3_kernels.wgsl.inc"
;

/* Alternative: Load from file at runtime */
static char *load_shader_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *source = (char *)malloc(size + 1);
    if (!source) {
        fclose(f);
        return NULL;
    }

    fread(source, 1, size, f);
    source[size] = '\0';
    fclose(f);

    return source;
}

/* ============================================================================
 * Error Handling
 * ========================================================================== */

static GEMMA3_THREAD_LOCAL char g_gpu_error[512] = {0};

static void set_gpu_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_gpu_error, sizeof(g_gpu_error), fmt, args);
    va_end(args);
    fprintf(stderr, "[WebGPU Error] %s\n", g_gpu_error);
}

const char *gemma3_gpu_get_error(void) {
    return g_gpu_error;
}

/* ============================================================================
 * WebGPU Callbacks (v27.x API)
 * ========================================================================== */

static void on_device_error(WGPUDevice const *device, WGPUErrorType type,
                            WGPUStringView message, void *userdata1, void *userdata2) {
    (void)device;
    (void)userdata1;
    (void)userdata2;
    const char *type_str = "Unknown";
    switch (type) {
        case WGPUErrorType_Validation: type_str = "Validation"; break;
        case WGPUErrorType_OutOfMemory: type_str = "OutOfMemory"; break;
        case WGPUErrorType_Internal: type_str = "Internal"; break;
        case WGPUErrorType_Unknown: type_str = "Unknown"; break;
        default: break;
    }
    set_gpu_error("Device error (%s): %.*s", type_str,
                  (int)message.length, message.data ? message.data : "");
}

static void on_adapter_request(WGPURequestAdapterStatus status,
                               WGPUAdapter adapter,
                               WGPUStringView message,
                               void *userdata1, void *userdata2) {
    (void)userdata2;
    if (status != WGPURequestAdapterStatus_Success) {
        set_gpu_error("Adapter request failed: %.*s",
                      (int)message.length, message.data ? message.data : "unknown");
        return;
    }
    *(WGPUAdapter *)userdata1 = adapter;
}

static void on_device_request(WGPURequestDeviceStatus status,
                              WGPUDevice device,
                              WGPUStringView message,
                              void *userdata1, void *userdata2) {
    (void)userdata2;
    if (status != WGPURequestDeviceStatus_Success) {
        set_gpu_error("Device request failed: %.*s",
                      (int)message.length, message.data ? message.data : "unknown");
        return;
    }
    *(WGPUDevice *)userdata1 = device;
}

/* Buffer map callback (v27.x signature) */
typedef struct {
    int done;
    WGPUMapAsyncStatus status;
} MapContext;

static void on_buffer_mapped(WGPUMapAsyncStatus status, WGPUStringView message,
                             void *userdata1, void *userdata2) {
    (void)message;
    (void)userdata2;
    MapContext *ctx = (MapContext *)userdata1;
    ctx->status = status;
    ctx->done = 1;
}

/* ============================================================================
 * Pipeline Creation Helpers
 * ========================================================================== */

static WGPUBindGroupLayout create_bind_group_layout(
    WGPUDevice device,
    const WGPUBindGroupLayoutEntry *entries,
    uint32_t entry_count,
    const char *label
) {
    WGPUBindGroupLayoutDescriptor desc = {0};
    desc.label = wgpu_str(label);
    desc.entryCount = entry_count;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

static WGPUComputePipeline create_compute_pipeline(
    WGPUDevice device,
    WGPUShaderModule shader,
    WGPUBindGroupLayout layout,
    const char *entry_point,
    const char *label
) {
    WGPUPipelineLayoutDescriptor layout_desc = {0};
    layout_desc.label = wgpu_str(label);
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = &layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);

    WGPUComputePipelineDescriptor desc = {0};
    desc.label = wgpu_str(label);
    desc.layout = pipeline_layout;
    desc.compute.module = shader;
    desc.compute.entryPoint = wgpu_str(entry_point);

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &desc);
    wgpuPipelineLayoutRelease(pipeline_layout);

    return pipeline;
}

/* ============================================================================
 * Buffer Management
 * ========================================================================== */

gemma3_gpu_buffer gemma3_gpu_create_buffer(
    gemma3_gpu_context *ctx,
    size_t size,
    WGPUBufferUsage usage
) {
    gemma3_gpu_buffer buf = {0};

    WGPUBufferDescriptor desc = {0};
    desc.size = size;
    desc.usage = usage;
    desc.mappedAtCreation = 0;

    buf.buffer = wgpuDeviceCreateBuffer(ctx->device, &desc);
    buf.size = size;
    buf.usage = usage;

    return buf;
}

void gemma3_gpu_write_buffer(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *buf,
    const void *data,
    size_t size
) {
    wgpuQueueWriteBuffer(ctx->queue, buf->buffer, 0, data, size);
}

void gemma3_gpu_read_buffer(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *buf,
    void *data,
    size_t size
) {
    /* Create staging buffer for readback */
    WGPUBufferDescriptor staging_desc = {0};
    staging_desc.size = size;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    staging_desc.mappedAtCreation = 0;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx->device, &staging_desc);

    /* Copy from GPU buffer to staging */
    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buf->buffer, 0, staging, 0, size);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    /* Wait for completion and map */
    wgpuDevicePoll(ctx->device, 1, NULL);

    /* Map buffer using v27 callback info struct */
    MapContext map_ctx = {0, WGPUMapAsyncStatus_Unknown};

    WGPUBufferMapCallbackInfo map_cb = {0};
    map_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    map_cb.callback = on_buffer_mapped;
    map_cb.userdata1 = &map_ctx;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, map_cb);

    while (!map_ctx.done) {
        wgpuDevicePoll(ctx->device, 1, NULL);
    }

    if (map_ctx.status == WGPUMapAsyncStatus_Success) {
        const void *mapped = wgpuBufferGetConstMappedRange(staging, 0, size);
        memcpy(data, mapped, size);
        wgpuBufferUnmap(staging);
    }

    wgpuBufferRelease(staging);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);
}

void gemma3_gpu_destroy_buffer(gemma3_gpu_buffer *buf) {
    if (buf->buffer) {
        wgpuBufferDestroy(buf->buffer);
        wgpuBufferRelease(buf->buffer);
        buf->buffer = NULL;
    }
}

/* ============================================================================
 * Initialization
 * ========================================================================== */

int gemma3_gpu_available(void) {
    /* Try to create a minimal WebGPU instance */
    WGPUInstanceDescriptor desc = {0};
    WGPUInstance instance = wgpuCreateInstance(&desc);
    if (!instance) return 0;
    wgpuInstanceRelease(instance);
    return 1;
}

gemma3_gpu_context *gemma3_gpu_init(void) {
    gemma3_gpu_context *ctx = (gemma3_gpu_context *)calloc(1, sizeof(gemma3_gpu_context));
    if (!ctx) {
        set_gpu_error("Failed to allocate GPU context");
        return NULL;
    }

    /* Create instance */
    WGPUInstanceDescriptor instance_desc = {0};
    ctx->instance = wgpuCreateInstance(&instance_desc);
    if (!ctx->instance) {
        set_gpu_error("Failed to create WebGPU instance");
        free(ctx);
        return NULL;
    }

    /* Request adapter using v27 callback info struct */
    WGPURequestAdapterOptions adapter_opts = {0};
    adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;

    WGPURequestAdapterCallbackInfo adapter_cb = {0};
    adapter_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    adapter_cb.callback = on_adapter_request;
    adapter_cb.userdata1 = &ctx->adapter;

    wgpuInstanceRequestAdapter(ctx->instance, &adapter_opts, adapter_cb);

    /* Poll for adapter completion */
    wgpuInstanceProcessEvents(ctx->instance);

    if (!ctx->adapter) {
        set_gpu_error("No suitable GPU adapter found");
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }

    /* Query adapter limits and request the device with the same limits.
     * This avoids the v27 pitfall where zero-initialized WGPULimits = {0}
     * would set maxBindGroups=0, etc. */
    WGPULimits adapter_limits = {0};
    wgpuAdapterGetLimits(ctx->adapter, &adapter_limits);

    /* Set error callback via device descriptor (v27 API) */
    WGPUUncapturedErrorCallbackInfo error_cb = {0};
    error_cb.callback = on_device_error;

    WGPUDeviceDescriptor device_desc = {0};
    device_desc.requiredLimits = &adapter_limits;
    device_desc.uncapturedErrorCallbackInfo = error_cb;

    WGPURequestDeviceCallbackInfo device_cb = {0};
    device_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    device_cb.callback = on_device_request;
    device_cb.userdata1 = &ctx->device;

    wgpuAdapterRequestDevice(ctx->adapter, &device_desc, device_cb);

    /* Poll for device completion */
    wgpuInstanceProcessEvents(ctx->instance);

    if (!ctx->device) {
        set_gpu_error("Failed to create GPU device");
        wgpuAdapterRelease(ctx->adapter);
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }

    /* Get queue */
    ctx->queue = wgpuDeviceGetQueue(ctx->device);

    /* Load shader source (try file first, then embedded) */
    char *shader_source = load_shader_file("shaders/gemma3_kernels.wgsl");
    if (!shader_source) {
        shader_source = (char *)SHADER_SOURCE;
    }

    /* Create shader module (v27 uses WGPUShaderSourceWGSL) */
    WGPUShaderSourceWGSL wgsl_desc = {0};
    wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_desc.code = wgpu_str(shader_source);

    WGPUShaderModuleDescriptor shader_desc = {0};
    shader_desc.nextInChain = (WGPUChainedStruct *)&wgsl_desc;
    shader_desc.label = wgpu_str("gemma3_kernels");

    ctx->shader_module = wgpuDeviceCreateShaderModule(ctx->device, &shader_desc);

    if (shader_source != SHADER_SOURCE) {
        free(shader_source);
    }

    if (!ctx->shader_module) {
        set_gpu_error("Failed to compile shader module");
        gemma3_gpu_free(ctx);
        return NULL;
    }

    /* Create bind group layouts */

    /* Matvec layout: params, A, x, y */
    WGPUBindGroupLayoutEntry matvec_entries[] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 2, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 3, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    ctx->matvec_layout = create_bind_group_layout(ctx->device, matvec_entries, 4, "matvec_layout");

    /* RMSNorm layout: params, x, weight, y, scratch */
    WGPUBindGroupLayoutEntry rmsnorm_entries[] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 2, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 3, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
        {.binding = 4, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    ctx->rmsnorm_layout = create_bind_group_layout(ctx->device, rmsnorm_entries, 5, "rmsnorm_layout");

    /* GELU layout: params, x */
    WGPUBindGroupLayoutEntry gelu_entries[] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    ctx->gelu_layout = create_bind_group_layout(ctx->device, gelu_entries, 2, "gelu_layout");

    /* Softmax layout: params, x */
    ctx->softmax_layout = create_bind_group_layout(ctx->device, gelu_entries, 2, "softmax_layout");

    /* RoPE layout: params, x */
    WGPUBindGroupLayoutEntry rope_entries[] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    ctx->rope_layout = create_bind_group_layout(ctx->device, rope_entries, 2, "rope_layout");

    /* GQA layout: params, q, k, v, mask, output, scores */
    WGPUBindGroupLayoutEntry gqa_entries[] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 2, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 3, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 4, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 5, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
        {.binding = 6, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    ctx->gqa_layout = create_bind_group_layout(ctx->device, gqa_entries, 7, "gqa_layout");

    /* Vec op layout: params, a, b, y */
    WGPUBindGroupLayoutEntry vec_op_entries[] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 2, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 3, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    ctx->vec_op_layout = create_bind_group_layout(ctx->device, vec_op_entries, 4, "vec_op_layout");

    /* Embed layout: params, embed_table, output */
    WGPUBindGroupLayoutEntry embed_entries[] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 2, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    ctx->embed_layout = create_bind_group_layout(ctx->device, embed_entries, 3, "embed_layout");

    /* Create compute pipelines */
    ctx->matvec_bf16_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->matvec_layout,
        "matvec_bf16_kernel", "matvec_bf16");

    ctx->rmsnorm_bf16_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->rmsnorm_layout,
        "rmsnorm_bf16_kernel", "rmsnorm_bf16");

    ctx->gelu_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->gelu_layout,
        "gelu_kernel", "gelu");

    ctx->softmax_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->softmax_layout,
        "softmax_kernel", "softmax");

    ctx->rope_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->rope_layout,
        "rope_kernel", "rope");

    ctx->gqa_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->gqa_layout,
        "gqa_kernel", "gqa");

    ctx->vec_add_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->vec_op_layout,
        "vec_add_kernel", "vec_add");

    ctx->vec_mul_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->vec_op_layout,
        "vec_mul_kernel", "vec_mul");

    ctx->embed_bf16_pipeline = create_compute_pipeline(
        ctx->device, ctx->shader_module, ctx->embed_layout,
        "embed_bf16_kernel", "embed_bf16");

    /* Set default workgroup sizes */
    ctx->workgroup_size_1d = 256;
    ctx->workgroup_size_2d_x = 16;
    ctx->workgroup_size_2d_y = 16;

    fprintf(stderr, "WebGPU initialized successfully\n");
    return ctx;
}

int gemma3_gpu_init_buffers(
    gemma3_gpu_context *ctx,
    int hidden_size,
    int intermediate_size,
    int vocab_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_context
) {
    ctx->hidden_size = hidden_size;
    ctx->intermediate_size = intermediate_size;
    ctx->vocab_size = vocab_size;
    ctx->num_heads = num_heads;
    ctx->num_kv_heads = num_kv_heads;
    ctx->head_dim = head_dim;
    ctx->max_context = max_context;

    WGPUBufferUsage storage_rw = WGPUBufferUsage_Storage |
                                  WGPUBufferUsage_CopyDst |
                                  WGPUBufferUsage_CopySrc;

    /* Activation buffers */
    ctx->buf_x = gemma3_gpu_create_buffer(ctx, hidden_size * sizeof(float), storage_rw);
    ctx->buf_x_norm = gemma3_gpu_create_buffer(ctx, hidden_size * sizeof(float), storage_rw);
    ctx->buf_q = gemma3_gpu_create_buffer(ctx, num_heads * head_dim * sizeof(float), storage_rw);
    ctx->buf_k = gemma3_gpu_create_buffer(ctx, num_kv_heads * head_dim * sizeof(float), storage_rw);
    ctx->buf_v = gemma3_gpu_create_buffer(ctx, num_kv_heads * head_dim * sizeof(float), storage_rw);
    ctx->buf_attn_out = gemma3_gpu_create_buffer(ctx, num_heads * head_dim * sizeof(float), storage_rw);
    ctx->buf_proj_out = gemma3_gpu_create_buffer(ctx, hidden_size * sizeof(float), storage_rw);
    ctx->buf_mlp_gate = gemma3_gpu_create_buffer(ctx, intermediate_size * sizeof(float), storage_rw);
    ctx->buf_mlp_up = gemma3_gpu_create_buffer(ctx, intermediate_size * sizeof(float), storage_rw);
    ctx->buf_mlp_out = gemma3_gpu_create_buffer(ctx, hidden_size * sizeof(float), storage_rw);
    ctx->buf_logits = gemma3_gpu_create_buffer(ctx, vocab_size * sizeof(float), storage_rw);
    ctx->buf_attn_scores = gemma3_gpu_create_buffer(ctx, max_context * sizeof(float), storage_rw);
    ctx->buf_mask = gemma3_gpu_create_buffer(ctx, max_context * sizeof(float), storage_rw);

    /* Uniform buffer for parameters (largest needed) */
    ctx->buf_params = gemma3_gpu_create_buffer(ctx, 256,
                                                WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

    /* Staging buffers */
    size_t max_staging = (size_t)intermediate_size * hidden_size * sizeof(uint16_t);
    ctx->staging_write = gemma3_gpu_create_buffer(ctx, max_staging,
                                                   WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite);
    ctx->staging_read = gemma3_gpu_create_buffer(ctx, vocab_size * sizeof(float),
                                                  WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    return 0;
}

void gemma3_gpu_free(gemma3_gpu_context *ctx) {
    if (!ctx) return;

    /* Release buffers */
    gemma3_gpu_destroy_buffer(&ctx->buf_x);
    gemma3_gpu_destroy_buffer(&ctx->buf_x_norm);
    gemma3_gpu_destroy_buffer(&ctx->buf_q);
    gemma3_gpu_destroy_buffer(&ctx->buf_k);
    gemma3_gpu_destroy_buffer(&ctx->buf_v);
    gemma3_gpu_destroy_buffer(&ctx->buf_attn_out);
    gemma3_gpu_destroy_buffer(&ctx->buf_proj_out);
    gemma3_gpu_destroy_buffer(&ctx->buf_mlp_gate);
    gemma3_gpu_destroy_buffer(&ctx->buf_mlp_up);
    gemma3_gpu_destroy_buffer(&ctx->buf_mlp_out);
    gemma3_gpu_destroy_buffer(&ctx->buf_logits);
    gemma3_gpu_destroy_buffer(&ctx->buf_attn_scores);
    gemma3_gpu_destroy_buffer(&ctx->buf_mask);
    gemma3_gpu_destroy_buffer(&ctx->buf_params);
    gemma3_gpu_destroy_buffer(&ctx->staging_write);
    gemma3_gpu_destroy_buffer(&ctx->staging_read);

    /* Release pipelines */
    if (ctx->matvec_bf16_pipeline) wgpuComputePipelineRelease(ctx->matvec_bf16_pipeline);
    if (ctx->rmsnorm_bf16_pipeline) wgpuComputePipelineRelease(ctx->rmsnorm_bf16_pipeline);
    if (ctx->gelu_pipeline) wgpuComputePipelineRelease(ctx->gelu_pipeline);
    if (ctx->softmax_pipeline) wgpuComputePipelineRelease(ctx->softmax_pipeline);
    if (ctx->rope_pipeline) wgpuComputePipelineRelease(ctx->rope_pipeline);
    if (ctx->gqa_pipeline) wgpuComputePipelineRelease(ctx->gqa_pipeline);
    if (ctx->vec_add_pipeline) wgpuComputePipelineRelease(ctx->vec_add_pipeline);
    if (ctx->vec_mul_pipeline) wgpuComputePipelineRelease(ctx->vec_mul_pipeline);
    if (ctx->embed_bf16_pipeline) wgpuComputePipelineRelease(ctx->embed_bf16_pipeline);

    /* Release layouts */
    if (ctx->matvec_layout) wgpuBindGroupLayoutRelease(ctx->matvec_layout);
    if (ctx->rmsnorm_layout) wgpuBindGroupLayoutRelease(ctx->rmsnorm_layout);
    if (ctx->gelu_layout) wgpuBindGroupLayoutRelease(ctx->gelu_layout);
    if (ctx->softmax_layout) wgpuBindGroupLayoutRelease(ctx->softmax_layout);
    if (ctx->rope_layout) wgpuBindGroupLayoutRelease(ctx->rope_layout);
    if (ctx->gqa_layout) wgpuBindGroupLayoutRelease(ctx->gqa_layout);
    if (ctx->vec_op_layout) wgpuBindGroupLayoutRelease(ctx->vec_op_layout);
    if (ctx->embed_layout) wgpuBindGroupLayoutRelease(ctx->embed_layout);

    /* Release shader module */
    if (ctx->shader_module) wgpuShaderModuleRelease(ctx->shader_module);

    /* Release device and instance */
    if (ctx->queue) wgpuQueueRelease(ctx->queue);
    if (ctx->device) wgpuDeviceRelease(ctx->device);
    if (ctx->adapter) wgpuAdapterRelease(ctx->adapter);
    if (ctx->instance) wgpuInstanceRelease(ctx->instance);

    free(ctx);
}

/* ============================================================================
 * Synchronization
 * ========================================================================== */

void gemma3_gpu_sync(gemma3_gpu_context *ctx) {
    wgpuDevicePoll(ctx->device, 1, NULL);
    ctx->pending_commands = 0;
}

void gemma3_gpu_submit(gemma3_gpu_context *ctx) {
    (void)ctx;
    /* Commands are submitted immediately in current implementation */
}

/* ============================================================================
 * Compute Kernels
 * ========================================================================== */

void gemma3_matvec_bf16_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *y,
    const uint16_t *A,
    gemma3_gpu_buffer *x,
    int M, int K
) {
    /* Upload weight matrix A to temporary buffer */
    size_t A_size = (size_t)M * K * sizeof(uint16_t);
    gemma3_gpu_buffer A_buf = gemma3_gpu_create_buffer(ctx, A_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    gemma3_gpu_write_buffer(ctx, &A_buf, A, A_size);

    /* Set parameters */
    gemma3_matvec_params params = {0};
    params.M = (uint32_t)M;
    params.K = (uint32_t)K;
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    /* Create bind group */
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = A_buf.buffer, .size = A_size},
        {.binding = 2, .buffer = x->buffer, .size = (size_t)K * sizeof(float)},
        {.binding = 3, .buffer = y->buffer, .size = (size_t)M * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->matvec_layout;
    bg_desc.entryCount = 4;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    /* Dispatch */
    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->matvec_bf16_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);

    uint32_t workgroups = ((uint32_t)M + ctx->workgroup_size_1d - 1) / ctx->workgroup_size_1d;
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    /* Cleanup */
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    gemma3_gpu_destroy_buffer(&A_buf);
}

void gemma3_rmsnorm_bf16_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *y,
    gemma3_gpu_buffer *x,
    const uint16_t *weight,
    int n, float eps
) {
    /* Upload weight to temporary buffer */
    size_t weight_size = (size_t)n * sizeof(uint16_t);
    gemma3_gpu_buffer weight_buf = gemma3_gpu_create_buffer(ctx, weight_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    gemma3_gpu_write_buffer(ctx, &weight_buf, weight, weight_size);

    /* Scratch buffer for reduction */
    gemma3_gpu_buffer scratch_buf = gemma3_gpu_create_buffer(ctx, 256 * sizeof(float),
        WGPUBufferUsage_Storage);

    /* Set parameters */
    gemma3_rmsnorm_params params = {0};
    params.n = (uint32_t)n;
    params.eps = eps;
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    /* Create bind group */
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = x->buffer, .size = (size_t)n * sizeof(float)},
        {.binding = 2, .buffer = weight_buf.buffer, .size = weight_size},
        {.binding = 3, .buffer = y->buffer, .size = (size_t)n * sizeof(float)},
        {.binding = 4, .buffer = scratch_buf.buffer, .size = 256 * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->rmsnorm_layout;
    bg_desc.entryCount = 5;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    /* Dispatch (single workgroup for reduction) */
    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->rmsnorm_bf16_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    /* Cleanup */
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    gemma3_gpu_destroy_buffer(&weight_buf);
    gemma3_gpu_destroy_buffer(&scratch_buf);
}

void gemma3_gelu_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *x,
    int n
) {
    /* Set parameters */
    struct { uint32_t n; uint32_t _pad[3]; } params = { (uint32_t)n, {0} };
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    /* Create bind group */
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = x->buffer, .size = (size_t)n * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->gelu_layout;
    bg_desc.entryCount = 2;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    /* Dispatch */
    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->gelu_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);

    uint32_t workgroups = ((uint32_t)n + ctx->workgroup_size_1d - 1) / ctx->workgroup_size_1d;
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
}

void gemma3_softmax_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *x,
    int n
) {
    gemma3_softmax_params params = {0};
    params.n = (uint32_t)n;
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = x->buffer, .size = (size_t)n * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->softmax_layout;
    bg_desc.entryCount = 2;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->softmax_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
}

void gemma3_rope_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *x,
    int num_heads,
    int head_dim,
    int pos,
    float theta
) {
    gemma3_rope_params params = {0};
    params.head_dim = (uint32_t)head_dim;
    params.pos = (uint32_t)pos;
    params.theta = theta;
    params.num_heads = (uint32_t)num_heads;
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = x->buffer, .size = (size_t)num_heads * head_dim * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->rope_layout;
    bg_desc.entryCount = 2;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    int total_pairs = num_heads * (head_dim / 2);
    uint32_t workgroups = ((uint32_t)total_pairs + ctx->workgroup_size_1d - 1) / ctx->workgroup_size_1d;

    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->rope_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
}

void gemma3_gqa_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *output,
    gemma3_gpu_buffer *q,
    gemma3_gpu_buffer *k_cache,
    gemma3_gpu_buffer *v_cache,
    int n_heads, int n_kv_heads,
    int seq_len, int head_dim,
    float scale,
    gemma3_gpu_buffer *mask
) {
    gemma3_gqa_params params = {0};
    params.n_heads = (uint32_t)n_heads;
    params.n_kv_heads = (uint32_t)n_kv_heads;
    params.seq_len = (uint32_t)seq_len;
    params.head_dim = (uint32_t)head_dim;
    params.scale = scale;
    params.use_mask = mask ? 1 : 0;
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    WGPUBuffer mask_buffer = mask ? mask->buffer : ctx->buf_mask.buffer;
    size_t mask_size = (size_t)seq_len * sizeof(float);
    size_t kv_size = (size_t)seq_len * n_kv_heads * head_dim * sizeof(float);

    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = q->buffer, .size = (size_t)n_heads * head_dim * sizeof(float)},
        {.binding = 2, .buffer = k_cache->buffer, .size = kv_size},
        {.binding = 3, .buffer = v_cache->buffer, .size = kv_size},
        {.binding = 4, .buffer = mask_buffer, .size = mask_size},
        {.binding = 5, .buffer = output->buffer, .size = (size_t)n_heads * head_dim * sizeof(float)},
        {.binding = 6, .buffer = ctx->buf_attn_scores.buffer, .size = (size_t)seq_len * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->gqa_layout;
    bg_desc.entryCount = 7;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->gqa_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)n_heads, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
}

void gemma3_vec_add_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *y,
    gemma3_gpu_buffer *a,
    gemma3_gpu_buffer *b,
    int n
) {
    struct { uint32_t n; uint32_t _pad[3]; } params = { (uint32_t)n, {0} };
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = a->buffer, .size = (size_t)n * sizeof(float)},
        {.binding = 2, .buffer = b->buffer, .size = (size_t)n * sizeof(float)},
        {.binding = 3, .buffer = y->buffer, .size = (size_t)n * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->vec_op_layout;
    bg_desc.entryCount = 4;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    uint32_t workgroups = ((uint32_t)n + ctx->workgroup_size_1d - 1) / ctx->workgroup_size_1d;

    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->vec_add_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
}

void gemma3_vec_mul_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *y,
    gemma3_gpu_buffer *a,
    gemma3_gpu_buffer *b,
    int n
) {
    struct { uint32_t n; uint32_t _pad[3]; } params = { (uint32_t)n, {0} };
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = a->buffer, .size = (size_t)n * sizeof(float)},
        {.binding = 2, .buffer = b->buffer, .size = (size_t)n * sizeof(float)},
        {.binding = 3, .buffer = y->buffer, .size = (size_t)n * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->vec_op_layout;
    bg_desc.entryCount = 4;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    uint32_t workgroups = ((uint32_t)n + ctx->workgroup_size_1d - 1) / ctx->workgroup_size_1d;

    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->vec_mul_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
}

void gemma3_embed_bf16_gpu(
    gemma3_gpu_context *ctx,
    gemma3_gpu_buffer *output,
    const uint16_t *embed,
    int token_id,
    int hidden_size
) {
    size_t row_size = (size_t)hidden_size * sizeof(uint16_t);
    gemma3_gpu_buffer embed_buf = gemma3_gpu_create_buffer(ctx, row_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

    const uint16_t *row = embed + token_id * hidden_size;
    gemma3_gpu_write_buffer(ctx, &embed_buf, row, row_size);

    struct { uint32_t token_id; uint32_t hidden_size; uint32_t _pad[2]; } params = {
        0, (uint32_t)hidden_size, {0}
    };
    gemma3_gpu_write_buffer(ctx, &ctx->buf_params, &params, sizeof(params));

    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = ctx->buf_params.buffer, .size = sizeof(params)},
        {.binding = 1, .buffer = embed_buf.buffer, .size = row_size},
        {.binding = 2, .buffer = output->buffer, .size = (size_t)hidden_size * sizeof(float)},
    };

    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->embed_layout;
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);

    uint32_t workgroups = ((uint32_t)hidden_size + ctx->workgroup_size_1d - 1) / ctx->workgroup_size_1d;

    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);

    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

    wgpuComputePassEncoderSetPipeline(pass, ctx->embed_bf16_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);

    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBufferDescriptor cmd_desc = {0};
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(ctx->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    gemma3_gpu_destroy_buffer(&embed_buf);
}

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

const char *gemma3_gpu_device_name(gemma3_gpu_context *ctx) {
    if (!ctx || !ctx->device) return "Unknown";

    static char name[256];
    WGPUAdapterInfo info = {0};
    wgpuAdapterGetInfo(ctx->adapter, &info);

    const char *dev_name = (info.device.data && info.device.length > 0) ? info.device.data : "Unknown";
    const char *desc_name = (info.description.data && info.description.length > 0) ? info.description.data : "Unknown driver";

    snprintf(name, sizeof(name), "%.*s (%.*s)",
             (int)info.device.length, dev_name,
             (int)info.description.length, desc_name);

    wgpuAdapterInfoFreeMembers(info);
    return name;
}

#endif /* USE_WEBGPU */
