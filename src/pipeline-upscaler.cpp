// pipeline-upscaler.cpp: VAE-only VoxCPM2 48 kHz enhancer for OmniVoice.
//
// The graph topology and causal padding/cropping rules are adapted from the
// MIT-licensed CrispASR VoxCPM2 runtime (CrispStrobe/CrispASR,
// src/voxcpm2_tts.cpp).  This focused implementation loads only AudioVAE
// tensors, expects weight_norm to be folded by the converter, and uses the
// backend/scheduler already selected by omnivoice.cpp.

#include "pipeline-upscaler.h"

#include "audio-resample.h"
#include "ov-error.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kEncoderRates[4] = { 2, 5, 8, 8 };
constexpr int kDecoderRates[6] = { 8, 6, 5, 2, 2, 2 };
constexpr int kDilations[3]    = { 1, 3, 9 };
constexpr int kEncoderDim      = 128;
constexpr int kLatentDim       = 64;
constexpr int kDecoderDim      = 2048;
constexpr int kInputRate       = 16000;
constexpr int kOutputRate      = 48000;

// Keep each GPU dispatch below the conservative cross-vendor Vulkan limit
// documented by CrispASR.  102400 input samples = 6.4 s; the 1.6 s causal
// history exceeds the complete encoder+decoder receptive field (~1.42 s).
constexpr int kChunkPayload16k = 102400;
constexpr int kChunkHistory16k = 25600;

struct LoadScratch {
    std::vector<std::vector<uint8_t>> buffers;

    explicit LoadScratch(size_t reserve) { buffers.reserve(reserve); }

    void * alloc(size_t bytes) {
        buffers.emplace_back(bytes);
        return buffers.back().data();
    }
};

static struct ggml_tensor * source_tensor(const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * t = ggml_get_tensor(gf.meta, name.c_str());
    if (!t) {
        ov_throw("[Upscaler] tensor '%s' not found", name.c_str());
    }
    return t;
}

static void source_to_f32(const GGUFModel & gf, const std::string & name, std::vector<float> & out) {
    struct ggml_tensor * src = source_tensor(gf, name);
    size_t               n   = (size_t) ggml_nelements(src);
    out.resize(n);
    const void * raw = gf_get_data(gf, name.c_str());
    if (src->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw, n * sizeof(float));
    } else if (src->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, out.data(), (int64_t) n);
    } else if (src->type == GGML_TYPE_BF16) {
        const ggml_bf16_t * p = (const ggml_bf16_t *) raw;
        for (size_t i = 0; i < n; i++) {
            out[i] = ggml_bf16_to_fp32(p[i]);
        }
    } else {
        ov_throw("[Upscaler] unsupported tensor type %s for '%s'", ggml_type_name(src->type), name.c_str());
    }
}

static void stage_f32(WeightCtx * wctx,
                      LoadScratch * scratch,
                      struct ggml_tensor * dst,
                      const GGUFModel & gf,
                      const std::string & name) {
    std::vector<float> f32;
    source_to_f32(gf, name, f32);
    if ((size_t) ggml_nelements(dst) != f32.size()) {
        ov_throw("[Upscaler] shape mismatch for '%s' (%lld destination elements, %zu source)", name.c_str(),
                 (long long) ggml_nelements(dst), f32.size());
    }
    size_t bytes = f32.size() * sizeof(float);
    void * p     = scratch->alloc(bytes);
    memcpy(p, f32.data(), bytes);
    wctx->pending.push_back({ dst, p, bytes, 0 });
}

static void stage_f16(WeightCtx * wctx,
                      LoadScratch * scratch,
                      struct ggml_tensor * dst,
                      const GGUFModel & gf,
                      const std::string & name) {
    std::vector<float> f32;
    source_to_f32(gf, name, f32);
    if ((size_t) ggml_nelements(dst) != f32.size()) {
        ov_throw("[Upscaler] shape mismatch for '%s' (%lld destination elements, %zu source)", name.c_str(),
                 (long long) ggml_nelements(dst), f32.size());
    }
    size_t bytes = f32.size() * sizeof(ggml_fp16_t);
    auto * p     = (ggml_fp16_t *) scratch->alloc(bytes);
    ggml_fp32_to_fp16_row(f32.data(), p, (int64_t) f32.size());
    wctx->pending.push_back({ dst, p, bytes, 0 });
}

static enum ggml_type conv_weight_type(const GGUFModel & gf, const std::string & prefix, bool force_f16) {
    enum ggml_type type = gf_get_type(gf, prefix + ".weight");
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_F32) {
        ov_throw("[Upscaler] convolution '%s.weight' must be F16 or F32, got %s", prefix.c_str(),
                 ggml_type_name(type));
    }
    // ggml's CPU im2col requires F16 convolution weights on every supported
    // architecture. Preserve native F32 for GPU parity, but safely narrow on
    // CPU as the existing DAC runtime does.
    if (force_f16 && type == GGML_TYPE_F32) {
        type = GGML_TYPE_F16;
    }
    return type;
}

static bool metadata_array_equals(const GGUFModel & gf,
                                  const char * key,
                                  const int * expected,
                                  size_t expected_count) {
    int64_t id = gguf_find_key(gf.gguf, key);
    if (id < 0 || gguf_get_kv_type(gf.gguf, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_n(gf.gguf, id) != expected_count) {
        return false;
    }
    enum gguf_type type = gguf_get_arr_type(gf.gguf, id);
    const void * data   = gguf_get_arr_data(gf.gguf, id);
    for (size_t i = 0; i < expected_count; i++) {
        int value;
        if (type == GGUF_TYPE_UINT32) {
            value = (int) ((const uint32_t *) data)[i];
        } else if (type == GGUF_TYPE_INT32) {
            value = (int) ((const int32_t *) data)[i];
        } else {
            return false;
        }
        if (value != expected[i]) {
            return false;
        }
    }
    return true;
}

static UpscalerConv alloc_conv(struct ggml_context * ctx,
                               const GGUFModel & gf,
                               const std::string & prefix,
                               int kernel,
                               int in_channels,
                               int out_channels,
                               bool force_f16) {
    UpscalerConv c = {};
    struct ggml_tensor * src = source_tensor(gf, prefix + ".weight");
    if (src->ne[0] != kernel || src->ne[1] != in_channels || src->ne[2] != out_channels) {
        ov_throw("[Upscaler] convolution shape mismatch for '%s.weight': ne=(%lld,%lld,%lld), expected (%d,%d,%d)",
                 prefix.c_str(), (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], kernel,
                 in_channels, out_channels);
    }
    c.weight = ggml_new_tensor_3d(ctx, conv_weight_type(gf, prefix, force_f16), kernel, in_channels, out_channels);
    c.bias         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    return c;
}

static void load_conv(WeightCtx * wctx,
                      LoadScratch * scratch,
                      const GGUFModel & gf,
                      UpscalerConv * c,
                      const std::string & prefix) {
    if (c->weight->type == GGML_TYPE_F32) {
        stage_f32(wctx, scratch, c->weight, gf, prefix + ".weight");
    } else {
        stage_f16(wctx, scratch, c->weight, gf, prefix + ".weight");
    }
    stage_f32(wctx, scratch, c->bias, gf, prefix + ".bias");
}

static UpscalerSnake alloc_snake(struct ggml_context * ctx, int channels) {
    UpscalerSnake s = {};
    s.alpha         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, channels);
    s.inv_alpha     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, channels);
    return s;
}

static void load_snake(WeightCtx * wctx,
                       LoadScratch * scratch,
                       const GGUFModel & gf,
                       UpscalerSnake * s,
                       const std::string & name) {
    std::vector<float> alpha;
    source_to_f32(gf, name, alpha);
    if ((size_t) ggml_nelements(s->alpha) != alpha.size()) {
        ov_throw("[Upscaler] alpha shape mismatch for '%s'", name.c_str());
    }
    size_t bytes = alpha.size() * sizeof(float);
    auto * a     = (float *) scratch->alloc(bytes);
    auto * inv   = (float *) scratch->alloc(bytes);
    for (size_t i = 0; i < alpha.size(); i++) {
        a[i]   = alpha[i];
        inv[i] = 1.0f / (alpha[i] + 1e-9f);
    }
    wctx->pending.push_back({ s->alpha, a, bytes, 0 });
    wctx->pending.push_back({ s->inv_alpha, inv, bytes, 0 });
}

static void alloc_res_unit(struct ggml_context * ctx,
                           const GGUFModel & gf,
                           const std::string & prefix,
                           UpscalerResUnit * ru,
                           int channels,
                           int dilation,
                           bool force_f16) {
    ru->snake[0] = alloc_snake(ctx, channels);
    ru->conv[0]  = alloc_conv(ctx, gf, prefix + ".1", 7, 1, channels, force_f16);  // depthwise
    ru->snake[1] = alloc_snake(ctx, channels);
    ru->conv[1]  = alloc_conv(ctx, gf, prefix + ".3", 1, channels, channels, force_f16);
    ru->dilation = dilation;
}

static void load_res_unit(WeightCtx * wctx,
                          LoadScratch * scratch,
                          const GGUFModel & gf,
                          UpscalerResUnit * ru,
                          const std::string & prefix) {
    load_snake(wctx, scratch, gf, &ru->snake[0], prefix + ".0.alpha");
    load_conv(wctx, scratch, gf, &ru->conv[0], prefix + ".1");
    load_snake(wctx, scratch, gf, &ru->snake[1], prefix + ".2.alpha");
    load_conv(wctx, scratch, gf, &ru->conv[1], prefix + ".3");
}

static void load_transposed_weight(WeightCtx * wctx,
                                   LoadScratch * scratch,
                                   const GGUFModel & gf,
                                   UpscalerConv * c,
                                   const std::string & prefix,
                                   int input_channels,
                                   int output_channels,
                                   int kernel) {
    const std::string name = prefix + ".weight";
    struct ggml_tensor * src = source_tensor(gf, name);
    if (src->ne[0] != kernel || src->ne[1] != output_channels || src->ne[2] != input_channels) {
        ov_throw("[Upscaler] ConvTranspose shape mismatch for '%s': ne=(%lld,%lld,%lld), expected (%d,%d,%d)",
                 name.c_str(), (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], kernel,
                 output_channels, input_channels);
    }

    std::vector<float> f32;
    source_to_f32(gf, name, f32);
    size_t n     = f32.size();
    const bool f32_dst = c->weight->type == GGML_TYPE_F32;
    size_t bytes       = n * (f32_dst ? sizeof(float) : sizeof(ggml_fp16_t));
    void * packed      = scratch->alloc(bytes);
    // PyTorch ConvTranspose1d [IC, OC, K] -> col2im [IC, K*OC].
    for (int ic = 0; ic < input_channels; ic++) {
        for (int oc = 0; oc < output_channels; oc++) {
            for (int k = 0; k < kernel; k++) {
                size_t src_i = (size_t) ic * output_channels * kernel + (size_t) oc * kernel + k;
                size_t dst_i = (size_t) (oc * kernel + k) * input_channels + ic;
                if (f32_dst) {
                    ((float *) packed)[dst_i] = f32[src_i];
                } else {
                    ((ggml_fp16_t *) packed)[dst_i] = ggml_fp32_to_fp16(f32[src_i]);
                }
            }
        }
    }
    wctx->pending.push_back({ c->weight, packed, bytes, 0 });
    stage_f32(wctx, scratch, c->bias, gf, prefix + ".bias");
}

static void load_sr_bucket(WeightCtx * wctx,
                           LoadScratch * scratch,
                           const GGUFModel & gf,
                           struct ggml_tensor * dst,
                           const std::string & name,
                           int bucket) {
    struct ggml_tensor * src = source_tensor(gf, name);
    int channels = (int) ggml_nelements(dst);
    if (src->ne[0] != channels || src->ne[1] <= bucket) {
        ov_throw("[Upscaler] SR embedding shape mismatch for '%s'", name.c_str());
    }
    std::vector<float> f32;
    source_to_f32(gf, name, f32);
    size_t bytes = (size_t) channels * sizeof(float);
    void * p     = scratch->alloc(bytes);
    memcpy(p, f32.data() + (size_t) bucket * channels, bytes);
    wctx->pending.push_back({ dst, p, bytes, 0 });
}

static struct ggml_tensor * snake_graph(struct ggml_context * ctx,
                                        struct ggml_tensor * x,
                                        const UpscalerSnake & s) {
    struct ggml_tensor * ax = ggml_mul(ctx, x, s.alpha);
    struct ggml_tensor * v  = ggml_sin(ctx, ax);
    v                       = ggml_sqr(ctx, v);
    v                       = ggml_mul(ctx, v, s.inv_alpha);
    return ggml_add(ctx, x, v);
}

static struct ggml_tensor * conv_graph(struct ggml_context * ctx,
                                       struct ggml_tensor * x,
                                       const UpscalerConv & c,
                                       int stride,
                                       int pad,
                                       int dilation,
                                       bool depthwise) {
    struct ggml_tensor * x3 = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);
    struct ggml_tensor * y  = depthwise ? ggml_conv_1d_dw(ctx, c.weight, x3, stride, pad, dilation)
                                        : ggml_conv_1d(ctx, c.weight, x3, stride, pad, dilation);
    y                       = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (c.bias) {
        struct ggml_tensor * b = ggml_reshape_2d(ctx, c.bias, 1, c.bias->ne[0]);
        y                      = ggml_add(ctx, y, b);
    }
    return y;
}

static struct ggml_tensor * causal_conv_graph(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const UpscalerConv & c,
                                              int dilation,
                                              bool depthwise) {
    int K = (int) c.weight->ne[0];
    int T = (int) x->ne[0];
    struct ggml_tensor * y = conv_graph(ctx, x, c, 1, (K - 1) * dilation, dilation, depthwise);
    if (y->ne[0] > T) {
        y = ggml_view_2d(ctx, y, T, y->ne[1], y->nb[1], 0);
        y = ggml_cont(ctx, y);
    }
    return y;
}

static struct ggml_tensor * strided_conv_graph(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const UpscalerConv & c,
                                               int stride) {
    int K        = (int) c.weight->ne[0];
    int left_pad = 2 * ((stride + 1) / 2) - (stride % 2);
    int T_out    = ((int) x->ne[0] + left_pad - K) / stride + 1;
    struct ggml_tensor * y = conv_graph(ctx, x, c, stride, left_pad, 1, false);
    if (y->ne[0] > T_out) {
        y = ggml_view_2d(ctx, y, T_out, y->ne[1], y->nb[1], 0);
        y = ggml_cont(ctx, y);
    }
    return y;
}

static struct ggml_tensor * transposed_conv_graph(struct ggml_context * ctx,
                                                  struct ggml_tensor * x,
                                                  const UpscalerDecoderBlock & b) {
    struct ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor * col = ggml_mul_mat(ctx, b.up.weight, xt);
    struct ggml_tensor * y   = ggml_col2im_1d(ctx, col, b.stride, b.output_channels, 0);
    int T_out                = (int) x->ne[0] * b.stride;
    y                        = ggml_view_2d(ctx, y, T_out, b.output_channels, y->nb[1], 0);
    y                        = ggml_cont(ctx, y);
    struct ggml_tensor * bias = ggml_reshape_2d(ctx, b.up.bias, 1, b.output_channels);
    return ggml_add(ctx, y, bias);
}

static struct ggml_tensor * res_unit_graph(struct ggml_context * ctx,
                                           struct ggml_tensor * x,
                                           const UpscalerResUnit & ru) {
    struct ggml_tensor * skip = x;
    x = snake_graph(ctx, x, ru.snake[0]);
    x = causal_conv_graph(ctx, x, ru.conv[0], ru.dilation, true);
    x = snake_graph(ctx, x, ru.snake[1]);
    x = causal_conv_graph(ctx, x, ru.conv[1], 1, false);
    return ggml_add(ctx, skip, x);
}

static std::vector<float> run_chunk(PipelineUpscaler * pu, const float * pcm_16k, int n_samples) {
    if (!pu || !pu->loaded || !pcm_16k || n_samples <= 0) {
        return {};
    }

    int padded = ((n_samples + pu->encoder_hop - 1) / pu->encoder_hop) * pu->encoder_hop;
    const int n_max_nodes = 4096;
    size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);
    struct ggml_init_params gp = { graph_ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        ov_log(OV_LOG_ERROR, "[Upscaler] failed to allocate graph context");
        return {};
    }
    struct GraphGuard {
        PipelineUpscaler * pu;
        struct ggml_context * ctx;
        ~GraphGuard() {
            ggml_backend_sched_reset(pu->sched);
            ggml_free(ctx);
        }
    } guard{ pu, ctx };

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, 1);
    ggml_set_name(input, "upscaler_input_16k");
    ggml_set_input(input);

    // Encoder mu.  Padding and left-causal crops match AudioVAE.preprocess.
    struct ggml_tensor * x = causal_conv_graph(ctx, input, pu->enc_in, 1, false);
    for (int i = 0; i < 4; i++) {
        for (int r = 0; r < 3; r++) {
            x = res_unit_graph(ctx, x, pu->enc_block[i].res[r]);
        }
        x = snake_graph(ctx, x, pu->enc_block[i].snake);
        x = strided_conv_graph(ctx, x, pu->enc_block[i].down, pu->enc_block[i].stride);
    }
    x = causal_conv_graph(ctx, x, pu->enc_mu, 1, false);

    // Decoder, 48 kHz SR bucket (bucketize(48000,[20k,30k,40k]) == 3).
    x = causal_conv_graph(ctx, x, pu->dec_in_depthwise, 1, true);
    x = causal_conv_graph(ctx, x, pu->dec_in_pointwise, 1, false);
    for (int i = 0; i < 6; i++) {
        const UpscalerDecoderBlock & b = pu->dec_block[i];
        x = ggml_mul(ctx, x, b.sr_scale);
        x = ggml_add(ctx, x, b.sr_bias);
        x = snake_graph(ctx, x, b.snake);
        x = transposed_conv_graph(ctx, x, b);
        for (int r = 0; r < 3; r++) {
            x = res_unit_graph(ctx, x, b.res[r]);
        }
    }
    x = snake_graph(ctx, x, pu->dec_final_snake);
    x = causal_conv_graph(ctx, x, pu->dec_out, 1, false);
    x = ggml_tanh(ctx, x);
    ggml_set_name(x, "upscaler_output_48k");
    ggml_set_output(x);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, x);
    ggml_backend_sched_reset(pu->sched);
    if (!ggml_backend_sched_alloc_graph(pu->sched, graph)) {
        ov_log(OV_LOG_ERROR, "[Upscaler] scheduler allocation failed for %d input samples", padded);
        return {};
    }

    std::vector<float> host((size_t) padded, 0.0f);
    memcpy(host.data(), pcm_16k, (size_t) n_samples * sizeof(float));
    ggml_backend_tensor_set(input, host.data(), 0, host.size() * sizeof(float));

    enum ggml_status status = ggml_backend_sched_graph_compute(pu->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ov_log(OV_LOG_ERROR, "[Upscaler] graph compute failed with status %d", (int) status);
        return {};
    }

    size_t n_out = (size_t) ggml_nelements(x);
    std::vector<float> out(n_out);
    ggml_backend_tensor_get(x, out.data(), 0, n_out * sizeof(float));
    return out;
}

}  // namespace

bool pipeline_upscaler_load(PipelineUpscaler * pu, const char * gguf_path, BackendPair bp) {
    if (!pu || !gguf_path || !*gguf_path || !bp.backend) {
        ov_log(OV_LOG_ERROR, "[Upscaler] invalid load parameters");
        return false;
    }
    *pu    = {};
    pu->bp = bp;

    GGUFModel gf = {};
    try {
        if (!gf_load(&gf, gguf_path)) {
            return false;
        }
        const char * arch = gf_get_str(gf, "general.architecture");
        if (strcmp(arch, "voxcpm2-audiovae") != 0) {
            ov_throw("[Upscaler] '%s' has architecture '%s', expected 'voxcpm2-audiovae'", gguf_path, arch);
        }
        if (!gf_get_bool(gf, "voxcpm2.vae.weight_norm_folded")) {
            ov_throw("[Upscaler] model must contain pre-folded weight_norm tensors");
        }
        if (!gf_get_bool(gf, "voxcpm2.vae.depthwise")) {
            ov_throw("[Upscaler] model must use the depthwise AudioVAE topology");
        }
        if (gf_get_bool(gf, "voxcpm2.vae.use_noise_block")) {
            ov_throw("[Upscaler] noise-block AudioVAE models are not supported");
        }
        const char * cond = gf_get_str(gf, "voxcpm2.vae.cond_type");
        if (strcmp(cond, "scale_bias") != 0) {
            ov_throw("[Upscaler] unsupported sample-rate conditioning '%s'", cond);
        }
        if (gf_get_bool(gf, "voxcpm2.vae.cond_out_layer")) {
            ov_throw("[Upscaler] cond_out_layer AudioVAE models are not supported");
        }

        int encoder_dim = (int) gf_get_u32(gf, "voxcpm2.vae.encoder_dim");
        int latent_dim  = (int) gf_get_u32(gf, "voxcpm2.vae.latent_dim");
        int decoder_dim = (int) gf_get_u32(gf, "voxcpm2.vae.decoder_dim");
        int in_rate     = (int) gf_get_u32(gf, "voxcpm2.vae.sample_rate");
        int out_rate    = (int) gf_get_u32(gf, "voxcpm2.vae.out_sample_rate");
        int cond_dim    = (int) gf_get_u32(gf, "voxcpm2.vae.cond_dim");
        if (encoder_dim != kEncoderDim || latent_dim != kLatentDim || decoder_dim != kDecoderDim ||
            in_rate != kInputRate || out_rate != kOutputRate || cond_dim != 128) {
            ov_throw("[Upscaler] unsupported AudioVAE dimensions/rates: enc=%d latent=%d dec=%d cond=%d sr=%d->%d",
                     encoder_dim, latent_dim, decoder_dim, cond_dim, in_rate, out_rate);
        }
        static const int sr_boundaries[3] = { 20000, 30000, 40000 };
        if (!metadata_array_equals(gf, "voxcpm2.vae.encoder_rates", kEncoderRates, 4) ||
            !metadata_array_equals(gf, "voxcpm2.vae.decoder_rates", kDecoderRates, 6) ||
            !metadata_array_equals(gf, "voxcpm2.vae.sr_bin_boundaries", sr_boundaries, 3)) {
            ov_throw("[Upscaler] unsupported AudioVAE rate arrays or SR bucket boundaries");
        }

        wctx_init(&pu->wctx, 400);
        struct ggml_context * wctx = pu->wctx.ctx;
        LoadScratch scratch(400);
        const bool force_f16 = !bp.has_gpu;

        pu->enc_in = alloc_conv(wctx, gf, "vae.enc.conv0", 7, 1, kEncoderDim, force_f16);
        int channels = kEncoderDim;
        for (int b = 0; b < 4; b++) {
            pu->enc_block[b].stride = kEncoderRates[b];
            for (int r = 0; r < 3; r++) {
                std::string rp = "vae.enc.blk." + std::to_string(b) + ".res." + std::to_string(r);
                alloc_res_unit(wctx, gf, rp, &pu->enc_block[b].res[r], channels, kDilations[r], force_f16);
            }
            pu->enc_block[b].snake = alloc_snake(wctx, channels);
            std::string down = "vae.enc.blk." + std::to_string(b) + ".sub.4";
            pu->enc_block[b].down  =
                alloc_conv(wctx, gf, down, 2 * kEncoderRates[b], channels, channels * 2, force_f16);
            channels *= 2;
        }
        pu->enc_mu = alloc_conv(wctx, gf, "vae.enc.fc_mu", 3, channels, kLatentDim, force_f16);

        pu->dec_in_depthwise = alloc_conv(wctx, gf, "vae.dec.layer.0", 7, 1, kLatentDim, force_f16);
        pu->dec_in_pointwise = alloc_conv(wctx, gf, "vae.dec.layer.1", 1, kLatentDim, kDecoderDim, force_f16);
        channels = kDecoderDim;
        for (int b = 0; b < 6; b++) {
            UpscalerDecoderBlock & block = pu->dec_block[b];
            block.stride         = kDecoderRates[b];
            block.input_channels = channels;
            block.output_channels = channels / 2;
            block.sr_scale = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, channels);
            block.sr_bias  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, channels);
            block.snake    = alloc_snake(wctx, channels);
            std::string layer_prefix = "vae.dec.layer." + std::to_string(b + 2);
            enum ggml_type up_type = conv_weight_type(gf, layer_prefix + ".block.1", force_f16);
            block.up.weight =
                ggml_new_tensor_2d(wctx, up_type, channels, 2 * block.stride * block.output_channels);
            block.up.bias = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, block.output_channels);
            for (int r = 0; r < 3; r++) {
                std::string rp = layer_prefix + ".block." + std::to_string(r + 2) + ".block";
                alloc_res_unit(wctx, gf, rp, &block.res[r], block.output_channels, kDilations[r], force_f16);
            }
            channels = block.output_channels;
        }
        pu->dec_final_snake = alloc_snake(wctx, channels);
        pu->dec_out         = alloc_conv(wctx, gf, "vae.dec.layer.9", 7, channels, 1, force_f16);

        load_conv(&pu->wctx, &scratch, gf, &pu->enc_in, "vae.enc.conv0");
        channels = kEncoderDim;
        for (int b = 0; b < 4; b++) {
            std::string prefix = "vae.enc.blk." + std::to_string(b);
            for (int r = 0; r < 3; r++) {
                load_res_unit(&pu->wctx, &scratch, gf, &pu->enc_block[b].res[r],
                              prefix + ".res." + std::to_string(r));
            }
            load_snake(&pu->wctx, &scratch, gf, &pu->enc_block[b].snake, prefix + ".sub.3.alpha");
            load_conv(&pu->wctx, &scratch, gf, &pu->enc_block[b].down, prefix + ".sub.4");
            channels *= 2;
        }
        load_conv(&pu->wctx, &scratch, gf, &pu->enc_mu, "vae.enc.fc_mu");

        load_conv(&pu->wctx, &scratch, gf, &pu->dec_in_depthwise, "vae.dec.layer.0");
        load_conv(&pu->wctx, &scratch, gf, &pu->dec_in_pointwise, "vae.dec.layer.1");
        for (int b = 0; b < 6; b++) {
            UpscalerDecoderBlock & block = pu->dec_block[b];
            int layer = b + 2;
            std::string prefix = "vae.dec.layer." + std::to_string(layer);
            std::string sr = "vae.dec.sr_cond." + std::to_string(layer);
            load_sr_bucket(&pu->wctx, &scratch, gf, block.sr_scale, sr + ".scale_embed", 3);
            load_sr_bucket(&pu->wctx, &scratch, gf, block.sr_bias, sr + ".bias_embed", 3);
            load_snake(&pu->wctx, &scratch, gf, &block.snake, prefix + ".block.0.alpha");
            load_transposed_weight(&pu->wctx, &scratch, gf, &block.up, prefix + ".block.1",
                                   block.input_channels, block.output_channels, 2 * block.stride);
            for (int r = 0; r < 3; r++) {
                load_res_unit(&pu->wctx, &scratch, gf, &block.res[r],
                              prefix + ".block." + std::to_string(r + 2) + ".block");
            }
        }
        load_snake(&pu->wctx, &scratch, gf, &pu->dec_final_snake, "vae.dec.layer.8.alpha");
        load_conv(&pu->wctx, &scratch, gf, &pu->dec_out, "vae.dec.layer.9");

        if (!wctx_alloc(&pu->wctx, bp.backend)) {
            ov_throw("[Upscaler] failed to allocate VAE weights on %s", ggml_backend_name(bp.backend));
        }
        gf_close(&gf);

        pu->sched = backend_sched_new(bp, 4096);
        if (!pu->sched) {
            ov_throw("[Upscaler] failed to create scheduler");
        }
        pu->input_sample_rate  = kInputRate;
        pu->output_sample_rate = kOutputRate;
        pu->encoder_hop        = kEncoderRates[0] * kEncoderRates[1] * kEncoderRates[2] * kEncoderRates[3];
        pu->loaded             = true;
        ov_log(OV_LOG_INFO, "[Upscaler] loaded VoxCPM2 AudioVAE on %s (24k -> 16k -> 48k)",
               ggml_backend_name(bp.backend));
        return true;
    } catch (const std::exception & e) {
        gf_close(&gf);
        ov_log(OV_LOG_ERROR, "%s", e.what());
        pipeline_upscaler_free(pu);
        return false;
    }
}

void pipeline_upscaler_free(PipelineUpscaler * pu) {
    if (!pu) {
        return;
    }
    if (pu->sched) {
        ggml_backend_sched_free(pu->sched);
    }
    wctx_free(&pu->wctx);
    *pu = {};
}

std::vector<float> pipeline_upscaler_process_16k(PipelineUpscaler * pu, const float * pcm_16k, int n_16k) {
    if (!pu || !pu->loaded || !pcm_16k || n_16k <= 0) {
        ov_log(OV_LOG_ERROR, "[Upscaler] process called without a loaded model or audio");
        return {};
    }

    const int target_48k = ((n_16k + pu->encoder_hop - 1) / pu->encoder_hop) * pu->encoder_hop * 3;
    std::vector<float> out;
    out.reserve((size_t) target_48k);

    for (int start = 0; start < n_16k; start += kChunkPayload16k) {
        int seg_start = std::max(0, start - kChunkHistory16k);
        int seg_end   = std::min(n_16k, start + kChunkPayload16k);
        int seg_n     = seg_end - seg_start;
        std::vector<float> chunk = run_chunk(pu, pcm_16k + seg_start, seg_n);
        if (chunk.empty()) {
            return {};
        }
        size_t discard = (size_t) (start - seg_start) * 3;
        size_t remaining = (size_t) target_48k - out.size();
        // The final graph contains AudioVAE's right-padding to the 640-sample
        // encoder hop. Keep that tail exactly as torch does. Interior chunks
        // contribute only their unpadded payload.
        size_t wanted = seg_end == n_16k ? remaining : (size_t) kChunkPayload16k * 3;
        wanted = std::min(wanted, remaining);
        if (discard + wanted > chunk.size()) {
            ov_log(OV_LOG_ERROR, "[Upscaler] internal chunk length mismatch (%zu + %zu > %zu)", discard, wanted,
                   chunk.size());
            return {};
        }
        out.insert(out.end(), chunk.begin() + (ptrdiff_t) discard,
                   chunk.begin() + (ptrdiff_t) (discard + wanted));
    }
    return out;
}

std::vector<float> pipeline_upscaler_process(PipelineUpscaler * pu, const float * pcm_24k, int n_samples) {
    if (!pu || !pu->loaded || !pcm_24k || n_samples <= 0) {
        ov_log(OV_LOG_ERROR, "[Upscaler] process called without a loaded model or audio");
        return {};
    }
    int n_16k = 0;
    std::unique_ptr<float, decltype(&free)> resampled(
        audio_resample(pcm_24k, n_samples, 24000, pu->input_sample_rate, 1, &n_16k), &free);
    if (!resampled || n_16k <= 0) {
        ov_log(OV_LOG_ERROR, "[Upscaler] 24 kHz -> 16 kHz resample failed");
        return {};
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> out = pipeline_upscaler_process_16k(pu, resampled.get(), n_16k);
    auto t1 = std::chrono::steady_clock::now();
    if (!out.empty()) {
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double audio_sec = (double) out.size() / 48000.0;
        ov_log(OV_LOG_INFO,
               "[Upscaler] %d samples at 24 kHz -> %zu at 48 kHz in %.3f s (%.3f s audio, RTF %.3f)", n_samples,
               out.size(), elapsed, audio_sec, audio_sec > 0.0 ? elapsed / audio_sec : 0.0);
    }
    return out;
}
