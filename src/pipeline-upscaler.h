#pragma once
// pipeline-upscaler.h: VoxCPM2 AudioVAE 24 kHz -> 48 kHz enhancer.
//
// The network is intentionally a separate, optional GGUF.  It shares the
// OmniVoice BackendPair, but owns its weights and scheduler.  A NULL
// upscaler_path at the public ABI level leaves this module unloaded and the
// existing native 24 kHz path unchanged.

#include "backend.h"
#include "gguf-weights.h"
#include "omnivoice.h"

#include <vector>

struct UpscalerSnake {
    struct ggml_tensor * alpha;
    struct ggml_tensor * inv_alpha;
};

struct UpscalerConv {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;
};

struct UpscalerResUnit {
    UpscalerSnake snake[2];
    UpscalerConv  conv[2];
    int           dilation;
};

struct UpscalerEncoderBlock {
    UpscalerResUnit res[3];
    UpscalerSnake   snake;
    UpscalerConv    down;
    int              stride;
};

struct UpscalerDecoderBlock {
    struct ggml_tensor * sr_scale;
    struct ggml_tensor * sr_bias;
    UpscalerSnake        snake;
    // Pre-permuted [input_channels, kernel * output_channels] F16 weight for
    // the ggml mul_mat + col2im ConvTranspose1d implementation.
    UpscalerConv         up;
    UpscalerResUnit      res[3];
    int                  stride;
    int                  input_channels;
    int                  output_channels;
};

struct PipelineUpscaler {
    WeightCtx            wctx;
    ggml_backend_sched_t sched;
    BackendPair          bp;

    UpscalerConv         enc_in;
    UpscalerEncoderBlock enc_block[4];
    UpscalerConv         enc_mu;

    UpscalerConv         dec_in_depthwise;
    UpscalerConv         dec_in_pointwise;
    UpscalerDecoderBlock dec_block[6];
    UpscalerSnake        dec_final_snake;
    UpscalerConv         dec_out;

    int input_sample_rate;
    int output_sample_rate;
    int encoder_hop;
    bool loaded;
};

// Load and validate a VAE-only `voxcpm2-audiovae` GGUF.  Weight norm must
// already be folded by the converter.  Returns false on any mismatch.
bool pipeline_upscaler_load(PipelineUpscaler * pu, const char * gguf_path, BackendPair bp);

void pipeline_upscaler_free(PipelineUpscaler * pu);

// Input is post-processed OmniVoice mono PCM at 24 kHz.  The implementation
// performs the same 24 -> 16 kHz sinc resample as the torch path, encodes mu,
// and decodes with the 48 kHz SR-conditioning bucket.  Empty means failure or
// cancellation; cancelled is set only for the latter. CLI callers omit the
// optional callback arguments and remain uncancelled.
std::vector<float> pipeline_upscaler_process(PipelineUpscaler * pu,
                                             const float * pcm_24k,
                                             int n_samples,
                                             ov_cancel_cb cancel = nullptr,
                                             void * cancel_user_data = nullptr,
                                             bool * cancelled = nullptr);

// Test/parity entry: bypass the 24 -> 16 kHz resampler and run AudioVAE on
// already-resampled mono PCM. Not part of the public C ABI.
std::vector<float> pipeline_upscaler_process_16k(PipelineUpscaler * pu,
                                                 const float * pcm_16k,
                                                 int n_samples,
                                                 ov_cancel_cb cancel = nullptr,
                                                 void * cancel_user_data = nullptr,
                                                 bool * cancelled = nullptr);
