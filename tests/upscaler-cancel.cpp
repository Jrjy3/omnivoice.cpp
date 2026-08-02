// Model-backed cooperative cancellation harness for the standalone AudioVAE.
#include "backend.h"
#include "pipeline-upscaler.h"
#include "utf8.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

struct CancelState {
    int calls;
    int cancel_on_call;
};

static bool cancel_at_threshold(void * user_data) {
    CancelState * state = static_cast<CancelState *>(user_data);
    state->calls++;
    return state->calls >= state->cancel_on_call;
}

static bool expect_cancel_16k(PipelineUpscaler * pu, int threshold, const char * phase) {
    std::vector<float> input(640, 0.0f);
    CancelState state = { 0, threshold };
    bool cancelled = false;
    std::vector<float> out = pipeline_upscaler_process_16k(
        pu, input.data(), (int) input.size(), cancel_at_threshold, &state, &cancelled);
    if (!cancelled || !out.empty() || state.calls != threshold) {
        std::fprintf(stderr, "[Upscaler-cancel] %s failed: cancelled=%d output=%zu polls=%d expected=%d\n", phase,
                     cancelled ? 1 : 0, out.size(), state.calls, threshold);
        return false;
    }
    return true;
}

static int main_impl(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <voxcpm2-audiovae-f16.gguf>\n", argv[0]);
        return 2;
    }

    BackendPair bp = backend_init("AudioVAE cancellation test");
    if (!bp.backend) {
        return 1;
    }
    PipelineUpscaler pu = {};
    if (!pipeline_upscaler_load(&pu, argv[1], bp)) {
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    std::vector<float> input_24k(960, 0.0f);
    CancelState before_work = { 0, 1 };
    bool cancelled = false;
    std::vector<float> out = pipeline_upscaler_process(
        &pu, input_24k.data(), (int) input_24k.size(), cancel_at_threshold, &before_work, &cancelled);
    bool ok = cancelled && out.empty() && before_work.calls == 1;
    if (!ok) {
        std::fprintf(stderr, "[Upscaler-cancel] 24 kHz pre-work poll failed\n");
    }

    // The 16 kHz path polls once before work, then immediately before each
    // chunk and immediately after it. The third case computes one minimal
    // 640-sample AudioVAE chunk, so it exercises genuine model-backed cancel.
    ok = expect_cancel_16k(&pu, 2, "pre-chunk poll") && ok;
    ok = expect_cancel_16k(&pu, 3, "post-chunk poll") && ok;

    pipeline_upscaler_free(&pu);
    backend_release(bp.backend, bp.cpu_backend);
    if (ok) {
        std::fprintf(stderr, "[Upscaler-cancel] all cancellation phases passed\n");
    }
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[Upscaler-cancel] FATAL: %s\n", error.what());
        return 1;
    }
}
