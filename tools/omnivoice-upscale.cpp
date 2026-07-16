// omnivoice-upscale.cpp: standalone VoxCPM2 AudioVAE parity/benchmark CLI.

#include "audio-io.h"
#include "backend.h"
#include "pipeline-upscaler.h"
#include "utf8.h"
#include "version.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
            "omnivoice.cpp %s\n\n"
            "Usage: %s --model <voxcpm2-audiovae-f16.gguf> -i <input.wav> [-o output.wav] [--native-16k]\n"
            "  --native-16k  Treat input WAV as the post-resampler 16 kHz parity input\n",
            OMNIVOICE_VERSION, prog);
}

static std::string default_output(const char * input) {
    std::string p(input ? input : "output");
    size_t dot = p.find_last_of('.');
    size_t sep = p.find_last_of("/\\");
    if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) {
        p.resize(dot);
    }
    return p + "-48k.wav";
}

static int main_impl(int argc, char ** argv) {
    const char * model = nullptr;
    const char * input = nullptr;
    const char * output_arg = nullptr;
    bool native_16k = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model = argv[++i];
        } else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            input = argv[++i];
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            output_arg = argv[++i];
        } else if (!strcmp(argv[i], "--native-16k")) {
            native_16k = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (!model || !input) {
        usage(argv[0]);
        return 1;
    }
    std::string output = output_arg ? output_arg : default_output(input);

    BackendPair bp = backend_init("AudioVAE");
    if (!bp.backend) {
        return 1;
    }
    PipelineUpscaler pu = {};
    if (!pipeline_upscaler_load(&pu, model, bp)) {
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    int n = 0;
    float * pcm = audio_read_mono(input, native_16k ? 16000 : 24000, &n);
    if (!pcm || n <= 0) {
        fprintf(stderr, "[Upscale-CLI] failed to read %s\n", input);
        free(pcm);
        pipeline_upscaler_free(&pu);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> out = native_16k ? pipeline_upscaler_process_16k(&pu, pcm, n)
                                        : pipeline_upscaler_process(&pu, pcm, n);
    auto t1 = std::chrono::steady_clock::now();
    free(pcm);
    int rc = 0;
    if (out.empty()) {
        fprintf(stderr, "[Upscale-CLI] enhancement failed\n");
        rc = 1;
    } else if (!audio_write_wav(output.c_str(), out.data(), (int) out.size(), 48000, WAV_F32)) {
        rc = 1;
    } else {
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double audio_sec = (double) out.size() / 48000.0;
        fprintf(stderr, "[Upscale-CLI] wrote %s: %.3f s audio in %.3f s (RTF %.3f)\n", output.c_str(), audio_sec,
                sec, audio_sec > 0.0 ? sec / audio_sec : 0.0);
    }
    pipeline_upscaler_free(&pu);
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[Upscale-CLI] FATAL: %s\n", e.what());
        return 1;
    }
}
