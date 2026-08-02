/* Frozen ABI-v3 caller canary. This translation unit must only include the
 * historical header snapshot, then link against the current library. */
#include "omnivoice-v3.h"

#include <stdint.h>
#include <stdio.h>

struct guarded_init_params {
    struct ov_init_params params;
    uint64_t canary;
};

int main(void) {
    struct guarded_init_params guarded = { { 0 }, UINT64_C(0x5a17c0de5a17c0de) };
    struct ov_tts_params tts;

    if (OV_ABI_VERSION != 3) {
        fprintf(stderr, "[ABI-v3] frozen header no longer reports ABI 3\n");
        return 1;
    }

    ov_init_default_params(&guarded.params);
    if (guarded.params.abi_version != 3 || guarded.canary != UINT64_C(0x5a17c0de5a17c0de)) {
        fprintf(stderr, "[ABI-v3] init defaults changed ABI or overwrote the v3 struct\n");
        return 2;
    }

    ov_tts_default_params(&tts);
    if (tts.abi_version != 3 || tts.mg_num_step != 32 || !tts.postproc) {
        fprintf(stderr, "[ABI-v3] TTS defaults do not preserve the v3 contract\n");
        return 3;
    }

    return 0;
}
