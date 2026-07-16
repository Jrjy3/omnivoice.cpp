# omnivoice.cpp

Local AI text-to-speech with voice cloning and voice design, powered
by GGML. C++17 port of OmniVoice (k2-fsa/OmniVoice). 646 languages,
native 24 kHz mono output with an optional VoxCPM2 AudioVAE path to
48 kHz, runs on CPU, CUDA, ROCm, Metal, Vulkan.

## Features

- Voice cloning from a reference WAV plus its transcript
- Voice design via attribute keywords (gender, age, pitch, style,
  volume, emotion)
- Auto voice with consistent speaker identity across long inputs
- Long-form synthesis with punctuation-aware text chunking, voice
  prompt promotion, cross-fade and pydub-strict silence removal
- Bit deterministic generation in greedy mode, seedable Philox PRNG
  for stochastic sampling
- Q8_0 quantisation of the 612 M parameter Qwen3 backbone
- Optional VAE-only GGUF enhancement from 24 kHz to 48 kHz on the same
  backend, with bounded-memory long-audio chunking
- Three CLI tools : `omnivoice-tts` (text -> WAV), `omnivoice-codec`
  (WAV <-> RVQ codes), and `omnivoice-upscale` (standalone AudioVAE)

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git
cd omnivoice.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildcpu.sh                    # CPU only
./buildall.sh                    # all backends, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

## Model conversion

Pre-converted GGUFs are available on Hugging Face :

  https://huggingface.co/Serveurperso/OmniVoice-GGUF

Drop them in `models/` and skip to the quick start. To convert from
the original checkpoint :

```
./checkpoints.sh      # hf download k2-fsa/OmniVoice -> checkpoints/
./convert.py          # 2 GGUFs in BF16 -> models/
./quantize.sh         # base LM Q8_0 (tokenizer stays at native dtype)
```

The optional 48 kHz path uses a third, independently converted model:

```sh
python tools/convert-voxcpm2-audiovae.py \
    --input /path/to/openbmb/VoxCPM2 \
    --output models/voxcpm2-audiovae-f16.gguf
```

The current mixed F16/F32 VAE-only GGUF is 187,868,032 bytes. The
converter folds weight normalization, omits the unused `fc_logvar`
head, strictly audits every source tensor, and embeds the source
checkpoint SHA-256 and license provenance. See
[docs/VOXCPM2_AUDIOVAE_TOOLING.md](docs/VOXCPM2_AUDIOVAE_TOOLING.md).

## Quick start

```
echo "Hello world." | ./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --lang English -o hello.wav
```

Voice cloning :

```
./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --ref-wav ref.wav --ref-text ref.txt \
    --lang English -o out.wav < prompt.txt
```

Pre-encoded reference (`clone.sh`): `omnivoice-codec` encodes a reference WAV
into a compact `.rvq` latent (it applies the exact TTS reference
preprocessing, so the codes are bit-identical to the `--ref-wav` path).
Passing it via `--ref-rvq` skips the codec encode on every synthesis:

```
build/omnivoice-codec --model models/omnivoice-tokenizer-F32.gguf -i ref.wav
build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --ref-rvq ref.rvq --ref-text ref.txt \
    --lang English -o out.wav < prompt.txt
```

## Embedding the library

The CLI tools are thin wrappers over a public ABI. Single-header,
single-name-prefix, plain C linkage so that C, C++, Python ctypes,
Rust bindgen and Go cgo all consume it the same way.

```c
#include "omnivoice.h"

struct ov_init_params iparams;
ov_init_default_params(&iparams);
iparams.model_path = "models/omnivoice-base-Q8_0.gguf";
iparams.codec_path = "models/omnivoice-tokenizer-F32.gguf";
/* Optional ABI v4 field. Omit for native 24 kHz output. */
iparams.upscaler_path = "models/voxcpm2-audiovae-f16.gguf";

struct ov_context * ov = ov_init(&iparams);

struct ov_tts_params params;
ov_tts_default_params(&params);
params.text = "Hello world.";
params.lang = "English";

struct ov_audio audio = { 0 };
ov_synthesize(ov, &params, &audio);
/* audio.samples, audio.n_samples, audio.sample_rate, audio.channels */
ov_audio_free(&audio);
ov_free(ov);
```

`OV_ABI_VERSION` is 4. A non-NULL `upscaler_path` eagerly loads the
VAE on the same backend and makes buffered synthesis return 48 kHz.
Native `on_chunk` callback streaming currently requires the 24 kHz
path; use buffered synthesis when the upscaler is loaded.

The standalone parity/benchmark path accepts either 24 kHz input or
an already-resampled 16 kHz reference:

```sh
./build/omnivoice-upscale \
    --model models/voxcpm2-audiovae-f16.gguf \
    -i input.wav -o output-48k.wav
```

`tests/abi-c.c` is built with `-std=c99 -Wall -Werror -pedantic` on
every build, so any regression that breaks plain C consumability fails
the build, not just an opt-in target.

For a binding-friendly shared library (libomnivoice.so / .dll / .dylib),
configure with `cmake -DOMNIVOICE_SHARED=ON ...`. The shared target
exports only the `ov_*` symbols ; every internal `pipeline_*` and
`backend_*` stays hidden inside the .so.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, the
GGUF layout, the inference pipeline, every CLI flag, the public API
reference and the validation results.

## License

MIT. See [LICENSE](LICENSE).

Upstream model : OmniVoice by Xiaomi / k2-fsa, Apache 2.0.
Audio codec : Higgs Audio v2 (`bosonai/higgs-audio-v2-tokenizer`),
Apache 2.0.
Optional AudioVAE : VoxCPM2 by OpenBMB, Apache 2.0. Tensor naming and
graph mapping were informed by CrispASR, MIT.
