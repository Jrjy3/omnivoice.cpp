# OmniVoice runtime for Sonorus v1.1.0

Portable Windows x64 runtime for the Sonorus `omnivoice_cpp` integration.

- CPU baseline: AVX2, FMA, F16C, and BMI2
- Vulkan backend included
- `GGML_NATIVE=OFF`
- AVX-512 and AVX-VNNI variants disabled
- OmniVoice ABI version 5
- omnivoice.cpp runtime source: `ec6b9f266ffd536aaed4db9922102dd435e30553`
- ggml source: `9fcaed18adc9f6ecc6ef1c8e58e2285e893d8319`

The archive contains the five required runtime DLLs, the omnivoice.cpp and
ggml MIT license notices, and a per-file size/SHA-256 manifest.

Archive SHA-256:
`e8ed705fc441fe8276f50c42adf701f7f86d874be83d6681efe530582cd3773c`

## What changed since v1.0.0

The codec's voice encoder (DAC encoder, SemanticEncoder, HuBERT stack) is no
longer resident for the life of the handle. It is only needed to turn a raw
reference waveform into RVQ codes, so a caller that caches codes per voice was
holding 469 MB of VRAM for nothing once its voices were pre-processed.

- `ov_init_params.encoder_mode` (ABI v5 tail field): `OV_ENCODER_EAGER`
  (default, v1.0.0 behaviour), `OV_ENCODER_LAZY`, `OV_ENCODER_ON_DEMAND`
- `ov_release_voice_encoder` hands the memory back at any time
- `ov_voice_encoder_bytes` reports what the encoder currently holds
- Every `vN` default-params symbol now reports exactly `N`, so a caller built
  against the v4 header cannot make `ov_init` read an `encoder_mode` field its
  struct does not have. `ov_init_default_params_v5` and
  `ov_tts_default_params_v5` are the current entry points.

ABI v3 and v4 callers are unaffected and keep an always-resident encoder.

Validation:

- Native CTest suite: 5/5 passed, including the ABI v3 canary.
- `eager`, `lazy` and `ondemand` produce byte-identical `.rvq` output, and
  re-encoding after an explicit release matches the first encode
  (`omnivoice-codec --verify-reload`), verified on Vulkan (Radeon AI PRO
  R9700).
- `dumpbin` disassembly audit: AVX2 instructions present; no `zmm` or AVX-512
  opmask register instructions detected.
