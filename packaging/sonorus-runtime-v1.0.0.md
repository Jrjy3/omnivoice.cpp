# OmniVoice runtime for Sonorus v1.0.0

Portable Windows x64 runtime for the Sonorus `omnivoice_cpp` integration.

- CPU baseline: AVX2, FMA, F16C, and BMI2
- Vulkan backend included
- `GGML_NATIVE=OFF`
- AVX-512 and AVX-VNNI variants disabled
- OmniVoice ABI version 4
- omnivoice.cpp runtime source: `cac4a81766700838cb1cbcc0ff36112cda628c3d`
- ggml source: `9fcaed18adc9f6ecc6ef1c8e58e2285e893d8319`

The archive contains the five required runtime DLLs, the omnivoice.cpp and
ggml MIT license notices, and a per-file size/SHA-256 manifest.

Archive SHA-256:
`a1fd71977e424c110a0ff86e70a37b87022128668709d3c6fb9ed0f9275dbbe9`

Validation:

- Native CTest suite: 5/5 passed.
- Sonorus focused Python regression suite: 14/14 passed.
- `dumpbin` disassembly audit: AVX2 instructions present; no `zmm` or AVX-512
  opmask register instructions detected.
