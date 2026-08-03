---
license: apache-2.0
base_model:
- openbmb/VoxCPM2
tags:
- gguf
- audio
- text-to-speech
- voxcpm2
---

# Sonorus OmniVoice AudioVAE

This repository contains the VAE-only GGUF used by Sonorus and the
`voxcpm2-upscaler` branch of omnivoice.cpp to convert native OmniVoice output
to 48 kHz audio.

## File

| File | Size | SHA-256 |
|---|---:|---|
| `voxcpm2-audiovae-f16.gguf` | 187,868,032 bytes | `a5fb091c0a95172bdee2ee7230335dac7d3dc318d77ca100f095d023cabd5d97` |

The file was converted from the official
[OpenBMB/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
`audiovae.pth` checkpoint using
`tools/convert-voxcpm2-audiovae.py` from the
[omnivoice.cpp fork](https://github.com/Jrjy3/omnivoice.cpp/tree/voxcpm2-upscaler).
The original model and weights are Apache-2.0. Conversion provenance is also
embedded in the GGUF metadata.

This is only the AudioVAE. The OmniVoice language model and tokenizer GGUFs
remain hosted in
[Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF).
