# VoxCPM2 AudioVAE tooling

These tools support the optional 24 kHz-to-48 kHz VoxCPM2 AudioVAE path. They
do not download a model or modify an existing OmniVoice GGUF.

## Provenance

- Model and PyTorch architecture: [OpenBMB/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
  and [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM), Apache-2.0.
- Tensor naming and graph mapping reference:
  [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR), MIT.

The converter records the source checkpoint's SHA-256 and provenance in every
GGUF. Do not substitute a checkpoint merely because its filename matches;
OpenBMB has published more than one `audiovae.pth` revision.

## Convert the VAE

Requirements are already covered by the normal conversion environment:
Python, PyTorch, NumPy, and the `gguf` package.

```powershell
python tools/convert-voxcpm2-audiovae.py `
  --input C:\models\VoxCPM2 `
  --output models\voxcpm2-audiovae-f16.gguf
```

`--input` may instead point directly to `audiovae.pth`. If `config.json` is not
beside it, the converter uses the official V2 defaults; `--config` overrides
that lookup. `--precision f32` produces a larger debugging model.

The default mixed file stores large folded convolution weights in F16 while
keeping biases, Snake parameters, sample-rate embeddings, and small weights in
F32. Weight normalization is folded with PyTorch's `dim=0` convention. The
unused `encoder.fc_logvar` head is intentionally omitted. Conversion fails on
any unrecognized, orphaned, duplicated, or otherwise unconsumed tensor.

Use this dependency-free command to inspect the stable native contract:

```powershell
python tools/convert-voxcpm2-audiovae.py --print-contract
```

The current official checkpoint produces 233 GGUF tensors from 312 source
tensors: 75 weight-normalization pairs are folded and four source tensors are
omitted (three `fc_logvar` tensors and the sample-rate-boundary buffer). The
mixed payload is approximately 179 MiB.

## Dump PyTorch reference tensors

The parity harness imports OpenBMB's actual `audio_vae_v2.py`; it does not carry
a second implementation. The official module requires PyTorch, NumPy, and
Pydantic. WAV input additionally requires `soundfile`. Non-16-kHz audio uses
`torchaudio.functional.resample` when available, or SciPy's polyphase resampler.

```powershell
python tools/dump-voxcpm2-audiovae-reference.py `
  --checkpoint C:\models\VoxCPM2\audiovae.pth `
  --config C:\models\VoxCPM2\config.json `
  --module-file C:\src\VoxCPM\src\voxcpm\modules\audiovae\audio_vae_v2.py `
  --input-wav input-24k.wav `
  --output-dir reference-dump
```

If `audio_vae_v2.py` is beside the checkpoint, `--module-file` is unnecessary.
An official source checkout may also be supplied with `--voxcpm-root`.
Omitting `--input-wav` generates a deterministic test signal.

The output directory contains:

- Original and 16 kHz input arrays, plus a float32 16 kHz parity WAV.
- Encoder block outputs and the encoder `mu` latent.
- Decoder sample-rate-conditioning and block outputs.
- Final float32 waveform array and a 48 kHz PCM16 WAV.
- `manifest.json` with shapes, extrema, RMS values, model hash, configuration,
  implementation path, resampler, device, and dtype.

For exact graph debugging, pass the dumped `input_16k.wav` to the native test
harness with its native-16-kHz option. The WAV is IEEE float32 and contains the
same samples as `input_16k.npy`, removing resampler and PCM-quantization
differences from layer-by-layer comparisons.
CPU float32 is the reference mode; reduced precision is available only for
diagnosing backend-specific behavior.

Long audio produces very large intermediate dumps. The default one-second cap
is deliberate. For overlap/seam testing, use `--final-only --max-seconds 0` to
produce full-length input and output without writing every intermediate layer.
