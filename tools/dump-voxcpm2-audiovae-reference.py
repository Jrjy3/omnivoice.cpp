#!/usr/bin/env python3
"""Dump authoritative PyTorch VoxCPM2 AudioVAE reference tensors.

The harness imports OpenBMB's ``audio_vae_v2.py`` (Apache-2.0) rather than a
reimplementation.  It records the encoder/decoder stages needed to compare the
native ggml graph layer by layer.  A local official VoxCPM checkout, a standalone
module file, or a sibling ``audio_vae_v2.py`` next to audiovae.pth may be used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "encoder_dim": 128,
    "encoder_rates": [2, 5, 8, 8],
    "latent_dim": 64,
    "decoder_dim": 2048,
    "decoder_rates": [8, 6, 5, 2, 2, 2],
    "depthwise": True,
    "sample_rate": 16000,
    "out_sample_rate": 48000,
    "use_noise_block": False,
    "sr_bin_boundaries": [20000, 30000, 40000],
    "cond_type": "scale_bias",
    "cond_dim": 128,
    "cond_out_layer": False,
}


def log(message: str) -> None:
    print(f"[AudioVAE-ref] {message}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    for key in ("encoder_rates", "decoder_rates", "sr_bin_boundaries"):
        config[key] = list(DEFAULT_CONFIG[key])
    if path is None:
        return config
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw_vae = raw.get("audio_vae_config", raw)
    for key in config:
        if key in raw_vae:
            config[key] = raw_vae[key]
    return config


def import_official_module(module_file: Path | None, voxcpm_root: Path | None) -> tuple[ModuleType, str]:
    if module_file is not None:
        module_file = module_file.expanduser().resolve()
        if not module_file.is_file():
            raise FileNotFoundError(f"AudioVAE module not found: {module_file}")
        spec = importlib.util.spec_from_file_location("voxcpm2_audio_vae_reference", module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {module_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, str(module_file)

    if voxcpm_root is not None:
        root = voxcpm_root.expanduser().resolve()
        src = root / "src"
        sys.path.insert(0, str(src if src.is_dir() else root))
    try:
        module = importlib.import_module("voxcpm.modules.audiovae.audio_vae_v2")
    except ImportError as exc:
        raise RuntimeError(
            "cannot import the official AudioVAE; pass --module-file audio_vae_v2.py "
            "or --voxcpm-root /path/to/VoxCPM"
        ) from exc
    return module, str(Path(module.__file__).resolve())


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a state_dict")
    normalized = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        for prefix in ("_orig_mod.", "module.", "audio_vae."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = value
    return normalized, metadata if isinstance(metadata, dict) else {}


def read_audio(path: Path) -> tuple[Any, int]:
    import numpy as np

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for --input-wav") from exc
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio), int(sample_rate)


def make_synthetic(kind: str, n_samples: int, sample_rate: int, seed: int) -> Any:
    import numpy as np

    if n_samples < 1:
        raise ValueError("--synthetic-samples must be positive")
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    if kind == "impulse":
        audio = np.zeros(n_samples, dtype=np.float32)
        audio[0] = 0.8
    elif kind == "noise":
        audio = np.random.default_rng(seed).standard_normal(n_samples).astype(np.float32) * 0.05
    else:
        # Deterministic, non-bin-centred speech-band mixture that exercises
        # phase, amplitude and the upscaler's high-frequency reconstruction.
        audio = (
            0.22 * np.sin(2 * np.pi * 173.0 * t)
            + 0.11 * np.sin(2 * np.pi * 997.0 * t + 0.37)
            + 0.04 * np.sin(2 * np.pi * 3521.0 * t + 1.1)
        ).astype(np.float32)
    return np.ascontiguousarray(audio)


def resample_audio(audio: Any, source_rate: int, target_rate: int, backend: str) -> tuple[Any, str]:
    import numpy as np
    import torch

    if source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32), "none"

    if backend in {"auto", "torchaudio"}:
        try:
            import torchaudio.functional as AF

            tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
            result = AF.resample(tensor, source_rate, target_rate).squeeze(0).numpy()
            return np.ascontiguousarray(result, dtype=np.float32), "torchaudio.functional.resample"
        except ImportError:
            if backend == "torchaudio":
                raise RuntimeError("torchaudio was requested but is not installed")

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("install torchaudio or scipy to resample non-16-kHz input") from exc
    divisor = math.gcd(source_rate, target_rate)
    result = resample_poly(audio, target_rate // divisor, source_rate // divisor)
    return np.ascontiguousarray(result, dtype=np.float32), "scipy.signal.resample_poly"


class StageWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.stages: list[dict[str, Any]] = []
        self.sequence = 0

    def save(self, name: str, value: Any) -> None:
        import numpy as np
        import torch

        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise ValueError(f"stage {name} returned {len(value)} values")
            value = value[0]
        if not torch.is_tensor(value):
            raise TypeError(f"stage {name} did not return a tensor: {type(value).__name__}")
        array = np.ascontiguousarray(value.detach().float().cpu().numpy(), dtype=np.float32)
        filename = f"{self.sequence:03d}_{name}.npy"
        np.save(self.output_dir / filename, array, allow_pickle=False)
        self.stages.append(
            {
                "sequence": self.sequence,
                "name": name,
                "file": filename,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "min": float(array.min()) if array.size else None,
                "max": float(array.max()) if array.size else None,
                "rms": float(np.sqrt(np.mean(array.astype(np.float64) ** 2))) if array.size else None,
            }
        )
        self.sequence += 1


def register_stage_hooks(model: Any, writer: StageWriter) -> list[Any]:
    handles = []

    def output_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            writer.save(name, output)

        return hook

    def input_hook(name: str):
        def hook(_module: Any, inputs: Any) -> None:
            writer.save(name, inputs[0])

        return hook

    handles.append(model.encoder.block[0].register_forward_pre_hook(input_hook("encoder_padded_input")))
    for index, layer in enumerate(model.encoder.block):
        handles.append(layer.register_forward_hook(output_hook(f"encoder_block_{index:02d}")))

    handles.append(model.decoder.model[0].register_forward_pre_hook(input_hook("decoder_latent_input")))
    for index, layer in enumerate(model.decoder.model):
        handles.append(layer.register_forward_hook(output_hook(f"decoder_layer_{index:02d}")))
    if hasattr(model.decoder, "sr_cond_model"):
        for name, layer in model.decoder.sr_cond_model.named_children():
            handles.append(layer.register_forward_hook(output_hook(f"decoder_sr_cond_{int(name):02d}")))
    return handles


def write_wav(path: Path, audio: Any, sample_rate: int, *, float32: bool = False) -> None:
    import numpy as np

    try:
        import soundfile as sf

        subtype = "FLOAT" if float32 else "PCM_16"
        sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype=subtype)
    except ImportError:
        if float32:
            pcm = np.ascontiguousarray(audio, dtype="<f4")
            data = pcm.tobytes()
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + len(data),
                b"WAVE",
                b"fmt ",
                16,
                3,  # IEEE float
                1,
                sample_rate,
                sample_rate * 4,
                4,
                32,
                b"data",
                len(data),
            )
            with path.open("wb") as handle:
                handle.write(header)
                handle.write(data)
            return
        import wave

        pcm = np.clip(np.asarray(audio), -1.0, 1.0)
        pcm = (pcm * 32767.0).round().astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump layer-by-layer tensors from OpenBMB's authoritative VoxCPM2 AudioVAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="official audiovae.pth")
    parser.add_argument("--config", type=Path, help="VoxCPM2 config.json")
    parser.add_argument("--module-file", type=Path, help="official audio_vae_v2.py")
    parser.add_argument("--voxcpm-root", type=Path, help="official OpenBMB/VoxCPM source checkout")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-wav", type=Path, help="mono/stereo input; omitted uses a deterministic signal")
    parser.add_argument("--synthetic", choices=("tones", "impulse", "noise"), default="tones")
    parser.add_argument("--synthetic-samples", type=int, default=1280)
    parser.add_argument("--input-sample-rate", type=int, default=24000, help="sample rate for synthetic input")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-seconds", type=float, default=1.0, help="truncate input; 0 disables truncation")
    parser.add_argument("--resampler", choices=("auto", "torchaudio", "scipy"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="save only inputs/final output (recommended for long seam tests)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import numpy as np
        import torch

        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        config_path = args.config.expanduser().resolve() if args.config else checkpoint.parent / "config.json"
        if not config_path.is_file():
            config_path = None
        module_file = args.module_file
        if module_file is None and (checkpoint.parent / "audio_vae_v2.py").is_file():
            module_file = checkpoint.parent / "audio_vae_v2.py"
        module, module_source = import_official_module(module_file, args.voxcpm_root)
        config = load_json_config(config_path)

        model_config = module.AudioVAEConfig(**config)
        model = module.AudioVAE(model_config)
        state, checkpoint_metadata = load_checkpoint(checkpoint)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"state_dict mismatch: {incompatible}")

        dtype = getattr(torch, args.dtype)
        device = torch.device(args.device)
        if device.type == "cpu" and dtype != torch.float32:
            log("warning: F16/BF16 CPU operators may be unsupported; float32 is recommended for parity")
        model = model.to(device=device, dtype=dtype).eval()

        if args.input_wav:
            source_audio, source_rate = read_audio(args.input_wav.expanduser().resolve())
            source_description = str(args.input_wav.expanduser().resolve())
        else:
            source_rate = args.input_sample_rate
            source_audio = make_synthetic(args.synthetic, args.synthetic_samples, source_rate, args.seed)
            source_description = f"synthetic:{args.synthetic}"
        if args.max_seconds > 0:
            source_audio = source_audio[: max(1, int(round(source_rate * args.max_seconds)))]
        audio_16k, resampler = resample_audio(source_audio, source_rate, int(config["sample_rate"]), args.resampler)

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "input_source.npy", np.asarray(source_audio, dtype=np.float32), allow_pickle=False)
        np.save(output_dir / "input_16k.npy", audio_16k, allow_pickle=False)
        # Native parity CLIs generally consume WAV rather than NPY. Preserve
        # float32 here so the shared input is byte-for-byte equivalent to the
        # array used by PyTorch, without PCM16 quantization noise.
        write_wav(output_dir / "input_16k.wav", audio_16k, int(config["sample_rate"]), float32=True)

        writer = StageWriter(output_dir)
        handles = [] if args.final_only else register_stage_hooks(model, writer)
        input_tensor = torch.from_numpy(audio_16k).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.inference_mode():
            latent = model.encode(input_tensor, sample_rate=int(config["sample_rate"]))
            if not args.final_only:
                writer.save("encoder_mu", latent)
            sr_cond = torch.tensor([int(config["out_sample_rate"])], device=device, dtype=torch.int32)
            final = model.decode(latent, sr_cond=sr_cond)
            writer.save("final_waveform", final)
        for handle in handles:
            handle.remove()

        final_audio = final.detach().float().cpu().numpy()[0, 0]
        write_wav(output_dir / "final_48k.wav", final_audio, int(config["out_sample_rate"]))
        manifest = {
            "format": "omnivoice.cpp.voxcpm2-audiovae-reference.v1",
            "implementation": {
                "project": "OpenBMB/VoxCPM",
                "license": "Apache-2.0",
                "module": module_source,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "metadata": checkpoint_metadata,
            },
            "config": config,
            "execution": {
                "torch": torch.__version__,
                "device": str(device),
                "dtype": args.dtype,
                "resampler": resampler,
                "final_only": bool(args.final_only),
            },
            "input": {
                "source": source_description,
                "source_sample_rate": source_rate,
                "source_samples": int(len(source_audio)),
                "input_16k_samples": int(len(audio_16k)),
                "input_16k_wav": "input_16k.wav",
            },
            "output": {
                "sample_rate": int(config["out_sample_rate"]),
                "samples": int(len(final_audio)),
                "wav": "final_48k.wav",
            },
            "stages": writer.stages,
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        log(f"wrote {len(writer.stages)} stages and final_48k.wav to {output_dir}")
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        log(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
