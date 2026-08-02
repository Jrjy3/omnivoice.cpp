#!/usr/bin/env python3
"""Convert the official VoxCPM2 AudioVAE checkpoint to a VAE-only GGUF.

The tensor naming scheme is derived from CrispASR's VoxCPM2 converter/runtime
(MIT), but this converter intentionally folds PyTorch weight normalization and
omits the unused encoder log-variance head.  The source model and architecture
are OpenBMB VoxCPM2 (Apache-2.0).

This file does not download models.  Point it at either ``audiovae.pth`` or a
local VoxCPM2 model directory containing that file and, ideally, config.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


SOURCE_REPO = "openbmb/VoxCPM2"
SOURCE_URL = "https://huggingface.co/openbmb/VoxCPM2"
SOURCE_LICENSE = "Apache-2.0"
MAPPING_SOURCE = "CrispStrobe/CrispASR"
MAPPING_LICENSE = "MIT"

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

SKIPPED_SOURCE_KEYS = {
    "decoder.sr_bin_boundaries": "stored as GGUF metadata",
}


def log(message: str) -> None:
    print(f"[AudioVAE] {message}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inputs(input_path: Path, explicit_config: Path | None) -> tuple[Path, Path | None]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        checkpoint = input_path / "audiovae.pth"
        config = explicit_config or (input_path / "config.json")
    else:
        checkpoint = input_path
        config = explicit_config or (input_path.parent / "config.json")

    if not checkpoint.is_file():
        raise FileNotFoundError(f"AudioVAE checkpoint not found: {checkpoint}")
    if checkpoint.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("the official audiovae.pth PyTorch checkpoint is required")
    if config is not None:
        config = config.expanduser().resolve()
        if not config.is_file():
            if explicit_config is not None:
                raise FileNotFoundError(f"config not found: {config}")
            config = None
    return checkpoint, config


def _same_config_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_config_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["encoder_rates"] = list(DEFAULT_CONFIG["encoder_rates"])
    config["decoder_rates"] = list(DEFAULT_CONFIG["decoder_rates"])
    config["sr_bin_boundaries"] = list(DEFAULT_CONFIG["sr_bin_boundaries"])
    if path is None:
        log("config.json not found; using the official VoxCPM2 AudioVAE defaults")
        return config

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config in {path} is not an object")
    raw_vae = raw.get("audio_vae_config", raw)
    if not isinstance(raw_vae, dict):
        raise ValueError(f"audio_vae_config in {path} is not an object")
    for key in config:
        if key in raw_vae:
            config[key] = raw_vae[key]

    # The native runtime has no dynamic topology path: every recognized
    # AudioVAE setting is fixed to the official V2 contract. Ignore unrelated
    # config keys, but reject a recognized override instead of writing a GGUF
    # that the runtime cannot load.
    mismatches = []
    for key, expected in DEFAULT_CONFIG.items():
        actual = config[key]
        if not _same_config_value(actual, expected):
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError("unsupported AudioVAE runtime configuration: " + ", ".join(mismatches))
    return config


def _find_state_dict(obj: Any) -> dict[str, Any]:
    """Find the tensor dictionary in official and common wrapped checkpoints."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read audiovae.pth") from exc

    if isinstance(obj, dict):
        tensors = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
        if tensors and len(tensors) == len(obj):
            return tensors
        for key in ("state_dict", "model", "module", "audio_vae"):
            value = obj.get(key)
            if isinstance(value, dict):
                try:
                    return _find_state_dict(value)
                except ValueError:
                    pass
    raise ValueError("could not locate a tensor state_dict in the checkpoint")


def load_state_dict(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install the repository conversion requirements") from exc

    log(f"loading {path} (CPU; this needs roughly 0.8 GB of temporary RAM)")
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        checkpoint = torch.load(str(path), map_location="cpu")
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    raw_state = _find_state_dict(checkpoint)

    state: dict[str, Any] = {}
    for source_name, tensor in raw_state.items():
        name = source_name
        for prefix in ("_orig_mod.", "module.", "audio_vae."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name in state:
            raise ValueError(f"duplicate normalized tensor name: {name}")
        state[name] = tensor.detach().cpu()
    return state, metadata if isinstance(metadata, dict) else {}


def remap_tensor_name(source_name: str) -> str | None:
    """Map an unfolded PyTorch name to the CrispASR-compatible GGUF contract."""
    if source_name == "encoder.block.0.weight" or source_name == "encoder.block.0.bias":
        return "vae.enc.conv0." + source_name.rsplit(".", 1)[1]

    match = re.fullmatch(r"encoder\.block\.(\d+)\.block\.(\d+)\.(weight|bias|alpha)", source_name)
    if match:
        block = int(match.group(1)) - 1
        if block < 0:
            return None
        return f"vae.enc.blk.{block}.sub.{match.group(2)}.{match.group(3)}"

    match = re.fullmatch(
        r"encoder\.block\.(\d+)\.block\.(\d+)\.block\.(\d+)\.(weight|bias|alpha)", source_name
    )
    if match:
        block = int(match.group(1)) - 1
        if block < 0:
            return None
        return f"vae.enc.blk.{block}.res.{match.group(2)}.{match.group(3)}.{match.group(4)}"

    match = re.fullmatch(r"encoder\.fc_(mu|logvar)\.(weight|bias)", source_name)
    if match:
        return f"vae.enc.fc_{match.group(1)}.{match.group(2)}"

    match = re.fullmatch(r"decoder\.model\.(\d+)\.(weight|bias|alpha)", source_name)
    if match:
        return f"vae.dec.layer.{match.group(1)}.{match.group(2)}"

    for depth in (1, 2, 3):
        block_parts = "".join(r"\.block\.(\d+)" for _ in range(depth))
        match = re.fullmatch(
            r"decoder\.model\.(\d+)" + block_parts + r"\.(weight|bias|alpha)", source_name
        )
        if match:
            indices = ".block.".join(match.groups()[:-1])
            return f"vae.dec.layer.{indices}.{match.group(depth + 2)}"

    match = re.fullmatch(
        r"decoder\.sr_cond_model\.(\d+)\.(scale_embed|bias_embed)\.weight", source_name
    )
    if match:
        return f"vae.dec.sr_cond.{match.group(1)}.{match.group(2)}"

    # Current official configuration has no conditioning output layer, but map
    # it explicitly so future checkpoints fail on topology rather than names.
    match = re.fullmatch(
        r"decoder\.sr_cond_model\.(\d+)\.out_layer\.(\d+)\.(weight|bias|alpha)", source_name
    )
    if match:
        return f"vae.dec.sr_cond.{match.group(1)}.out.{match.group(2)}.{match.group(3)}"
    return None


def fold_weight_norm(g_tensor: Any, v_tensor: Any) -> Any:
    """Return torch weight_norm(v, g, dim=0) as a contiguous float32 array."""
    import numpy as np

    g = g_tensor.float().numpy()
    v = v_tensor.float().numpy()
    if v.ndim < 2:
        raise ValueError(f"weight_v must have at least 2 dimensions, got {v.shape}")
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v * v, axis=axes, keepdims=True, dtype=np.float64)).astype(np.float32)
    if np.any(norm == 0):
        raise ValueError("weight normalization encountered a zero-norm filter")
    try:
        weight = v * (g / norm)
    except ValueError as exc:
        raise ValueError(f"weight_g {g.shape} cannot broadcast over weight_v {v.shape}") from exc
    return np.ascontiguousarray(weight, dtype=np.float32)


def convert_tensors(
    state: dict[str, Any], precision: str, large_tensor_threshold: int
) -> tuple[list[tuple[str, Any]], dict[str, str], int]:
    """Fold, remap, cast, and strictly audit all source tensors."""
    import numpy as np

    output: list[tuple[str, Any]] = []
    consumed: set[str] = set()
    skipped: dict[str, str] = {}
    target_names: set[str] = set()
    n_folded = 0

    for name in sorted(state):
        if name.startswith("encoder.fc_logvar."):
            skipped[name] = "inference encode() returns mu; logvar is unused"
            consumed.add(name)
            continue
        if name in SKIPPED_SOURCE_KEYS:
            skipped[name] = SKIPPED_SOURCE_KEYS[name]
            consumed.add(name)
            continue
        if name in consumed:
            continue

        g_suffix = None
        v_suffix = None
        if name.endswith(".weight_g"):
            g_suffix, v_suffix = ".weight_g", ".weight_v"
        elif name.endswith(".parametrizations.weight.original0"):
            g_suffix = ".parametrizations.weight.original0"
            v_suffix = ".parametrizations.weight.original1"

        if g_suffix is not None and v_suffix is not None:
            base = name[: -len(g_suffix)]
            v_name = base + v_suffix
            if v_name not in state:
                raise ValueError(f"missing weight-norm partner {v_name} for {name}")
            target = remap_tensor_name(base + ".weight")
            if target is None:
                raise ValueError(f"no GGUF mapping for folded tensor {base}.weight")
            array = fold_weight_norm(state[name], state[v_name])
            consumed.update((name, v_name))
            n_folded += 1
        elif name.endswith(".weight_v") or name.endswith(".parametrizations.weight.original1"):
            raise ValueError(f"orphan weight-norm vector without preceding scale: {name}")
        else:
            target = remap_tensor_name(name)
            if target is None:
                raise ValueError(f"no GGUF mapping for source tensor {name}")
            tensor = state[name]
            if not tensor.is_floating_point():
                raise ValueError(f"unexpected non-floating tensor {name}: {tensor.dtype}")
            array = np.ascontiguousarray(tensor.float().numpy(), dtype=np.float32)
            consumed.add(name)

        if target in target_names:
            raise ValueError(f"duplicate GGUF tensor name: {target}")
        target_names.add(target)

        # F16 only large convolution weights. Biases, Snake alpha, SR
        # embeddings, and small weights remain F32 for numerical stability.
        if precision == "f16" and target.endswith(".weight") and array.size >= large_tensor_threshold:
            array = np.ascontiguousarray(array.astype(np.float16))
        output.append((target, array))

    unconsumed = sorted(set(state) - consumed)
    if unconsumed:
        raise ValueError("strict tensor audit failed; unconsumed: " + ", ".join(unconsumed))
    return output, skipped, n_folded


def add_metadata(writer: Any, config: dict[str, Any], source_hash: str, precision: str) -> None:
    writer.add_name("VoxCPM2 AudioVAE")
    writer.add_string("voxcpm2.vae.source.repo", SOURCE_REPO)
    writer.add_string("voxcpm2.vae.source.url", SOURCE_URL)
    writer.add_string("voxcpm2.vae.source.file", "audiovae.pth")
    writer.add_string("voxcpm2.vae.source.sha256", source_hash)
    writer.add_string("voxcpm2.vae.source.license", SOURCE_LICENSE)
    writer.add_string("voxcpm2.vae.mapping.reference", MAPPING_SOURCE)
    writer.add_string("voxcpm2.vae.mapping.license", MAPPING_LICENSE)
    writer.add_string("voxcpm2.vae.precision", "mixed-f16-f32" if precision == "f16" else "f32")
    writer.add_bool("voxcpm2.vae.weight_norm_folded", True)

    for key in ("encoder_dim", "latent_dim", "decoder_dim", "sample_rate", "out_sample_rate", "cond_dim"):
        writer.add_uint32(f"voxcpm2.vae.{key}", int(config[key]))
    for key in ("encoder_rates", "decoder_rates", "sr_bin_boundaries"):
        writer.add_array(f"voxcpm2.vae.{key}", [int(value) for value in config[key]])
    for key in ("depthwise", "use_noise_block", "cond_out_layer"):
        writer.add_bool(f"voxcpm2.vae.{key}", bool(config[key]))
    writer.add_string("voxcpm2.vae.cond_type", str(config["cond_type"]))


def write_gguf(
    output_path: Path,
    tensors: list[tuple[str, Any]],
    config: dict[str, Any],
    source_hash: str,
    precision: str,
) -> None:
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("the gguf Python package is required") from exc

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".incomplete")
    if temp_path.exists():
        temp_path.unlink()

    writer = gguf.GGUFWriter(str(temp_path), "voxcpm2-audiovae", use_temp_file=True)
    try:
        add_metadata(writer, config, source_hash, precision)
        for name, array in tensors:
            writer.add_tensor(name, array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        os.replace(temp_path, output_path)
    except Exception:
        try:
            writer.close()
        finally:
            if temp_path.exists():
                temp_path.unlink()
        raise


def print_contract() -> None:
    print(
        """GGUF architecture: voxcpm2-audiovae
Metadata prefix: voxcpm2.vae.*
Tensor prefixes:
  vae.enc.conv0.*
  vae.enc.blk.{0..3}.res.* / .sub.*
  vae.enc.fc_mu.*
  vae.dec.layer.{0..9}.*
  vae.dec.sr_cond.{2..7}.{scale_embed,bias_embed}
All convolution weight_norm pairs are folded to one .weight tensor.
fc_logvar and decoder.sr_bin_boundaries are omitted (metadata supplies the latter)."""
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the official VoxCPM2 audiovae.pth to a VAE-only GGUF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, help="audiovae.pth or its local VoxCPM2 directory")
    parser.add_argument("--output", type=Path, help="output GGUF path")
    parser.add_argument("--config", type=Path, help="optional model config.json override")
    parser.add_argument("--precision", choices=("f16", "f32"), default="f16")
    parser.add_argument(
        "--large-tensor-threshold",
        type=int,
        default=4096,
        help="minimum element count for folded weights to be stored as F16",
    )
    parser.add_argument("--print-contract", action="store_true", help="print the runtime GGUF contract and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_contract:
        print_contract()
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --print-contract is used")
    if args.large_tensor_threshold < 1:
        parser.error("--large-tensor-threshold must be positive")

    try:
        checkpoint, config_path = resolve_inputs(args.input, args.config)
        config = load_config(config_path)
        state, checkpoint_metadata = load_state_dict(checkpoint)
        log(f"checkpoint contains {len(state)} tensors")
        if checkpoint_metadata:
            log("checkpoint metadata: " + json.dumps(checkpoint_metadata, sort_keys=True))
        tensors, skipped, n_folded = convert_tensors(state, args.precision, args.large_tensor_threshold)
        source_hash = sha256_file(checkpoint)
        f16_count = sum(1 for _, array in tensors if str(array.dtype) == "float16")
        total_mib = sum(array.nbytes for _, array in tensors) / (1 << 20)
        log(
            f"audit passed: {len(tensors)} output tensors, {n_folded} folded weights, "
            f"{len(skipped)} skipped source tensors"
        )
        log(f"payload: {total_mib:.1f} MiB ({f16_count} F16, {len(tensors) - f16_count} F32)")
        log(f"source SHA-256: {source_hash}")
        write_gguf(args.output, tensors, config, source_hash, args.precision)
        log(f"wrote {args.output.resolve()} ({args.output.stat().st_size / (1 << 20):.1f} MiB)")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        log(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
