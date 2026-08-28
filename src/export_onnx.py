"""Export a frozen MOMENT encoder and reconstruction decoder to ONNX.

The exporter prefers the reconstruction checkpoint from MOMENT and falls back
to a small autoencoder when MOMENT or its checkpoint cannot be loaded. Inputs
are expected to have shape (batch, channels, window).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)


class FallbackEncoder(nn.Module):
    def __init__(self, input_size: int, latent_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class FallbackDecoder(nn.Module):
    def __init__(self, channels: int, window: int, latent_size: int) -> None:
        super().__init__()
        output_size = channels * window
        self.network = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )
        self.channels = channels
        self.window = window

    def forward(self, latent: Tensor) -> Tensor:
        out = self.network(latent)
        return out.reshape(-1, self.channels, self.window)


class MomentEncoder(nn.Module):
    """Expose MOMENT's preprocessing and frozen transformer as one graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, values: Tensor) -> Tensor:
        batch_size, channels, window = values.shape
        # RevIN normalization without non-exportable nanmean
        normalized = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean = normalized.mean(dim=-1, keepdim=True)
        variance = ((normalized - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (normalized - mean) / torch.sqrt(variance + 1e-5)
        
        patch_length = self.model.patch_len
        stride = self.model.tokenizer.stride
        patch_count = (window - patch_length) // stride + 1
        patches = torch.stack(
            [normalized[..., start : start + patch_length] for start in range(0, patch_count * stride, stride)],
            dim=2,
        )
        embedded = self.model.patch_embedding.value_embedding(patches)
        if self.model.patch_embedding.add_positional_embedding:
            embedded = embedded + self.model.patch_embedding.position_embedding(embedded)
            
        patch_count = embedded.shape[2]
        embedded = embedded.reshape(batch_size * channels, patch_count, -1)
        attention_mask = torch.ones((batch_size * channels, patch_count), device=values.device)
        encoded = self.model.encoder(inputs_embeds=embedded, attention_mask=attention_mask)
        
        # 4D tensor: (batch, channels, patch_count, d_model)
        full_latent = encoded.last_hidden_state.reshape(batch_size, channels, patch_count, -1)
        
        # Mean-pool over channel and patch axes to output a clean 2D vector: (batch_size, d_model)
        return full_latent.mean(dim=(1, 2))


class MomentDecoder(nn.Module):
    """Expose MOMENT's reconstruction head accepting a pooled latent vector."""

    def __init__(self, model: nn.Module, channels: int, window: int) -> None:
        super().__init__()
        self.model = model
        self.channels = channels
        self.window = window
        self.d_model = model.d_model
        
        # Determine patch count matching encoder
        patch_length = model.patch_len
        stride = model.tokenizer.stride
        self.patch_count = (window - patch_length) // stride + 1
        
        self.expand = nn.Linear(self.d_model, channels * self.patch_count * self.d_model)
        self.head = model.head

    def forward(self, latent: Tensor) -> Tensor:
        batch_size = latent.shape[0]
        # Expand 2D latent vector back to 4D tensor for MOMENT's head
        expanded = self.expand(latent)
        reshaped = expanded.reshape(batch_size, self.channels, self.patch_count, self.d_model)
        return self.head(reshaped)


def freeze(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module.eval()


def load_models(
    channels: int, window: int, latent_size: int
) -> Tuple[nn.Module, nn.Module, str]:
    """Load MOMENT, or return a deterministic lightweight fallback."""
    try:
        from momentfm import MOMENTPipeline

        moment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-base",
            model_kwargs={"task_name": "reconstruction"},
        )
        moment.init()
        LOGGER.info("Successfully loaded MOMENT-1-base with %d channels.", channels)
        return (
            freeze(MomentEncoder(moment)),
            freeze(MomentDecoder(moment, channels, window)),
            "MOMENT-1-base",
        )
    except Exception as error:
        LOGGER.warning("Could not load MOMENT (%s); using the lightweight fallback.", error)
        input_size = channels * window
        encoder = FallbackEncoder(input_size, latent_size)
        decoder = FallbackDecoder(channels, window, latent_size)
        return freeze(encoder), freeze(decoder), "fallback-autoencoder"


def export_graphs(
    encoder: nn.Module,
    decoder: nn.Module,
    example_input: Tensor,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        latent = encoder(example_input)

    torch.onnx.export(
        encoder,
        example_input,
        output_directory / "encoder.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["latent"],
        dynamic_axes={"input": {0: "batch"}, "latent": {0: "batch"}},
        opset_version=14,
    )
    torch.onnx.export(
        decoder,
        latent,
        output_directory / "decoder.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["latent"],
        output_names=["reconstruction"],
        dynamic_axes={"latent": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=14,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Set default channels to 2 (DE and FE channels from CWRU) and window to 512
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).parents[1] / "models")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.channels < 1 or args.window < 1 or args.latent_size < 1:
        raise ValueError("channels, window, and latent-size must be positive")
    torch.manual_seed(42)
    encoder, decoder, model_name = load_models(args.channels, args.window, args.latent_size)
    example_input = torch.zeros(1, args.channels, args.window)
    export_graphs(encoder, decoder, example_input, args.output_directory)
    LOGGER.info("Exported frozen %s encoder and decoder to %s", model_name, args.output_directory)


if __name__ == "__main__":
    main()