from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by installs without chart-ml
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


MODEL_ARCHITECTURE = "beatforge.chart-transformer.encoder.v1"


class TorchUnavailableError(RuntimeError):
    """Raised when local chart inference is requested without PyTorch installed."""


def torch_available() -> bool:
    return torch is not None and nn is not None


def require_torch() -> Any:
    if not torch_available():
        raise TorchUnavailableError(
            "PyTorch is required for the chart sequence model; install beatforge-api[chart-ml]."
        )
    return torch


@dataclass(frozen=True, slots=True)
class ChartTransformerConfig:
    input_dim: int
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 512
    lane_count: int = 5
    maximum_difficulty: int = 15

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.d_model <= 0 or self.d_model % self.nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.num_layers <= 0 or self.dim_feedforward <= 0:
            raise ValueError("Transformer layer sizes must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.lane_count != 5:
            raise ValueError("the SPEED single-panel model requires exactly five lanes")
        if self.maximum_difficulty != 15:
            raise ValueError("the chart engine difficulty contract is fixed to 1-15")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChartTransformerConfig:
        return cls(**value)


if nn is not None:

    class ChartTransformer(nn.Module):
        """Transformer encoder over ordered BeatForge candidate events.

        Five independent lane logits allow jumps, while a separate event-level
        hold logit estimates whether at least one predicted lane should start a hold.
        Difficulty is a learned sequence condition and is always in the public 1-15
        range.
        """

        def __init__(self, config: ChartTransformerConfig) -> None:
            super().__init__()
            self.config = config
            self.input_projection = nn.Linear(config.input_dim, config.d_model)
            self.position_embedding = nn.Embedding(config.max_sequence_length, config.d_model)
            self.difficulty_embedding = nn.Embedding(config.maximum_difficulty + 1, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=config.num_layers,
                norm=nn.LayerNorm(config.d_model),
                enable_nested_tensor=False,
            )
            self.output_norm = nn.LayerNorm(config.d_model)
            self.lane_head = nn.Linear(config.d_model, config.lane_count)
            self.hold_head = nn.Linear(config.d_model, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.normal_(self.position_embedding.weight, std=0.02)
            nn.init.normal_(self.difficulty_embedding.weight, std=0.02)
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
            nn.init.xavier_uniform_(self.lane_head.weight)
            nn.init.zeros_(self.lane_head.bias)
            nn.init.xavier_uniform_(self.hold_head.weight)
            nn.init.zeros_(self.hold_head.bias)

        def forward(
            self,
            features: Any,
            difficulties: Any,
            padding_mask: Any | None = None,
        ) -> dict[str, Any]:
            if features.ndim != 3:
                raise ValueError("features must have shape [batch, sequence, feature]")
            batch_size, sequence_length, feature_count = features.shape
            if feature_count != self.config.input_dim:
                raise ValueError(
                    f"expected {self.config.input_dim} features, received {feature_count}"
                )
            if sequence_length > self.config.max_sequence_length:
                raise ValueError(
                    "sequence length exceeds the configured positional embedding limit"
                )
            if difficulties.shape != (batch_size,):
                raise ValueError("difficulties must have shape [batch]")
            if bool(
                ((difficulties < 1) | (difficulties > self.config.maximum_difficulty)).any().item()
            ):
                raise ValueError("difficulty must be between 1 and 15")

            positions = torch.arange(sequence_length, device=features.device)
            positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
            encoded = self.input_projection(features)
            encoded = encoded + self.position_embedding(positions)
            encoded = encoded + self.difficulty_embedding(difficulties).unsqueeze(1)
            encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
            encoded = self.output_norm(encoded)
            return {
                "lane_logits": self.lane_head(encoded),
                "hold_logits": self.hold_head(encoded).squeeze(-1),
            }

else:

    class ChartTransformer:  # pragma: no cover - only defined without PyTorch
        def __init__(self, _config: ChartTransformerConfig) -> None:
            require_torch()
