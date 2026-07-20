from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import get_settings
from .base import (
    AdapterDiagnostics,
    AdapterOutput,
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentContext,
    executable_from_environment,
)

PUBLIC_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "schufo/lyrics-aligner",
        "source": "https://github.com/schufo/lyrics-aligner",
        "paper": "Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation",
        "license": "MIT",
        "checkpointPublic": True,
        "checkpointBytes": 40_357_640,
        "supportedLanguages": ["en"],
        "phoneInventory": "39 English ARPAbet phones",
        "runtime": "Python 3.6 / PyTorch 1.5 reference implementation; CPU or CUDA only",
        "rejectedCode": "UNSUPPORTED_LANGUAGE",
        "rejectedReason": "The public checkpoint has no Japanese phone inventory.",
    },
    {
        "id": "jhuang448/LyricsAlignment-Multilingual",
        "source": "https://github.com/jhuang448/LyricsAlignment-Multilingual",
        "paper": "Multilingual lyrics alignment baseline trained on singing datasets",
        "license": "MIT",
        "checkpointPublic": True,
        "checkpointBytes": 57_491_286,
        "supportedLanguages": ["en", "fr", "de", "it", "es"],
        "phoneInventory": "Western-language IPA inventory supplied by the repository phonemizer",
        "runtime": "Reference pins PyTorch 2.1.2 and legacy NumPy/librosa versions",
        "rejectedCode": "UNSUPPORTED_LANGUAGE",
        "rejectedReason": "The public phonemizer/checkpoint does not support Japanese.",
    },
)


class SingingAlignmentAdapter(AlignmentAdapter):
    """Capability record for public singing-specific lyrics aligners.

    None of the verified, downloadable singing-specific checkpoints supports
    Japanese.  Returning an explicit unavailable result is preferable to
    mapping Japanese phones into an English/Western inventory and fabricating a
    plausible-looking timeline.
    """

    method = "singing"
    name = "Public Singing Voice Aligner"

    @staticmethod
    def _candidate_root(context: AlignmentContext | None = None) -> Path:
        settings = get_settings() if context is None else None
        models_dir = context.models_dir if context else settings.models_dir
        return executable_from_environment(
            "BEATFORGE_SINGING_MODEL_ROOT",
            models_dir / "singing-alignment",
        )

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        candidate_root = self._candidate_root(context)
        candidates: list[dict[str, Any]] = []
        for candidate in PUBLIC_CANDIDATES:
            candidate_id = str(candidate["id"])
            local_name = candidate_id.replace("/", "--")
            local_path = candidate_root / local_name
            candidates.append(
                {
                    **candidate,
                    "localPath": str(local_path),
                    "localCheckpointPresent": any(
                        path.is_file()
                        for path in (
                            local_path / "model_parameters.pth",
                            local_path / "checkpoint_Baseline",
                            local_path / "checkpoint.pt",
                        )
                    ),
                }
            )
        reason = (
            "已验证的公开 singing lyrics alignment checkpoints 均不支持日语；"
            "为避免伪结果，本方法对当前日语实验明确标记为 unavailable。"
        )
        return AdapterDiagnostics(
            available=False,
            reason=reason,
            model="No Japanese-compatible public checkpoint",
            automatic_downloads_enabled=False,
            details={
                "requestedLanguage": "ja",
                "failureStage": "preflight.language",
                "failureCode": "UNSUPPORTED_LANGUAGE",
                "retryable": False,
                "candidateRoot": str(candidate_root),
                "candidates": candidates,
                "selectionPolicy": (
                    "public code + public checkpoint + local deployment + singing-specific + "
                    "complete Japanese phone inventory"
                ),
                "checkedAt": "2026-07-19",
                "approximatePhoneMappingAllowed": False,
                "automaticDownloadSuppressedReason": (
                    "Downloading an incompatible checkpoint cannot make Japanese inference valid."
                ),
            },
        )

    def run(self, context: AlignmentContext) -> AdapterOutput:
        diagnostics = self.diagnostics(context)
        raise AlignmentAdapterError(
            "SINGING_MODEL_UNSUPPORTED_LANGUAGE",
            diagnostics.reason or "No public singing alignment checkpoint supports Japanese.",
            status="unavailable",
            details=diagnostics.details,
        )
