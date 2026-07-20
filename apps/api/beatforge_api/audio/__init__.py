"""BeatForge audio analysis public API."""

from .config import PRESETS, AnalysisConfig, AnalysisMode, get_config
from .io import AudioDecodeError, audio_from_array, load_audio
from .melody import MelodyExtractionResult, extract_melody_candidates
from .models import AnalysisResult, AudioData, OnsetCandidate, TempoEstimate
from .onsets import classify_band, detect_onsets, merge_candidates, refine_candidate_sample
from .pipeline import ANALYSIS_VERSION, analyze_audio, analyze_samples
from .rhythm import RhythmConstraintConfig, constrain_hits_to_rhythm_grid
from .separation import DemucsSeparator, NoopSeparator, SeparationResult, StemSeparator
from .tempo import estimate_tempo
from .waveform import build_waveform_lods, waveform_lods_from_samples

__all__ = [
    "ANALYSIS_VERSION",
    "RhythmConstraintConfig",
    "AnalysisConfig",
    "AnalysisMode",
    "AnalysisResult",
    "AudioData",
    "AudioDecodeError",
    "MelodyExtractionResult",
    "DemucsSeparator",
    "NoopSeparator",
    "OnsetCandidate",
    "PRESETS",
    "SeparationResult",
    "StemSeparator",
    "TempoEstimate",
    "analyze_audio",
    "analyze_samples",
    "constrain_hits_to_rhythm_grid",
    "audio_from_array",
    "build_waveform_lods",
    "classify_band",
    "detect_onsets",
    "estimate_tempo",
    "extract_melody_candidates",
    "get_config",
    "load_audio",
    "merge_candidates",
    "refine_candidate_sample",
    "waveform_lods_from_samples",
]
