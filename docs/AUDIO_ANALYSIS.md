# Audio analysis

The production analyzer accepts an audio path or an in-memory signal. It has no import of the
demo synthesizer, manifest or ground-truth files. The current production analysis version is
`0.5.0`. Every persisted time anchor is an integer index at the original sample rate.

## Base mix pipeline

`recall`, `balanced` and `clean` share the following real audio pipeline:

1. Probe the original format, sample rate, channels and exact frame count. Compatible formats
   are decoded through FFmpeg when they are not directly readable.
2. Build a 44.1 kHz working copy, remove DC, apply percentile-based robust normalization,
   preserve the original file and measure leading silence. A channel-aligned stereo copy is
   retained for optional source separation; base analysis uses mono.
3. Compute harmonic/percussive separation for the mix.
4. Extract STFT evidence at 1024, 2048 and 4096 samples with a 128-sample coarse hop, plus a
   512-sample fine detector with a 64-sample hop.
5. Produce full-mix and percussive positive spectral flux, sub/low (20–180 Hz), low-mid
   (180–800 Hz), mid (800–3000 Hz), high (3000–8000 Hz), air, energy derivative, RMS
   derivative and high-frequency-content change.
6. Normalize every curve against a sliding local median and MAD and fuse it using
   `AnalysisConfig` weights.
7. Group correlated curves into independent detector families before voting. Mix/fine flux,
   percussive flux, waveform-energy attack and multi-band flux are separate families; correlated
   band curves cannot manufacture multiple consensus votes.
8. Refine each coarse peak in a bounded local window using the analyzed waveform and percussive
   envelope rise. Preserve both `detectedSample` and `refinedSample`.
9. Merge only an 8–11 ms detector cluster and suppress weaker decay echoes without folding a
   genuine dense double hit into its predecessor.
10. Classify comparable band evidence at one shared attack frame and calculate confidence
    (whether an attack exists) separately from salience (whether it is chart-relevant).
11. Estimate tempo from the fused envelope and detected event intervals. Score BPM/2, BPM and
    BPM×2 hypotheses against an absolute grid and estimate phase/offset.
12. Select chart candidates. Acoustic anchors, strong/salient events and local standouts are
    retained; the base mix path may rescue sufficiently evidenced events near a fine rhythm grid.
13. Map the analysis index once to the original sample rate, compute a non-destructive snap
    recommendation and persist the integer sample.

Tempo estimation uses the full detected event set; chart-oriented selection does not feed back
into BPM detection. `analysisMetadata.candidateSelection` records detected/selected counts and
selection reasons.

## Accurate source-aware pipeline

`accurate` first runs the locally cached `htdemucs` four-source model and produces time-aligned
`vocals`, `drums`, `bass` and `other` stems. It then differs from the base path in four important
ways:

1. Mix and every stem run independent feature extraction and onset detection. Their feature
   curves are not max-blended into a single envelope.
2. A section-focus pass still summarizes local dominance, but it is soft routing evidence rather
   than an exclusive gate. Acoustically active vocal, drum and melodic lanes can all retain
   candidates in the same interval; `focusMap.alternatives` exposes the competing scores.
3. `other` uses a local `librosa.pyin` melody extractor. Voiced pitch changes and energy
   re-attacks create melody candidates; failure leaves that lane empty with a warning instead of
   pretending every `other` transient is melody.
4. Every existing event is scored with `0.35 sourceEvidence + 0.25 acousticConfidence +
   0.20 rhythmAlignment + 0.20 semanticEvidence`. Strong off-grid evidence may survive. BPM never
   fills an empty cell. The final chart keeps the strongest event per grid cell and 30 ms density
   neighborhood, while accepted, uncertain and rejected alternatives remain `CandidateEvent`s.

This is an inspectable source-conditioned heuristic, not semantic transcription:

- `vocals` activity is not lyric recognition, ASR or phoneme alignment. It only makes vocal
  acoustic attacks available as chart candidates.
- `other` is not a piano classifier. It can contain piano, guitar, synths, orchestral sources and
  separation leakage.
- A drum focus means the separated drum stem dominates under the current thresholds; it does not
  identify kick, snare or a human-declared solo.

## Japanese vocal lyric alignment and coverage fallback

Lyric timing is an explicit second pipeline; it never treats the source-focus vocal onset list as
phonetic truth:

1. Require a persisted `vocals.flac` from accurate analysis.
2. Accept corrected Japanese/kana lyrics, or run local `Qwen3-ASR-1.7B` to create a visible draft.
3. Divide the vocal stem into non-overlapping 20-second cores. Absolute short-time activity gates
   prevent silent chunks from reaching ASR. `Qwen3-ASR-1.7B` transcribes each remaining chunk
   without the known lyrics as context, so the guidance cannot simply echo the supplied text.
4. A monotonic dynamic program matches contiguous saved lyric lines to those singing-ASR chunks.
   Instrumental chunks and unmatched lyric lines may be skipped; unrelated text is never forced
   into an otherwise silent interval.
5. Run `Qwen3-ForcedAligner-0.6B` only on each matched short chunk, with 1.25 seconds of boundary
   context. Every returned boundary is immediately mapped back to an integer whole-song sample.
6. Treat each Qwen span conservatively as a phrase-level semantic region. It does not claim a
   phoneme or mora boundary. Inside that region, select one unused local vocal acoustic candidate
   using onset strength, envelope rise, pitch change and spectral transition; otherwise use the
   bounded local attack refiner.
7. Persist pitch/transition/acoustic evidence. `phonemeConfidence` remains zero unless a future
   singing-domain phoneme model supplies it. ASR timestamps alone never become final hits.
8. Record every 20-second interval as `success`, `silent`, `asr_failed`, `unassigned`,
   `alignment_failed`, `alignment_collapse`, `insufficient_anchors` or `low_confidence`, including
   exact sample bounds, confidence, anchor count and raw timestamp count.
9. Replacement is interval-local. Only a successful interval with enough reliable anchors can
   replace its old automatic vocal points. Failed, empty and low-confidence intervals retain the
   previous vocal acoustic fallback. If the full aligner fails, the local vocal detector supplies
   real acoustic fallbacks; if neither source has evidence, the job fails explicitly.
10. Persist `acousticSample` and `chartSample` separately. The timeline renders both with a
   connector instead of hiding timing error.

The official aligner returns character/word spans but no calibrated confidence value. Displayed
anchor confidence is therefore a labeled heuristic derived from span duration and local
voiced-attack strength, not an official Qwen probability.

Qwen runs in `.venv-qwen` via a filesystem-only subprocess. `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` are enabled for every job. Downloads occur only through
`python scripts/beatforge.py prepare-vocal-models`; missing dependencies or weights yield
`VOCAL_MODEL_NOT_READY`.

The selected stem candidate is locally refined on that stem and then mapped to the original
integer-sample timeline. For an accepted candidate, `detectedSample` retains the model boundary,
`acousticSample`, `refinedSample` and legacy `sample` retain the acoustic attack;
`chartSample` and legacy `snappedSample` store the integer playable 1/16 suggestion. `snapErrorMs`
records the difference. Stems are kept the same
duration as the original audio. A second original-mix attack-boundary refinement after mapping is
not implemented yet, so source-separation bleed or phase behavior can still shift an event and
must remain visible for manual correction.

## Accurate-mode installation and cache policy

The base install never imports Demucs or PyTorch. Install the optional dependency group:

```bash
python scripts/beatforge.py install-accurate
```

Model download is deliberately disabled inside analysis jobs. Explicitly prepare the model once:

```bash
python scripts/beatforge.py prepare-model
```

The task runner stores Demucs weights under `storage/models/torch/hub/checkpoints/`; no weight is
committed to this repository. The first explicit
prepare command needs network access, while analysis after that is local. Device selection is
MPS, then CUDA, then CPU. The worker runs one analysis job at a time because source separation can
use substantially more memory than the base pipeline.

If the Python dependencies or a local checkpoint are missing, `accurate` records a warning,
changes `effectiveMode` to `balanced` and completes normally. It never waits on an implicit model
download.

## Presets

- `recall` lowers peak and vote thresholds and uses a 12 ms minimum interval.
- `balanced` is the default compromise for editor candidates.
- `clean` raises evidence/prominence requirements and uses a 24 ms minimum interval.
- `accurate` requests local source separation and section focus; unavailable dependencies or
  weights cause the explicit `balanced` fallback described above.

Sensitivity adjusts configured evidence thresholds but never changes sample mapping.

## Reanalysis and user edits

A completed reanalysis transaction replaces only unedited algorithm-derived hit points. Hits
with `manuallyEdited=true`, `locked=true`, or `source=manual` keep their IDs, integer samples and
editor metadata. Newly detected hits inside the merge window of a preserved user hit are omitted.
Manually edited tempo segments are also retained. A project that retains either kind of user edit
remains in the `edited` state.

## Waveform LOD and stem artifacts

For every mix and available stem level, the worker stores exact min/max pairs per window. Window
size doubles across levels. The API returns an explicitly requested level or chooses the first
level below a requested point budget, so a long track never transfers every decoded sample.

Successful accurate jobs also save local FLAC files under `storage/stems/<track-id>/`. The API
serves every stem with Range support for browser seek. Mix and all visible stem lanes use the same
original integer sample coordinate; the UI may hide a lane, but hiding it never changes hit times.

## Evaluation

`scripts/evaluate_onsets.py` performs distance-sorted one-to-one assignment at ±10, ±20, ±30
and ±50 ms. It reports precision, recall, F1, median/P95 timing error, false positives and
negatives per minute, predictions clustered within 8 ms, BPM/offset error, and straight-1/16
grid-cell occupancy accuracy. Strong events are reported separately using synthesizer strength.
Reports preserve actual results even when a target is missed.

`scripts/evaluate_source_conditioned.py` measures active vocal-frame and active-section proximity
on a stored local vocal stem. Its after case is a non-persisted acoustic fallback dry run. This is
a coverage proxy, not event precision/recall. Saved lyric-line assignment coverage is reported as
a labeled proxy; boundary P/R/F1, chart accuracy and editing time remain null when the user song
has no independent human labels or telemetry.

### Current deterministic demo baseline

The checked-in report exercises the `balanced` mix path. The v0.5 accurate source-conditioned
1/16 constraint do not read or receive demo ground truth.

| Track | Truth / prediction | Strong F1 @ 10/20 ms | All-event F1 @ 10 ms | Median / P95 error |
| --- | ---: | ---: | ---: | ---: |
| Neon Pulse | 144 / 136 | 0.967742 / 0.967742 | 0.971429 | 0.226757 / 0.498866 ms |
| Iron Rift | 216 / 202 | 0.953020 / 0.953020 | 0.966507 | 0.249433 / 4.399093 ms |
| Glass Tide | 38 / 38 | 1.000000 / 1.000000 | 1.000000 | 0.215420 / 1.201814 ms |

### Evaluation without ground truth

For any locally supplied track without independently annotated onset truth, candidate counts,
focus-source proportions, separated waveforms and visual density can verify only that the
source-aware pipeline executed. They are **not** precision, recall, F1 or an accuracy result. No
accuracy claim should be made until an independent annotation set exists and is evaluated without
being supplied to the detector.
