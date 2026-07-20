# BeatForge Studio implementation plan

## Vertical slices

1. **Persistent audio project** — upload or generate an audio file, retain exact original sample metadata, create a project and survive an API restart.
2. **Real analysis** — decode to a working mono signal, produce multi-resolution waveform peaks, run multi-scale and multi-band onset detection, refine every candidate against the original mix, estimate tempo/offset, and persist the result.
3. **Reproducible demos** — synthesize three copyright-free tracks from fixed seeds, keep sample-index ground truth separate from detection, seed idempotently, and publish one-to-one evaluation metrics.
4. **Workspace to editor** — browse/filter projects, upload and monitor real stages, then open a project with playable audio, LOD waveform, sample-aligned grid and hit points.
5. **Sample-accurate editing** — add, drag, nudge, classify, lock, delete, filter, snap, undo/redo and debounce-save integer sample positions without replacing them with snapped positions.
6. **Delivery gate** — backend tests, frontend unit/interaction tests, one local Playwright journey, lint, production builds, Docker configuration and reproducible documentation.
7. **Japanese vocal charting** — isolate Qwen dependencies, explicitly cache local models, persist
   lyrics and jobs, align corrected lyrics to a vocals stem, refine voiced attacks, solve a
   sentence-level 1/16 mapping, expose aligned/refined/grid samples and replace only automatic
   vocal hits after an explicit user action.
8. **Source-conditioned charting** — persist every real acoustic candidate independently of the
   final chart, retain explicit acoustic/chart sample coordinates, route vocals/melody/drums as
   concurrent lanes, extract local melody pitch onsets, fuse Qwen phrase regions with vocal
   acoustic attacks, and replace vocal points only inside successful coverage intervals.

## Architectural decisions

- The integer original-audio sample index is authoritative. Seconds are derived at API and render boundaries.
- SQLite stores project state and job stages; original audio, waveform LOD data and analysis artifacts live under `storage/`.
- The API performs CPU work outside request handlers and records each real stage instead of simulating percentage progress.
- The timeline uses a bounded Canvas viewport. Grid positions are computed from an absolute beat index, never by repeated floating-point addition.
- Accurate mode is an optional `StemSeparator` implementation. When Demucs is unavailable it records a warning and executes the balanced pipeline.
- The three demo ground-truth files are evaluation inputs only. The analysis package has no ground-truth import path.
- ASR text is always a draft. Known-lyrics alignment is authoritative only after user review; the
  system does not download copyrighted lyrics or treat romaji as the acoustic alignment target.
- Qwen runs in `.venv-qwen` with offline environment flags. The API worker cannot implicitly
  download a model and reports a structured missing-model error instead.
- Qwen phrase timing is semantic context, never a source of fabricated mora/phoneme attacks. A
  final vocal candidate must be backed by a real local acoustic event.
- `CandidateEvent` is the audit layer. `acousticSample` remains the heard event;
  `chartSample` is the playable grid recommendation; rejected alternatives remain inspectable.
- BPM can score and place an existing event, but cannot synthesize an event in an empty grid cell.

## Completion gates

- Three generated WAV files, cover SVGs, database rows, real analyses and `reports/demo-evaluation.json`.
- Optional source-conditioned diagnostics are generated only from a developer's local track and
  remain ignored; unavailable human-label metrics are explicit nulls.
- Upload, analysis, playback/seek, editor persistence and JSON/CSV export operate end to end.
- Backend pytest, frontend Vitest, critical Playwright journey, lint and production builds have each been run at least once.
