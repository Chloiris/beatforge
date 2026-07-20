# Third-party notices

BeatForge Studio is an original local application. The following direct dependencies are
redistributed or installed by the project. Their own license texts and copyright notices
remain controlling; this summary is not a replacement for those notices.

## Frontend

| Dependency | Locked version | License |
| --- | ---: | --- |
| React / React DOM | 19.1.0 | MIT |
| React Router DOM | 7.18.1 | MIT |
| TanStack Query | 5.83.0 | MIT |
| Zustand | 5.0.6 | MIT |
| Vite | 7.0.6 | MIT |
| TypeScript | 5.8.3 | Apache-2.0 |
| Vitest | 3.2.4 | MIT |
| Testing Library | 16.3.0 | MIT |
| Playwright | 1.54.1 | Apache-2.0 |
| ESLint | 9.31.0 | MIT |

## Backend and audio processing

| Dependency | Locked version | License |
| --- | ---: | --- |
| FastAPI | 0.115.12 | MIT |
| Uvicorn | 0.34.2 | BSD-3-Clause |
| Pydantic | 2.11.5 | MIT |
| SQLAlchemy | 2.0.41 | MIT |
| NumPy | 2.2.6 | BSD-3-Clause |
| SciPy | 1.15.3 | BSD-3-Clause |
| librosa | 0.11.0 | ISC |
| soxr (librosa resampling runtime) | 1.1.0 | LGPL-2.1-or-later |
| SoundFile | 0.13.1 | BSD-3-Clause |
| libsndfile (runtime dependency) | system package | LGPL-2.1-or-later |
| pytest | 8.3.5 | MIT |
| Ruff | 0.11.11 | MIT |

## Optional accurate-mode audio processing

These packages are not installed by the base install task. They are activated only through the
`apps/api[accurate]` optional dependency group. Versions below are the exact versions declared and
validated in the current local workspace; model files are not committed to the repository.

| Dependency | Locked version | License |
| --- | ---: | --- |
| Demucs | 4.0.1 | MIT |
| PyTorch | 2.13.0 | BSD-3-Clause |
| TorchAudio (Demucs runtime) | 2.11.0 | BSD-2-Clause |
| Julius (Demucs runtime) | 0.2.8 validated | MIT |
| Open-Unmix (Demucs runtime) | 1.3.0 validated | MIT |
| Dora Search (Demucs runtime) | 0.1.12 validated | MIT |
| lameenc (Demucs runtime) | 1.8.4 validated | LGPL-3.0-or-later |

The `htdemucs` checkpoint is downloaded separately, only after an explicit developer command,
into `storage/models/torch/hub/checkpoints/` when using the project task runner. It is not bundled,
committed, copied into Docker images, or redistributed by this repository. The upstream Demucs
project and checkpoint distribution terms remain controlling.

## Optional local Japanese vocal alignment

These dependencies are installed into the isolated `.venv-qwen` environment only after running
`python scripts/beatforge.py install-vocal`. Model files are fetched only by the explicit
`prepare-vocal-models` task and are excluded from Git and application images.

| Dependency/model | Locked version | License |
| --- | ---: | --- |
| Qwen3-ASR Python runtime | 0.0.6 | Apache-2.0 |
| Qwen3-ASR-1.7B weights | 2026 public checkpoint | Apache-2.0 |
| Qwen3-ForcedAligner-0.6B weights | 2026 public checkpoint | Apache-2.0 |
| Transformers (Qwen runtime pin) | 4.57.6 | Apache-2.0 |
| Accelerate (Qwen runtime pin) | 1.12.0 | Apache-2.0 |
| PyTorch | 2.13.0 validated | BSD-3-Clause |
| pyopenjtalk | 0.4.1 | MIT; bundled Open JTalk components use Modified BSD |

BeatForge never calls the commercial DashScope/Qwen API. The local adapter resolves explicit
directories or an already present cache, sets offline mode during inference, and reports missing
weights as a structured error instead of downloading them from an analysis job.

## Optional Singing Alignment Lab

Alignment Lab models and tools are optional, excluded from Git and application images, and run
only on local audio. The CTC preparation command pins an immutable revision and records per-file
SHA-256 values in `storage/models`; inference forces Hugging Face and Transformers offline modes.

| Dependency/model | Pinned version or revision | License |
| --- | ---: | --- |
| Japanese HuBERT phoneme CTC (`prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`) | `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99` | Apache-2.0 |
| Montreal Forced Aligner (optional external CLI) | user-installed compatible release | MIT |
| `japanese_mfa` acoustic model | `2.0.1a` | Upstream model terms control; downloaded by MFA, not redistributed |

The public singing-specific checkpoints inspected by the adapter are capability records only and
are not downloaded or redistributed: `schufo/lyrics-aligner` and
`jhuang448/LyricsAlignment-Multilingual` do not support the Japanese phone inventory required by
the included experiment. BeatForge reports that limitation instead of transforming Japanese
phones into an unsupported inventory.

## System software

- FFmpeg is invoked as a separate executable for compatible decoding. FFmpeg is
  available under LGPL-2.1-or-later or GPL-2.0-or-later depending on how a particular
  system build was configured; BeatForge does not bundle a binary in this repository.
- libsndfile is loaded by SoundFile. Its system package is LGPL-2.1-or-later and is not bundled by
  this repository.

The generated demo recordings and geometric covers are produced by this repository and
contain no third-party music, samples, images or model weights.
