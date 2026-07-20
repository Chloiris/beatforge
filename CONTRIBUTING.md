# Contributing to BeatForge Studio

Thank you for helping improve BeatForge Studio. Keep changes focused, reproducible, and safe for
people working with private audio.

## Development setup

Install Python 3.11 or newer, Node.js 20 or newer, pnpm 11, FFmpeg, and ffprobe. Python 3.12 and
Node.js 22 match the continuous-integration environment.

From the repository root, run:

```text
python scripts/beatforge.py install
python scripts/beatforge.py seed
python scripts/beatforge.py dev
```

The task runner resolves `.venv/bin/python` on macOS/Linux and
`.venv/Scripts/python.exe` on Windows. Do not commit a virtual environment or downloaded model
weights.

Optional model runtimes are intentionally separate from the base installation. Prepare them only
when the change being tested requires them; the normal test suite must not download models.

## Quality checks

Run the same base gates used by CI before opening a pull request:

```text
python scripts/beatforge.py test
python scripts/beatforge.py lint
python scripts/beatforge.py build
```

Changes to an end-to-end workflow should also include an appropriate Playwright check. Keep tests
deterministic and use the generated demo corpus or small synthetic fixtures.

## Privacy and test data

- Never commit uploaded songs, separated stems, lyrics, local databases, model weights, machine
  paths, project identifiers, or reports derived from private projects.
- Do not use commercial song or artist names as convenient fixtures. Use fictional Unicode text
  when a test needs to exercise non-ASCII filenames.
- Screenshots and evaluation reports must come from the generated demo projects and must not expose
  local paths or user data.
- Do not weaken ignore rules for `storage/`, caches, build output, or local environment files.

## Pull requests

Explain the user-visible outcome, implementation boundary, and verification performed. Include or
update tests for behavior changes, keep API/data-schema changes explicit, and update documentation
when commands or supported platforms change. Avoid mixing unrelated formatting or refactors into
the same pull request.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
For suspected vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
