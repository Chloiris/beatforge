# BeatForge API

FastAPI service for local projects, audio uploads, sample-accurate hit-point editing, waveform
LODs, five-track marker export and optional local audio models.

Use the cross-platform task runner from the repository root:

```bash
python scripts/beatforge.py install
python scripts/beatforge.py dev
```

For an API-only process after installation:

```bash
.venv/bin/python -m uvicorn beatforge_api.main:app --app-dir apps/api --host 127.0.0.1 --port 8000
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. The service has no
authentication and is intended to remain bound to the local loopback interface.
