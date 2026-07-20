#!/usr/bin/env python3
"""Cross-platform development and model-preparation tasks for BeatForge Studio."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "apps" / "api"
WEB_DIR = PROJECT_ROOT / "apps" / "web"
sys.path.insert(0, str(API_DIR))

from beatforge_api.platform_paths import venv_executable  # noqa: E402


def _load_project_environment() -> None:
    """Load the simple KEY=VALUE project file without a bootstrap dependency."""

    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


_load_project_environment()

BASE_VENV = PROJECT_ROOT / ".venv"
VOCAL_VENV = PROJECT_ROOT / ".venv-qwen"
os.environ.setdefault(
    "TORCH_HOME", str(PROJECT_ROOT / "storage" / "models" / "torch")
)


class TaskError(RuntimeError):
    pass


def _tool(value: str, label: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if resolved.is_file():
            return str(resolved)
    discovered = shutil.which(value)
    if discovered:
        return discovered
    raise TaskError(f"{label} was not found: {value}")


def _bootstrap_python() -> str:
    return _tool(os.environ.get("PYTHON", sys.executable), "Python")


def _pnpm() -> str:
    return _tool(os.environ.get("PNPM", "pnpm"), "pnpm")


def _base_python() -> str:
    path = venv_executable(PROJECT_ROOT, BASE_VENV)
    if not path.is_file():
        raise TaskError("Base environment is missing; run `python scripts/beatforge.py install`.")
    return str(path)


def _vocal_python() -> str:
    path = venv_executable(PROJECT_ROOT, VOCAL_VENV)
    if not path.is_file():
        raise TaskError(
            "Vocal environment is missing; run `python scripts/beatforge.py install-vocal`."
        )
    return str(path)


def _ci_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CI"] = "true"
    return environment


def _web_dev_environment(host: str, api_port: str) -> dict[str, str]:
    environment = os.environ.copy()
    proxy_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    environment.setdefault("VITE_API_PROXY", f"http://{proxy_host}:{api_port}")
    return environment


def _native_command(command: Sequence[str]) -> list[str]:
    """Wrap Windows batch launchers so shell=False works for pnpm.cmd/pnpm.bat."""

    parts = [str(part) for part in command]
    if os.name != "nt" or Path(parts[0]).suffix.casefold() not in {".cmd", ".bat"}:
        return parts
    command_processor = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
    if not command_processor:
        raise TaskError("Windows command processor was not found (COMSPEC/cmd.exe).")
    return [command_processor, "/d", "/s", "/c", subprocess.list2cmdline(parts)]


def _run(
    command: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        _native_command(command),
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def install() -> None:
    python = _bootstrap_python()
    base_python = venv_executable(PROJECT_ROOT, BASE_VENV)
    if not base_python.is_file():
        _run([python, "-m", "venv", str(BASE_VENV)])
    _run([str(base_python), "-m", "pip", "install", "-e", f"{API_DIR}[dev]"])
    _run([_pnpm(), "install", "--frozen-lockfile"], environment=_ci_environment())


def seed() -> None:
    python = _base_python()
    _run([python, str(PROJECT_ROOT / "scripts" / "generate_demo_audio.py")])
    _run([python, str(PROJECT_ROOT / "scripts" / "seed_demo_projects.py")])
    _run([python, str(PROJECT_ROOT / "scripts" / "evaluate_onsets.py")])


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def dev() -> None:
    host = os.environ.get("BEATFORGE_HOST", "127.0.0.1").strip() or "127.0.0.1"
    api_port = os.environ.get("BEATFORGE_API_PORT", "8000")
    web_port = os.environ.get("BEATFORGE_WEB_PORT", "5173")
    os.environ.setdefault(
        "BEATFORGE_ALLOWED_ORIGINS",
        f"http://localhost:{web_port},http://127.0.0.1:{web_port}",
    )
    api = subprocess.Popen(
        [
            _base_python(),
            "-m",
            "uvicorn",
            "beatforge_api.main:app",
            "--app-dir",
            str(API_DIR),
            "--host",
            host,
            "--port",
            api_port,
            "--reload",
        ],
        cwd=PROJECT_ROOT,
    )
    web: subprocess.Popen[bytes] | None = None
    try:
        web = subprocess.Popen(
            _native_command(
                [
                    _pnpm(),
                    "--dir",
                    str(WEB_DIR),
                    "dev",
                    "--host",
                    host,
                    "--port",
                    web_port,
                ]
            ),
            cwd=PROJECT_ROOT,
            env=_web_dev_environment(host, api_port),
        )
        while True:
            api_status = api.poll()
            web_status = web.poll()
            if api_status is not None:
                if api_status:
                    raise TaskError(f"API process exited with status {api_status}.")
                return
            if web_status is not None:
                if web_status:
                    raise TaskError(f"Web process exited with status {web_status}.")
                return
            time.sleep(0.2)
    except KeyboardInterrupt:
        return
    finally:
        if web is not None:
            _stop(web)
        _stop(api)


def test() -> None:
    python = _base_python()
    # The copyright-free WAV fixtures are intentionally kept out of Git. Generate
    # any missing files so the public test command also works from a clean clone.
    _run([python, str(PROJECT_ROOT / "scripts" / "generate_demo_audio.py")])
    _run([python, "-m", "pytest"])
    _run(
        [_pnpm(), "--dir", str(WEB_DIR), "test", "--run"],
        environment=_ci_environment(),
    )


def e2e() -> None:
    _run([_pnpm(), "--dir", str(WEB_DIR), "e2e"], environment=_ci_environment())


def lint() -> None:
    _run(
        [
            _base_python(),
            "-m",
            "ruff",
            "check",
            "apps/api/beatforge_api",
            "apps/api/tests",
            "scripts",
            "tests",
        ]
    )
    _run([_pnpm(), "--dir", str(WEB_DIR), "lint"], environment=_ci_environment())


def build() -> None:
    _run([_base_python(), "-m", "build", str(API_DIR), "--outdir", "dist"])
    _run([_pnpm(), "--dir", str(WEB_DIR), "build"], environment=_ci_environment())


def clean_generated() -> None:
    _run([_base_python(), str(PROJECT_ROOT / "scripts" / "clean_generated.py")])


def evaluate() -> None:
    _run([_base_python(), str(PROJECT_ROOT / "scripts" / "evaluate_onsets.py")])


def install_accurate() -> None:
    _run([_base_python(), "-m", "pip", "install", "-e", f"{API_DIR}[accurate]"])


def prepare_model() -> None:
    _run([_base_python(), str(PROJECT_ROOT / "scripts" / "prepare_demucs.py")])


def install_vocal() -> None:
    python = _bootstrap_python()
    vocal_python = venv_executable(PROJECT_ROOT, VOCAL_VENV)
    if not vocal_python.is_file():
        _run([python, "-m", "venv", str(VOCAL_VENV)])
    _run([str(vocal_python), "-m", "pip", "install", "--upgrade", "pip"])
    _run(
        [
            str(vocal_python),
            "-m",
            "pip",
            "install",
            "-r",
            str(API_DIR / "requirements-vocal.lock.txt"),
        ]
    )


def prepare_vocal_models() -> None:
    _run([_vocal_python(), str(PROJECT_ROOT / "scripts" / "prepare_qwen_models.py")])


def prepare_alignment_models() -> None:
    _run(
        [_vocal_python(), str(PROJECT_ROOT / "scripts" / "prepare_ctc_alignment_model.py")]
    )


def doctor() -> None:
    checks = {
        "Python >= 3.11": sys.version_info >= (3, 11),
        "Node.js": shutil.which("node") is not None,
        "pnpm": shutil.which(os.environ.get("PNPM", "pnpm")) is not None,
        "FFmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
    }
    for name, available in checks.items():
        print(f"{'ok' if available else 'missing':>7}  {name}")
    if not all(checks.values()):
        raise TaskError("One or more base prerequisites are missing.")


TASKS: dict[str, Callable[[], None]] = {
    "install": install,
    "seed": seed,
    "dev": dev,
    "test": test,
    "e2e": e2e,
    "lint": lint,
    "build": build,
    "clean-generated": clean_generated,
    "evaluate": evaluate,
    "install-accurate": install_accurate,
    "prepare-model": prepare_model,
    "prepare-demucs": prepare_model,
    "install-vocal": install_vocal,
    "prepare-vocal-models": prepare_vocal_models,
    "prepare-alignment-models": prepare_alignment_models,
    "doctor": doctor,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASKS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        TASKS[arguments.task]()
    except (TaskError, subprocess.CalledProcessError) as error:
        print(f"BeatForge task failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
