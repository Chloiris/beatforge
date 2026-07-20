from __future__ import annotations

import os
from pathlib import Path

from beatforge_api.config import get_settings
from beatforge_api.platform_paths import venv_executable


def test_venv_executable_uses_native_environment_layout(tmp_path: Path) -> None:
    executable = venv_executable(tmp_path, ".venv-qwen")

    if os.name == "nt":
        assert executable == tmp_path / ".venv-qwen" / "Scripts" / "python.exe"
    else:
        assert executable == tmp_path / ".venv-qwen" / "bin" / "python"


def test_venv_executable_accepts_absolute_environment_path(tmp_path: Path) -> None:
    environment = tmp_path / "runtime"

    executable = venv_executable(tmp_path / "project", environment, "mfa")

    directory = "Scripts" if os.name == "nt" else "bin"
    name = "mfa.exe" if os.name == "nt" else "mfa"
    assert executable == environment / directory / name


def test_project_root_can_be_overridden_for_installed_and_container_runs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BEATFORGE_PROJECT_ROOT", str(tmp_path))

    settings = get_settings()

    assert settings.project_root == tmp_path.resolve()
