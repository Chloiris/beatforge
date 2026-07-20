from __future__ import annotations

import os
from pathlib import Path


def venv_executable(
    project_root: str | Path,
    environment: str | Path,
    executable: str = "python",
) -> Path:
    """Return a virtual-environment executable on Windows, macOS, or Linux."""

    root = Path(project_root).expanduser()
    environment_path = Path(environment).expanduser()
    if not environment_path.is_absolute():
        environment_path = root / environment_path

    if os.name == "nt":
        executable_name = executable if Path(executable).suffix else f"{executable}.exe"
        return environment_path / "Scripts" / executable_name
    return environment_path / "bin" / executable
