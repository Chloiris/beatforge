from pathlib import Path

from scripts import beatforge


def test_project_environment_loads_simple_dotenv_without_overriding_process(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(beatforge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("BEATFORGE_API_PORT", "9000")
    monkeypatch.delenv("BEATFORGE_WEB_PORT", raising=False)
    monkeypatch.delenv("BEATFORGE_HOST", raising=False)
    (tmp_path / ".env").write_text(
        "BEATFORGE_API_PORT=8100\n"
        "export BEATFORGE_WEB_PORT='5200'\n"
        "BEATFORGE_HOST=127.0.0.1 # local only\n",
        encoding="utf-8",
    )

    beatforge._load_project_environment()

    assert beatforge.os.environ["BEATFORGE_API_PORT"] == "9000"
    assert beatforge.os.environ["BEATFORGE_WEB_PORT"] == "5200"
    assert beatforge.os.environ["BEATFORGE_HOST"] == "127.0.0.1"


def test_web_proxy_follows_api_port_and_preserves_an_explicit_target(monkeypatch) -> None:
    monkeypatch.delenv("VITE_API_PROXY", raising=False)
    assert beatforge._web_dev_environment("0.0.0.0", "8123")["VITE_API_PROXY"] == (
        "http://127.0.0.1:8123"
    )

    monkeypatch.setenv("VITE_API_PROXY", "http://127.0.0.1:8999")
    assert beatforge._web_dev_environment("127.0.0.1", "8123")["VITE_API_PROXY"] == (
        "http://127.0.0.1:8999"
    )


def test_test_task_generates_public_audio_fixtures_before_pytest(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(beatforge, "_base_python", lambda: "python-from-venv")
    monkeypatch.setattr(beatforge, "_pnpm", lambda: "pnpm")
    monkeypatch.setattr(
        beatforge,
        "_run",
        lambda command, **_kwargs: commands.append(list(command)),
    )

    beatforge.test()

    assert commands[0] == [
        "python-from-venv",
        str(beatforge.PROJECT_ROOT / "scripts" / "generate_demo_audio.py"),
    ]
    assert commands[1] == ["python-from-venv", "-m", "pytest"]
    assert commands[2][:4] == ["pnpm", "--dir", str(beatforge.WEB_DIR), "test"]
