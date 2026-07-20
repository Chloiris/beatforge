from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()
_settings.ensure_directories()
_connect_args = {"check_same_thread": False, "timeout": 30} if _settings.database_url.startswith(
    "sqlite"
) else {}
engine = create_engine(_settings.database_url, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@event.listens_for(Engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
    if _settings.database_url.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def init_db() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    # The local MVP intentionally avoids a migration service, but existing user
    # databases must still gain additive source-aware columns without data loss.
    if engine.dialect.name == "sqlite":
        hit_columns = {column["name"] for column in inspect(engine).get_columns("hit_points")}
        track_columns = {column["name"] for column in inspect(engine).get_columns("tracks")}
        vocal_job_columns = {
            column["name"]
            for column in inspect(engine).get_columns("vocal_alignment_jobs")
        }
        candidate_columns = {
            column["name"] for column in inspect(engine).get_columns("candidate_events")
        }
        with engine.begin() as connection:
            if "primary_stem" not in hit_columns:
                connection.execute(
                    text(
                        "ALTER TABLE hit_points ADD COLUMN primary_stem "
                        "VARCHAR(24) NOT NULL DEFAULT 'mix'"
                    )
                )
            if "stem_evidence_json" not in hit_columns:
                connection.execute(
                    text(
                        "ALTER TABLE hit_points ADD COLUMN stem_evidence_json "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )
                )
            if "acoustic_sample" not in hit_columns:
                connection.execute(
                    text("ALTER TABLE hit_points ADD COLUMN acoustic_sample INTEGER")
                )
            if "chart_sample" not in hit_columns:
                connection.execute(
                    text("ALTER TABLE hit_points ADD COLUMN chart_sample INTEGER")
                )
            # Repair nulls on every startup as well as during the first migration.
            # A pre-upgrade desktop API process can remain alive briefly after the
            # schema changes and insert legacy rows that do not populate new fields.
            connection.execute(
                text(
                    "UPDATE hit_points SET acoustic_sample = CASE "
                    "WHEN manually_edited = 1 OR locked = 1 THEN sample "
                    "ELSE refined_sample END WHERE acoustic_sample IS NULL"
                )
            )
            connection.execute(
                text(
                    "UPDATE hit_points SET chart_sample = snapped_sample "
                    "WHERE chart_sample IS NULL"
                )
            )
            if "lyrics_text" not in track_columns:
                connection.execute(
                    text("ALTER TABLE tracks ADD COLUMN lyrics_text TEXT NOT NULL DEFAULT ''")
                )
            if "lyrics_format" not in track_columns:
                connection.execute(
                    text(
                        "ALTER TABLE tracks ADD COLUMN lyrics_format "
                        "VARCHAR(24) NOT NULL DEFAULT 'japanese'"
                    )
                )
            if "vocal_alignment_json" not in track_columns:
                connection.execute(
                    text(
                        "ALTER TABLE tracks ADD COLUMN vocal_alignment_json "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )
                )
            if "replace_vocal_hits" not in vocal_job_columns:
                connection.execute(
                    text(
                        "ALTER TABLE vocal_alignment_jobs ADD COLUMN replace_vocal_hits "
                        "BOOLEAN NOT NULL DEFAULT 1"
                    )
                )
            candidate_additions = {
                "source": "VARCHAR(32) NOT NULL DEFAULT 'mix'",
                "generator": "VARCHAR(32) NOT NULL DEFAULT 'analysis'",
                "character": "TEXT",
                "mora": "TEXT",
                "phoneme": "TEXT",
                "event_level": "VARCHAR(24) NOT NULL DEFAULT 'analysis'",
                "event_policy": "VARCHAR(48)",
                "alignment_unit_id": "VARCHAR(160)",
                "alignment_unit_index": "INTEGER",
                "alignment_run_id": "VARCHAR(36)",
                "character_indices_json": "TEXT NOT NULL DEFAULT '[]'",
                "phonemes_json": "TEXT NOT NULL DEFAULT '[]'",
                "aligned_sample": "INTEGER",
                "refined_sample": "INTEGER",
                "evidence_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for name, definition in candidate_additions.items():
                if name not in candidate_columns:
                    connection.execute(
                        text(f"ALTER TABLE candidate_events ADD COLUMN {name} {definition}")
                    )
            connection.execute(
                text(
                    "UPDATE candidate_events SET event_level = 'character', "
                    "event_policy = 'character' "
                    "WHERE generator = 'hubert_ctc' AND source = 'vocals' "
                    "AND event_policy IS NULL"
                )
            )
            # Legacy candidates had only acoustic_sample. Preserve that real
            # measured location as both provenance samples during migration.
            connection.execute(
                text(
                    "UPDATE candidate_events SET aligned_sample = acoustic_sample "
                    "WHERE aligned_sample IS NULL"
                )
            )
            connection.execute(
                text(
                    "UPDATE candidate_events SET refined_sample = acoustic_sample "
                    "WHERE refined_sample IS NULL"
                )
            )


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
