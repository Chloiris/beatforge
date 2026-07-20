from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_storage = Path(tempfile.mkdtemp(prefix="beatforge-api-tests-"))
os.environ["BEATFORGE_STORAGE_DIR"] = str(_storage)
os.environ["BEATFORGE_DATABASE_URL"] = f"sqlite:///{_storage / 'tests.db'}"

from beatforge_api.database import Base, engine  # noqa: E402
from beatforge_api.main import app  # noqa: E402


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        yield test_client
