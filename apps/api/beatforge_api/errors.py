from __future__ import annotations

from typing import Any


class BeatForgeError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def not_found(resource: str, resource_id: str) -> BeatForgeError:
    return BeatForgeError(
        f"{resource.upper()}_NOT_FOUND",
        f"{resource} not found",
        status_code=404,
        details={"id": resource_id},
    )
