"""Configuration for the app, read from the environment.

Everything the app can be told differently at startup lives here, so there is one
place to look and one place to test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import MODEL_ID

# 3080 is taken on this machine by another service, so the app must not default
# to it (acceptance criterion 24).
DEFAULT_PORT = 8765
# Bound to loopback only: this is an unauthenticated single-user app (ADR-0003).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_RETENTION_SECONDS = 3600.0


@dataclass
class AppConfig:
    """What the app needs to know to start."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    threads: int = field(default_factory=lambda: os.cpu_count() or 1)
    # Fixed by the spec (Out of Scope: "PixelML/sea-asr-id is fixed"), so it is not
    # configurable. A second model would mean a second language, which this app
    # does not offer; keeping the field records what produced each Transcript.
    model_id: str = MODEL_ID
    retention_seconds: float = DEFAULT_RETENTION_SECONDS
    work_root: Path = field(default_factory=lambda: Path(".jobs"))
    # A model that cannot load stops the server, rather than failing later.
    require_model: bool = True

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> AppConfig:
        values = os.environ if environ is None else environ
        port = values.get("MYTRANSCRIBE_PORT") or values.get("PORT")
        threads = values.get("MYTRANSCRIBE_THREADS")
        retention = values.get("MYTRANSCRIBE_RETENTION_SECONDS")
        return cls(
            host=values.get("MYTRANSCRIBE_HOST", DEFAULT_HOST),
            port=int(port) if port else DEFAULT_PORT,
            threads=int(threads) if threads else (os.cpu_count() or 1),
            model_id=MODEL_ID,
            retention_seconds=float(retention) if retention else DEFAULT_RETENTION_SECONDS,
            work_root=Path(values.get("MYTRANSCRIBE_WORK_ROOT", ".jobs")),
            require_model=values.get("MYTRANSCRIBE_REQUIRE_MODEL", "1") != "0",
        )