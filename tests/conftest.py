"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The slow parity tests compare this app's output against a recording and a
# transcript produced by the one-shot script. Neither is shipped: a real recording
# is somebody's speech, and its transcript is their words. Point these at your own
# pair to run the parity suite; without them it skips rather than fails.
REFERENCE_AUDIO = Path(
    os.environ.get("MYTRANSCRIBE_REFERENCE_AUDIO", REPO_ROOT / "reference.mp3")
)
REFERENCE_TRANSCRIPT = Path(
    os.environ.get("MYTRANSCRIBE_REFERENCE_TRANSCRIPT", REPO_ROOT / "reference.txt")
)

MISSING_REFERENCE = (
    "no reference recording. Set MYTRANSCRIBE_REFERENCE_AUDIO (and "
    "MYTRANSCRIBE_REFERENCE_TRANSCRIPT for the byte-equality check) to run the "
    "parity suite against your own audio."
)


def require_reference_audio() -> Path:
    """The reference recording, or a skip explaining how to supply one."""
    if not REFERENCE_AUDIO.is_file():
        pytest.skip(MISSING_REFERENCE)
    return REFERENCE_AUDIO


def require_reference_transcript() -> Path:
    """The one-shot transcript to compare against, or a skip."""
    require_reference_audio()
    if not REFERENCE_TRANSCRIPT.is_file():
        pytest.skip(
            "no reference transcript. Set MYTRANSCRIBE_REFERENCE_TRANSCRIPT to a "
            "file produced by `./transcribe.sh` for the same recording."
        )
    return REFERENCE_TRANSCRIPT


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def reference_audio() -> Path:
    return require_reference_audio()


@pytest.fixture(scope="session")
def reference_transcript() -> Path:
    return require_reference_transcript()


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    """A scratch directory that a test may fill however it likes."""
    target = tmp_path / "work"
    target.mkdir()
    return target


def pytest_collection_modifyitems(config, items):
    """Slow tests need the model; run them only when asked."""
    if os.environ.get("MYTRANSCRIBE_RUN_SLOW") == "1":
        return
    skip_slow = pytest.mark.skip(reason="slow model tests: run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)