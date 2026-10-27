"""t13 — the page: upload, watch, cancel, download.

The page is served by the same process that transcribes, as plain files. There is
no build step and no dev server, so "working on it" and "using it" are one command.

These tests check the served assets, not a browser rendering: that the page is
self-contained, references only the app's own endpoints, states the things the
spec requires it to state, and is operable by keyboard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig


@pytest.fixture()
def client(work_dir: Path):
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=FakeTranscriber(), model_loaded=True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def page(client) -> str:
    response = client.get("/")
    assert response.status_code == 200
    return response.text


@pytest.fixture()
def script(client) -> str:
    response = client.get("/app.js")
    assert response.status_code == 200
    return response.text


@pytest.fixture()
def styles(client) -> str:
    response = client.get("/app.css")
    assert response.status_code == 200
    return response.text


# --------------------------------------------------------------- no build step

def test_the_page_is_served_as_html(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_the_page_loads_only_its_own_assets(page: str) -> None:
    """No CDN, no bundler: the page must work with the network unplugged."""
    references = re.findall(r'(?:src|href)="([^"]+)"', page)

    assert references, "the page references nothing at all"
    for reference in references:
        assert reference.startswith("/"), f"external asset would need the network: {reference}"
        assert "//" not in reference[2:], f"absolute URL in the page: {reference}"


def test_every_asset_the_page_asks_for_is_actually_served(client, page: str) -> None:
    for reference in re.findall(r'(?:src|href)="([^"]+)"', page):
        response = client.get(reference)
        assert response.status_code == 200, f"{reference} -> {response.status_code}"


def test_the_script_uses_no_module_loading_that_needs_building(script: str) -> None:
    assert "import " not in script.replace("import(", "")
    assert "require(" not in script


def test_the_page_and_script_agree_on_the_endpoints(client, script: str) -> None:
    """Every endpoint the script calls must exist on the server."""
    # Whole string literals, including any `${...}` interpolation inside them.
    # Any /api/... path the script mentions, inside a template or plain string.
    literals = re.findall(r"/api/[A-Za-z0-9_/${}.-]*", script)
    endpoints = {route.path for route in client.app.routes if hasattr(route, "path")}

    assert literals, "the script never calls the API"
    for path in literals:
        normalised = re.sub(r"\$\{[^}]*\}", "{job_id}", path)
        normalised = re.sub(r"\{[^}]*\}", "{job_id}", normalised)
        assert any(
            re.sub(r"\{[^}]*\}", "{job_id}", endpoint) == normalised for endpoint in endpoints
        ), f"the page calls {path}, which the server does not serve"


# ------------------------------------------------------------ stated honestly

def test_the_page_says_the_model_is_indonesian_only(page: str) -> None:
    assert re.search(r"Bahasa Indonesia", page)
    assert re.search(r"saja|only", page, re.IGNORECASE)


def test_the_page_offers_no_language_picker(page: str) -> None:
    """One model, one language: a picker would be a lie."""
    assert not re.search(r"<select", page)
    assert not re.search(r'type="radio"[^>]*name="lang', page)
    assert "<option" not in page


def test_the_page_offers_the_four_formats(page: str) -> None:
    for output_format in ("txt", "srt", "vtt", "json"):
        assert f'value="{output_format}"' in page


def test_txt_is_preselected_in_the_page(page: str) -> None:
    match = re.search(r'<input[^>]*value="txt"[^>]*>', page)
    assert match and "checked" in match.group(0)


def test_the_page_states_the_cancel_wait(page: str) -> None:
    assert "30 detik" in page or "30 detik" in page.lower()


def test_the_page_states_retention_in_plain_language(page: str, script: str) -> None:
    combined = page + script
    assert re.search(r"kedaluwarsa|expired", combined)
    assert re.search(r"60", combined)


# ------------------------------------------------------------------ keyboard

def test_the_upload_control_is_a_labelled_file_input(page: str) -> None:
    assert re.search(r'<label[^>]*for="file"', page)
    assert re.search(r'<input[^>]*id="file"[^>]*type="file"', page)


def test_the_submit_control_is_a_real_button(page: str) -> None:
    """A clickable div is not reachable by keyboard; a button is."""
    assert re.search(r'<button[^>]*type="submit"', page)
    assert "onclick" not in page, "inline handlers hide the keyboard contract"


def test_the_format_choices_are_real_radio_inputs(page: str) -> None:
    radios = re.findall(r'<input[^>]*type="radio"[^>]*name="format"', page)
    assert len(radios) == 4
    # Each is inside a label, so the whole row is a click target and is announced.
    assert len(re.findall(r'<label><input[^>]*type="radio"', page)) == 4


def test_the_page_has_no_positive_tabindex(page: str) -> None:
    """Positive tabindex breaks the document order keyboard users rely on."""
    values = re.findall(r'tabindex="(-?\d+)"', page)
    assert not [value for value in values if int(value) > 0]


def test_the_download_is_a_link_not_script_only(page: str, script: str) -> None:
    """A link can be reached and activated from the keyboard, and it shows a URL."""
    assert re.search(r'<a[^>]*class="download"', page)
    assert ".href = " in script
    assert "window.location" not in script.split(".href = ")[0][-200:]


def test_live_regions_announce_changes(page: str) -> None:
    """Progress that changes silently is progress a screen reader never hears."""
    assert re.search(r'aria-live="polite"', page)
    assert re.search(r'role="alert"', page)


def test_the_progress_bar_is_a_progressbar(page: str) -> None:
    assert re.search(r'role="progressbar"', page)


# -------------------------------------------------------------------- layout

def test_the_page_declares_a_laptop_viewport(page: str) -> None:
    assert re.search(r'<meta[^>]*name="viewport"', page)


def test_the_stylesheet_avoids_a_fixed_width_that_would_scroll(styles: str) -> None:
    """A fixed pixel width wider than a laptop viewport forces horizontal scroll."""
    widths = [int(value) for value in re.findall(r"width:\s*(\d{3,})px", styles)]
    assert not [value for value in widths if value > 900], widths
    # And content must be allowed to shrink.
    assert "max-width" in styles


def test_the_stylesheet_states_a_line_length_limit(styles: str) -> None:
    """Readable measure: a transcript page of full-width text is unreadable."""
    assert re.search(r"max-width:\s*\d+rem", styles)


def test_the_stylesheet_respects_reduced_motion(styles: str) -> None:
    assert "prefers-reduced-motion" in styles


def test_the_stylesheet_supports_dark_mode(styles: str) -> None:
    assert "prefers-color-scheme" in styles


# --------------------------------------------------------------- state labels

def test_every_state_has_a_plain_language_label(script: str) -> None:
    """Each state the server can report needs words a person understands."""
    for state in ("queued", "running", "finished", "cancelled", "failed"):
        assert f"{state}:" in script, f"no label for {state}"
    assert "menunggu" in script
    assert "selesai" in script


def test_the_script_never_invents_progress_locally(script: str) -> None:
    """Progress is the server's to report; the page must not estimate a bar width."""
    assert "Math.random" not in script
    assert "progress" in script