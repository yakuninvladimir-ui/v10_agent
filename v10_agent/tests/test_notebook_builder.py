"""Unit tests for build_notebook_v10 and Kaggle notebook validation."""

from __future__ import annotations

import base64
import io
import json
import pathlib
import zipfile
import pytest

from build_notebook_v10 import generate_notebook, MAX_NOTEBOOK_BYTES


def test_notebook_builder_generates_valid_notebook():
    nb_path = generate_notebook()
    assert nb_path.is_file(), f"Notebook not found at {nb_path}"

    content = nb_path.read_text(encoding="utf-8")
    size_bytes = len(content.encode("utf-8"))

    # Invariant: Must be strictly under 985,000 bytes
    assert size_bytes < MAX_NOTEBOOK_BYTES, f"Notebook size {size_bytes} exceeds {MAX_NOTEBOOK_BYTES}"

    nb = json.loads(content)
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 5

    # Check that payload unpacking cell contains valid base64 LZMA archive
    unpack_cell = None
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if "PAYLOAD_B64" in source:
            unpack_cell = source
            break

    assert unpack_cell is not None, "Payload cell missing from notebook"

    # Extract base64 string and verify archive contents
    first_q = unpack_cell.find("PAYLOAD_B64 = '") + len("PAYLOAD_B64 = '")
    last_q = unpack_cell.find("'", first_q)
    b64_str = unpack_cell[first_q:last_q]

    zip_bytes = base64.b64decode(b64_str)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert "kaggle_agent.py" in namelist
        assert "submission.py" in namelist
        assert "lcld_competition_child.py" in namelist
        assert "lcld_preflight.py" in namelist
        assert "v10_agent/__init__.py" in namelist
        assert "v10_agent/session.py" in namelist
        assert "v10_agent/brusentsov_logic.py" in namelist
