from __future__ import annotations

from pathlib import Path

import fitz
from shapely.geometry import LineString

from app import build_stage1_box_geometry, validate_stage1_box_geometry


PDF_PATH = Path(__file__).resolve().parents[1] / "20182462_POWER_TRUNK 3P_USA_50x120x180mm.pdf"


def test_stage1_geometry_for_sample_pdf() -> None:
    assert PDF_PATH.exists(), f"Missing sample PDF: {PDF_PATH}"

    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)

    validate_stage1_box_geometry(box)

    body = box["body_panels"]
    assert len(body) == 5

    widths = [panel["bbox_mm"][2] for panel in body]
    heights = [panel["bbox_mm"][3] for panel in body]
    body_heights = [panel["bbox_mm"][3] for panel in body if panel["id"] != "glue"]

    expected_widths = [50.0, 120.0, 50.0, 120.0, 15.0]
    assert all(abs(width - expected) / expected <= 0.02 for width, expected in zip(widths, expected_widths))
    assert any(abs(width - round(width)) > 1e-6 for width in widths), widths

    assert max(body_heights) < 190.0 and min(body_heights) > 150.0
    assert abs(box["D_mm"] - 50.0) < 2.0
    assert abs(box["W_mm"] - 120.0) < 2.0
    assert abs(box["H_mm"] - 180.0) < 2.0

    flaps = box["flaps"]
    assert len(flaps) >= 8
    flap_heights = [max(0.0, flap["polygon_mm"].bounds[3] - flap["polygon_mm"].bounds[1]) for flap in flaps]
    assert max(flap_heights) - min(flap_heights) > 10.0, flap_heights
    # Assert heights are not all multiples of 12 (proving they come from real die lines, not synthesis)
    assert not all(abs(h % 12.0) < 1e-4 or abs(h % 12.0 - 12.0) < 1e-4 for h in flap_heights)

    raw_dump = box.get("raw_dump", [])
    assert len(raw_dump) >= 2
    for row in raw_dump:
        assert row["segments"] > 0
        assert row["bbox_pt"][2] > row["bbox_pt"][0]
        assert row["bbox_pt"][3] > row["bbox_pt"][1]

    lines_to_check = box["red_cut_segments"] + box["green_crease_segments"]
    assert lines_to_check

    panel_map = {panel["id"]: panel for panel in body + flaps}
    for flap in flaps:
        assert flap["parent"] in panel_map
        assert flap["hinge_p0"] is not None and flap["hinge_p1"] is not None
        parent_poly = panel_map[flap["parent"]]["polygon_mm"]
        assert flap["polygon_mm"].touches(parent_poly) or flap["polygon_mm"].intersects(parent_poly)
        crease = LineString([flap["hinge_p0"], flap["hinge_p1"]])
        shared = flap["polygon_mm"].boundary.intersection(crease.buffer(0.5)).length
        assert shared >= 5.0, f"Flap {flap['id']} shares only {shared}mm with parent crease"

    flap_areas = [flap["polygon_mm"].area for flap in flaps]
    assert min(flap_areas) > 50.0, flap_areas
