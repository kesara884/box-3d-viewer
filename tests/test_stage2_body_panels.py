from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np

from app import (
    build_stage1_box_geometry,
    build_stage2_body_panel_model,
    compute_panel_outward_normals,
    compute_stage2_numpy_box_bounds,
)

PDF_PATH = Path(__file__).resolve().parents[1] / "20182462_POWER_TRUNK 3P_USA_50x120x180mm.pdf"


def test_stage2_body_panel_fold_tree_matches_mm_box() -> None:
    assert PDF_PATH.exists(), f"Missing sample PDF: {PDF_PATH}"

    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    model = build_stage2_body_panel_model(box)

    assert [panel["id"] for panel in model["panels"]] == ["front", "side_a", "side_b", "back", "glue"]
    assert model["root"] == "front"

    np_box = compute_stage2_numpy_box_bounds(model, fold_progress=1.0, flip=False)
    size = np.array(np_box["size_mm"], dtype=float)
    assert size[0] > 40.0 and size[0] < 60.0
    assert size[1] > 110.0 and size[1] < 130.0
    assert size[2] > 160.0 and size[2] < 190.0

    flat = compute_stage2_numpy_box_bounds(model, fold_progress=0.0, flip=False)
    flat_size = np.array(flat["size_mm"], dtype=float)
    assert flat_size[0] == 0.0
    assert flat_size[1] > 120.0 and flat_size[2] > 150.0

    js_payload = json.dumps(model)
    assert 'scene_scale' in js_payload
    assert 'panels' in js_payload
    assert 'flaps' in js_payload

    print("numpy box size at foldProgress=1:", size)


def test_stage3_flaps_and_viewer_bounds() -> None:
    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    model = build_stage2_body_panel_model(box)

    flaps = model["flaps"]
    assert len(flaps) == 8
    flap_ids = [f["id"] for f in flaps]
    expected_ids = [
        "side_a_top", "side_a_bottom",
        "front_top", "top_tuck", "front_bottom",
        "side_b_top", "side_b_bottom",
        "back_bottom"
    ]
    assert flap_ids == expected_ids

    # Parent hierarchy checks
    flap_by_id = {f["id"]: f for f in flaps}
    assert flap_by_id["top_tuck"]["parent"] == "front_top"
    assert flap_by_id["front_top"]["parent"] == "front"
    assert flap_by_id["side_a_top"]["parent"] == "side_a"
    assert flap_by_id["side_b_top"]["parent"] == "side_b"
    assert flap_by_id["back_bottom"]["parent"] == "back"

    # Progress 0.0: flat sheet
    flat = compute_stage2_numpy_box_bounds(model, fold_progress=0.0)
    flat_size = flat["size_mm"]
    assert flat_size[0] == 0.0  # depth
    assert abs(flat_size[1] - 355.0) < 1.0  # width
    assert abs(flat_size[2] - 286.8) < 1.0  # height

    # Progress 0.5: walls folded into tube, flaps open
    tube = compute_stage2_numpy_box_bounds(model, fold_progress=0.5)
    tube_size = tube["size_mm"]
    assert abs(tube_size[0] - 50.42) < 1.0  # depth
    assert abs(tube_size[1] - 120.66) < 1.0  # width
    assert abs(tube_size[2] - 286.81) < 1.0  # height

    # Progress 1.0: fully closed carton
    closed = compute_stage2_numpy_box_bounds(model, fold_progress=1.0)
    closed_size = closed["size_mm"]
    assert abs(closed_size[0] - 50.42) < 1.0  # depth ~ 50 mm
    assert abs(closed_size[1] - 120.66) < 1.0  # width ~ 120 mm
    assert abs(closed_size[2] - 180.00) < 1.0  # height ~ 180 mm

    # Flipped progress 1.0
    closed_flipped = compute_stage2_numpy_box_bounds(model, fold_progress=1.0, flip=True)
    flip_size = closed_flipped["size_mm"]
    assert abs(flip_size[0] - 50.42) < 1.0
    assert abs(flip_size[1] - 120.66) < 1.0
    assert abs(flip_size[2] - 180.00) < 1.0


def test_stage4_artwork_textures() -> None:
    from app import make_viewer_html

    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    model = build_stage2_body_panel_model(box, page=page, dpi=200)

    # Verify each body panel has a texture
    assert len(model["panels"]) == 5
    for p in model["panels"]:
        tex = p.get("texture", "")
        assert tex.startswith("data:image/jpeg;base64,"), f"Panel {p['id']} missing valid texture"

    # Verify each flap has a texture
    assert len(model["flaps"]) == 8
    for f in model["flaps"]:
        tex = f.get("texture", "")
        assert tex.startswith("data:image/jpeg;base64,"), f"Flap {f['id']} missing valid texture"

    # Total texture size check (must be well under 15 MB)
    total_tex_bytes = sum(len(p.get("texture", "")) for p in model["panels"]) + sum(len(f.get("texture", "")) for f in model["flaps"])
    total_mb = total_tex_bytes / (1024 * 1024)
    print(f"Total base64 texture payload size: {total_mb:.2f} MB")
    assert total_mb < 5.0, f"Texture payload too large: {total_mb:.2f} MB"

    # Test viewer HTML generation
    html = make_viewer_html(model)
    assert "normalizeUVs" in html
    assert "TextureLoader" in html
    assert "THREE.FrontSide" in html
    assert "THREE.BackSide" in html
    assert "GLTFExporter" in html
    assert "polygonOffset" in html


def test_stage4_panel_outward_normals_and_bbox() -> None:
    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    model = build_stage2_body_panel_model(box, page=page, dpi=200)

    # Compute outward normals at fold progress 1.0 with flip=False
    normals_data = compute_panel_outward_normals(model, fold_progress=1.0, flip=False)

    expected_panels = ["front", "side_a", "side_b", "back", "glue"]
    assert list(normals_data.keys()) == expected_panels

    for pid in expected_panels:
        pdata = normals_data[pid]
        dot = pdata["dot"]
        print(f"Panel '{pid}': normal={pdata['normal']}, dot={dot:.3f}")
        assert dot > 0.0, f"Panel {pid} printed face does not point outward (dot = {dot})"

    # Assert closed bounding box is still ~ 50 x 120 x 180 mm
    closed = compute_stage2_numpy_box_bounds(model, fold_progress=1.0, flip=False)
    closed_size = closed["size_mm"]
    assert abs(closed_size[0] - 50.42) < 1.0  # depth ~ 50 mm
    assert abs(closed_size[1] - 120.66) < 1.0  # width ~ 120 mm
    assert abs(closed_size[2] - 180.00) < 1.0  # height ~ 180 mm


def test_stage4_flap_closing_bounds_and_tuck() -> None:
    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    model = build_stage2_body_panel_model(box, page=page, dpi=200)

    panels = model["panels"]
    flaps = model["flaps"]
    panel_by_id = {p["id"]: p for p in panels}
    flap_by_id = {f["id"]: f for f in flaps}

    def remap_p(p: float, s: float, e: float) -> float:
        return float(np.clip((p - s) / (e - s), 0.0, 1.0))

    p_wall = remap_p(1.0, 0.0, 0.5)
    p_dust = remap_p(1.0, 0.5, 0.7)
    p_tuck = remap_p(1.0, 0.7, 1.0)

    def hinge_position(pid: str) -> np.ndarray:
        pr = panel_by_id[pid]["parent"]
        if not pr:
            return np.zeros(3, dtype=float)
        parent = panel_by_id[pr]
        if pr == "front" and pid == "side_a":
            return np.zeros(3, dtype=float)
        return np.array([float(parent["x1"] - parent["x0"]), 0.0, 0.0], dtype=float)

    def panel_T(pid: str) -> np.ndarray:
        p = panel_by_id[pid]
        if not p["parent"]:
            return np.eye(4, dtype=float)
        pT = panel_T(p["parent"])
        h = hinge_position(pid)
        sign = -1.0 if pid == "side_a" else 1.0
        th = sign * p_wall * (np.pi / 2.0)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)
        Tr = np.array([[1, 0, 0, h[0]], [0, 1, 0, h[1]], [0, 0, 1, h[2]], [0, 0, 0, 1]], dtype=float)
        return pT @ Tr @ R

    def flap_T(fid: str) -> np.ndarray:
        f = flap_by_id[fid]
        pr = f["parent"]
        pT = panel_T(pr) if pr in panel_by_id else flap_T(pr)
        piv = f["pivot_position"]
        sign = float(f.get("sign", 1.0))
        sub_p = remap_p(1.0, 0.7, 0.95) if fid == "top_tuck" else (p_dust if f.get("kind") == "dust" else p_tuck)
        th = sign * sub_p * (np.pi / 2.0)
        c, s = np.cos(th), np.sin(th)
        Rx = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)
        Tr = np.array([[1, 0, 0, piv[0]], [0, 1, 0, piv[1]], [0, 0, 1, piv[2]], [0, 0, 0, 1]], dtype=float)
        return pT @ Tr @ Rx

    # Nominal box bbox in mm:
    box_x_min, box_x_max = 0.0, 119.61
    box_y_min, box_y_max = 0.0, 180.0
    box_z_min, box_z_max = -50.42, 0.0

    for f in flaps:
        fid = f["id"]
        poly = f.get("polygon", [])
        local_z = float(f.get("z_offset", 0.0))
        lpts = np.array([[pt[0], pt[1], local_z, 1.0] for pt in poly], dtype=float)
        w = (flap_T(fid) @ lpts.T).T[:, :3]

        dev_x_min = max(0.0, box_x_min - float(w[:, 0].min()))
        dev_x_max = max(0.0, float(w[:, 0].max()) - box_x_max)
        dev_y_min = max(0.0, box_y_min - float(w[:, 1].min()))
        dev_y_max = max(0.0, float(w[:, 1].max()) - box_y_max)
        dev_z_min = max(0.0, box_z_min - float(w[:, 2].min()))
        dev_z_max = max(0.0, float(w[:, 2].max()) - box_z_max)

        max_dev = max(dev_x_min, dev_x_max, dev_y_min, dev_y_max, dev_z_min, dev_z_max)
        print(f"Flap '{fid}': max_dev_outside_bbox = {max_dev:.3f} mm")
        assert max_dev <= 1.0, f"Flap {fid} vertex exceeds box bbox by {max_dev:.3f} mm (> 1.0 mm)"

    # Assert top_tuck lies inside the box depth (z strictly between front and back walls)
    tuck_lpts = np.array([[pt[0], pt[1], float(flap_by_id["top_tuck"].get("z_offset", 0.0)), 1.0] for pt in flap_by_id["top_tuck"]["polygon"]], dtype=float)
    tuck_w = (flap_T("top_tuck") @ tuck_lpts.T).T[:, :3]
    tuck_z_min = float(tuck_w[:, 2].min())
    tuck_z_max = float(tuck_w[:, 2].max())
    print(f"top_tuck z range: [{tuck_z_min:.2f}, {tuck_z_max:.2f}] mm (depth: [{box_z_min}, {box_z_max}])")
    assert tuck_z_min > box_z_min, f"top_tuck pokes out past back wall: {tuck_z_min} <= {box_z_min}"
    assert tuck_z_max < box_z_max, f"top_tuck pokes out past front wall: {tuck_z_max} >= {box_z_max}"

    # Overall closed bounds check:
    closed = compute_stage2_numpy_box_bounds(model, fold_progress=1.0)
    closed_size = closed["size_mm"]
    assert abs(closed_size[0] - 50.42) < 1.0
    assert abs(closed_size[1] - 120.66) < 1.0
    assert abs(closed_size[2] - 180.00) < 1.0




