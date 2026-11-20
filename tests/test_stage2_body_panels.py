from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np

from app import (
    build_stage1_box_geometry,
    build_stage2_body_panel_model,
    compute_flap_outward_normals,
    compute_panel_outward_normals,
    compute_stage2_numpy_box_bounds,
    detect_layout_rows,
    extract_structural_lines,
    measure_pdf_nominal_dims,
)

PDF_PATH = Path(__file__).resolve().parents[1] / "20182462_POWER_TRUNK 3P_USA_50x120x180mm.pdf"
FITNESS_PDF_PATH = Path(__file__).resolve().parents[1] / "FITNESS TUBE BOX_V7.pdf"
TEA1_PDF_PATH = Path(__file__).resolve().parents[1] / "BRITISH TEA - 20TB EARL GREY_EDITED_SA_V3.pdf"
TEA2_PDF_PATH = Path(__file__).resolve().parents[1] / "Nordqvist_Moomin_20TB_37465_AFRICA_ Rooibos Tea.pdf"


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
    assert abs(flat_size[1] - 355.0) < 2.0  # width (~353.5mm from measured linework)
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


def test_stage4_flap_fold_direction_and_normals() -> None:
    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    model = build_stage2_body_panel_model(box, page=page, dpi=200)

    # 1. Flip OFF (default closed carton): all flap printed faces point outward (dot > 0)
    data = compute_flap_outward_normals(model, fold_progress=1.0, flip=False)
    expected_ids = [
        "side_a_top", "side_a_bottom",
        "front_top", "top_tuck", "front_bottom",
        "side_b_top", "side_b_bottom",
        "back_bottom"
    ]
    assert list(data.keys()) == expected_ids

    print("\n--- FLAP DIRECTION TABLE (Flip OFF) ---")
    print(f"{'Flap ID':15s} | {'Sign':4s} | {'Axis':7s} | {'Dot':6s} | {'p=0.5':7s} | {'p=0.7':7s} | {'In/Out'}")
    print("-" * 75)
    for fid, f in data.items():
        print(f"{fid:15s} | {f['sign']:4.1f} | {f['axis']:7s} | {f['dot']:6.2f} | {f['movement_at_p05']:7s} | {f['movement_at_p07']:7s} | {f['movement_vs_p0']}")
        # All flaps must point outward (dot > 0)
        assert f["dot"] > 0.0, f"Flap {fid} dot product {f['dot']} <= 0 (not pointing outward)"
        # All flaps must fold inward toward the box center axis
        assert f["movement_vs_p0"] == "inward", f"Flap {fid} did not fold inward (moved outward)"

    # 2. Flip ON: all flap printed faces point inward (dot < 0)
    flip_data = compute_flap_outward_normals(model, fold_progress=1.0, flip=True)
    for fid, f in flip_data.items():
        assert f["dot"] < 0.0, f"Flip ON flap {fid} dot product {f['dot']} >= 0 (must point inward)"

    # 3. Closed bbox check
    closed = compute_stage2_numpy_box_bounds(model, fold_progress=1.0)
    closed_size = closed["size_mm"]
    assert abs(closed_size[0] - 50.42) < 1.0
    assert abs(closed_size[1] - 120.66) < 1.0
    assert abs(closed_size[2] - 180.00) < 1.0


def test_stage4_glue_panel_inside_side_a() -> None:
    doc = fitz.open(str(PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    model = build_stage2_body_panel_model(box, page=page, dpi=200)

    # 1. At progress=1 compute the world position of the glue panel (all 4 corners) and side_a
    normals_data = compute_panel_outward_normals(model, fold_progress=1.0, flip=False)
    glue_data = normals_data["glue"]
    side_a_data = normals_data["side_a"]

    glue_corners = np.array(glue_data["corners"], dtype=float)
    side_a_corners = np.array(side_a_data["corners"], dtype=float)
    box_center = np.array(glue_data["box_center"], dtype=float)

    # a) glue lies INSIDE the box bbox in X, i.e. its x range is strictly between side_a's outer plane and the box centre
    outer_plane_x = float(side_a_corners[:, 0].max())
    print(f"\nglue X range: [{glue_corners[:, 0].min():.2f}, {glue_corners[:, 0].max():.2f}], outer_plane_x={outer_plane_x:.2f}, box_center_x={box_center[0]:.2f}")
    assert glue_corners[:, 0].min() > outer_plane_x, f"glue X min {glue_corners[:, 0].min()} <= {outer_plane_x}"
    assert glue_corners[:, 0].max() < box_center[0], f"glue X max {glue_corners[:, 0].max()} >= {box_center[0]}"
    assert glue_corners[:, 2].min() >= -50.42 - 1e-3 and glue_corners[:, 2].max() <= 0.0 + 1e-3

    # b) glue's centre is closer to the box centre than side_a's centre is (inside, behind side_a)
    c_glue_x = float(glue_corners[:, 0].mean())
    c_side_a_x = float(side_a_corners[:, 0].mean())
    print(f"X dist to box center: glue={abs(box_center[0] - c_glue_x):.2f}, side_a={abs(box_center[0] - c_side_a_x):.2f}")
    assert abs(box_center[0] - c_glue_x) < abs(box_center[0] - c_side_a_x)

    # c) the gap between glue and side_a's inner surface is small (<= 1 mm, tiny offset to avoid z-fighting)
    gap = abs(c_glue_x - outer_plane_x)
    print(f"gap between glue and side_a inner surface: {gap:.2f} mm")
    assert gap <= 1.0

    # d) glue's printed normal points toward the box centre or away from side_a's outer face (hidden from outside view)
    n_glue = np.array(glue_data["normal"], dtype=float)
    n_side_a = np.array(side_a_data["normal"], dtype=float)
    dot_box = float(np.dot(n_glue, box_center - glue_corners.mean(axis=0)))
    dot_side_a = float(np.dot(n_glue, -n_side_a))
    print(f"glue normal: {n_glue}, dot_box={dot_box:.2f}, dot_away_from_side_a={dot_side_a:.2f}")
    assert dot_box > 0.0, f"glue normal {n_glue} does not point toward box center"
    assert dot_side_a > 0.0, f"glue normal {n_glue} does not point away from side_a outer face"

    # Flip toggle check: with flip=True, glue still sits inside side_a
    flip_normals = compute_panel_outward_normals(model, fold_progress=1.0, flip=True)
    glue_flip = np.array(flip_normals["glue"]["corners"], dtype=float)
    side_a_flip = np.array(flip_normals["side_a"]["corners"], dtype=float)
    outer_x_flip = float(side_a_flip[:, 0].max())
    assert glue_flip[:, 0].min() > outer_x_flip


def test_stage5a_fitness_tube_box() -> None:
    assert FITNESS_PDF_PATH.exists(), f"Missing Fitness Tube PDF: {FITNESS_PDF_PATH}"

    doc = fitz.open(str(FITNESS_PDF_PATH))
    page = doc[0]

    # Verify single-row layout detection
    lines = extract_structural_lines(page, doc=doc, min_length=0)
    green = [l for l in lines if l.get("kind") == "crease"]
    red = [l for l in lines if l.get("kind") == "cut"]
    rows, reason = detect_layout_rows(green, red)
    assert rows == 1, f"Expected single-row layout for Fitness Tube, got {rows} rows ({reason})"

    # User-entered / assumed dimensions: 69 x 160 x 170 mm (Depth x Width x Height)
    dims = (69.0, 160.0, 170.0)
    box = build_stage1_box_geometry(page, doc, dimensions=dims)

    # Assert at least 4 body panels
    body_panels = box["body_panels"]
    assert len(body_panels) >= 4, f"Expected at least 4 body panels, found {len(body_panels)}"

    # Assert at least 4 flaps
    flaps = box["flaps"]
    assert len(flaps) >= 4, f"Expected at least 4 flaps, found {len(flaps)}"

    model = build_stage2_body_panel_model(box)

    # Assert outward normals positive for all body panels
    normals = compute_panel_outward_normals(model, fold_progress=1.0, flip=False)
    for pid, n_info in normals.items():
        assert n_info["dot"] > 0.0, f"Body panel {pid} normal is not positive/outward: dot={n_info['dot']}"
        assert n_info["is_outward"] is True

    # Assert bbox within 5% of the entered dimensions
    model_body = {**model, "flaps": []}
    bounds = compute_stage2_numpy_box_bounds(model_body, fold_progress=1.0)
    sz = np.array(bounds["size_mm"])
    print(f"\nFitness Tube folded body bbox: D={sz[0]:.2f}mm, W={sz[1]:.2f}mm, H={sz[2]:.2f}mm")

    assert abs(sz[0] - dims[0]) / dims[0] <= 0.05, f"Depth {sz[0]:.2f}mm deviates >5% from entered {dims[0]}mm"
    assert abs(sz[1] - dims[1]) / dims[1] <= 0.05, f"Width {sz[1]:.2f}mm deviates >5% from entered {dims[1]}mm"
    assert abs(sz[2] - dims[2]) / dims[2] <= 0.05, f"Height {sz[2]:.2f}mm deviates >5% from entered {dims[2]}mm"

    # Assert no two tuck flaps share the same parent region
    tuck_flaps = [f for f in model["flaps"] if f.get("kind") == "tuck"]
    tuck_parents = [f["parent"] for f in tuck_flaps]
    assert len(tuck_parents) == len(set(tuck_parents)), f"Duplicate tuck flap parent regions: {tuck_parents}"

    # Assert at progress=1 every flap's world bbox is within 2 mm of the box bbox [0..W, 0..H, -D..0]
    W, H, D = dims[1], dims[2], dims[0]
    panel_by_id = {p["id"]: p for p in model["panels"]}
    flap_by_id = {f["id"]: f for f in model["flaps"]}

    def remap_p(t: float, a: float, b: float) -> float:
        return float(np.clip((t - a) / (b - a), 0.0, 1.0))

    p_wall = remap_p(1.0, 0.0, 0.5)
    p_dust = remap_p(1.0, 0.5, 0.7)
    p_tuck = remap_p(1.0, 0.7, 1.0)
    p_tuck_flap = remap_p(1.0, 0.7, 0.95)

    def panel_T(pid: str) -> np.ndarray:
        p = panel_by_id[pid]
        if p["parent"] is None:
            return np.eye(4, dtype=float)
        parent_T = panel_T(p["parent"])
        piv = np.array(p.get("pivot_position", [0, 0, 0]), dtype=float).copy()
        if p.get("glue_offset", 0.0) != 0.0:
            piv[0] -= p["glue_offset"] * p_wall
        sign = float(p.get("fold_sign", 1.0))
        theta = sign * p_wall * (np.pi / 2.0)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)
        Tr = np.array([[1, 0, 0, piv[0]], [0, 1, 0, piv[1]], [0, 0, 1, piv[2]], [0, 0, 0, 1]], dtype=float)
        return parent_T @ Tr @ R

    def flap_T(fid: str) -> np.ndarray:
        f = flap_by_id[fid]
        pr = f["parent"]
        parent_T = panel_T(pr) if pr in panel_by_id else flap_T(pr)
        piv = f["pivot_position"]
        sign = float(f.get("sign", 1.0))
        sub_p = p_tuck_flap if (fid == "top_tuck" or fid.endswith("_tuck")) else (p_dust if f.get("kind") == "dust" else p_tuck)
        theta = sign * sub_p * (np.pi / 2.0)
        c, s = np.cos(theta), np.sin(theta)
        Rx = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)
        Tr = np.array([[1, 0, 0, piv[0]], [0, 1, 0, piv[1]], [0, 0, 1, piv[2]], [0, 0, 0, 1]], dtype=float)
        return parent_T @ Tr @ Rx

    for f in model["flaps"]:
        fid = f["id"]
        poly = f.get("polygon", [])
        pts = np.array([[pt[0], pt[1], 0.0, 1.0] for pt in poly], dtype=float)
        w1 = (flap_T(fid) @ pts.T).T[:, :3]

        min_x, max_x = float(w1[:, 0].min()), float(w1[:, 0].max())
        min_y, max_y = float(w1[:, 1].min()), float(w1[:, 1].max())
        min_z, max_z = float(w1[:, 2].min()), float(w1[:, 2].max())

        assert min_x >= -2.0 and max_x <= W + 2.0, f"Flap {fid} X out of bounds: [{min_x:.1f}, {max_x:.1f}]"
        assert min_y >= -2.0 and max_y <= H + 2.0, f"Flap {fid} Y out of bounds: [{min_y:.1f}, {max_y:.1f}]"
        assert min_z >= -D - 2.0 and max_z <= 2.0, f"Flap {fid} Z out of bounds: [{min_z:.1f}, {max_z:.1f}]"

    # Assert at p=1 every flap's centroid is within the box bbox (2 mm tolerance)
    flap_normals_fit = compute_flap_outward_normals(model, fold_progress=1.0)
    for fid, n_data in flap_normals_fit.items():
        c = np.array(n_data["centroid"])
        assert -2.0 <= c[0] <= W + 2.0, f"Flap {fid} centroid X out of bounds: {c[0]}"
        assert -2.0 <= c[1] <= H + 2.0, f"Flap {fid} centroid Y out of bounds: {c[1]}"
        assert -D - 2.0 <= c[2] <= 2.0, f"Flap {fid} centroid Z out of bounds: {c[2]}"

    # Assert every bottom flap's world normal is (0, +/-1, 0)
    for f in model["flaps"]:
        if "bottom" in f["id"] and f.get("kind") != "tuck":
            n = np.array(flap_normals_fit[f["id"]]["normal"])
            assert np.allclose(np.abs(n), [0.0, 1.0, 0.0], atol=1e-2), (
                f"Bottom flap {f['id']} world normal is not (0, +/-1, 0): {n}"
            )

    # Assert glue panel folds inside the box behind side_a
    glue_p = panel_by_id["glue"]
    glue_corners = np.array([
        [glue_p["x0"], 0.0, 0.0, 1.0],
        [glue_p["x1"], 0.0, 0.0, 1.0],
        [glue_p["x1"], glue_p["height"], 0.0, 1.0],
        [glue_p["x0"], glue_p["height"], 0.0, 1.0],
    ], dtype=float)
    glue_world = (panel_T("glue") @ glue_corners.T).T[:, :3]
    assert glue_world[:, 0].min() >= -0.5 and glue_world[:, 0].max() <= 1.0
    assert glue_world[:, 2].min() >= -D - 0.5 and glue_world[:, 2].max() <= 0.5


def test_stage5a_unsupported_tea_layouts() -> None:
    for tea_path in [TEA1_PDF_PATH, TEA2_PDF_PATH]:
        assert tea_path.exists(), f"Missing tea PDF: {tea_path}"
        doc = fitz.open(str(tea_path))
        page = doc[0]
        try:
            build_stage1_box_geometry(page, doc)
            raise AssertionError(f"Expected Unsupported layout error for {tea_path.name}")
        except ValueError as exc:
            msg = str(exc)
            assert "Unsupported layout" in msg or "Multi-row" in msg, (
                f"Expected clean unsupported layout error message for {tea_path.name}, got: {msg}"
            )
            print(f"\n{tea_path.name} cleanly raised unsupported layout error: {msg}")


def test_stage5a_auto_suggest_and_proportion_warnings() -> None:
    # 1. Auto-suggest D/W/H from PDF linework at 72/25.4 pt per mm
    doc_fit = fitz.open(str(FITNESS_PDF_PATH))
    page_fit = doc_fit[0]
    sugg_fit = measure_pdf_nominal_dims(page_fit, doc_fit)
    assert sugg_fit == (69.0, 160.0, 170.0), f"Expected (69, 160, 170) for Fitness Tube, got {sugg_fit}"

    doc_boss = fitz.open(str(PDF_PATH))
    page_boss = doc_boss[0]
    sugg_boss = measure_pdf_nominal_dims(page_boss, doc_boss)
    assert sugg_boss[0] == 50.0 and sugg_boss[1] == 120.0 and abs(sugg_boss[2] - 180.0) <= 1.0, f"Unexpected suggestion for BOSS: {sugg_boss}"

    # 2. When proportions differ by > 5%, readable warning is generated with measured ratio and suggested dimensions
    # Pass mismatched dimensions (e.g. 50 x 120 x 180) to Fitness Tube (which is 69 x 160 x 170)
    mismatched_dims = (50.0, 120.0, 180.0)
    box = build_stage1_box_geometry(page_fit, doc_fit, dimensions=mismatched_dims)
    warnings = box.get("dimension_warnings", [])
    assert warnings, "Expected proportion warning when entering mismatched dimensions"
    warning_text = warnings[0]
    print(f"\nProportion warning text: {warning_text}")
    assert "differ from entered proportions" in warning_text or "differs from entered" in warning_text
    assert "69 x 160 x 170" in warning_text








