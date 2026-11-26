from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np
from streamlit.testing.v1 import AppTest

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
NO_DIELINE_PDF_PERFUME = Path(__file__).resolve().parents[1] / "02_Luxury_Perfume_Noir_55x85x160mm.pdf"
NO_DIELINE_PDF_VITAMIN = Path(__file__).resolve().parents[1] / "06_Vitamin_C_Serum_38x42x125mm.pdf"
NO_DIELINE_PDF_ESPRESSO = Path(__file__).resolve().parents[1] / "08_Espresso_Coffee_Pods_65x120x150mm.pdf"
BAMBOO_PDF_PATH = Path(__file__).resolve().parents[1] / "NORDIC_FORM_BAMBOO_SOCKS_3P_ARTWORK_V1.pdf"
AEROFIT_PDF_PATH = Path(__file__).resolve().parents[1] / "AEROFIT_RESISTANCE_BAND_SET_ARTWORK_V1.pdf"
CEYLON_PDF_PATH = Path(__file__).resolve().parents[1] / "CEYLON_SPICE_TEA_25TB_ARTWORK_V1.pdf"
HIGHLAND_PDF_PATH = Path(__file__).resolve().parents[1] / "HIGHLAND_ROAST_ESPRESSO_PODS_10CT_V1.pdf"
LUMIERE_PDF_PATH = Path(__file__).resolve().parents[1] / "LUMIERE_BOTANICAL_SERUM_30ML_V1.pdf"


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


def test_stage8b_two_row_layouts_support() -> None:
    for tea_path in [TEA1_PDF_PATH, TEA2_PDF_PATH]:
        assert tea_path.exists(), f"Missing tea PDF: {tea_path}"
        doc = fitz.open(str(tea_path))
        page = doc[0]

        # 1. Single-row layout detected
        lines = extract_structural_lines(page, doc=doc, min_length=0)
        green = [l for l in lines if l.get("kind") == "crease"]
        red = [l for l in lines if l.get("kind") == "cut"]
        rows, reason = detect_layout_rows(green, red)
        assert rows == 1, f"Expected single-row layout for {tea_path.name}, got {rows} rows ({reason})"

        box = build_stage1_box_geometry(page, doc)
        model = build_stage2_body_panel_model(box)

        # Assert 5 body panels
        panels = model["panels"]
        assert len(panels) == 5, f"Expected 5 body panels for {tea_path.name}, got {len(panels)}"
        expected_pids = ["glue", "front", "side_b", "back", "side_a"]
        assert [p["id"] for p in box["body_panels"]] == expected_pids
        assert set(p["id"] for p in panels) == set(expected_pids)
        assert model["root"] == "front"

        # Assert >= 8 flaps
        flaps = model["flaps"]
        assert len(flaps) >= 8, f"Expected >= 8 flaps for {tea_path.name}, got {len(flaps)}"

        W = box["W_mm"]
        H = box["H_mm"]
        D = box["D_mm"]

        # Assert outward normals positive for all flaps
        fn = compute_flap_outward_normals(model, fold_progress=1.0)
        for fid, d in fn.items():
            assert d["is_outward"], f"Flap {fid} normal is inward! dot={d['dot']}"
            assert d["dot"] > 0.0

        panel_by_id = {panel["id"]: panel for panel in panels}
        flap_by_id = {flap["id"]: flap for flap in flaps}

        def remap_p(p: float, start: float, end: float) -> float:
            return float(np.clip((p - start) / (end - start), 0.0, 1.0))

        p_wall = 1.0
        p_dust = 1.0
        p_tuck = 1.0

        def panel_T(pid: str) -> np.ndarray:
            p = panel_by_id[pid]
            if p["parent"] is None:
                return np.eye(4)
            parent_transform = panel_T(p["parent"])
            hinge = np.array(p.get("pivot_position", [0.0, 0.0, 0.0]), dtype=float).copy()
            glue_off = float(p.get("glue_offset", 0.0))
            if glue_off != 0.0:
                hinge[0] -= glue_off * p_wall
            sign = float(p.get("fold_sign", 1.0))
            theta = sign * p_wall * (np.pi / 2.0)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])
            Tr = np.array([[1.0, 0.0, 0.0, hinge[0]], [0.0, 1.0, 0.0, hinge[1]], [0.0, 0.0, 1.0, hinge[2]], [0.0, 0.0, 0.0, 1.0]])
            return parent_transform @ Tr @ R

        def flap_T(fid: str) -> np.ndarray:
            f = flap_by_id[fid]
            pr = f["parent"]
            parent_transform = panel_T(pr) if pr in panel_by_id else flap_T(pr)
            piv = f["pivot_position"]
            sign = float(f.get("sign", 1.0))
            sub_p = remap_p(1.0, 0.7, 0.95) if (fid == "top_tuck" or fid.endswith("_tuck")) else (p_dust if f.get("kind") == "dust" else p_tuck)
            theta = sign * sub_p * (np.pi / 2.0)
            c, s = np.cos(theta), np.sin(theta)
            Rx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]])
            Tr = np.array([[1.0, 0.0, 0.0, piv[0]], [0.0, 1.0, 0.0, piv[1]], [0.0, 0.0, 1.0, piv[2]], [0.0, 0.0, 0.0, 1.0]])
            return parent_transform @ Tr @ Rx

        for flap in flaps:
            fid = flap["id"]
            poly = flap.get("polygon", [])
            if not poly:
                continue
            sub_p = 1.0
            local_z = float(flap.get("z_offset", 0.0)) * sub_p
            local_pts = np.array([[pt[0], pt[1], local_z, 1.0] for pt in poly], dtype=float)
            world = (flap_T(fid) @ local_pts.T).T[:, :3]
            min_x, max_x = float(world[:, 0].min()), float(world[:, 0].max())
            min_y, max_y = float(world[:, 1].min()), float(world[:, 1].max())
            min_z, max_z = float(world[:, 2].min()), float(world[:, 2].max())

            dev_x = max(0.0, -2.0 - min_x, max_x - (W + 2.0))
            dev_y = max(0.0, -2.0 - min_y, max_y - (H + 2.0))
            dev_z = max(0.0, -D - 2.0 - min_z, max_z - 2.0)
            max_dev = max(dev_x, dev_y, dev_z)
            assert max_dev <= 2.0, (
                f"Flap {fid} of {tea_path.name} exceeds closed box bounds! max_dev={max_dev:.2f}mm "
                f"(x:[{min_x:.1f}, {max_x:.1f}], y:[{min_y:.1f}, {max_y:.1f}], z:[{min_z:.1f}, {max_z:.1f}])"
            )

        # Assert glue panel tucked behind neighbour
        glue_p = panel_by_id["glue"]
        h_glue = float(glue_p.get("height", H))
        glue_corners = np.array([
            [glue_p["x0"], 0.0, 0.0, 1.0],
            [glue_p["x1"], 0.0, 0.0, 1.0],
            [glue_p["x1"], h_glue, 0.0, 1.0],
            [glue_p["x0"], h_glue, 0.0, 1.0],
        ], dtype=float)
        glue_world = (panel_T("glue") @ glue_corners.T).T[:, :3]
        assert glue_world[:, 0].min() >= -0.5 and glue_world[:, 0].max() <= 1.0
        assert glue_world[:, 2].min() >= -D - 0.5 and glue_world[:, 2].max() <= 0.5


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

    doc_tea1 = fitz.open(str(TEA1_PDF_PATH))
    page_tea1 = doc_tea1[0]
    sugg_tea1 = measure_pdf_nominal_dims(page_tea1, doc_tea1)
    assert sugg_tea1 == (79.0, 150.0, 69.0), f"Expected (79, 150, 69) for British Tea, got {sugg_tea1}"

    doc_tea2 = fitz.open(str(TEA2_PDF_PATH))
    page_tea2 = doc_tea2[0]
    sugg_tea2 = measure_pdf_nominal_dims(page_tea2, doc_tea2)
    assert sugg_tea2 == (66.5, 119.0, 78.0), f"Expected (66.5, 119, 78) for Nordqvist, got {sugg_tea2}"

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


def test_stage5a_streamlit_apptest_pipeline() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    assert app_path.exists(), f"Missing app.py: {app_path}"

    # 1. BOSS PDF: renders without error or exception and shows viewer iframe
    at_boss = AppTest.from_file(str(app_path), default_timeout=90).run()
    at_boss.file_uploader[0].set_value((PDF_PATH.name, PDF_PATH.read_bytes(), "application/pdf")).run()
    assert len(at_boss.exception) == 0, f"Unexpected exception in BOSS run: {[e.value for e in at_boss.exception]}"
    assert len(at_boss.error) == 0, f"Unexpected error in BOSS run: {[e.value for e in at_boss.error]}"
    iframes_boss = at_boss.get("iframe")
    assert len(iframes_boss) >= 1, "Expected viewer HTML iframe to be rendered for BOSS"
    assert "box-viewer" in iframes_boss[0].proto.srcdoc or "Three.js" in iframes_boss[0].proto.srcdoc

    # 2. FITNESS TUBE PDF: renders without error or exception and shows viewer iframe
    at_fit = AppTest.from_file(str(app_path), default_timeout=90).run()
    at_fit.file_uploader[0].set_value((FITNESS_PDF_PATH.name, FITNESS_PDF_PATH.read_bytes(), "application/pdf")).run()
    assert len(at_fit.exception) == 0, f"Unexpected exception in Fitness Tube run: {[e.value for e in at_fit.exception]}"
    assert len(at_fit.error) == 0, f"Unexpected error in Fitness Tube run: {[e.value for e in at_fit.error]}"
    iframes_fit = at_fit.get("iframe")
    assert len(iframes_fit) >= 1, "Expected viewer HTML iframe to be rendered for Fitness Tube"
    assert "box-viewer" in iframes_fit[0].proto.srcdoc or "Three.js" in iframes_fit[0].proto.srcdoc

    # 3. BRITISH TEA PDF: renders without error or exception and shows viewer iframe
    at_tea = AppTest.from_file(str(app_path), default_timeout=90).run()
    at_tea.file_uploader[0].set_value((TEA1_PDF_PATH.name, TEA1_PDF_PATH.read_bytes(), "application/pdf")).run()
    assert len(at_tea.exception) == 0, f"Unexpected exception in Tea run: {[e.value for e in at_tea.exception]}"
    assert len(at_tea.error) == 0, f"Unexpected error in Tea run: {[e.value for e in at_tea.error]}"
    iframes_tea = at_tea.get("iframe")
    assert len(iframes_tea) >= 1, "Expected viewer HTML iframe to be rendered for British Tea"
    assert "box-viewer" in iframes_tea[0].proto.srcdoc or "Three.js" in iframes_tea[0].proto.srcdoc

    # 4. NORDQVIST PDF: renders without error or exception and shows viewer iframe
    at_nord = AppTest.from_file(str(app_path), default_timeout=90).run()
    at_nord.file_uploader[0].set_value((TEA2_PDF_PATH.name, TEA2_PDF_PATH.read_bytes(), "application/pdf")).run()
    assert len(at_nord.exception) == 0, f"Unexpected exception in Nordqvist run: {[e.value for e in at_nord.exception]}"
    assert len(at_nord.error) == 0, f"Unexpected error in Nordqvist run: {[e.value for e in at_nord.error]}"
    iframes_nord = at_nord.get("iframe")
    assert len(iframes_nord) >= 1, "Expected viewer HTML iframe to be rendered for Nordqvist"
    assert "box-viewer" in iframes_nord[0].proto.srcdoc or "Three.js" in iframes_nord[0].proto.srcdoc


def test_no_dieline_friendly_error_apptest() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    assert app_path.exists(), f"Missing app.py: {app_path}"

    test_pdfs = [
        NO_DIELINE_PDF_PERFUME,
        NO_DIELINE_PDF_VITAMIN,
        NO_DIELINE_PDF_ESPRESSO,
    ]

    for pdf_path in test_pdfs:
        assert pdf_path.exists(), f"Missing test PDF: {pdf_path}"
        at = AppTest.from_file(str(app_path), default_timeout=90).run()
        at.file_uploader[0].set_value((pdf_path.name, pdf_path.read_bytes(), "application/pdf")).run()

        # Assert no unhandled Python exceptions
        assert len(at.exception) == 0, f"Unexpected exception for {pdf_path.name}: {[e.value for e in at.exception]}"

        # Assert friendly error message is shown
        assert len(at.error) >= 1, f"Expected st.error for {pdf_path.name}, but none found."
        error_text = at.error[0].value
        assert "No Fold/Die die-line found in this PDF" in error_text, f"Unexpected error message: {error_text}"
        assert "Upload a die-line PDF exported with 'Fold' (crease) and 'Die' (cut) spot colours/layers." in error_text
        assert "12 vector paths" in error_text
        assert "6 text blocks" in error_text


def test_bamboo_socks_panel_normals_bbox_and_apptest() -> None:
    assert BAMBOO_PDF_PATH.exists(), f"Missing Bamboo Socks PDF: {BAMBOO_PDF_PATH}"
    doc = fitz.open(str(BAMBOO_PDF_PATH))
    page = doc[0]
    box = build_stage1_box_geometry(page, doc, dimensions=(42.0, 95.0, 185.0))
    model = build_stage2_body_panel_model(box, page=page, dpi=200)

    # 1. All body-panel dot products > 0
    panel_normals = compute_panel_outward_normals(model, fold_progress=1.0)
    for pid, pdata in panel_normals.items():
        assert pdata["dot"] > 0.0, f"Panel {pid} dot product is non-positive: {pdata['dot']}"

    # 2. Closed-state bbox within 5% of 42x95x185
    bounds = compute_stage2_numpy_box_bounds(model, fold_progress=1.0)
    size_mm = bounds["size_mm"]
    assert abs(size_mm[0] - 42.0) / 42.0 < 0.05, f"Depth {size_mm[0]} not within 5% of 42.0"
    assert abs(size_mm[1] - 95.0) / 95.0 < 0.05, f"Width {size_mm[1]} not within 5% of 95.0"
    assert abs(size_mm[2] - 185.0) / 185.0 < 0.05, f"Height {size_mm[2]} not within 5% of 185.0"

    # 3. Streamlit AppTest: no unhandled exception, no error, viewer rendered
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=90).run()
    at.file_uploader[0].set_value((BAMBOO_PDF_PATH.name, BAMBOO_PDF_PATH.read_bytes(), "application/pdf")).run()

    assert len(at.exception) == 0, f"Unexpected exception in Bamboo Socks AppTest: {[e.value for e in at.exception]}"
    assert len(at.error) == 0, f"Unexpected error in Bamboo Socks AppTest: {[e.value for e in at.error]}"
    iframes = at.get("iframe")
    assert len(iframes) >= 1, "Expected viewer HTML iframe for Bamboo Socks"
    assert "box-viewer" in iframes[0].proto.srcdoc or "Three.js" in iframes[0].proto.srcdoc


def test_glue_panel_bbox_and_gap_and_apptest() -> None:
    test_cases = [
        (AEROFIT_PDF_PATH, "AEROFIT"),
        (FITNESS_PDF_PATH, "FITNESS TUBE"),
        (PDF_PATH, "BOSS"),
    ]

    for pdf_path, label in test_cases:
        assert pdf_path.exists(), f"Missing PDF for {label}: {pdf_path}"
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        box = build_stage1_box_geometry(page, doc)
        model = build_stage2_body_panel_model(box, page=page)

        panels = model["panels"]
        panel_by_id = {p["id"]: p for p in panels}
        panel_order = [p["id"] for p in box["body_panels"]]

        assert "glue" in panel_by_id, f"Missing glue panel in {label}"

        glue_idx = panel_order.index("glue")
        neighbour_id = panel_order[0] if glue_idx != 0 else panel_order[-1]
        parent_id = panel_order[-2] if glue_idx != 0 else panel_order[1]

        assert panel_by_id["glue"]["parent"] == parent_id, (
            f"Expected glue parent {parent_id}, got {panel_by_id['glue']['parent']}"
        )

        def transform_for(pid: str) -> np.ndarray:
            p_data = panel_by_id[pid]
            if p_data["parent"] is None:
                return np.eye(4, dtype=float)
            p_t = transform_for(p_data["parent"])
            piv_pt = np.array(p_data.get("pivot_position", [0.0, 0.0, 0.0]), dtype=float).copy()
            glue_off = float(p_data.get("glue_offset", 0.0))
            if glue_off != 0.0:
                piv_pt[0] -= glue_off
            sgn = float(p_data.get("fold_sign", 1.0))
            th = sgn * (np.pi / 2.0)
            c, s = np.cos(th), np.sin(th)
            rot = np.array([
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float)
            tr = np.array([
                [1.0, 0.0, 0.0, piv_pt[0]],
                [0.0, 1.0, 0.0, piv_pt[1]],
                [0.0, 0.0, 1.0, piv_pt[2]],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float)
            return p_t @ tr @ rot

        def corners_for(pid: str) -> np.ndarray:
            p_data = panel_by_id[pid]
            x0 = float(p_data["x0"])
            x1 = float(p_data["x1"])
            h = float(p_data["height"])
            pts = np.array([
                [x0, 0.0, 0.0, 1.0],
                [x1, 0.0, 0.0, 1.0],
                [x1, h, 0.0, 1.0],
                [x0, h, 0.0, 1.0],
            ], dtype=float)
            return (transform_for(pid) @ pts.T).T[:, :3]

        main_panels = [p for p in panel_order if p != "glue"]
        all_main_pts = np.vstack([corners_for(pid) for pid in main_panels])
        box_min = all_main_pts.min(axis=0)
        box_max = all_main_pts.max(axis=0)
        c_box = 0.5 * (box_min + box_max)

        glue_corners = corners_for("glue")
        glue_min = glue_corners.min(axis=0)
        glue_max = glue_corners.max(axis=0)

        # 1. Glue bbox lies inside closed box bbox (tolerance 1 mm)
        assert glue_min[0] >= box_min[0] - 1.0, f"{label}: glue min x {glue_min[0]} < box min x {box_min[0]} - 1"
        assert glue_max[0] <= box_max[0] + 1.0, f"{label}: glue max x {glue_max[0]} > box max x {box_max[0]} + 1"
        assert glue_min[1] >= box_min[1] - 1.0, f"{label}: glue min y {glue_min[1]} < box min y {box_min[1]} - 1"
        assert glue_max[1] <= box_max[1] + 1.0, f"{label}: glue max y {glue_max[1]} > box max y {box_max[1]} + 1"
        assert glue_min[2] >= box_min[2] - 1.0, f"{label}: glue min z {glue_min[2]} < box min z {box_min[2]} - 1"
        assert glue_max[2] <= box_max[2] + 1.0, f"{label}: glue max z {glue_max[2]} > box max z {box_max[2]} + 1"

        # 2. Gap to neighbour <= 0.5 mm
        neigh_corners = corners_for(neighbour_id)
        c_neigh = neigh_corners.mean(axis=0)
        v1 = neigh_corners[1] - neigh_corners[0]
        v2 = neigh_corners[3] - neigh_corners[0]
        n_raw = np.cross(v1, v2)
        n_unit = n_raw / np.linalg.norm(n_raw)
        n_in = n_unit if np.dot(n_unit, c_box - c_neigh) > 0 else -n_unit

        gap = float(np.dot(n_in, glue_corners.mean(axis=0) - c_neigh))
        assert abs(gap) <= 0.5, f"{label}: gap to neighbour {neighbour_id} is {gap:.3f} mm (> 0.5 mm)"

        # 3. Streamlit AppTest: no unhandled exception, no error, viewer rendered
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=90).run()
        at.file_uploader[0].set_value((pdf_path.name, pdf_path.read_bytes(), "application/pdf")).run()
        assert len(at.exception) == 0, f"Unexpected exception in {label} AppTest: {[e.value for e in at.exception]}"
        assert len(at.error) == 0, f"Unexpected error in {label} AppTest: {[e.value for e in at.error]}"
        iframes = at.get("iframe")
        assert len(iframes) >= 1, f"Expected viewer HTML iframe for {label}"


def test_proportion_warning_british_tea_green_tea() -> None:
    assert TEA1_PDF_PATH.exists(), f"Missing British Tea PDF: {TEA1_PDF_PATH}"
    doc = fitz.open(str(TEA1_PDF_PATH))
    page = doc[0]

    # Matching proportions (79 x 150 x 69) in either order (D, W, H) or (W, D, H) must NOT warn
    box1 = build_stage1_box_geometry(page, doc, dimensions=(79.0, 150.0, 69.0))
    warns1 = [w for w in box1.get("dimension_warnings", []) if "proportions" in w]
    assert len(warns1) == 0, f"Unexpected proportion warning for (79, 150, 69): {warns1}"

    box2 = build_stage1_box_geometry(page, doc, dimensions=(150.0, 79.0, 69.0))
    warns2 = [w for w in box2.get("dimension_warnings", []) if "proportions" in w]
    assert len(warns2) == 0, f"Unexpected proportion warning for (150, 79, 69): {warns2}"

    # Also test Nordqvist (66.5 x 119 x 78) in either order
    doc_nord = fitz.open(str(TEA2_PDF_PATH))
    box3 = build_stage1_box_geometry(doc_nord[0], doc_nord, dimensions=(66.5, 119.0, 78.0))
    warns3 = [w for w in box3.get("dimension_warnings", []) if "proportions" in w]
    assert len(warns3) == 0, f"Unexpected proportion warning for Nordqvist (66.5, 119, 78): {warns3}"

    box4 = build_stage1_box_geometry(doc_nord[0], doc_nord, dimensions=(119.0, 66.5, 78.0))
    warns4 = [w for w in box4.get("dimension_warnings", []) if "proportions" in w]
    assert len(warns4) == 0, f"Unexpected proportion warning for Nordqvist (119, 66.5, 78): {warns4}"


def test_green_tea_box_flaps_and_warnings_and_apptest() -> None:
    assert CEYLON_PDF_PATH.exists(), f"Missing Ceylon Spice Tea PDF: {CEYLON_PDF_PATH}"
    doc = fitz.open(str(CEYLON_PDF_PATH))
    page = doc[0]

    # 1. No false proportion or dimension warnings for (68, 82, 138), (82, 68, 138), or None
    box1 = build_stage1_box_geometry(page, doc, dimensions=(68.0, 82.0, 138.0))
    warns1 = box1.get("dimension_warnings", [])
    assert len(warns1) == 0, f"Unexpected warnings for (68, 82, 138): {warns1}"

    box2 = build_stage1_box_geometry(page, doc, dimensions=(82.0, 68.0, 138.0))
    warns2 = box2.get("dimension_warnings", [])
    assert len(warns2) == 0, f"Unexpected warnings for (82, 68, 138): {warns2}"

    box_none = build_stage1_box_geometry(page, doc, dimensions=None)
    warns_none = box_none.get("dimension_warnings", [])
    assert len(warns_none) == 0, f"Unexpected warnings for None: {warns_none}"
    assert abs(box_none["D_mm"] - 68.0) <= 1.0
    assert abs(box_none["W_mm"] - 82.0) <= 1.0
    assert abs(box_none["H_mm"] - 138.0) <= 1.0

    # 2. Flap world bboxes within 2 mm of closed box bbox [0..W, 0..H, -D..0]
    W, H, D = 82.0, 138.0, 68.0
    model = build_stage2_body_panel_model(box1, page=page)
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
        w_pts = (flap_T(fid) @ pts.T).T[:, :3]

        min_x, max_x = float(w_pts[:, 0].min()), float(w_pts[:, 0].max())
        min_y, max_y = float(w_pts[:, 1].min()), float(w_pts[:, 1].max())
        min_z, max_z = float(w_pts[:, 2].min()), float(w_pts[:, 2].max())

        assert min_x >= -2.0 and max_x <= W + 2.0, f"Flap {fid} X out of bounds: [{min_x:.1f}, {max_x:.1f}]"
        assert min_y >= -2.0 and max_y <= H + 2.0, f"Flap {fid} Y out of bounds: [{min_y:.1f}, {max_y:.1f}]"
        assert min_z >= -D - 2.0 and max_z <= 2.0, f"Flap {fid} Z out of bounds: [{min_z:.1f}, {max_z:.1f}]"

    # 3. Streamlit AppTest: renders without error or exception
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=90).run()
    at.file_uploader[0].set_value((CEYLON_PDF_PATH.name, CEYLON_PDF_PATH.read_bytes(), "application/pdf")).run()
    assert len(at.exception) == 0, f"Unexpected exception in AppTest: {[e.value for e in at.exception]}"
    assert len(at.error) == 0, f"Unexpected error in AppTest: {[e.value for e in at.error]}"
    iframes = at.get("iframe")
    assert len(iframes) >= 1, "Expected viewer HTML iframe to be rendered"


def test_square_cartons_flaps_and_warnings_and_apptest() -> None:
    cases = [
        (HIGHLAND_PDF_PATH, 85.0, 85.0, 85.0),
        (LUMIERE_PDF_PATH, 45.0, 45.0, 115.0),
    ]

    def remap_p(t: float, a: float, b: float) -> float:
        return float(np.clip((t - a) / (b - a), 0.0, 1.0))

    p_wall = remap_p(1.0, 0.0, 0.5)
    p_dust = remap_p(1.0, 0.5, 0.7)
    p_tuck = remap_p(1.0, 0.7, 1.0)
    p_tuck_flap = remap_p(1.0, 0.7, 0.95)
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    for pdf_path, exp_d, exp_w, exp_h in cases:
        assert pdf_path.exists(), f"Missing square carton PDF: {pdf_path}"
        doc = fitz.open(str(pdf_path))
        page = doc[0]

        # 1. No false warnings with nominal dims or None
        box = build_stage1_box_geometry(page, doc, dimensions=None)
        warns = box.get("dimension_warnings", [])
        assert len(warns) == 0, f"Unexpected warnings for {pdf_path.name} (None): {warns}"
        assert abs(box["D_mm"] - exp_d) <= 1.0
        assert abs(box["W_mm"] - exp_w) <= 1.0
        assert abs(box["H_mm"] - exp_h) <= 1.0

        box_exp = build_stage1_box_geometry(page, doc, dimensions=(exp_d, exp_w, exp_h))
        warns_exp = box_exp.get("dimension_warnings", [])
        assert len(warns_exp) == 0, f"Unexpected warnings for {pdf_path.name} ({exp_d}, {exp_w}, {exp_h}): {warns_exp}"

        # 2. Flap world bboxes at p=1 within 2 mm of closed box bbox [0..W, 0..H, -D..0]
        W, H, D = exp_w, exp_h, exp_d
        model = build_stage2_body_panel_model(box_exp, page=page)
        panel_by_id = {p["id"]: p for p in model["panels"]}
        flap_by_id = {f["id"]: f for f in model["flaps"]}

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
            w_pts = (flap_T(fid) @ pts.T).T[:, :3]

            min_x, max_x = float(w_pts[:, 0].min()), float(w_pts[:, 0].max())
            min_y, max_y = float(w_pts[:, 1].min()), float(w_pts[:, 1].max())
            min_z, max_z = float(w_pts[:, 2].min()), float(w_pts[:, 2].max())

            assert min_x >= -2.0 and max_x <= W + 2.0, f"Flap {fid} X out of bounds in {pdf_path.name}: [{min_x:.1f}, {max_x:.1f}]"
            assert min_y >= -2.0 and max_y <= H + 2.0, f"Flap {fid} Y out of bounds in {pdf_path.name}: [{min_y:.1f}, {max_y:.1f}]"
            assert min_z >= -D - 2.0 and max_z <= 2.0, f"Flap {fid} Z out of bounds in {pdf_path.name}: [{min_z:.1f}, {max_z:.1f}]"

        # 3. Lid/lock area covers >= 80% of opening
        top_open = W * D
        lids = [f for f in model["flaps"] if f.get("kind") == "lid"]
        locks = [f for f in model["flaps"] if f.get("kind") == "lock"]
        assert len(lids) >= 1, f"Expected at least 1 lid in {pdf_path.name}"
        assert len(locks) >= 1, f"Expected at least 1 lock in {pdf_path.name}"
        for lid in lids:
            l_xs = [pt[0] for pt in lid["polygon"]]
            l_ys = [pt[1] for pt in lid["polygon"]]
            l_area = (max(l_xs) - min(l_xs)) * (max(l_ys) - min(l_ys))
            assert l_area >= 0.8 * top_open, f"Lid area {l_area:.1f} covers < 80% of opening {top_open:.1f} in {pdf_path.name}"
        for lock in locks:
            l_xs = [pt[0] for pt in lock["polygon"]]
            l_ys = [pt[1] for pt in lock["polygon"]]
            l_area = (max(l_xs) - min(l_xs)) * (max(l_ys) - min(l_ys))
            assert l_area >= 0.8 * top_open, f"Lock area {l_area:.1f} covers < 80% of opening {top_open:.1f} in {pdf_path.name}"

        # 4. Streamlit AppTest: renders without error or exception
        at = AppTest.from_file(str(app_path), default_timeout=90).run()
        at.file_uploader[0].set_value((pdf_path.name, pdf_path.read_bytes(), "application/pdf")).run()
        assert len(at.exception) == 0, f"Unexpected exception in AppTest for {pdf_path.name}: {[e.value for e in at.exception]}"
        assert len(at.error) == 0, f"Unexpected error in AppTest for {pdf_path.name}: {[e.value for e in at.error]}"
        iframes = at.get("iframe")
        assert len(iframes) >= 1, f"Expected viewer HTML iframe to be rendered for {pdf_path.name}"














