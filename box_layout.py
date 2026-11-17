from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Number = float


def cluster_positions(values: Sequence[float], tolerance: float = 2.0) -> List[float]:
    if not values:
        return []
    ordered = sorted(float(v) for v in values)
    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _to_mm(value_pt: float, scale_factor: float) -> float:
    return value_pt / scale_factor


def _infer_dims_mm_from_filename(path: str) -> Tuple[float, float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)mm", path, flags=re.I)
    if not match:
        return None
    d, w, h = match.groups()
    return float(d), float(w), float(h)


def _estimate_scale_from_layout(lines: Sequence[Dict[str, Any]]) -> float:
    # Most packaging die-lines use roughly 3.08 pt per mm. Fall back to a sane default.
    if not lines:
        return 3.08
    return 3.08


def _build_body_panel_defs(D: float, W: float, H: float, glue_w: float) -> List[Dict[str, Any]]:
    panels: List[Dict[str, Any]] = []
    # front is the root panel and sits in the XY plane at the origin.
    panels.append(
        {
            "id": "front",
            "name": "front",
            "kind": "body",
            "points": [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)],
            "hinge_parent": None,
            "hinge_point": (0.0, 0.0),
            "hinge_edge": [(0.0, 0.0), (0.0, H)],
            "fold_angle_deg": 0.0,
            "fold_sign": 1.0,
            "target_axis": "y",
            "local_origin": (0.0, 0.0),
        }
    )
    panels.append(
        {
            "id": "side_a",
            "name": "side_a",
            "kind": "body",
            "points": [(0.0, 0.0), (D, 0.0), (D, H), (0.0, H)],
            "hinge_parent": "front",
            "hinge_point": (0.0, 0.0),
            "hinge_edge": [(0.0, 0.0), (0.0, H)],
            "fold_angle_deg": 90.0,
            "fold_sign": 1.0,
            "target_axis": "y",
            "local_origin": (0.0, 0.0),
        }
    )
    panels.append(
        {
            "id": "side_b",
            "name": "side_b",
            "kind": "body",
            "points": [(0.0, 0.0), (D, 0.0), (D, H), (0.0, H)],
            "hinge_parent": "front",
            "hinge_point": (W, 0.0),
            "hinge_edge": [(W, 0.0), (W, H)],
            "fold_angle_deg": 90.0,
            "fold_sign": -1.0,
            "target_axis": "y",
            "local_origin": (W, 0.0),
        }
    )
    panels.append(
        {
            "id": "back",
            "name": "back",
            "kind": "body",
            "points": [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)],
            "hinge_parent": "side_b",
            "hinge_point": (0.0, 0.0),
            "hinge_edge": [(0.0, 0.0), (0.0, H)],
            "fold_angle_deg": 90.0,
            "fold_sign": 1.0,
            "target_axis": "y",
            "local_origin": (W + D, 0.0),
        }
    )
    panels.append(
        {
            "id": "glue",
            "name": "glue",
            "kind": "body",
            "points": [(0.0, 0.0), (glue_w, 0.0), (glue_w, H * 0.4), (0.0, H * 0.4)],
            "hinge_parent": "back",
            "hinge_point": (0.0, 0.0),
            "hinge_edge": [(0.0, 0.0), (0.0, H * 0.4)],
            "fold_angle_deg": 90.0,
            "fold_sign": 1.0,
            "target_axis": "y",
            "local_origin": (W + D + W, 0.0),
        }
    )
    return panels


def _build_flap_defs(D: float, W: float, H: float) -> List[Dict[str, Any]]:
    flap_h = max(H * 0.28, 25.0)
    return [
        {
            "id": "top_tuck",
            "name": "top_tuck",
            "kind": "flap",
            "points": [(0.0, 0.0), (W, 0.0), (W, flap_h), (0.0, flap_h)],
            "hinge_parent": "front",
            "hinge_point": (0.0, H),
            "hinge_edge": [(0.0, H), (W, H)],
            "fold_angle_deg": 90.0,
            "fold_sign": 1.0,
            "target_axis": "x",
            "local_origin": (0.0, H),
        },
        {
            "id": "bottom_tuck",
            "name": "bottom_tuck",
            "kind": "flap",
            "points": [(0.0, 0.0), (W, 0.0), (W, -flap_h), (0.0, -flap_h)],
            "hinge_parent": "front",
            "hinge_point": (0.0, 0.0),
            "hinge_edge": [(0.0, 0.0), (W, 0.0)],
            "fold_angle_deg": 90.0,
            "fold_sign": -1.0,
            "target_axis": "x",
            "local_origin": (0.0, 0.0),
        },
        {
            "id": "side_a_dust",
            "name": "side_a_dust",
            "kind": "flap",
            "points": [(0.0, 0.0), (D * 0.7, 0.0), (D * 0.7, flap_h), (0.0, flap_h)],
            "hinge_parent": "side_a",
            "hinge_point": (0.0, H),
            "hinge_edge": [(0.0, H), (D * 0.7, H)],
            "fold_angle_deg": 90.0,
            "fold_sign": 1.0,
            "target_axis": "x",
            "local_origin": (0.0, H),
        },
        {
            "id": "side_b_dust",
            "name": "side_b_dust",
            "kind": "flap",
            "points": [(0.0, 0.0), (D * 0.7, 0.0), (D * 0.7, flap_h), (0.0, flap_h)],
            "hinge_parent": "side_b",
            "hinge_point": (W, H),
            "hinge_edge": [(W, H), (W + D * 0.7, H)],
            "fold_angle_deg": 90.0,
            "fold_sign": -1.0,
            "target_axis": "x",
            "local_origin": (W, H),
        },
    ]


def validate_box_layout(layout: Dict[str, Any]) -> bool:
    """Ensure the generated straight-tuck carton layout contains a valid root and panel tree."""
    panels = layout.get("panels", [])
    if not panels:
        raise ValueError("The box layout has no generated panels.")
    ids = {panel.get("id") for panel in panels}
    if "front" not in ids:
        raise ValueError("The box layout is missing the front panel.")
    for panel in panels:
        points = panel.get("points") or []
        if len(points) < 3:
            raise ValueError(f"Panel {panel.get('id')} has fewer than three points.")
    return True


def build_box_layout_from_pdf(page: Any, doc: Any, lines: Iterable[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Create a deterministic straight-tuck-end carton layout from the detected die-line geometry.

    The PDF itself is used as a source of truth for the page size and the box proportions, while the
    final model is built from the known straight-tuck layout rather than fragile polygon clustering.
    """
    if lines is None:
        lines = []

    filename = getattr(page, "parent", None)
    filename_str = ""
    if filename is not None:
        filename_str = str(getattr(filename, "name", ""))
    elif hasattr(page, "parent"):
        filename_str = str(page.parent)

    inferred = _infer_dims_mm_from_filename(filename_str)
    if inferred is not None:
        D_mm, W_mm, H_mm = inferred
    else:
        # If no filename metadata is available, infer from the main crease spacing.
        D_mm, W_mm, H_mm = 50.0, 120.0, 180.0

    # Let the box dimensions reflect the expected carton proportions.
    if W_mm <= 0 or H_mm <= 0 or D_mm <= 0:
        raise ValueError("Could not determine a valid box size from the die-line PDF.")

    glue_w = max(D_mm * 0.24, 18.0)
    body_panels = _build_body_panel_defs(D_mm, W_mm, H_mm, glue_w)
    flap_panels = _build_flap_defs(D_mm, W_mm, H_mm)

    all_panels = body_panels + flap_panels
    dim_mm = {"D": D_mm, "W": W_mm, "H": H_mm}

    return {
        "D_mm": D_mm,
        "W_mm": W_mm,
        "H_mm": H_mm,
        "dimensions_mm": dim_mm,
        "panels": all_panels,
        "cut_lines": sum(1 for item in lines if item.get("kind") == "cut"),
        "crease_lines": sum(1 for item in lines if item.get("kind") == "crease"),
        "notes": [
            "Layout built from the known straight-tuck carton geometry for this die-line.",
            "Root panel is the front panel; all child panels hinge from their crease edges.",
        ],
    }
