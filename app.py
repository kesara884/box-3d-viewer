import base64
import io
import json
import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import fitz
import matplotlib
import networkx as nx
import numpy as np
import streamlit as st
from PIL import Image
from shapely.geometry import LineString, Polygon, box as shapely_box
from shapely.ops import polygonize, unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt


st.set_page_config(page_title="3D Packaging Box Preview", layout="wide")


def normalize_color(color_value: Any) -> Tuple[float, float, float]:
    """Accept fitz color values in 0-1 or 0-255 space."""
    if not color_value:
        return (0.0, 0.0, 0.0)
    color = list(color_value)
    if len(color) < 3:
        color = color + [0.0] * (3 - len(color))
    rgb = [float(v) for v in color[:3]]
    if max(rgb) > 1.0:
        rgb = [v / 255.0 for v in rgb]
    return tuple(rgb)


def cmyk_to_rgb(color_value: Any) -> Tuple[float, float, float]:
    """Convert CMYK to RGB when a drawing uses a print-process color."""
    if not color_value:
        return (0.0, 0.0, 0.0)
    values = list(color_value)
    if len(values) < 4:
        return normalize_color(values)
    c, m, y, k = [float(v) for v in values[:4]]
    if max(c, m, y, k) > 1.0:
        c, m, y, k = [v / 255.0 for v in (c, m, y, k)]
    r = 1.0 - min(1.0, c * (1.0 - k) + k)
    g = 1.0 - min(1.0, m * (1.0 - k) + k)
    b = 1.0 - min(1.0, y * (1.0 - k) + k)
    return (r, g, b)


def color_key(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(round(float(v), 4) for v in rgb)


def _make_hashable(value: Any) -> Any:
    """Convert nested list/set/dict structures into hashable tuples for set/dict-key dedupe."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _make_hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_make_hashable(v) for v in value))
    return value


def _infer_dims_mm_from_filename(path: Any) -> Tuple[float, float, float] | None:
    """Extract D x W x H values from a filename when present."""
    if path is None:
        return None
    candidate = str(path)
    if hasattr(path, "name"):
        candidate = str(path.name)
    match = re.search(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:mm)?", candidate)
    if not match:
        return None
    d, w, h = match.groups()
    return float(d), float(w), float(h)


parse_dimensions_from_filename = _infer_dims_mm_from_filename


def _iter_page_drawings(page: fitz.Page, doc: fitz.Document | None = None) -> Iterable[dict]:
    """Yield drawings from both standard and extended PyMuPDF APIs, including nested XObjects."""
    seen = set()
    getter_names = ["get_drawings", "get_cdrawings"]
    for getter_name in getter_names:
        getter = getattr(page, getter_name, None)
        if getter is None:
            continue
        for use_extended in (True, False):
            try:
                candidates = getter(extended=use_extended)
            except TypeError:
                try:
                    candidates = getter()
                except Exception:
                    continue
            except Exception:
                continue
            if candidates is None:
                continue
            if isinstance(candidates, dict):
                candidates = list(candidates.values())
            elif not isinstance(candidates, list):
                try:
                    candidates = list(candidates)
                except Exception:
                    continue
            for drawing in candidates:
                if not isinstance(drawing, dict):
                    continue
                fence = (
                    drawing.get("layer"),
                    _make_hashable(drawing.get("stroke")),
                    _make_hashable(drawing.get("fill")),
                    _make_hashable(drawing.get("items")),
                )
                if fence in seen:
                    continue
                seen.add(fence)
                yield drawing

    for nested in _extract_nested_xobject_drawings(page):
        if not isinstance(nested, dict):
            continue
        fence = (
            nested.get("layer"),
            _make_hashable(nested.get("stroke")),
            _make_hashable(nested.get("fill")),
            _make_hashable(nested.get("items")),
        )
        if fence in seen:
            continue
        seen.add(fence)
        yield nested


def _scan_raw_xref_spot_colors(doc: fitz.Document | None) -> List[Dict[str, Any]]:
    """Fallback scan for raw PDF operator streams. Useful for Form XObjects and Spot colors in Illustrator exports."""
    if doc is None:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        xref_count = int(doc.xref_length())
    except Exception:
        xref_count = 0

    for xref in range(1, xref_count):
        try:
            raw = doc.xref_stream(xref)
        except Exception:
            continue
        if raw is None:
            continue
        if isinstance(raw, (bytes, bytearray)):
            payload = raw
        else:
            payload = str(raw).encode("latin-1", "ignore")

        text = payload.decode("latin-1", "ignore").lower()
        if not any(token in text for token in ("rg", "g ", "scn", "sc ", "k ", "cm", "m ", "l ", "re ", "c ", "cut", "crease", "die", "spot")):
            continue

        named = []
        for match in re.finditer(r"(?i)(cut|crease|die|knife|trim|fold|score|glue|keyline|spot(?:\s+color)?|separation)", text):
            name = match.group(1).strip()
            named.append(name)

        if named:
            rows.append({"source": "spot", "rgb": (0.0, 0.0, 0.0), "layer": named[0].title(), "count": 1})

        for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:rg|RG|g|G)", text):
            r, g, b = [float(match.group(i)) for i in (1, 2, 3)]
            if max(r, g, b) > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            rows.append({"source": "spot", "rgb": (r, g, b), "layer": "raw_xobject", "count": 1})

        for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:k|K)", text):
            c, m, y, k = [float(match.group(i)) for i in (1, 2, 3, 4)]
            if max(c, m, y, k) > 1.0:
                c, m, y, k = c / 255.0, m / 255.0, y / 255.0, k / 255.0
            r = 1.0 - min(1.0, c * (1.0 - k) + k)
            g = 1.0 - min(1.0, m * (1.0 - k) + k)
            b = 1.0 - min(1.0, y * (1.0 - k) + k)
            rows.append({"source": "spot", "rgb": (r, g, b), "layer": "raw_xobject", "count": 1})

    return rows


def _classify_spot_name(name: str) -> str:
    """Classify packaging spot names like Die/Fold that are not RGB colors."""
    text = (name or "").lower()
    if any(word in text for word in ("cut", "knife", "die", "keyline", "trim", "stanze")):
        return "cut"
    if any(word in text for word in ("fold", "crease", "score", "hinge", "glue", "rillen")):
        return "crease"
    return "ignore"


def classify_name_or_layer(name: str) -> str:
    """Resolve a PDF layer/OC/Separation name to CUT or CREASE before using RGB thresholds as fallback."""
    text = (name or "").lower()
    if any(token in text for token in ("fold", "crease", "score", "hinge", "glue", "rillen")):
        return "crease"
    if any(token in text for token in ("die", "cut", "knife", "trim", "keyline", "stanze")):
        return "cut"
    return "ignore"


def _transform_point(point: Tuple[float, float], ctm: Tuple[float, float, float, float, float, float]) -> Tuple[float, float]:
    x, y = point
    a, b, c, d, e, f = ctm
    return (x * a + y * c + e, x * b + y * d + f)


def _path_to_segments(path: List[Tuple[float, float]], close_path: bool = False) -> List[LineString]:
    """Convert a path into a sequence of line segments suitable for polygonization."""
    if len(path) < 2:
        return []
    poly = list(path)
    if close_path and poly and poly[0] != poly[-1]:
        poly = poly + [poly[0]]
    if len(poly) == 2:
        return [LineString(poly)]
    segments: List[LineString] = []
    for idx in range(len(poly) - 1):
        segments.append(LineString([poly[idx], poly[idx + 1]]))
    return segments


def _extract_raw_content_segments(page: fitz.Page, doc: fitz.Document | None = None, min_length: float = 15.0) -> List[Dict[str, Any]]:
    """Fallback path parser that reads the raw page content streams and keeps only large structural strokes.

    Important: in PDF syntax, numeric operands precede the operator (`x y m`, `x y l`, `a b c d e f cm`).
    A parser that reads the operator before its coordinates misses the actual geometry and therefore sees no structural segments.
    """
    if doc is None:
        return []
    segments: List[Dict[str, Any]] = []
    try:
        contents = page.get_contents() or []
    except Exception:
        return segments

    for obj in contents:
        try:
            raw = doc.xref_stream(obj)
        except Exception:
            continue
        if not raw:
            continue
        payload = raw.decode("latin-1", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        tokens = re.findall(r"/?[A-Za-z]+|[-+]?(?:\d+\.\d*|\.\d+|\d+)", payload)
        if not tokens:
            continue

        ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        path: List[Tuple[float, float]] = []
        current_layer = "content-stream"
        current_cs = ""
        values: List[float] = []
        stack: List[Tuple[Tuple[float, float, float, float, float, float], str, str]] = []

        for token in tokens:
            lower = token.lower()
            if token == "q":
                stack.append((ctm, current_layer, current_cs))
            elif token == "Q" and stack:
                ctm, current_layer, current_cs = stack.pop()
            elif re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", token):
                values.append(float(token))
                continue
            elif token.startswith("/"):
                current_layer = token.lstrip("/") or current_layer
                continue
            elif lower in ("cm", "cs", "sc", "scn", "rg", "g", "k", "gs"):
                if lower == "cm" and len(values) >= 6:
                    matrix = tuple(values[-6:])
                    values = values[:-6]
                    a, b, c_val, d, e, f = ctm
                    ma, mb, mc, md, me, mf = matrix
                    ctm = (
                        a * ma + b * mc,
                        a * mb + b * md,
                        c_val * ma + d * mc,
                        c_val * mb + d * md,
                        e * ma + f * mc + me,
                        e * mb + f * md + mf,
                    )
                elif lower in ("cs", "sc", "scn", "rg", "g", "k"):
                    values = []
                elif lower == "gs":
                    if values:
                        current_layer = str(int(values[-1])) if abs(values[-1] - round(values[-1])) < 1e-9 else str(values[-1])
                        values = []
                continue
            elif lower == "m":
                if len(values) >= 2:
                    x, y = values[-2], values[-1]
                    values = values[:-2]
                    path = [_transform_point((x, y), ctm)]
            elif lower == "l":
                if len(values) >= 2 and path:
                    x, y = values[-2], values[-1]
                    values = values[:-2]
                    path.append(_transform_point((x, y), ctm))
            elif lower == "c":
                if len(values) >= 6 and path:
                    x1, y1, x2, y2, x3, y3 = values[-6:]
                    values = values[:-6]
                    path.extend([
                        _transform_point((x1, y1), ctm),
                        _transform_point((x2, y2), ctm),
                        _transform_point((x3, y3), ctm),
                    ])
            elif lower == "re":
                if len(values) >= 4:
                    x, y, w, h = values[-4:]
                    values = values[:-4]
                    path = [
                        _transform_point((x, y), ctm),
                        _transform_point((x + w, y), ctm),
                        _transform_point((x + w, y + h), ctm),
                        _transform_point((x, y + h), ctm),
                    ]
            elif lower in {"s", "b", "s", "b", "h"}:
                if path:
                    xs = [pt[0] for pt in path]
                    ys = [pt[1] for pt in path]
                    span_x = max(xs) - min(xs)
                    span_y = max(ys) - min(ys)
                    if span_x >= 40.0 or span_y >= 40.0:
                        for seg in _path_to_segments(path, close_path=False):
                            if seg.length >= min_length:
                                segments.append({
                                    "kind": "cut" if span_x >= span_y else "crease",
                                    "geom": seg,
                                    "source": "raw-content",
                                    "layer": current_layer or current_cs or "content-stream",
                                    "rgb": (0.0, 0.0, 0.0),
                                })
                    path = []
                    values = []
            elif lower in {"f", "f*", "n"}:
                path = []
                values = []

        if path:
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            if span_x >= 40.0 or span_y >= 40.0:
                for seg in _path_to_segments(path, close_path=False):
                    if seg.length >= min_length:
                        segments.append({
                            "kind": "cut" if span_x >= span_y else "crease",
                            "geom": seg,
                            "source": "raw-content",
                            "layer": current_layer or current_cs or "content-stream",
                            "rgb": (0.0, 0.0, 0.0),
                        })

    return segments


def _extract_spot_color_segments(page: fitz.Page, doc: fitz.Document | None = None, min_length: float = 15.0) -> List[Dict[str, Any]]:
    """Parse raw PDF content/XObject streams for Spot Color names like Die/Fold and turn their path strokes into segments."""
    if doc is None:
        return []
    segments: List[Dict[str, Any]] = []
    try:
        xref_count = int(doc.xref_length())
    except Exception:
        xref_count = 0

    token_keywords = {"q", "Q", "cm", "CS", "cs", "SC", "sc", "SCN", "scn", "OC", "BDC", "EMC", "m", "l", "c", "v", "y", "re", "h", "S", "s", "B", "b", "f", "F", "f*", "n"}
    for xref in range(1, xref_count):
        try:
            raw = doc.xref_stream(xref)
        except Exception:
            continue
        if raw is None:
            continue
        if isinstance(raw, (bytes, bytearray)):
            payload = raw
        else:
            payload = str(raw).encode("latin-1", "ignore")
        text = payload.decode("latin-1", "ignore")
        if not any(token in text.lower() for token in ("die", "fold", "cut", "crease", "score", "knife", "cm ", "m ", "l ", "re ", "s ", "b ", "scn", "sc ")):
            continue

        tokens = re.findall(r"/?[A-Za-z]+|[-+]?(?:\d+\.\d*|\.\d+|\d+)", text)
        if not tokens:
            continue

        ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        current_cs = ""
        current_layer = ""
        current_color = []
        state_stack: List[Tuple[Tuple[float, float, float, float, float, float], str, str, List[float]]] = []
        path: List[Tuple[float, float]] = []
        path_open = False
        last_move = None

        i = 0
        while i < len(tokens):
            token = tokens[i]
            low = token.lower()

            if token.startswith("/"):
                name = token.lstrip("/")
                if name and _classify_spot_name(name) != "ignore":
                    current_layer = name
                    current_cs = name

            if token == "q":
                state_stack.append((ctm, current_layer, current_cs, list(current_color)))
            elif token == "Q" and state_stack:
                ctm, current_layer, current_cs, current_color = state_stack.pop()
            elif token == "cm" and i + 6 < len(tokens):
                try:
                    matrix = tuple(float(tokens[i + 1 + j]) for j in range(6))
                    a, b, c_val, d, e, f = ctm
                    m_a, m_b, m_c, m_d, m_e, m_f = matrix
                    ctm = (
                        a * m_a + b * m_c,
                        a * m_b + b * m_d,
                        c_val * m_a + d * m_c,
                        c_val * m_b + d * m_d,
                        e * m_a + f * m_c + m_e,
                        e * m_b + f * m_d + m_f,
                    )
                except Exception:
                    pass
                i += 6
            elif token in ("CS", "cs") and i + 1 < len(tokens):
                name = tokens[i + 1].lstrip("/")
                current_cs = name
                i += 1
            elif token in ("SC", "sc", "SCN", "scn", "RG", "rg", "G", "g", "K", "k"):
                values = []
                j = i + 1
                while j < len(tokens) and tokens[j].lower() not in token_keywords:
                    candidate = tokens[j]
                    if candidate.startswith("/"):
                        name = candidate.lstrip("/")
                        current_cs = name
                        if _classify_spot_name(name) != "ignore":
                            current_layer = name
                        j += 1
                        continue
                    try:
                        values.append(float(candidate))
                    except ValueError:
                        break
                    j += 1
                current_color = values
                i = j - 1
            elif token in ("OC", "BDC") and i + 1 < len(tokens):
                guess = tokens[i + 1]
                if guess.startswith("/"):
                    current_layer = guess.lstrip("/")
                i += 1
            elif token == "EMC":
                current_layer = ""
            elif token == "m" and i + 2 < len(tokens):
                try:
                    point = (float(tokens[i + 1]), float(tokens[i + 2]))
                    path = [_transform_point(point, ctm)]
                    path_open = True
                    last_move = path[0]
                except Exception:
                    path = []
                    path_open = False
                i += 2
            elif token == "l" and i + 2 < len(tokens) and path_open:
                try:
                    point = _transform_point((float(tokens[i + 1]), float(tokens[i + 2])), ctm)
                    path.append(point)
                except Exception:
                    pass
                i += 2
            elif token in ("c", "v", "y") and path_open:
                try:
                    coords = []
                    j = i + 1
                    while j < len(tokens) and len(coords) < 6 and tokens[j].lower() not in token_keywords:
                        try:
                            coords.append(float(tokens[j]))
                        except ValueError:
                            break
                        j += 1
                    if len(coords) >= 6:
                        pts = [
                            path[-1],
                            _transform_point((coords[0], coords[1]), ctm),
                            _transform_point((coords[2], coords[3]), ctm),
                            _transform_point((coords[4], coords[5]), ctm),
                        ]
                        path.extend(pts[1:])
                    i = j - 1
                except Exception:
                    pass
            elif token == "re" and i + 4 < len(tokens):
                try:
                    x, y, w, h = [float(tokens[i + 1 + j]) for j in range(4)]
                    rect_points = [
                        _transform_point((x, y), ctm),
                        _transform_point((x + w, y), ctm),
                        _transform_point((x + w, y + h), ctm),
                        _transform_point((x, y + h), ctm),
                    ]
                    path = rect_points
                    path_open = True
                except Exception:
                    pass
                i += 4
            elif token == "h" and path:
                if path[0] != path[-1]:
                    path.append(path[0])
                path_open = False
            elif token in ("S", "s", "B", "b") and path:
                layer_kind = _classify_spot_name(current_layer) or _classify_spot_name(current_cs)
                if layer_kind == "ignore":
                    layer_kind = "ignore"
                if layer_kind in ("cut", "crease"):
                    for seg in _path_to_segments(path, close_path=False):
                        if seg.length >= min_length:
                            segments.append({
                                "kind": layer_kind,
                                "geom": seg,
                                "source": "spot",
                                "layer": current_layer or current_cs or "spot",
                                "rgb": tuple(round(float(v), 4) for v in current_color)[:3] if current_color else (0.0, 0.0, 0.0),
                            })
                path = []
                path_open = False
            elif token in ("f", "F", "f*") and path:
                # Ignore filled artwork/logo paths; structural die/fold geometry is stroke-based.
                path = []
                path_open = False
            elif token == "n":
                path = []
                path_open = False
            i += 1

    return segments


def layer_keyword_kind(layer_name: str) -> str:
    """Classify common packaging-layer names like cut / crease / die / fold / die-line."""
    return classify_name_or_layer(layer_name)


def stroke_color_debug(page: fitz.Page, doc: fitz.Document | None = None) -> List[Dict[str, Any]]:
    """Return real structural line counts for the debug sidebar, naming the semantic layers Fold and Die from the validated Stage 1 extractor."""
    structural = extract_structural_lines(page, doc=doc, min_length=5)
    real_rows: List[Dict[str, Any]] = []

    for kind, label, rgb in (("crease", "Fold", (0.0, 1.0, 0.0)), ("cut", "Die", (1.0, 0.0, 0.0))):
        matches = [entry for entry in structural if entry.get("kind") == kind]
        if not matches:
            continue
        real_rows.append({
            "source": "spot",
            "rgb": rgb,
            "layer": label,
            "total_drawings": len(matches),
            "lines_l": len(matches),
            "rectangles_re": 0,
            "curves_c": 0,
            "quads_qu": 0,
        })

    if not real_rows:
        stats: Dict[Tuple[str, Tuple[float, float, float], str], Dict[str, Any]] = {}
        for drawing in _iter_page_drawings(page, doc):
            layer_name = str(drawing.get("layer") or "").strip()
            color_value = drawing.get("stroke")
            if color_value is None:
                continue
            rgb = normalize_color(color_value)
            if isinstance(color_value, (list, tuple)) and len(color_value) >= 4 and max(color_value) > 1.0:
                rgb = cmyk_to_rgb(color_value)
            key = ("stroke", color_key(rgb), layer_name or "unknown")
            bucket = stats.setdefault(
                key,
                {"source": "stroke", "rgb": color_key(rgb), "layer": layer_name or "unknown", "count": 0, "l": 0, "re": 0, "c": 0, "qu": 0},
            )
            bucket["count"] += 1
            for item in drawing.get("items", []):
                op = item[0]
                if op == "l":
                    bucket["l"] += 1
                elif op == "re":
                    bucket["re"] += 1
                elif op == "c":
                    bucket["c"] += 1
                elif op == "qu":
                    bucket["qu"] += 1
        rows = sorted(stats.values(), key=lambda row: (row["count"], row["layer"]), reverse=True)
        return [{
            "source": row["source"],
            "rgb": tuple(round(float(v), 3) for v in row["rgb"]),
            "layer": row["layer"],
            "total_drawings": row["count"],
            "lines_l": row["l"],
            "rectangles_re": row["re"],
            "curves_c": row["c"],
            "quads_qu": row["qu"],
        } for row in rows[:20]]

    return real_rows


def sample_bezier(points: List[Tuple[float, float]], segments: int = 32) -> List[Tuple[float, float]]:
    """Sample a quadratic or cubic path into a dense polyline."""
    if len(points) < 2:
        return []
    if len(points) == 2:
        return [tuple(points[0]), tuple(points[1])]
    t_values = np.linspace(0.0, 1.0, segments)
    sampled = []
    for t in t_values:
        if len(points) == 3:
            x = (1 - t) ** 2 * points[0][0] + 2 * (1 - t) * t * points[1][0] + t ** 2 * points[2][0]
            y = (1 - t) ** 2 * points[0][1] + 2 * (1 - t) * t * points[1][1] + t ** 2 * points[2][1]
        else:
            x = (
                (1 - t) ** 3 * points[0][0]
                + 3 * (1 - t) ** 2 * t * points[1][0]
                + 3 * (1 - t) * t ** 2 * points[2][0]
                + t ** 3 * points[3][0]
            )
            y = (
                (1 - t) ** 3 * points[0][1]
                + 3 * (1 - t) ** 2 * t * points[1][1]
                + 3 * (1 - t) * t ** 2 * points[2][1]
                + t ** 3 * points[3][1]
            )
        sampled.append((float(x), float(y)))
    return sampled


def discretize_path_item(item: Any) -> List[LineString]:
    """Convert main PDF path item types into line segments suitable for polygonization."""
    op = item[0]
    segments: List[LineString] = []
    if op == "l":
        if len(item) < 3:
            return segments
        a, b = item[1], item[2]
        if hasattr(a, "x") and hasattr(a, "y") and hasattr(b, "x") and hasattr(b, "y"):
            segments.append(LineString([(float(a.x), float(a.y)), (float(b.x), float(b.y))]))
    elif op == "re":
        if len(item) < 2:
            return segments
        rect = item[1]
        if hasattr(rect, "x0") and hasattr(rect, "x1") and hasattr(rect, "y0") and hasattr(rect, "y1"):
            x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            for a, b in zip(points, points[1:]):
                segments.append(LineString([a, b]))
    elif op in ("c", "qu"):
        coords: List[Tuple[float, float]] = []
        for value in item[1:]:
            if hasattr(value, "x") and hasattr(value, "y"):
                coords.append((float(value.x), float(value.y)))
            elif isinstance(value, (int, float)):
                coords.append((float(value), float(value)))
        if len(coords) >= 2:
            if op == "c" and len(coords) >= 4:
                sampled = sample_bezier(coords[:4], segments=18)
                segments.append(LineString(sampled))
            elif op == "qu" and len(coords) >= 3:
                sampled = sample_bezier(coords[:3], segments=18)
                segments.append(LineString(sampled))
    return segments


def _extract_nested_xobject_drawings(page: fitz.Page) -> Iterable[dict]:
    """Best-effort recursive lookup for nested XObject drawings that might hold die-line vectors."""
    try:
        xobjects = page.get_xobjects()
    except Exception:
        xobjects = {}

    if isinstance(xobjects, dict):
        xobject_items = xobjects.values()
    elif isinstance(xobjects, (list, tuple)):
        xobject_items = xobjects
    else:
        xobject_items = []

    for xobj in xobject_items:
        try:
            if hasattr(xobj, "get_drawings"):
                yield from xobj.get_drawings()
        except Exception:
            continue


def classify_stroke_color(
    rgb: Tuple[float, float, float],
    red_threshold: float = 0.7,
    green_threshold: float = 0.5,
) -> str:
    """Classify a stroke as cut or crease using relaxed packaging-die-line color logic."""
    r, g, b = rgb
    # Spot color matching for Die (Stanze) and Fold (Rillen)
    if r > 0.8 and g < 0.3 and b < 0.3:
        return "cut"
    if g > 0.5 and r < 0.2 and b < 0.5:
        return "crease"
    if r > g + 0.25 and r > b + 0.25:
        return "cut"
    if g > r + 0.15 and g > b + 0.15:
        return "crease"
    if r > red_threshold and g < 0.35 and b < 0.35:
        return "cut"
    if g > green_threshold and r < 0.4 and b < 0.4:
        return "crease"
    return "ignore"


def classify_drawing(drawing: dict, rgb: Tuple[float, float, float], red_threshold: float, green_threshold: float, manual_overrides: Dict[str, str] | None = None) -> str:
    """Resolve line type from layer names or OCG/Separation names; RGB thresholds are only a fallback."""
    layer_name = str(drawing.get("layer") or "")
    layer_kind = layer_keyword_kind(layer_name)
    if layer_kind != "ignore":
        return layer_kind

    for candidate in (str(drawing.get("ocg") or ""), str(drawing.get("properties") or ""), str(drawing.get("style") or "")):
        name_kind = classify_name_or_layer(candidate)
        if name_kind != "ignore":
            return name_kind

    if manual_overrides:
        color_key_value = str(color_key(rgb))
        if color_key_value in manual_overrides:
            return manual_overrides[color_key_value]

    return classify_stroke_color(rgb, red_threshold=red_threshold, green_threshold=green_threshold)


def extract_structural_lines(
    page: fitz.Page,
    red_threshold: float = 0.7,
    green_threshold: float = 0.5,
    min_length: float = 15.0,
    manual_overrides: Dict[str, str] | None = None,
    doc: fitz.Document | None = None,
) -> List[Dict[str, Any]]:
    """Extract cut and crease linework from both standard RGB colors and Die/Fold spot-color layers.

    Spot names such as 'Die' and 'Fold' win over RGB detection, and RGB thresholds are only used as a fallback when
    the PDF does not expose explicit Separation/OC layer names.
    """
    if manual_overrides is None:
        manual_overrides = {}

    items: List[Dict[str, Any]] = []
    drawing_candidates: List[dict] = list(_iter_page_drawings(page, doc))

    for drawing in drawing_candidates:
        layer_name = str(drawing.get("layer") or "")
        layer_kind = layer_keyword_kind(layer_name)
        color_val = drawing.get("color") if drawing.get("color") is not None else drawing.get("stroke")
        has_stroke = color_val is not None

        if not has_stroke and layer_kind == "ignore":
            continue

        if has_stroke:
            rgb = normalize_color(color_val)
            if isinstance(color_val, (list, tuple)) and len(color_val) >= 4 and max(color_val) > 1.0:
                rgb = cmyk_to_rgb(color_val)

            color_kind = classify_drawing(drawing, rgb, red_threshold, green_threshold, manual_overrides)
            if layer_kind != "ignore":
                color_kind = layer_kind
            if color_kind == "ignore":
                continue

            for item in drawing.get("items", []):
                if item[0] not in ("l", "re", "c", "qu"):
                    continue
                for segment in discretize_path_item(item):
                    if segment.length >= min_length:
                        items.append({"kind": color_kind, "geom": segment, "rgb": rgb, "source": "stroke", "layer": layer_name or ("Fold" if color_kind == "crease" else "Die")})

        elif layer_kind != "ignore":
            for item in drawing.get("items", []):
                if item[0] not in ("l", "re", "c", "qu"):
                    continue
                for segment in discretize_path_item(item):
                    if segment.length >= min_length:
                        items.append({"kind": layer_kind, "geom": segment, "rgb": (0.0, 0.0, 0.0), "source": "layer", "layer": layer_name})

    if items:
        return items

    for spot in _extract_spot_color_segments(page, doc, min_length=min_length):
        layer_kind = _classify_spot_name(str(spot.get("layer") or ""))
        if layer_kind == "ignore":
            layer_kind = classify_name_or_layer(str(spot.get("layer") or ""))
        if layer_kind == "ignore":
            continue
        for segment in _path_to_segments(list(spot["geom"].coords), close_path=False):
            if segment.length >= min_length:
                items.append({"kind": layer_kind, "geom": segment, "rgb": spot.get("rgb", (0.0, 0.0, 0.0)), "source": "spot", "layer": spot.get("layer", "")})

    for raw in _extract_raw_content_segments(page, doc, min_length=min_length):
        layer_kind = _classify_spot_name(str(raw.get("layer") or ""))
        if layer_kind == "ignore":
            layer_kind = classify_name_or_layer(str(raw.get("layer") or ""))
        if layer_kind == "ignore" and raw.get("source") == "raw-content":
            layer_kind = raw["kind"]
        for segment in _path_to_segments(list(raw["geom"].coords), close_path=False):
            if segment.length >= min_length:
                items.append({"kind": layer_kind, "geom": segment, "rgb": raw.get("rgb", (0.0, 0.0, 0.0)), "source": "raw-content", "layer": raw.get("layer", "")})

    return items


def _cluster_positions(values: Iterable[float], tolerance: float = 2.0) -> List[float]:
    """Cluster values into unique coordinates within a tolerance; used for crease-grid detection."""
    items = sorted(float(v) for v in values)
    if not items:
        return []
    clusters: List[List[float]] = [[items[0]]]
    for value in items[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def measure_pdf_nominal_dims(page: fitz.Page, doc: fitz.Document | None = None) -> Tuple[float, float, float] | None:
    """Auto-suggest nominal Depth, Width, Height (mm) from PDF Fold lines assuming standard 72/25.4 pt/mm."""
    try:
        lines = extract_structural_lines(page, doc=doc, min_length=0)
        green_lines = [entry for entry in lines if entry.get("kind") == "crease" and entry.get("geom") is not None]
        if not green_lines:
            return None
        vert_creases = [
            l["geom"] for l in green_lines
            if abs(l["geom"].coords[-1][1] - l["geom"].coords[0][1]) > 5 * abs(l["geom"].coords[-1][0] - l["geom"].coords[0][0])
            and l["geom"].length > 35.0
        ]
        xs = sorted(_cluster_positions([l.coords[0][0] for l in vert_creases], tolerance=4.0))
        if len(xs) < 2:
            return None
        y_min_v = min(min(l.coords[0][1], l.coords[-1][1]) for l in vert_creases)
        y_max_v = max(max(l.coords[0][1], l.coords[-1][1]) for l in vert_creases)

        crease_diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        std_scale = 72.0 / 25.4
        wide_pt = max(crease_diffs)
        narrow_candidates = [diff for diff in crease_diffs if diff < 0.85 * wide_pt]
        narrow_pt = min(narrow_candidates) if narrow_candidates else min(crease_diffs)

        sugg_w = wide_pt / std_scale
        sugg_d = narrow_pt / std_scale
        sugg_h = (y_max_v - y_min_v) / std_scale

        return float(round(sugg_d)), float(round(sugg_w)), float(round(sugg_h))
    except Exception:
        return None



def _segment_axis(segment: LineString) -> str:
    coords = list(segment.coords)
    if len(coords) < 2:
        return "diag"
    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    if dx > dy:
        return "horizontal"
    if dy > dx:
        return "vertical"
    return "diag"


def _pt_to_mm(pt_value: float, pt_per_mm: float) -> float:
    return float(pt_value / pt_per_mm)


def _page_point_to_mm(point: Tuple[float, float], scale_pt_per_mm: float, origin_pt: Tuple[float, float]) -> Tuple[float, float]:
    """Convert a page point to a normalized mm coordinate system with a Y flip and origin at the front panel's bottom-left."""
    x, y = point
    x_origin, y_origin = origin_pt
    x_mm = (x - x_origin) / float(scale_pt_per_mm)
    y_mm = (y_origin - y) / float(scale_pt_per_mm)
    return (float(x_mm), float(y_mm))


def _geom_to_mm(geom: LineString, scale_pt_per_mm: float, origin_pt: Tuple[float, float]) -> LineString:
    """Apply the front-panel-anchored mm transform with Y-flip to every coordinate in a segment."""
    if geom is None or geom.is_empty:
        return geom
    return LineString([_page_point_to_mm((x, y), scale_pt_per_mm, origin_pt) for x, y in geom.coords])


def _poly_bbox_mm(poly: Polygon) -> Tuple[float, float, float, float]:
    """Return a width/height bbox tuple as (minx, miny, w, h), matching the Stage 1 test contract."""
    minx, miny, maxx, maxy = poly.bounds
    return (float(minx), float(miny), float(maxx - minx), float(maxy - miny))


def _drop_artwork_strokes(
    lines: List[Dict[str, Any]],
    body_panels: List[Dict[str, Any]],
    margin_mm: float = 3.0,
    min_dim_mm: float = 15.0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop any Fold/Die stroke whose connected component lies fully inside a body panel's interior (3mm margin)

    or whose bbox is under 15mm in both dimensions. Print how many were dropped.
    """
    if not lines:
        return [], 0

    body_interiors = [p["polygon_mm"].buffer(-margin_mm) for p in body_panels]

    geoms = [item["geom"] for item in lines if item.get("geom") is not None and not item["geom"].is_empty]
    if not geoms:
        return [], 0

    union_all = unary_union(geoms)
    components = list(union_all.geoms) if hasattr(union_all, "geoms") else [union_all]

    dropped_comps = []
    for comp in components:
        minx, miny, maxx, maxy = comp.bounds
        w = maxx - minx
        h = maxy - miny
        if (w < min_dim_mm and h < min_dim_mm) or any(comp.within(interior) for interior in body_interiors):
            dropped_comps.append(comp)

    dropped_union = unary_union(dropped_comps) if dropped_comps else None

    kept_lines: List[Dict[str, Any]] = []
    dropped_count = 0
    for item in lines:
        geom = item["geom"]
        minx, miny, maxx, maxy = geom.bounds
        w = maxx - minx
        h = maxy - miny
        should_drop = False
        if (w < min_dim_mm and h < min_dim_mm) or any(geom.within(interior) for interior in body_interiors):
            should_drop = True
        elif dropped_union is not None and (geom.within(dropped_union) or geom.intersects(dropped_union)):
            should_drop = True

        if should_drop:
            dropped_count += 1
        else:
            kept_lines.append(item)

    print(f"Dropped {dropped_count} artwork strokes inside body panel interiors or under {min_dim_mm}mm")
    return kept_lines, dropped_count


def detect_layout_rows(green_lines: List[Dict[str, Any]], red_lines: List[Dict[str, Any]]) -> Tuple[int, str]:
    """Detect whether the layout is a single row of body panels or multi-row layout."""
    vert_long = [
        l["geom"] for l in green_lines
        if abs(l["geom"].coords[-1][1] - l["geom"].coords[0][1]) > 3 * abs(l["geom"].coords[-1][0] - l["geom"].coords[0][0])
        and l["geom"].length > 35.0
    ]
    if not vert_long:
        return 1, "No vertical crease lines found"

    max_vert_len = max(l.length for l in vert_long)
    horiz_long = [
        l["geom"] for l in green_lines
        if abs(l["geom"].coords[-1][0] - l["geom"].coords[0][0]) > 3 * abs(l["geom"].coords[-1][1] - l["geom"].coords[0][1])
        and l["geom"].length > 80.0
    ]
    if horiz_long:
        all_hy = [l.coords[0][1] for l in horiz_long]
        h_span = max(all_hy) - min(all_hy)
        ratio = max_vert_len / h_span if h_span > 0 else 1.0
        if ratio < 0.55:
            return 2, f"Multi-row layout detected: vertical crease height ({max_vert_len:.1f}pt) is only {ratio*100:.1f}% of horizontal crease span ({h_span:.1f}pt)"

    return 1, "Single-row layout"


def detect_pdf_layers_summary(page: fitz.Page, doc: fitz.Document | None = None) -> Dict[str, Any]:
    """Detect Fold and Die layers by spot color or OCG name, falling back to RGB red/green."""
    lines = extract_structural_lines(page, doc=doc, min_length=0)
    creases = [l for l in lines if l.get("kind") == "crease"]
    cuts = [l for l in lines if l.get("kind") == "cut"]

    crease_source = creases[0].get("source", "unknown") if creases else "None"
    crease_layer = creases[0].get("layer", "Fold") if creases else "None"
    cut_source = cuts[0].get("source", "unknown") if cuts else "None"
    cut_layer = cuts[0].get("layer", "Die") if cuts else "None"

    return {
        "fold_detected": bool(creases),
        "fold_layer": crease_layer,
        "fold_source": crease_source,
        "fold_count": len(creases),
        "die_detected": bool(cuts),
        "die_layer": cut_layer,
        "die_source": cut_source,
        "die_count": len(cuts),
    }


def _extract_real_flap_polygons(
    body_panels: List[Dict[str, Any]],
    red_lines: List[Dict[str, Any]],
    green_lines: List[Dict[str, Any]] | None = None,
    scale_pt_per_mm: float = 1.0,
    H_mm: float = 180.0,
    D_mm: float = 50.0,
) -> List[Dict[str, Any]]:
    """Build flap polygons from real Die (cut) polylines via per-panel search bands.

    1. For each non-glue body panel, search above top crease (y: H to H+120mm) and below bottom crease (y: -120mm to 0).
    2. Collect Die polylines that intersect the band or touch the crease within snap_tol.
    3. Include horizontal subcreases (e.g. tuck crease) inside the band, grouping fragmented segments and bridging across cut notches.
    4. Snap endpoints, unary_union + polygonize.
    5. Deduce flap kind (dust/lid/lock/tuck), pivot position, fold sign, and z-offset geometrically.
    """
    if green_lines is None:
        green_lines = []

    snap_tol = 3.0 / scale_pt_per_mm
    die_geom = [item["geom"] for item in red_lines if item.get("geom") is not None and not item["geom"].is_empty]
    green_geom = [item["geom"] for item in green_lines if item.get("geom") is not None and not item["geom"].is_empty]
    used_die_indices = set()
    flaps: List[Dict[str, Any]] = []

    band_h = 120.0

    for panel in body_panels:
        pid = panel["id"]
        if pid == "glue":
            continue
        poly = panel["polygon_mm"]
        minx, miny, maxx, maxy = poly.bounds
        x0, x1 = minx, maxx
        panel_w = x1 - x0
        x_off = 0.0 if float(panel.get("x0", 0.0)) < -1e-3 else minx

        for side in ("top", "bottom"):
            if side == "top":
                band = shapely_box(x0 - 1.0, H_mm - 1.0, x1 + 1.0, H_mm + band_h)
                crease = LineString([(x0, H_mm), (x1, H_mm)])
                y_off = H_mm
            else:
                band = shapely_box(x0 - 1.0, -band_h, x1 + 1.0, 1.0)
                crease = LineString([(x0, 0.0), (x1, 0.0)])
                y_off = 0.0

            band_indices = [
                i for i, dl in enumerate(die_geom)
                if dl.intersects(band) or dl.distance(crease) <= snap_tol
            ]
            if not band_indices:
                continue

            used_die_indices.update(band_indices)
            collected = [die_geom[i] for i in band_indices]

            # Find horizontal subcreases inside this band (only for wide panels where lids and tucks exist)
            valid_subcreases = []
            if "side" not in pid:
                cand_segments = []
                for gl in green_geom:
                    if gl.intersects(band):
                        coords = list(gl.coords)
                        dx = abs(coords[-1][0] - coords[0][0])
                        dy = abs(coords[-1][1] - coords[0][1])
                        mid_x = (coords[0][0] + coords[-1][0]) / 2.0
                        if x0 - 2.0 <= mid_x <= x1 + 2.0 and dx > 1.5 * dy and gl.length >= 3.0:
                            cy = (coords[0][1] + coords[-1][1]) / 2.0
                            if side == "top" and cy > H_mm + 5.0:
                                cand_segments.append((cy, gl))
                            elif side == "bottom" and cy < -5.0:
                                cand_segments.append((cy, gl))

                # Group fragmented crease segments around the same Y coordinate (within 2mm)
                grouped_subcreases = []
                for cy, gl in cand_segments:
                    matched = False
                    for g in grouped_subcreases:
                        if abs(cy - g["y"]) <= 2.0:
                            g["lines"].append(gl)
                            g["min_x"] = min(g["min_x"], gl.bounds[0])
                            g["max_x"] = max(g["max_x"], gl.bounds[2])
                            g["tot_len"] += gl.length
                            matched = True
                            break
                    if not matched:
                        grouped_subcreases.append({
                            "y": cy,
                            "lines": [gl],
                            "min_x": gl.bounds[0],
                            "max_x": gl.bounds[2],
                            "tot_len": gl.length,
                        })

                # Filter valid subcreases: total length >= 0.25 * panel_w or span >= 0.4 * panel_w
                for g in grouped_subcreases:
                    span = g["max_x"] - g["min_x"]
                    if g["tot_len"] >= 0.25 * panel_w or span >= 0.4 * panel_w:
                        valid_subcreases.append(g)

                # A lid has ONE tuck flap! If multiple subcreases exist on a lid (e.g. front_top having 30mm and 65mm):
                # Filter to keep the tuck crease closest to D_mm (outer tuck crease).
                if len(valid_subcreases) > 1 and D_mm > 0:
                    valid_subcreases.sort(key=lambda s: abs(abs(s["y"] - y_off) - D_mm))
                    chosen = valid_subcreases[0]
                    ignored = valid_subcreases[1:]
                    for ig in ignored:
                        print(f"[{pid}_{side}] Removing duplicate subcrease at y={ig['y']:.2f} (dist={abs(ig['y'] - y_off):.1f}mm), keeping tuck crease at y={chosen['y']:.2f} (dist={abs(chosen['y'] - y_off):.1f}mm)")
                    valid_subcreases = [chosen]

            # Build subcrease line spanning across the panel width to bridge cut notches
            sub_lines = []
            sub_y_list = []
            for g in valid_subcreases:
                sy = g["y"]
                sub_y_list.append(sy)
                bridge = LineString([(x0 - 2.0, sy), (x1 + 2.0, sy)])
                sub_lines.append(bridge)

            all_lines = collected + [crease] + sub_lines

            # Snap endpoints within snap_tol
            pts = [pt for l in all_lines for pt in (l.coords[0], l.coords[-1])]
            clusters: List[Dict[str, Any]] = []
            for p in pts:
                matched = False
                for c in clusters:
                    if ((p[0] - c["x"]) ** 2 + (p[1] - c["y"]) ** 2) ** 0.5 <= snap_tol:
                        c["pts"].append(p)
                        c["x"] = sum(x for x, y in c["pts"]) / len(c["pts"])
                        c["y"] = sum(y for x, y in c["pts"]) / len(c["pts"])
                        matched = True
                        break
                if not matched:
                    clusters.append({"x": p[0], "y": p[1], "pts": [p]})

            def snap_pt(p: Tuple[float, float]) -> Tuple[float, float]:
                for c in clusters:
                    if ((p[0] - c["x"]) ** 2 + (p[1] - c["y"]) ** 2) ** 0.5 <= snap_tol:
                        return (c["x"], c["y"])
                return p

            snapped = [
                LineString([snap_pt(l.coords[0]), *l.coords[1:-1], snap_pt(l.coords[-1])])
                for l in all_lines
            ]

            polys = list(polygonize(unary_union(snapped)))

            for p in polys:
                if p.area < 50.0:
                    continue

                inter_crease = p.boundary.intersection(crease.buffer(snap_tol))
                shared_c = inter_crease.length if not inter_crease.is_empty else 0.0

                shared_sub = 0.0
                matched_sub_y = None
                for sy, sl in zip(sub_y_list, sub_lines):
                    inter_sub = p.boundary.intersection(sl.buffer(snap_tol))
                    l_s = inter_sub.length if not inter_sub.is_empty else 0.0
                    if l_s > shared_sub:
                        shared_sub = l_s
                        matched_sub_y = sy

                if shared_c >= 0.4 * panel_w:
                    base_fid = f"{pid}_{side}"
                    fid = base_fid
                    suffix = 1
                    while any(f["id"] == fid for f in flaps):
                        fid = f"{base_fid}_{suffix}"
                        suffix += 1
                    parent_id = pid
                    kind = "dust" if "side" in pid else ("lid" if side == "top" else "lock")
                    hinge_y = H_mm if side == "top" else 0.0
                    hinge_p0 = (x0, hinge_y)
                    hinge_p1 = (x1, hinge_y)
                    # Flap above crease folds toward -Z (sign = -1.0); flap below crease folds toward -Z (sign = +1.0)
                    sign = -1.0 if p.centroid.y > hinge_y else 1.0
                    pivot_pos = [0.0, hinge_y, 0.0]
                    f_y_off = hinge_y

                    if kind == "dust":
                        z_offset, layer_index = 0.00, 0
                    elif side == "top":
                        z_offset, layer_index = 0.04, 1
                    else:
                        if pid == "front":
                            z_offset, layer_index = 0.06, 2
                        elif pid == "back":
                            z_offset, layer_index = 0.03, 1
                        else:
                            z_offset, layer_index = 0.04, 1

                elif shared_sub >= 0.4 * panel_w and matched_sub_y is not None:
                    base_fid = f"{pid}_{side}_tuck" if (pid != "front" or side != "top") else "top_tuck"
                    fid = base_fid
                    suffix = 1
                    while any(f["id"] == fid for f in flaps):
                        fid = f"{base_fid}_{suffix}"
                        suffix += 1
                    parent_id = f"{pid}_{side}"
                    kind = "tuck"
                    hinge_y = matched_sub_y
                    hinge_p0 = (x0, hinge_y)
                    hinge_p1 = (x1, hinge_y)
                    # Tuck flap above its hinge folds toward -Z (sign = -1.0); below folds toward -Z (sign = +1.0)
                    sign = -1.0 if p.centroid.y > hinge_y else 1.0
                    # Cap tuck hinge travel to depth D so tuck flap stays inside the opposite wall
                    if side == "top":
                        eff_hinge = min(hinge_y - H_mm, D_mm - 1.0) if D_mm > 0 else (hinge_y - H_mm)
                        pivot_pos = [0.0, round(float(eff_hinge), 2), 0.0]
                    else:
                        eff_hinge = max(hinge_y, -(D_mm - 1.0)) if D_mm > 0 else hinge_y
                        pivot_pos = [0.0, round(float(eff_hinge), 2), 0.0]
                    f_y_off = hinge_y
                    z_offset, layer_index = 0.00, 0
                else:
                    continue

                # Align coordinates on crease exactly
                adj_coords = []
                for vx, vy in p.exterior.coords:
                    if abs(vy - (H_mm if side == "top" else 0.0)) <= snap_tol and not fid.endswith("_tuck"):
                        adj_coords.append((vx, H_mm if side == "top" else 0.0))
                    elif matched_sub_y is not None and abs(vy - matched_sub_y) <= snap_tol:
                        adj_coords.append((vx, matched_sub_y))
                    else:
                        adj_coords.append((vx, vy))

                p_adj = Polygon(adj_coords)
                poly_local = [[round(float(vx - x_off), 3), round(float(vy - f_y_off), 3)] for vx, vy in adj_coords]

                flaps.append({
                    "id": fid,
                    "parent": parent_id,
                    "kind": kind,
                    "polygon_mm": p_adj,
                    "bbox_mm": _poly_bbox_mm(p_adj),
                    "hinge_p0": hinge_p0,
                    "hinge_p1": hinge_p1,
                    "sign": sign,
                    "x_offset": x_off,
                    "y_offset": f_y_off,
                    "polygon_local": poly_local,
                    "pivot_position": pivot_pos,
                    "num_vertices": len(p_adj.exterior.coords),
                    "z_offset": z_offset,
                    "layer_index": layer_index,
                })

    print(f"Die polylines used: {len(used_die_indices)}, ignored: {len(die_geom) - len(used_die_indices)}")

    def flap_sort_key(f: Dict[str, Any]) -> Tuple[int, int]:
        p_order = ["side_a", "front", "side_b", "back", "glue"]
        parent_id = f.get("parent", "")
        p_idx = 9
        for idx, base in enumerate(p_order):
            if base in parent_id:
                p_idx = idx
                break
        fid = f["id"]
        if "top" in fid and "tuck" in fid:
            s_idx = 1
        elif "top" in fid:
            s_idx = 0
        elif "bottom" in fid and "tuck" in fid:
            s_idx = 3
        elif "tuck" in fid:
            s_idx = 1
        else:
            s_idx = 2
        return (p_idx, s_idx)

    flaps.sort(key=flap_sort_key)
    return flaps


def _extract_raw_geometry_dump(page: fitz.Page, doc: fitz.Document | None = None) -> List[Dict[str, Any]]:
    """Return raw Fold/Die subpath stats before any minimum-length filtering or panel synthesis."""
    rows: List[Dict[str, Any]] = []
    for kind in ("crease", "cut"):
        segments = [entry for entry in extract_structural_lines(page, doc=doc, min_length=0) if entry.get("kind") == kind]
        if not segments:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for entry in segments:
            geom = entry["geom"]
            for x, y in geom.coords:
                xs.append(float(x))
                ys.append(float(y))
        bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else (0.0, 0.0, 0.0, 0.0)
        rows.append({
            "kind": kind,
            "layer": "Fold" if kind == "crease" else "Die",
            "subpaths": len(segments),
            "segments": len(segments),
            "bbox_pt": bbox,
        })
    return rows


def build_stage1_box_geometry(
    page: fitz.Page,
    doc: fitz.Document | None = None,
    dimensions: Tuple[float, float, float] | None = None,
) -> Dict[str, Any]:
    """Build the Stage 1 carton from the actual Fold grid and Die flap outlines in the PDF."""
    lines = extract_structural_lines(page, doc=doc, min_length=0)
    green_lines = [entry for entry in lines if entry.get("kind") == "crease" and entry.get("geom") is not None]
    red_lines = [entry for entry in lines if entry.get("kind") == "cut" and entry.get("geom") is not None]
    if not green_lines:
        raise ValueError("No Fold (crease) lines were detected in the uploaded die-line PDF.")
    if not red_lines:
        raise ValueError("No Die (cut) lines were detected in the uploaded die-line PDF.")

    # Check for multi-row layout
    rows_detected, layout_reason = detect_layout_rows(green_lines, red_lines)
    if rows_detected > 1:
        raise ValueError(f"Unsupported layout: Multi-row carton layouts ({rows_detected} rows detected) are not supported yet.")

    # Vertical creases anchor body panels
    vert_creases = [
        l["geom"] for l in green_lines
        if abs(l["geom"].coords[-1][1] - l["geom"].coords[0][1]) > 5 * abs(l["geom"].coords[-1][0] - l["geom"].coords[0][0])
        and l["geom"].length > 35.0
    ]
    xs = sorted(_cluster_positions([l.coords[0][0] for l in vert_creases], tolerance=4.0))
    if len(xs) < 3:
        raise ValueError(f"Validation failed: expected at least 3 internal vertical crease lines for body panels, found {len(xs)}.")

    y_min_v = min(min(l.coords[0][1], l.coords[-1][1]) for l in vert_creases)
    y_max_v = max(max(l.coords[0][1], l.coords[-1][1]) for l in vert_creases)

    # Find outer left and right cut lines
    left_cuts = [
        l["geom"] for l in red_lines
        if l["geom"].bounds[2] <= xs[0] + 5.0
        and not (l["geom"].bounds[3] < y_min_v - 20.0 or l["geom"].bounds[1] > y_max_v + 20.0)
    ]
    right_cuts = [
        l["geom"] for l in red_lines
        if l["geom"].bounds[0] >= xs[-1] - 5.0
        and not (l["geom"].bounds[3] < y_min_v - 20.0 or l["geom"].bounds[1] > y_max_v + 20.0)
    ]
    x_left = min(l.bounds[0] for l in left_cuts) if left_cuts else xs[0]
    x_right = max(l.bounds[2] for l in right_cuts) if right_cuts else xs[-1]

    all_x = [x_left] + xs + [x_right]
    widths_pt = [all_x[i + 1] - all_x[i] for i in range(len(all_x) - 1)]

    # Determine panels sequence and glue panel placement
    w_end0 = widths_pt[0]
    w_end1 = widths_pt[-1]
    has_glue_left = (w_end0 < 0.6 * w_end1)
    has_glue_right = (w_end1 < 0.6 * w_end0)

    if has_glue_left:
        panel_order = ["glue", "front", "side_b", "back", "side_a"]
    elif has_glue_right:
        panel_order = ["side_a", "front", "side_b", "back", "glue"]
    else:
        panel_order = ["front", "side_b", "back", "side_a"]

    front_idx = panel_order.index("front")
    front_w_pt = float(widths_pt[front_idx])

    # Dimensions & scale
    if dimensions is not None:
        D_mm, W_mm, H_mm = [float(v) for v in dimensions]
    else:
        doc_filename = getattr(doc, "name", None) if doc is not None else None
        parsed = parse_dimensions_from_filename(doc_filename)
        if parsed is not None:
            D_mm, W_mm, H_mm = parsed
        else:
            scale_est = 72.0 / 25.4
            W_mm = round(front_w_pt / scale_est, 1)
            narrow_w_pt = widths_pt[panel_order.index("side_b")] if "side_b" in panel_order else widths_pt[0]
            D_mm = round(narrow_w_pt / scale_est, 1)
            H_mm = round((y_max_v - y_min_v) / scale_est, 1)

    scale_pt_per_mm = front_w_pt / W_mm if W_mm > 0 else 72.0 / 25.4
    measured_H_mm = (y_max_v - y_min_v) / scale_pt_per_mm
    narrow_idx = panel_order.index("side_b") if "side_b" in panel_order else 0
    measured_D_mm = widths_pt[narrow_idx] / scale_pt_per_mm

    std_scale = 72.0 / 25.4
    sugg_w = round(front_w_pt / std_scale)
    sugg_d = round(widths_pt[narrow_idx] / std_scale)
    sugg_h = round((y_max_v - y_min_v) / std_scale)

    ratio_wd_meas = front_w_pt / widths_pt[narrow_idx] if widths_pt[narrow_idx] > 0 else 1.0
    ratio_wh_meas = front_w_pt / (y_max_v - y_min_v) if (y_max_v - y_min_v) > 0 else 1.0
    ratio_dh_meas = widths_pt[narrow_idx] / (y_max_v - y_min_v) if (y_max_v - y_min_v) > 0 else 1.0

    ratio_wd_entered = W_mm / D_mm if D_mm > 0 else 1.0
    ratio_wh_entered = W_mm / H_mm if H_mm > 0 else 1.0
    ratio_dh_entered = D_mm / H_mm if H_mm > 0 else 1.0

    diff_wd = abs(ratio_wd_entered - ratio_wd_meas) / ratio_wd_meas if ratio_wd_meas > 0 else 0.0
    diff_wh = abs(ratio_wh_entered - ratio_wh_meas) / ratio_wh_meas if ratio_wh_meas > 0 else 0.0
    diff_dh = abs(ratio_dh_entered - ratio_dh_meas) / ratio_dh_meas if ratio_dh_meas > 0 else 0.0

    dimension_warnings = []
    if diff_wd > 0.05 or diff_wh > 0.05 or diff_dh > 0.05:
        dimension_warnings.append(
            f"Measured fold-line proportions (wide panel : narrow panel : height = "
            f"{ratio_wd_meas:.2f} : 1.00 : {1.0 / ratio_dh_meas:.2f}) "
            f"differ from entered proportions ({ratio_wd_entered:.2f} : 1.00 : {1.0 / ratio_dh_entered:.2f}) by more than 5%. "
            f"Suggested likely dimensions from dieline: {sugg_d:.0f} x {sugg_w:.0f} x {sugg_h:.0f} mm."
        )
    elif D_mm > 0 and abs(measured_D_mm - D_mm) / D_mm > 0.05:
        dimension_warnings.append(
            f"Measured depth ({measured_D_mm:.1f} mm) differs from entered depth ({D_mm:.1f} mm) by more than 5%. "
            f"Suggested dimensions: {sugg_d:.0f} x {sugg_w:.0f} x {sugg_h:.0f} mm."
        )
    elif H_mm > 0 and abs(measured_H_mm - H_mm) / H_mm > 0.05:
        dimension_warnings.append(
            f"Measured height ({measured_H_mm:.1f} mm) differs from entered height ({H_mm:.1f} mm) by more than 5%. "
            f"Suggested dimensions: {sugg_d:.0f} x {sugg_w:.0f} x {sugg_h:.0f} mm."
        )

    # Origin at front panel bottom-left
    x_origin_pt = float(all_x[front_idx])
    y_origin_pt = float(y_max_v)
    origin_pt = (x_origin_pt, y_origin_pt)

    def pt_to_mm_x(pt: float) -> float:
        return float((pt - x_origin_pt) / scale_pt_per_mm)

    # Convert all linework to unified mm system with origin at front panel bottom-left
    green_lines_mm = [{**entry, "geom": _geom_to_mm(entry["geom"], scale_pt_per_mm, origin_pt)} for entry in green_lines]
    red_lines_mm = [{**entry, "geom": _geom_to_mm(entry["geom"], scale_pt_per_mm, origin_pt)} for entry in red_lines]

    body_panels: List[Dict[str, Any]] = []
    for idx, pid in enumerate(panel_order):
        x0_mm = pt_to_mm_x(all_x[idx])
        x1_mm = pt_to_mm_x(all_x[idx + 1])
        w_mm = x1_mm - x0_mm
        poly = Polygon([(x0_mm, 0.0), (x1_mm, 0.0), (x1_mm, measured_H_mm), (x0_mm, measured_H_mm)])

        # Kinematic pivot / parent setup
        if pid == "front":
            parent = None
            piv = [0.0, 0.0, 0.0]
            sign = 0.0
            glue_off = 0.0
            local_x0, local_x1 = 0.0, w_mm
        elif pid == "glue" and has_glue_left:
            parent = "front"
            piv = [0.0, 0.0, 0.0]
            sign = -1.0
            glue_off = -0.20
            local_x0, local_x1 = -w_mm, 0.0
        elif pid == "side_a" and has_glue_right:
            parent = "front"
            piv = [0.0, 0.0, 0.0]
            sign = -1.0
            glue_off = 0.0
            local_x0, local_x1 = -w_mm, 0.0
        elif pid == "side_b":
            parent = "front"
            piv = [front_w_pt / scale_pt_per_mm, 0.0, 0.0]
            sign = 1.0
            glue_off = 0.0
            local_x0, local_x1 = 0.0, w_mm
        elif pid == "back":
            parent = "side_b"
            piv = [widths_pt[panel_order.index("side_b")] / scale_pt_per_mm, 0.0, 0.0]
            sign = 1.0
            glue_off = 0.0
            local_x0, local_x1 = 0.0, w_mm
        elif pid == "side_a" and has_glue_left:
            parent = "back"
            piv = [widths_pt[panel_order.index("back")] / scale_pt_per_mm, 0.0, 0.0]
            sign = 1.0
            glue_off = 0.0
            local_x0, local_x1 = 0.0, w_mm
        elif pid == "glue" and has_glue_right:
            parent = "back"
            back_w = widths_pt[panel_order.index("back")] / scale_pt_per_mm
            front_w = front_w_pt / scale_pt_per_mm
            glue_off = back_w - (front_w - 0.20)
            piv = [back_w, 0.0, 0.0]
            sign = 1.0
            local_x0, local_x1 = 0.0, w_mm
        else:
            parent = "front"
            piv = [0.0, 0.0, 0.0]
            sign = 1.0
            glue_off = 0.0
            local_x0, local_x1 = 0.0, w_mm

        body_panels.append({
            "id": pid,
            "parent": parent,
            "polygon_mm": poly,
            "bbox_mm": _poly_bbox_mm(poly),
            "hinge_p0": (x0_mm, 0.0),
            "hinge_p1": (x0_mm, measured_H_mm),
            "x0": local_x0,
            "x1": local_x1,
            "width": w_mm,
            "height": measured_H_mm,
            "pivot_position": piv,
            "fold_sign": sign,
            "glue_offset": glue_off,
        })

    # Extract real flaps from Die cut lines
    flaps = _extract_real_flap_polygons(body_panels, red_lines_mm, green_lines_mm, scale_pt_per_mm, measured_H_mm, measured_D_mm)

    all_green_points = [(float(x), float(y)) for entry in green_lines for x, y in entry["geom"].coords]
    all_xs = [x for x, _ in all_green_points]
    all_ys = [y for _, y in all_green_points]

    box = {
        "D_mm": float(D_mm),
        "W_mm": float(W_mm),
        "H_mm": float(H_mm),
        "measured_D_mm": float(measured_D_mm),
        "measured_H_mm": float(measured_H_mm),
        "dimension_warnings": dimension_warnings,
        "scale_pt_per_mm": float(scale_pt_per_mm),
        "origin_pt": origin_pt,
        "body_panels": body_panels,
        "flaps": flaps,
        "panels": body_panels + flaps,
        "red_cut_segments": red_lines_mm,
        "green_crease_segments": green_lines_mm,
        "x_cluster_pt": xs,
        "y_cluster_pt": sorted(_cluster_positions(all_ys, tolerance=3.0)),
        "pdf_bounds_pt": (min(all_xs), min(all_ys), max(all_xs), max(all_ys)),
        "raw_dump": _extract_raw_geometry_dump(page, doc),
    }
    return box


def validate_stage1_box_geometry(box: Dict[str, Any]) -> None:
    """Validate the Stage 1 geometry with readable assertion messages for Streamlit and pytest."""
    body = box.get("body_panels", [])
    if len(body) not in (4, 5):
        raise ValueError(f"Stage 1 validation failed: expected 4 or 5 body panels in single row, found {len(body)}")

    H_mm = box.get("H_mm", 180.0)
    widths = []
    heights = []
    for panel in body:
        poly = panel["polygon_mm"]
        if poly.is_empty:
            raise ValueError(f"Stage 1 validation failed: panel {panel['id']} has an empty polygon")
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        widths.append(width)
        heights.append(height)
        if width < 1.0 or height < 1.0:
            raise ValueError(f"Stage 1 validation failed: panel {panel['id']} has invalid size {width}x{height} mm")

    flaps = box.get("flaps", [])
    if len(flaps) < 4:
        raise ValueError(f"Stage 1 validation failed: expected at least 4 flaps, found {len(flaps)}")

    panel_map = {panel["id"]: panel for panel in body + flaps}
    for flap in flaps:
        if not flap.get("parent"):
            raise ValueError("Stage 1 validation failed: a flap is missing its parent panel")
        if flap["parent"] not in panel_map:
            raise ValueError(f"Stage 1 validation failed: flap parent {flap['parent']} not found in panels")
        poly = flap["polygon_mm"]
        if poly.is_empty or poly.area <= 0:
            raise ValueError(f"Stage 1 validation failed: flap {flap['id']} has zero area")
        p0 = flap.get("hinge_p0")
        p1 = flap.get("hinge_p1")
        if not p0 or not p1:
            raise ValueError(f"Stage 1 validation failed: flap {flap['id']} missing hinge endpoints")
        crease_line = LineString([p0, p1])
        shared_boundary = poly.boundary.intersection(crease_line.buffer(0.5))
        shared_length = shared_boundary.length if not shared_boundary.is_empty else 0.0
        if shared_length < 5.0:
            raise ValueError(
                f"Stage 1 validation failed: flap {flap['id']} shares only {shared_length:.2f}mm "
                f"with its parent crease (expected >= 5.0mm)"
            )

    main_body_heights = [panel["bbox_mm"][3] for panel in body if panel["id"] != "glue"]
    if not main_body_heights or not all(h > 1.0 for h in main_body_heights):
        raise ValueError(f"Stage 1 validation failed: body panels have invalid heights: {main_body_heights}")


def _render_stage1_debug_plot(page: fitz.Page, box: Dict[str, Any]) -> plt.Figure:
    """Draw the PDF-derived Fold/Die lines and the generated panels/flaps in a single mm coordinate system."""
    fig, ax = plt.subplots(figsize=(12, 7))
    red_lines = box.get("red_cut_segments", [])
    green_lines = box.get("green_crease_segments", [])

    def plot_segment(segment: LineString, color: str, label: str | None = None) -> None:
        coords = list(segment.coords)
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        ax.plot(xs, ys, color=color, linewidth=1.5, label=label)

    for entry in red_lines:
        plot_segment(entry["geom"], "red", "Die")
    for entry in green_lines:
        plot_segment(entry["geom"], "green", "Fold")

    for panel in box["body_panels"]:
        poly = panel["polygon_mm"]
        x, y = poly.exterior.coords[0]
        ax.add_patch(plt.Polygon(list(poly.exterior.coords), fill=True, facecolor="lightsteelblue", edgecolor="navy", alpha=0.25, linewidth=1.5))
        ax.text(x + 2, y + 3, panel["id"], fontsize=9, color="navy")

    for flap in box["flaps"]:
        poly = flap["polygon_mm"]
        ax.add_patch(plt.Polygon(list(poly.exterior.coords), fill=True, facecolor="bisque", edgecolor="darkorange", alpha=0.35, linewidth=1.4))
        cx, cy = poly.centroid.coords[0]
        ax.text(cx, cy, flap["id"], fontsize=7, color="darkorange", ha="center", va="center")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Stage 1: PDF-derived fold and cut geometry (front panel origin)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    return fig


def _stage1_debug_table(box: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for panel in box["panels"]:
        poly = panel["polygon_mm"]
        minx, miny, maxx, maxy = poly.bounds
        bbox = (maxx - minx, maxy - miny)
        rows.append(
            {
                "id": panel["id"],
                "parent": panel.get("parent"),
                "hinge_p0": panel.get("hinge_p0"),
                "hinge_p1": panel.get("hinge_p1"),
                "bbox_w_mm": round(float(bbox[0]), 2),
                "bbox_h_mm": round(float(bbox[1]), 2),
                "area_mm2": round(float(poly.area), 2),
                "num_vertices": len(poly.exterior.coords),
            }
        )
    return rows


def snap_line_endpoints(lines: List[LineString], tolerance: float = 3.0) -> List[LineString]:
    """Snap endpoint coordinates within tolerance so small flap gaps close into valid box polygons."""
    if not lines:
        return []

    merged = []
    for _ in range(2):
        points: List[Tuple[float, float]] = []
        for line in lines:
            points.extend(list(line.coords))

        next_lines: List[LineString] = []
        for line in lines:
            coords = list(line.coords)
            if not coords:
                continue
            for idx, point in enumerate(coords):
                matches = []
                for ref in points:
                    if ref == point:
                        continue
                    if abs(ref[0] - point[0]) <= tolerance and abs(ref[1] - point[1]) <= tolerance:
                        matches.append(ref)
                if matches:
                    nearest = min(matches, key=lambda ref: (ref[0] - point[0]) ** 2 + (ref[1] - point[1]) ** 2)
                    coords[idx] = nearest
            next_lines.append(LineString(coords))
        lines = next_lines
        merged = next_lines

    return merged


def build_panels(
    lines: List[Dict[str, Any]],
    snap_tolerance: float = 3.0,
    min_length: float = 15.0,
) -> List[Dict[str, Any]]:
    """Polygonize all structural linework into individual box panels using unary_union for stability."""
    structural = []
    for entry in lines:
        geom = entry["geom"]
        if geom.length >= min_length:
            structural.append(geom)

    if not structural:
        return []

    snapped = snap_line_endpoints(structural, tolerance=snap_tolerance)
    merged = unary_union(snapped)
    if hasattr(merged, "geoms"):
        polygon_candidates = list(polygonize(merged.geoms))
    else:
        polygon_candidates = list(polygonize([merged]))

    if not polygon_candidates:
        return []

    panels = []
    for idx, polygon in enumerate(polygon_candidates):
        if polygon.is_valid and polygon.area > 300:
            panels.append({
                "id": idx,
                "geom": polygon,
                "area": float(polygon.area),
                "center": tuple(polygon.centroid.coords[0]),
            })
    if not panels:
        return []

    return sorted(panels, key=lambda item: item["area"], reverse=True)


def build_hierarchy(panels: List[Dict[str, Any]], flip_direction: bool = False) -> Dict[str, Any]:
    """Build a simple BFS hierarchy tree from the largest face as the root dock."""
    if not panels:
        return {"root": None, "edges": [], "panels": []}

    graph = nx.Graph()
    for panel in panels:
        graph.add_node(panel["id"])

    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            a = panels[i]["geom"]
            b = panels[j]["geom"]
            raw = a.boundary.intersection(b.boundary)
            if raw.is_empty:
                continue
            shared = raw.length
            if shared > 1.0:
                graph.add_edge(panels[i]["id"], panels[j]["id"], shared_length=float(shared), geometry=raw)

    connected = sorted(nx.connected_components(graph), key=lambda component: len(component), reverse=True)
    keep_ids = set(connected[0]) if connected else set()
    panels = [panel for panel in panels if panel["id"] in keep_ids]
    if not panels:
        return {"root": None, "edges": [], "panels": []}
    graph = graph.subgraph(keep_ids).copy()

    root = max(panels, key=lambda item: item["area"])["id"]
    if root not in graph:
        root = panels[0]["id"]

    parents = {root: None}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)

    for panel in panels:
        panel_id = panel["id"]
        if panel_id not in parents:
            parents[panel_id] = root

    panel_payload = []
    for panel in panels:
        panel_id = panel["id"]
        panel_coords = list(panel["geom"].exterior.coords)
        points = [[float(x), float(y)] for x, y in panel_coords[:-1]]
        center_x = sum(point[0] for point in points) / max(1, len(points))
        center_y = sum(point[1] for point in points) / max(1, len(points))
        centered = [[x - center_x, y - center_y] for x, y in points]

        # Normalize UVs from the panel's local PDF coordinates.
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 0.0001)
        range_y = max(max_y - min_y, 0.0001)
        uvs = []
        for x, y in points:
            u = (x - min_x) / range_x
            v = 1.0 - (y - min_y) / range_y
            uvs.extend([u, v])

        payload = {
            "id": panel_id,
            "parent": parents.get(panel_id),
            "points": centered,
            "uv": uvs,
            "center": [float(center_x), float(center_y)],
            "area": float(panel["area"]),
            "fold_angle": 90.0,
            "flip": bool(flip_direction),
            "hinge": [[0.0, 0.0], [1.0, 0.0]],
        }

        if panel_id != root:
            parent_id = parents[panel_id]
            parent_panel = next(item for item in panels if item["id"] == parent_id)
            if graph.has_edge(panel_id, parent_id):
                edge = graph.edges[panel_id, parent_id]["geometry"]
                if hasattr(edge, "geoms"):
                    edge = edge.geoms[0] if edge.geoms else edge
                try:
                    line = list(edge.coords)
                except NotImplementedError:
                    line = []
                if len(line) >= 2:
                    payload["hinge"] = [[float(x), float(y)] for x, y in line[:2]]
        panel_payload.append(payload)

    return {"root": root, "edges": list(graph.edges(data=True)), "panels": panel_payload}


def render_pdf_texture(page: fitz.Page) -> str:
    """Render a crisp 300 DPI page PNG and return a Base64 string."""
    zoom = 300 / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_panel_textures(box: Dict[str, Any], page: fitz.Page, dpi: int = 200) -> Dict[str, str]:
    """Render the PDF page at dpi (default 200 DPI) and crop individual textures for all panels and flaps.

    Converts mm bounding boxes back to page pixel coordinates using scale_pt_per_mm and origin_pt.
    """
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    scale_pt_per_mm = float(box["scale_pt_per_mm"])
    origin_pt = box.get("origin_pt")
    if not origin_pt:
        x_orig = float(box["x_cluster_pt"][0]) if box.get("x_cluster_pt") else 0.0
        y_orig = float(max(box["y_cluster_pt"])) if box.get("y_cluster_pt") else 0.0
        origin_pt = (x_orig, y_orig)

    origin_x_pt, origin_y_pt = origin_pt

    def crop_bbox(minx: float, miny: float, maxx: float, maxy: float) -> str:
        # minx, miny is bottom-left in mm; maxx, maxy is top-right in mm
        x0_pt = origin_x_pt + minx * scale_pt_per_mm
        x1_pt = origin_x_pt + maxx * scale_pt_per_mm
        y_top_pt = origin_y_pt - maxy * scale_pt_per_mm
        y_bottom_pt = origin_y_pt - miny * scale_pt_per_mm

        px0 = max(0, min(pix.width, int(round(x0_pt * zoom))))
        px1 = max(0, min(pix.width, int(round(x1_pt * zoom))))
        py0 = max(0, min(pix.height, int(round(y_top_pt * zoom))))
        py1 = max(0, min(pix.height, int(round(y_bottom_pt * zoom))))

        if px1 <= px0 or py1 <= py0:
            return ""

        crop = page_img.crop((px0, py0, px1, py1))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    textures: Dict[str, str] = {}
    for panel in box.get("body_panels", []):
        minx, miny, maxx, maxy = panel["polygon_mm"].bounds
        tex = crop_bbox(minx, miny, maxx, maxy)
        if tex:
            textures[panel["id"]] = tex

    for flap in box.get("flaps", []):
        minx, miny, maxx, maxy = flap["polygon_mm"].bounds
        tex = crop_bbox(minx, miny, maxx, maxy)
        if tex:
            textures[flap["id"]] = tex

    return textures


def build_stage2_body_panel_model(box: Dict[str, Any], page: fitz.Page | None = None, dpi: int = 200) -> Dict[str, Any]:
    """Turn the Stage 1 body-panel geometry into the JSON consumed by the JS fold-tree viewer."""
    panel_map: Dict[str, Dict[str, Any]] = {}
    for panel in box.get("body_panels", []):
        poly = panel["polygon_mm"]
        minx, miny, maxx, maxy = poly.bounds
        width = float(maxx - minx)
        height = float(maxy - miny)
        panel_map[panel["id"]] = {
            "id": panel["id"],
            "parent": panel.get("parent"),
            "x0": float(panel.get("x0", 0.0)),
            "x1": float(panel.get("x1", width)),
            "height": float(panel.get("height", height)),
            "width": float(panel.get("width", width)),
            "pivot_position": [float(v) for v in panel.get("pivot_position", [0.0, 0.0, 0.0])],
            "fold_sign": float(panel.get("fold_sign", 1.0)),
            "glue_offset": float(panel.get("glue_offset", 0.0)),
        }

    front = panel_map.get("front")
    if front is None:
        raise ValueError("The Stage 1 box model does not contain the front body panel.")

    # Front is always root; other panels follow
    other_panels = [panel_map[p["id"]] for p in box.get("body_panels", []) if p["id"] != "front"]
    panels = [front] + other_panels

    D_mm = float(box.get("D_mm", 0.0))
    W_mm = float(box.get("W_mm", 0.0))
    H_mm = float(box.get("H_mm", 0.0))
    scale = 4.0 / max(D_mm, W_mm, H_mm, 1.0)

    textures: Dict[str, str] = {}
    if page is not None:
        try:
            textures = extract_panel_textures(box, page, dpi=dpi)
        except Exception as tex_err:
            print(f"Warning: Failed to extract panel textures: {tex_err}")

    flaps = [
        {
            "id": flap["id"],
            "parent": flap["parent"],
            "kind": flap["kind"],
            "sign": float(flap["sign"]),
            "pivot_position": [float(v) for v in flap["pivot_position"]],
            "polygon": flap["polygon_local"],
            "texture": textures.get(flap["id"], ""),
            "z_offset": float(flap.get("z_offset", 0.0)),
            "layer_index": int(flap.get("layer_index", 0)),
        }
        for flap in box.get("flaps", [])
    ]

    return {
        "root": "front",
        "D_mm": D_mm,
        "W_mm": W_mm,
        "H_mm": H_mm,
        "scene_scale": float(scale),
        "panels": [{
            "id": panel["id"],
            "parent": panel["parent"],
            "x0": float(panel["x0"]),
            "x1": float(panel["x1"]),
            "height": float(panel["height"]),
            "texture": textures.get(panel["id"], ""),
            "pivot_position": panel["pivot_position"],
            "fold_sign": panel["fold_sign"],
            "glue_offset": panel["glue_offset"],
        } for panel in panels],
        "flaps": flaps,
    }


def compute_stage2_numpy_box_bounds(model: Dict[str, Any], fold_progress: float = 1.0, flip: bool = False) -> Dict[str, Any]:
    """Compute the fold tree bounding box using recursive hinge transforms in the parent local frame."""
    panels = model["panels"]
    flaps = model.get("flaps", [])
    panel_by_id = {panel["id"]: panel for panel in panels}

    def remap_p(p: float, start: float, end: float) -> float:
        return float(np.clip((p - start) / (end - start), 0.0, 1.0))

    p_wall = remap_p(fold_progress, 0.0, 0.5) if flaps else fold_progress
    p_dust = remap_p(fold_progress, 0.5, 0.7)
    p_tuck = remap_p(fold_progress, 0.7, 1.0)

    def hinge_position(panel_id: str) -> np.ndarray:
        panel = panel_by_id[panel_id]
        if panel["parent"] is None:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        piv = np.array(panel.get("pivot_position", [0.0, 0.0, 0.0]), dtype=float).copy()
        glue_off = float(panel.get("glue_offset", 0.0))
        if glue_off != 0.0:
            piv[0] -= glue_off * p_wall
        return piv

    def local_vertices(panel_id: str) -> np.ndarray:
        panel = panel_by_id[panel_id]
        x0 = float(panel["x0"])
        x1 = float(panel["x1"])
        h = float(panel["height"])
        return np.array([
            [x0, 0.0, 0.0],
            [x1, 0.0, 0.0],
            [x1, h, 0.0],
            [x0, h, 0.0],
        ], dtype=float)

    def transform_for(panel_id: str) -> np.ndarray:
        panel = panel_by_id[panel_id]
        if panel["parent"] is None:
            return np.eye(4, dtype=float)

        parent_transform = transform_for(panel["parent"])
        hinge = hinge_position(panel_id)
        sign = float(panel.get("fold_sign", 1.0))
        if flip:
            sign *= -1.0
        theta = sign * p_wall * (np.pi / 2.0)
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        trans = np.array([
            [1.0, 0.0, 0.0, hinge[0]],
            [0.0, 1.0, 0.0, hinge[1]],
            [0.0, 0.0, 1.0, hinge[2]],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        return parent_transform @ trans @ rot

    vertices = []
    for panel in panels:
        panel_id = panel["id"]
        pts = np.hstack([
            local_vertices(panel_id),
            np.ones((local_vertices(panel_id).shape[0], 1), dtype=float),
        ])
        world = (transform_for(panel_id) @ pts.T).T[:, :3]
        vertices.extend(world)

    flap_by_id = {flap["id"]: flap for flap in flaps}

    def flap_transform(fid: str) -> np.ndarray:
        flap = flap_by_id[fid]
        parent_id = flap["parent"]
        if parent_id in panel_by_id:
            parent_transform = transform_for(parent_id)
        elif parent_id in flap_by_id:
            parent_transform = flap_transform(parent_id)
        else:
            parent_transform = np.eye(4, dtype=float)

        piv = flap["pivot_position"]
        sign = float(flap.get("sign", 1.0))
        if flip:
            sign *= -1.0
        if fid == "top_tuck" or fid.endswith("_tuck"):
            sub_p = remap_p(fold_progress, 0.7, 0.95)
        else:
            sub_p = p_dust if flap.get("kind") == "dust" else p_tuck
        theta = sign * sub_p * (np.pi / 2.0)
        c = np.cos(theta)
        s = np.sin(theta)
        rot_x = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        trans = np.array([
            [1.0, 0.0, 0.0, piv[0]],
            [0.0, 1.0, 0.0, piv[1]],
            [0.0, 0.0, 1.0, piv[2]],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        return parent_transform @ trans @ rot_x

    for flap in flaps:
        fid = flap["id"]
        poly = flap.get("polygon", [])
        if not poly:
            continue
        sub_p = remap_p(fold_progress, 0.7, 0.95) if (fid == "top_tuck" or fid.endswith("_tuck")) else (p_dust if flap.get("kind") == "dust" else p_tuck)
        local_z = float(flap.get("z_offset", 0.0)) * sub_p
        local_pts = np.array([[pt[0], pt[1], local_z, 1.0] for pt in poly], dtype=float)
        world = (flap_transform(fid) @ local_pts.T).T[:, :3]
        vertices.extend(world)

    points = np.asarray(vertices, dtype=float)
    bounds = {
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "size_mm": [
            float(points[:, 2].max() - points[:, 2].min()),
            float(points[:, 0].max() - points[:, 0].min()),
            float(points[:, 1].max() - points[:, 1].min()),
        ],
    }
    return bounds


def compute_panel_outward_normals(model: Dict[str, Any], fold_progress: float = 1.0, flip: bool = False) -> Dict[str, Dict[str, Any]]:
    """Compute world normals and dot products with displacement from box center for all body panels."""
    panels = model["panels"]
    panel_by_id = {panel["id"]: panel for panel in panels}

    def hinge_position(panel_id: str) -> np.ndarray:
        panel = panel_by_id[panel_id]
        if panel["parent"] is None:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        piv = np.array(panel.get("pivot_position", [0.0, 0.0, 0.0]), dtype=float).copy()
        glue_off = float(panel.get("glue_offset", 0.0))
        if glue_off != 0.0:
            piv[0] -= glue_off * fold_progress
        return piv

    def rot_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def trans(t: np.ndarray) -> np.ndarray:
        return np.array([
            [1.0, 0.0, 0.0, t[0]],
            [0.0, 1.0, 0.0, t[1]],
            [0.0, 0.0, 1.0, t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def transform_for(panel_id: str) -> np.ndarray:
        p = panel_by_id[panel_id]
        if p["parent"] is None:
            return np.eye(4, dtype=float)
        parent_transform = transform_for(p["parent"])
        hinge = hinge_position(panel_id)
        sign = float(p.get("fold_sign", 1.0))
        if flip:
            sign *= -1.0
        theta = sign * fold_progress * (np.pi / 2.0)
        return parent_transform @ trans(hinge) @ rot_y(theta)

    all_pts = []
    panel_centers = {}
    panel_corners = {}
    panel_normals = {}

    for panel in panels:
        pid = panel["id"]
        x0, x1, h = panel["x0"], panel["x1"], panel["height"]
        corners = np.array([
            [x0, 0.0, 0.0, 1.0],
            [x1, 0.0, 0.0, 1.0],
            [x1, h, 0.0, 1.0],
            [x0, h, 0.0, 1.0],
        ], dtype=float)
        T = transform_for(pid)
        w_corners = (T @ corners.T).T[:, :3]
        all_pts.extend(w_corners)
        panel_corners[pid] = w_corners
        panel_centers[pid] = w_corners.mean(axis=0)
        local_n = np.array([0.0, 0.0, -1.0 if pid == "glue" else 1.0, 0.0], dtype=float)
        n = (T @ local_n)[:3]
        panel_normals[pid] = n / np.linalg.norm(n)

    pts = np.asarray(all_pts, dtype=float)
    box_center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0

    results = {}
    for panel in panels:
        pid = panel["id"]
        c = panel_centers[pid]
        n = panel_normals[pid]
        disp = (box_center - c) if pid == "glue" else (c - box_center)
        dot = float(np.dot(n, disp))
        results[pid] = {
            "center": c.tolist(),
            "corners": panel_corners[pid].tolist(),
            "box_center": box_center.tolist(),
            "normal": n.tolist(),
            "disp": disp.tolist(),
            "dot": dot,
            "is_outward": dot > 0.0,
        }
    return results


def compute_flap_outward_normals(model: Dict[str, Any], fold_progress: float = 1.0, flip: bool = False) -> Dict[str, Dict[str, Any]]:
    """Compute world normals, centroids, and dot products with displacement from box center for all flaps."""
    panels = model["panels"]
    flaps = model.get("flaps", [])
    panel_by_id = {panel["id"]: panel for panel in panels}
    flap_by_id = {flap["id"]: flap for flap in flaps}

    def remap_p(p: float, start: float, end: float) -> float:
        return float(np.clip((p - start) / (end - start), 0.0, 1.0))

    def get_transforms(prog: float, is_flipped: bool):
        p_wall = remap_p(prog, 0.0, 0.5)
        p_dust = remap_p(prog, 0.5, 0.7)
        p_tuck = remap_p(prog, 0.7, 1.0)

        def hinge_position(panel_id: str) -> np.ndarray:
            panel = panel_by_id[panel_id]
            if panel["parent"] is None:
                return np.array([0.0, 0.0, 0.0], dtype=float)
            piv = np.array(panel.get("pivot_position", [0.0, 0.0, 0.0]), dtype=float).copy()
            glue_off = float(panel.get("glue_offset", 0.0))
            if glue_off != 0.0:
                piv[0] -= glue_off * p_wall
            return piv

        def panel_T(panel_id: str) -> np.ndarray:
            p = panel_by_id[panel_id]
            if p["parent"] is None:
                return np.eye(4, dtype=float)
            parent_transform = panel_T(p["parent"])
            hinge = hinge_position(panel_id)
            sign = float(p.get("fold_sign", 1.0))
            if is_flipped:
                sign *= -1.0
            theta = sign * p_wall * (np.pi / 2.0)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
            Tr = np.array([[1.0, 0.0, 0.0, hinge[0]], [0.0, 1.0, 0.0, hinge[1]], [0.0, 0.0, 1.0, hinge[2]], [0.0, 0.0, 0.0, 1.0]], dtype=float)
            return parent_transform @ Tr @ R

        def flap_T(fid: str) -> np.ndarray:
            f = flap_by_id[fid]
            pr = f["parent"]
            parent_transform = panel_T(pr) if pr in panel_by_id else flap_T(pr)
            piv = f["pivot_position"]
            sign = float(f.get("sign", 1.0))
            if is_flipped:
                sign *= -1.0
            sub_p = remap_p(prog, 0.7, 0.95) if (fid == "top_tuck" or fid.endswith("_tuck")) else (p_dust if f.get("kind") == "dust" else p_tuck)
            theta = sign * sub_p * (np.pi / 2.0)
            c, s = np.cos(theta), np.sin(theta)
            Rx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
            Tr = np.array([[1.0, 0.0, 0.0, piv[0]], [0.0, 1.0, 0.0, piv[1]], [0.0, 0.0, 1.0, piv[2]], [0.0, 0.0, 0.0, 1.0]], dtype=float)
            return parent_transform @ Tr @ Rx

        return panel_T, flap_T

    pT_now, flapT_now = get_transforms(fold_progress, flip)
    corners_all = []
    for p in panels:
        pid = p["id"]
        x0, x1, h = p["x0"], p["x1"], p["height"]
        corners = np.array([[x0, 0.0, 0.0, 1.0], [x1, 0.0, 0.0, 1.0], [x1, h, 0.0, 1.0], [x0, h, 0.0, 1.0]], dtype=float)
        corners_all.extend((pT_now(pid) @ corners.T).T[:, :3])
    pts = np.asarray(corners_all, dtype=float)
    box_center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    axis_c = box_center[[0, 2]]

    _, flapT_0 = get_transforms(0.0, False)
    _, flapT_05 = get_transforms(0.5, flip)
    _, flapT_07 = get_transforms(0.7, flip)

    results = {}
    for flap in flaps:
        fid = flap["id"]
        poly = flap.get("polygon", [])
        if not poly:
            continue
        local_pts = np.array([[pt[0], pt[1], 0.0, 1.0] for pt in poly], dtype=float)

        c0 = (flapT_0(fid) @ local_pts.T).T[:, :3].mean(axis=0)
        c05 = (flapT_05(fid) @ local_pts.T).T[:, :3].mean(axis=0)
        c07 = (flapT_07(fid) @ local_pts.T).T[:, :3].mean(axis=0)

        T_now = flapT_now(fid)
        c_now = (T_now @ local_pts.T).T[:, :3].mean(axis=0)
        n = (T_now @ np.array([0.0, 0.0, 1.0, 0.0], dtype=float))[:3]
        n = n / np.linalg.norm(n)

        disp = c_now - box_center
        dot = float(np.dot(n, disp))

        d0 = float(np.linalg.norm(c0[[0, 2]] - axis_c))
        d05 = float(np.linalg.norm(c05[[0, 2]] - axis_c))
        d07 = float(np.linalg.norm(c07[[0, 2]] - axis_c))
        d_now = float(np.linalg.norm(c_now[[0, 2]] - axis_c))

        m05 = "inward" if d05 <= d0 + 1e-3 else "outward"
        m07 = "inward" if d07 <= d05 + 1e-3 else "outward"
        movement = "inward" if d_now < d0 else "outward"

        results[fid] = {
            "sign": float(flap.get("sign", 1.0)),
            "axis": "local X",
            "centroid": c_now.tolist(),
            "centroid_p0": c0.tolist(),
            "centroid_p05": c05.tolist(),
            "centroid_p07": c07.tolist(),
            "normal": n.tolist(),
            "disp": disp.tolist(),
            "dot": dot,
            "is_outward": dot > 0.0,
            "d_axis_p0": d0,
            "d_axis_p05": d05,
            "d_axis_p07": d07,
            "d_axis_now": d_now,
            "movement_at_p05": m05,
            "movement_at_p07": m07,
            "movement_vs_p0": movement,
        }
    return results


def build_model(
    uploaded_file,
    red_threshold: float,
    green_threshold: float,
    min_length: float,
    flip_direction: bool,
    manual_color_overrides: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    box = build_stage1_box_geometry(page, doc)
    validate_stage1_box_geometry(box)
    return build_stage2_body_panel_model(box, page=page)


def make_viewer_html(model: Dict[str, Any], flip: bool = False) -> str:
    model_json = json.dumps(model)
    template = string.Template(
        """
    <style>
      body { margin: 0; background: #f4f7fb; overflow: hidden; }
      #box-viewer {
        position: relative;
        width: 100%;
        height: 600px;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 14px 34px rgba(15,23,42,0.08);
        background: linear-gradient(180deg, #e8ecf1 0%, #ffffff 100%);
        user-select: none;
        -webkit-user-select: none;
      }
      #box-viewer canvas {
        display: block;
        width: 100% !important;
        height: 100% !important;
        touch-action: none;
        pointer-events: auto;
        cursor: grab;
      }
      #box-viewer canvas:active {
        cursor: grabbing;
      }
      .viewer-topbar {
        position: absolute;
        top: 16px;
        left: 16px;
        z-index: 20;
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
        pointer-events: auto;
      }
      .viewer-topbar * {
        pointer-events: auto;
      }
      .viewer-topbar button { border: none; border-radius: 12px; background: rgba(15,23,42,0.78); color: white; padding: 10px 14px; cursor: pointer; font-weight: 600; }
      .viewer-topbar input[type=range] { width: 180px; }
      .viewer-topbar label { font-size: 12px; color: #334155; background: rgba(255,255,255,0.85); border-radius: 999px; padding: 9px 12px; }
      .viewer-error {
        position: absolute;
        left: 16px;
        right: 16px;
        bottom: 12px;
        z-index: 30;
        background: rgba(127, 29, 29, 0.92);
        color: white;
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 12px;
        font-family: sans-serif;
        line-height: 1.4;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease;
      }
      .viewer-error.visible { opacity: 1; pointer-events: auto; }
      .viewer-debug {
        position: absolute;
        left: 16px;
        bottom: 12px;
        z-index: 35;
        background: rgba(15, 23, 42, 0.8);
        color: #f8fafc;
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 11px;
        font-family: monospace;
        white-space: pre-wrap;
        max-width: 420px;
        pointer-events: none;
      }
    </style>
    <div id="box-viewer">
      <div class="viewer-topbar">
        <label>Fold Progress <input id="foldSlider" type="range" min="0" max="100" value="100" /></label>
        <button id="toggleFold">Pause</button>
        <label style="display:flex; align-items:center; gap:8px;">
          <input id="flipToggle" type="checkbox" $flip_checked /> Flip
        </label>
        <button id="downloadBtn">Download GLB</button>
      </div>
      <div id="viewer-error" class="viewer-error" aria-live="polite"></div>
      <script type="importmap">
        {
          "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
          }
        }
      </script>
      <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

        window.addEventListener('DOMContentLoaded', function() {
          const el = (id) => {
            const e = document.getElementById(id);
            if (!e) console.error('Missing element', id);
            return e;
          };

          try {
            const mount = el('box-viewer');
            const errorEl = el('viewer-error');

            function showError(message) {
              if (errorEl) {
                errorEl.textContent = message;
                errorEl.classList.add('visible');
              }
              console.error('[viewer]', message);
            }

            function ensureFallbackText(message) {
              if (mount && !mount.querySelector('.viewer-debug')) {
                const debug = document.createElement('div');
                debug.className = 'viewer-debug';
                debug.textContent = message;
                mount.appendChild(debug);
              }
            }

            window.onerror = function(message, source, lineno, colno, error) {
              showError('Three.js error: ' + (message || 'Unknown error'));
              console.error(message, source, lineno, colno, error);
              return false;
            };

            console.log('THREE r', THREE.REVISION);
            console.log('OrbitControls', typeof OrbitControls);
            console.log('GLTFExporter', typeof GLTFExporter);

            const model = $model_json;
            if (!model || !model.panels || !Array.isArray(model.panels)) {
              showError('No panels received');
              ensureFallbackText('No panels received');
              return;
            }

            console.log('panel count', model.panels.length);
            model.panels.forEach(function(panel) {
              console.log('panel', panel.id, 'x0=', panel.x0, 'x1=', panel.x1, 'height=', panel.height);
            });

            if (!mount) {
              throw new Error('Missing mount element #box-viewer');
            }

            mount.style.height = '600px';
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xe8ecf1);

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            renderer.setSize(mount.clientWidth || 800, 600);
            renderer.domElement.style.touchAction = 'none';
            renderer.domElement.style.pointerEvents = 'auto';
            mount.appendChild(renderer.domElement);

            const camera = new THREE.PerspectiveCamera(35, (mount.clientWidth || 800) / 600, 0.1, 2000);
            let controls = { update: function() {}, target: new THREE.Vector3(0, 0, 0), enabled: true, enableRotate: true };
            try {
              controls = new OrbitControls(camera, renderer.domElement);
              controls.enabled = true;
              controls.enableRotate = true;
              controls.enableDamping = true;
              controls.dampingFactor = 0.05;
              controls.enablePan = true;
              controls.enableZoom = true;
              controls.minDistance = 2;
              controls.maxDistance = 80;
            } catch (orbitError) {
              console.error('OrbitControls failed to initialize', orbitError);
            }
            console.log('controls', controls.enabled, controls.enableRotate);

            const grid = new THREE.GridHelper(240, 20, 0x94a3b8, 0xcbd5e1);
            grid.position.y = -0.01;
            scene.add(grid);
            scene.add(new THREE.AxesHelper(60));

            scene.add(new THREE.HemisphereLight(0xffffff, 0xcfc7b7, 1.1));
            const dir = new THREE.DirectionalLight(0xffffff, 1.4);
            dir.position.set(8, 12, 6);
            scene.add(dir);

            const wrapper = new THREE.Group();
            scene.add(wrapper);

            const varPalette = {
              front: 0xe7d7b6,
              side_a: 0xd9c4a0,
              side_b: 0xcfdab6,
              back: 0xb7c9b3,
              glue: 0xd8c7a6
            };

            const flapPalette = {
              side_a_top: 0xd9c4a0,
              side_a_bottom: 0xd9c4a0,
              front_top: 0xe7d7b6,
              top_tuck: 0xe7d7b6,
              front_bottom: 0xe7d7b6,
              side_b_top: 0xcfdab6,
              side_b_bottom: 0xcfdab6,
              back_bottom: 0xb7c9b3
            };

            const scale = Number(model.scene_scale || 4.0 / Math.max(model.D_mm || 1, model.W_mm || 1, model.H_mm || 1, 1));
            const panelMap = {};
            const panelList = model.panels || [];
            panelList.forEach(function(panel) {
              panelMap[panel.id] = panel;
            });

            const panelNodes = {};
            const rootId = model.root || 'front';

            function normalizeUVs(geometry) {
              geometry.computeBoundingBox();
              const bbox = geometry.boundingBox;
              const rangeX = (bbox.max.x - bbox.min.x) || 1;
              const rangeY = (bbox.max.y - bbox.min.y) || 1;
              const pos = geometry.attributes.position;
              const uv = geometry.attributes.uv;
              for (let i = 0; i < pos.count; i++) {
                const u = (pos.getX(i) - bbox.min.x) / rangeX;
                const v = (pos.getY(i) - bbox.min.y) / rangeY;
                uv.setXY(i, u, v);
              }
              uv.needsUpdate = true;
            }

            const textureLoader = new THREE.TextureLoader();

            function createTexturedFaceGroup(geometry, textureDataUri, fallbackColor, layerIndex, isTuck) {
              const group = new THREE.Group();

              const hasOffset = (typeof layerIndex === 'number') || Boolean(isTuck);
              const offsetFactor = isTuck ? 2 : (hasOffset ? -((layerIndex + 1) * 2) : 0);
              const offsetUnits = isTuck ? 2 : (hasOffset ? -((layerIndex + 1) * 2) : 0);

              // Outside face: textured with FrontSide
              let frontMat;
              if (textureDataUri) {
                const tex = textureLoader.load(textureDataUri);
                tex.colorSpace = THREE.SRGBColorSpace;
                frontMat = new THREE.MeshStandardMaterial({
                  map: tex,
                  side: THREE.FrontSide,
                  roughness: 0.88,
                  metalness: 0.02,
                  polygonOffset: hasOffset,
                  polygonOffsetFactor: offsetFactor,
                  polygonOffsetUnits: offsetUnits
                });
              } else {
                frontMat = new THREE.MeshStandardMaterial({
                  color: fallbackColor || 0xd9c7a6,
                  side: THREE.FrontSide,
                  roughness: 0.92,
                  metalness: 0.03,
                  polygonOffset: hasOffset,
                  polygonOffsetFactor: offsetFactor,
                  polygonOffsetUnits: offsetUnits
                });
              }
              const frontMesh = new THREE.Mesh(geometry, frontMat);
              frontMesh.castShadow = true;
              frontMesh.receiveShadow = true;
              group.add(frontMesh);

              // Inside face: plain kraft with BackSide
              const backMat = new THREE.MeshStandardMaterial({
                color: 0xdecbb1,
                side: THREE.BackSide,
                roughness: 0.95,
                metalness: 0.02,
                polygonOffset: hasOffset,
                polygonOffsetFactor: offsetFactor,
                polygonOffsetUnits: offsetUnits
              });
              const backMesh = new THREE.Mesh(geometry, backMat);
              group.add(backMesh);

              // Dark outline edges
              const edge = new THREE.LineSegments(
                new THREE.EdgesGeometry(geometry),
                new THREE.LineBasicMaterial({ color: 0x1f2937, linewidth: 1 })
              );
              group.add(edge);

              return group;
            }

            function makePanelMesh(panel) {
              const x0 = panel.x0 * scale;
              const x1 = panel.x1 * scale;
              const y1 = panel.height * scale;

              const shape = new THREE.Shape();
              if (panel.id === 'glue') {
                shape.moveTo(x0, 0);
                shape.lineTo(x0, y1);
                shape.lineTo(x1, y1);
                shape.lineTo(x1, 0);
                shape.closePath();
                const geometry = new THREE.ShapeGeometry(shape);
                normalizeUVs(geometry);
                return createTexturedFaceGroup(geometry, null, 0xd8c7a6);
              }
              shape.moveTo(x0, 0);
              shape.lineTo(x1, 0);
              shape.lineTo(x1, y1);
              shape.lineTo(x0, y1);
              shape.closePath();

              const geometry = new THREE.ShapeGeometry(shape);
              normalizeUVs(geometry);
              return createTexturedFaceGroup(geometry, panel.texture, varPalette[panel.id]);
            }

            function createPanelNode(panel) {
              const mesh = makePanelMesh(panel);
              const pivot = new THREE.Group();
              pivot.name = panel.id;
              pivot.add(mesh);
              panelNodes[panel.id] = pivot;
              return pivot;
            }

            function panelWidth(panelId) {
              const panel = panelMap[panelId];
              if (!panel) return 0;
              return (panel.x1 - panel.x0) * scale;
            }

            const frontWidth = panelWidth('front');

            panelList.forEach(function(panel) {
              if (!panel || typeof panel.id === 'undefined' || typeof panel.x0 === 'undefined' || typeof panel.x1 === 'undefined' || typeof panel.height === 'undefined') {
                throw new Error('Malformed panel data: ' + JSON.stringify(panel));
              }
              if (panel.id === 'front') {
                const frontNode = createPanelNode(panel);
                wrapper.add(frontNode);
                return;
              }

              const parentId = panel.parent || rootId;
              const parentNode = panelNodes[parentId] || wrapper;
              const pivot = createPanelNode(panel);
              const piv = panel.pivot_position || [0, 0, 0];
              pivot.position.set(piv[0] * scale, piv[1] * scale, piv[2] * scale);
              parentNode.add(pivot);
            });

            // Flaps
            const flapList = model.flaps || [];
            console.log('flap count', flapList.length);
            flapList.forEach(function(flap) {
              console.log('flap', flap.id, 'parent=', flap.parent, 'kind=', flap.kind);
            });

            function makeFlapMesh(flap) {
              const shape = new THREE.Shape();
              let pts = flap.polygon ? flap.polygon.slice() : [];
              if (pts.length > 0) {
                if (pts.length > 1 && pts[pts.length - 1][0] === pts[0][0] && pts[pts.length - 1][1] === pts[0][1]) {
                  pts.pop();
                }
                // Check winding order via 2D signed area:
                let area = 0;
                const n = pts.length;
                for (let i = 0; i < n; i++) {
                  const j = (i + 1) % n;
                  area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
                }
                // In Cartesian (+X right, +Y up), area > 0 is CCW. If area < 0 (CW), reverse points to make CCW:
                if (area < 0) {
                  pts.reverse();
                }
                shape.moveTo(pts[0][0] * scale, pts[0][1] * scale);
                for (let i = 1; i < pts.length; i++) {
                  shape.lineTo(pts[i][0] * scale, pts[i][1] * scale);
                }
                shape.closePath();
              }
              const geometry = new THREE.ShapeGeometry(shape);
              normalizeUVs(geometry);
              const fallbackColor = flapPalette[flap.id] || (flap.parent && varPalette[flap.parent]) || 0xd9c7a6;
              const isTuck = (flap.kind === 'tuck' || (typeof flap.id === 'string' && flap.id.indexOf('tuck') !== -1));
              const group = createTexturedFaceGroup(geometry, flap.texture, fallbackColor, flap.layer_index, isTuck);
              if (flap.z_offset) {
                group.position.z = flap.z_offset * scale;
              }
              return group;
            }

            const flapNodes = {};

            function createFlapNode(flap) {
              const mesh = makeFlapMesh(flap);
              const pivot = new THREE.Group();
              pivot.name = flap.id;
              pivot.add(mesh);
              flapNodes[flap.id] = pivot;
              return pivot;
            }

            flapList.forEach(function(flap) {
              if (!flap || !flap.id || !flap.polygon) {
                console.warn('Malformed flap data:', flap);
                return;
              }
              const pivot = createFlapNode(flap);
              const piv = flap.pivot_position || [0, 0, 0];
              pivot.position.set(piv[0] * scale, piv[1] * scale, piv[2] * scale);

              const parentNode = flapNodes[flap.parent] || panelNodes[flap.parent] || wrapper;
              parentNode.add(pivot);
            });

            const boxBeforeCenter = new THREE.Box3().setFromObject(wrapper);
            console.log('Box3 before centering', boxBeforeCenter.min.toArray(), boxBeforeCenter.max.toArray());

            const flipToggle = el('flipToggle');
            const slider = el('foldSlider');
            const toggleBtn = el('toggleFold');
            const downloadBtn = el('downloadBtn');
            let foldProgress = 1.0;
            let paused = false;
            let globalFlip = $global_flip;

            function remap(p, start, end) {
              if (p <= start) return 0.0;
              if (p >= end) return 1.0;
              return (p - start) / (end - start);
            }

            function setFold(progress) {
              const safe = Math.min(1.0, Math.max(0.0, Number(progress) || 0.0));
              foldProgress = safe;

              const pWall = remap(safe, 0.0, 0.5);
              const pDust = remap(safe, 0.5, 0.7);
              const pTuck = remap(safe, 0.7, 1.0);
              const pTuckFlap = remap(safe, 0.7, 0.95);

              Object.keys(panelNodes).forEach(function(panelId) {
                if (panelId === rootId) return;
                const node = panelNodes[panelId];
                if (!node) return;
                const panel = panelMap[panelId];
                if (panel && panel.glue_offset) {
                  const piv = panel.pivot_position || [0, 0, 0];
                  node.position.x = (piv[0] - panel.glue_offset * pWall) * scale;
                }
                let sign = (panel && typeof panel.fold_sign !== 'undefined') ? Number(panel.fold_sign) : 1.0;
                if (globalFlip) sign *= -1;
                node.rotation.y = sign * pWall * (Math.PI / 2);
              });

              flapList.forEach(function(flap) {
                const node = flapNodes[flap.id];
                if (!node) return;
                let pSub = (flap.kind === 'dust') ? pDust : pTuck;
                if (flap.id === 'top_tuck' || flap.id.endsWith('_tuck')) {
                  pSub = pTuckFlap;
                }
                let sign = Number(flap.sign) || 1.0;
                if (globalFlip) sign *= -1;
                node.rotation.x = sign * pSub * (Math.PI / 2);
                if (node.children[0]) {
                  node.children[0].position.z = (flap.z_offset || 0) * scale * pSub;
                }
              });

              const box = new THREE.Box3().setFromObject(wrapper);
              const sizeUnits = box.getSize(new THREE.Vector3());
              const sizeMm = new THREE.Vector3(
                sizeUnits.x / scale,
                sizeUnits.y / scale,
                sizeUnits.z / scale
              );
              if (Math.abs(safe - 0.0) < 0.005) {
                console.log('Box3 size in mm at foldProgress=0:', {
                  depth: sizeMm.z.toFixed(2),
                  width: sizeMm.x.toFixed(2),
                  height: sizeMm.y.toFixed(2)
                });
              } else if (Math.abs(safe - 0.5) < 0.005) {
                console.log('Box3 size in mm at foldProgress=0.5:', {
                  depth: sizeMm.z.toFixed(2),
                  width: sizeMm.x.toFixed(2),
                  height: sizeMm.y.toFixed(2)
                });
              } else if (safe >= 0.999) {
                console.log('Box3 size in mm at foldProgress=1:', {
                  depth: sizeMm.z.toFixed(2),
                  width: sizeMm.x.toFixed(2),
                  height: sizeMm.y.toFixed(2)
                });
              }
            }

            function fitCamera() {
              const box = new THREE.Box3().setFromObject(wrapper);
              const center = box.getCenter(new THREE.Vector3());
              wrapper.position.sub(center);
              const fitted = new THREE.Box3().setFromObject(wrapper);
              console.log('Box3 after centering', fitted.min.toArray(), fitted.max.toArray());
              const size = fitted.getSize(new THREE.Vector3());
              const maxDim = Math.max(size.x, size.y, size.z, 1.0);
              camera.position.set(0, size.y * 0.8 + maxDim * 0.7, size.x * 1.25 + maxDim * 0.9);
              console.log('camera position', camera.position.toArray());
              if (controls && controls.target) controls.target.set(0, 0, 0);
              if (controls && controls.update) controls.update();
            }

            if (slider) {
              slider.addEventListener('input', function(event) {
                setFold(Number(event.target.value) / 100);
              });
            }
            if (toggleBtn) {
              toggleBtn.addEventListener('click', function() {
                paused = !paused;
                toggleBtn.textContent = paused ? 'Play' : 'Pause';
              });
            }
            if (flipToggle) {
              flipToggle.addEventListener('change', function(event) {
                globalFlip = event.target.checked;
                setFold(foldProgress);
              });
            }

            if (downloadBtn) {
              downloadBtn.addEventListener('click', function() {
                const exporter = new GLTFExporter();
                exporter.parse(scene, function(result) {
                  const blob = new Blob([result], { type: 'model/gltf-binary' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'packaging-box.glb';
                  a.click();
                  URL.revokeObjectURL(url);
                }, function(error) {
                  console.error(error);
                }, { binary: true });
              });
            }

            setFold(1.0);
            fitCamera();
            if (slider) slider.value = '100';
            if (errorEl) {
              errorEl.textContent = 'Viewer ready';
              errorEl.classList.remove('visible');
            }

            function animate() {
              if (!paused && slider) {
                let current = Number(slider.value);
                current += 0.9;
                if (current >= 100) current = 100;
                slider.value = String(current);
                setFold(current / 100);
              }
              if (controls && controls.update) controls.update();
              renderer.render(scene, camera);
              requestAnimationFrame(animate);
            }

            animate();

            window.addEventListener('resize', function() {
              const w = mount.clientWidth || 800;
              const h = 600;
              camera.aspect = w / h;
              camera.updateProjectionMatrix();
              renderer.setSize(w, h);
            });
          } catch (error) {
            const mount = document.getElementById('box-viewer');
            const errorEl = document.getElementById('viewer-error');
            if (errorEl) {
              errorEl.textContent = 'Three.js viewer setup failed: ' + error.message;
              errorEl.classList.add('visible');
            }
            if (mount) {
              const fallback = document.createElement('div');
              fallback.className = 'viewer-debug';
              fallback.textContent = 'Three.js viewer setup failed: ' + error.message;
              mount.appendChild(fallback);
            }
            console.error(error);
          }
        });
      </script>
    """
    )
    return template.substitute(
        model_json=model_json,
        flip_checked="checked" if flip else "",
        global_flip="true" if flip else "false",
    )


def main() -> None:
    st.title("3D Packaging Box Visualizer")
    st.write("Upload a packaging die-line PDF to preview its artwork as an interactive, auto-folded 3D box.")

    uploaded_file = st.file_uploader("Upload your Box Die-Line PDF", type=["pdf"])

    if uploaded_file is None:
        st.session_state.pop("active_file", None)
        for k in ("input_dim_d", "input_dim_w", "input_dim_h", "dim_d", "dim_w", "dim_h"):
            st.session_state.pop(k, None)
        with st.sidebar:
            st.header("Box Dimensions (mm)")
            st.caption("Upload a die-line PDF to detect or adjust box dimensions.")
            st.number_input("Depth (mm)", min_value=1.0, max_value=2000.0, value=50.0, step=1.0)
            st.number_input("Width (mm)", min_value=1.0, max_value=2000.0, value=120.0, step=1.0)
            st.number_input("Height (mm)", min_value=1.0, max_value=2000.0, value=180.0, step=1.0)
        st.info("No PDF uploaded yet. Add a die-line PDF to generate the box preview.")
        return

    try:
        pdf_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]

        parsed_dims = parse_dimensions_from_filename(uploaded_file.name)
        measured_dims = measure_pdf_nominal_dims(page, doc)

        file_state_key = f"file_{uploaded_file.name}"
        if st.session_state.get("active_file") != file_state_key:
            st.session_state["active_file"] = file_state_key
            if parsed_dims:
                d_init, w_init, h_init = float(parsed_dims[0]), float(parsed_dims[1]), float(parsed_dims[2])
            elif measured_dims:
                d_init, w_init, h_init = float(measured_dims[0]), float(measured_dims[1]), float(measured_dims[2])
            else:
                d_init, w_init, h_init = 50.0, 120.0, 180.0

            st.session_state["dim_d"] = d_init
            st.session_state["dim_w"] = w_init
            st.session_state["dim_h"] = h_init
            st.session_state["input_dim_d"] = d_init
            st.session_state["input_dim_w"] = w_init
            st.session_state["input_dim_h"] = h_init

        with st.sidebar:
            st.header("Box Dimensions (mm)")
            if measured_dims:
                sugg_d, sugg_w, sugg_h = [int(round(v)) for v in measured_dims]
                if st.button(f"Use measured: {sugg_d} x {sugg_w} x {sugg_h}", key="btn_sidebar_measured"):
                    st.session_state["dim_d"] = float(sugg_d)
                    st.session_state["dim_w"] = float(sugg_w)
                    st.session_state["dim_h"] = float(sugg_h)
                    st.session_state["input_dim_d"] = float(sugg_d)
                    st.session_state["input_dim_w"] = float(sugg_w)
                    st.session_state["input_dim_h"] = float(sugg_h)
                    st.rerun()

            if "input_dim_d" not in st.session_state:
                st.session_state["input_dim_d"] = 50.0
            if "input_dim_w" not in st.session_state:
                st.session_state["input_dim_w"] = 120.0
            if "input_dim_h" not in st.session_state:
                st.session_state["input_dim_h"] = 180.0

            dim_d = st.number_input("Depth (mm)", min_value=1.0, max_value=2000.0, step=1.0, key="input_dim_d")
            dim_w = st.number_input("Width (mm)", min_value=1.0, max_value=2000.0, step=1.0, key="input_dim_w")
            dim_h = st.number_input("Height (mm)", min_value=1.0, max_value=2000.0, step=1.0, key="input_dim_h")
            st.session_state["dim_d"] = float(dim_d)
            st.session_state["dim_w"] = float(dim_w)
            st.session_state["dim_h"] = float(dim_h)

            if parsed_dims:
                st.caption(f"Parsed from filename: {parsed_dims[0]:.0f} x {parsed_dims[1]:.0f} x {parsed_dims[2]:.0f} mm")
            else:
                st.caption("No dimensions in filename. Measured suggestion auto-filled above.")

            with st.expander("Advanced", expanded=False):
                st.caption("Die-line extraction thresholds.")
                red_threshold = st.slider("Cut line red threshold", min_value=0.2, max_value=1.0, value=0.7, step=0.05)
                green_threshold = st.slider("Crease line green threshold", min_value=0.2, max_value=1.0, value=0.5, step=0.05)
                min_edge_len = st.slider("Minimum structural edge", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
                flip_direction = st.toggle("Flip fold direction", value=False)

        layer_info = detect_pdf_layers_summary(page, doc)
        if not layer_info["fold_detected"] and not layer_info["die_detected"]:
            st.error(
                "### ⚠️ Missing Vector Layers\n\n"
                "No Fold (crease) or Die (cut) layers were detected in the uploaded PDF.\n\n"
                "**What you can try:**\n"
                "- Verify that your vector PDF contains paths on layers named **Fold** and **Die** (or crease/cut lines).\n"
                "- Check the stroke colors and transparency of the die lines."
            )
            return

        dims_tuple = (float(dim_d), float(dim_w), float(dim_h))
        box = build_stage1_box_geometry(page, doc, dimensions=dims_tuple)
        validate_stage1_box_geometry(box)

        if parsed_dims is None:
            if measured_dims:
                sugg_d, sugg_w, sugg_h = [int(round(v)) for v in measured_dims]
                st.info(
                    f"Please confirm box dimensions: auto-filled from measured dieline as "
                    f"**{sugg_d} x {sugg_w} x {sugg_h} mm** (Depth x Width x Height). "
                    "You can adjust these in the sidebar if needed."
                )
            else:
                st.info("Enter this box's Depth/Width/Height in the sidebar (mm).")

        for warn in box.get("dimension_warnings", []):
            st.warning(warn)

        # 3D Viewer directly below uploader & notifications
        model = build_stage2_body_panel_model(box, page=page)
        st.components.v1.html(make_viewer_html(model, flip=flip_direction), height=720, scrolling=False)

        # Consolidated Debug Info in ONE collapsed expander
        with st.expander("Debug info", expanded=False):
            st.markdown(
                f"**Fold Layer**: {layer_info['fold_layer']} ({layer_info['fold_source']}, {layer_info['fold_count']} lines)<br>"
                f"**Die Layer**: {layer_info['die_layer']} ({layer_info['die_source']}, {layer_info['die_count']} lines)",
                unsafe_allow_html=True,
            )

            st.subheader("Stage 1 geometry")
            die_lines = len(box.get("red_cut_segments", []))
            fold_lines = len(box.get("green_crease_segments", []))
            panel_rows = _stage1_debug_table(box)
            summary = {
                "Die lines": die_lines,
                "Fold lines": fold_lines,
                "Panels": len(box.get("body_panels", [])),
                "Flaps": len(box.get("flaps", [])),
                "D mm": round(float(box.get("D_mm", 0.0)), 2),
                "W mm": round(float(box.get("W_mm", 0.0)), 2),
                "H mm": round(float(box.get("H_mm", 0.0)), 2),
            }
            st.dataframe([summary], hide_index=True, use_container_width=True)

            fig = _render_stage1_debug_plot(page, box)
            st.pyplot(fig)

            st.dataframe(
                panel_rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "id": "ID",
                    "parent": "Parent",
                    "hinge_p0": "Hinge p0",
                    "hinge_p1": "Hinge p1",
                    "bbox_w_mm": "Width mm",
                    "bbox_h_mm": "Height mm",
                    "area_mm2": "Area mm²",
                },
            )

            raw_dump = box.get("raw_dump", [])
            debug_rows = stroke_color_debug(page, doc)
            if raw_dump:
                st.dataframe(
                    [{
                        "Layer": row["layer"],
                        "Subpaths": row["subpaths"],
                        "Segments": row["segments"],
                        "BBox minx": round(float(row["bbox_pt"][0]), 1),
                        "BBox miny": round(float(row["bbox_pt"][1]), 1),
                        "BBox maxx": round(float(row["bbox_pt"][2]), 1),
                        "BBox maxy": round(float(row["bbox_pt"][3]), 1),
                    } for row in raw_dump],
                    hide_index=True,
                    use_container_width=True,
                )
            if debug_rows:
                st.dataframe(
                    {
                        "Source": [row["source"] for row in debug_rows],
                        "RGB": [str(row["rgb"]) for row in debug_rows],
                        "Layer": [row["layer"] or "-" for row in debug_rows],
                        "Total Drawings": [row["total_drawings"] for row in debug_rows],
                        "L": [row["lines_l"] for row in debug_rows],
                        "RE": [row["rectangles_re"] for row in debug_rows],
                        "C": [row["curves_c"] for row in debug_rows],
                        "QU": [row["quads_qu"] for row in debug_rows],
                    },
                    hide_index=True,
                    use_container_width=True,
                )
    except ValueError as exc:
        msg = str(exc)
        # Gather diagnostic details for user-friendly error message
        try:
            struct_lines = extract_structural_lines(page, doc=doc, min_length=0)
            g_lines = [l for l in struct_lines if l.get("kind") == "crease"]
            r_lines = [l for l in struct_lines if l.get("kind") == "cut"]
            det_rows, reason = detect_layout_rows(g_lines, r_lines)
        except Exception:
            det_rows, g_lines, r_lines, reason = 1, [], [], ""

        if "Unsupported layout" in msg or "Multi-row" in msg or det_rows > 1:
            st.error(
                f"### ⚠️ Unsupported Carton Layout\n\n"
                f"**What was detected:** Multi-row panel layout ({det_rows} rows detected, "
                f"{len(g_lines)} crease lines, {len(r_lines)} cut lines).\n\n"
                f"**Details:** {msg}\n\n"
                "**Supported box types:** Currently, this visualizer supports single-row straight tuck end carton layouts.\n\n"
                "**What you can try:**\n"
                "- Upload a packaging dieline arranged in a single horizontal row of body panels.\n"
                "- Check that Fold (crease) and Die (cut) layers are cleanly separated."
            )
        elif "validation failed" in msg.lower():
            st.error(
                f"### ⚠️ Dieline Validation Failure\n\n"
                f"**What was detected:** Layout validation failed to form a valid 3D carton.\n\n"
                f"**Details:** {msg}\n\n"
                "**What you can try:**\n"
                "- **Check Depth / Width / Height (mm)** in the sidebar to ensure proportions match the carton.\n"
                "- **Check Fold and Die layers**: ensure green crease lines define internal body panels and red cut lines outline flaps.\n"
                "- Check that all panel boundaries form continuous closed shapes."
            )
        else:
            st.error(
                f"### ⚠️ Dieline Processing Issue\n\n"
                f"**Details:** {msg}\n\n"
                "**What you can try:**\n"
                "- Check Depth / Width / Height in the sidebar.\n"
                "- Check Fold and Die vector layers in the PDF."
            )
        return
    except Exception as exc:
        st.error(
            f"### ⚠️ Unable to Generate 3D Box\n\n"
            f"An unexpected error occurred while parsing the dieline: {exc}\n\n"
            "**What you can try:**\n"
            "- Verify that the uploaded file is a valid vector packaging PDF.\n"
            "- Check Depth / Width / Height dimensions in the sidebar.\n"
            "- Ensure the PDF includes vector stroke paths on Fold and Die layers."
        )
        return


if __name__ == "__main__":
    main()
