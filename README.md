# 3D Packaging Box Visualizer

An interactive Streamlit and Three.js application that parses vector packaging die-line PDFs, extracts panel geometry and artwork textures, and folds the carton into a realistic 3D box model.

---

## Quick Start

### 1. Installation

Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the App

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 3. Running the Test Suite

Execute the automated test suite with pytest:

```bash
pytest -v tests/
```

---

## Supported Box Types & Dieline Specifications

- **Box Style**: Single-row straight tuck end (STE) cartons (e.g., standard folding cartons like BOSS Power Trunk and Fitness Tube Box).
- **Layout**: Single horizontal row of body panels (`side_a`, `front`, `side_b`, `back`, and optional `glue` flap).
- **Layers & Linework**:
  - **Fold / Crease**: Vector strokes indicating fold lines (typically green strokes or paths placed on a layer named `Fold` or `Crease`).
  - **Die / Cut**: Vector strokes indicating perimeter cuts and flap outlines (typically red strokes or paths placed on a layer named `Die` or `Cut`).
- **Artwork**: Vector or embedded raster artwork rendered directly onto each panel and flap using UV-aligned texture mapping.

---

## Key Features

- **Automated Geometry Extraction**: Detects body panels, glue tabs, dust flaps, lid/lock flaps, and tuck closures directly from PDF vector layers.
- **Dimension Auto-Detection & Suggestion**: Automatically extracts carton dimensions (Depth x Width x Height in mm) from filenames or measures linework geometry at the standard 72 pt / 25.4 mm scale, with a one-click "Use measured" suggestion button.
- **Realistic Inward Folding & Layering**:
  - Dust flaps fold inward first.
  - Lid / lock flaps fold over dust flaps.
  - Tuck flaps fold an additional 90 degrees inside the carton depth ($D - 1.0$ mm) without poking out or bleeding through the outer walls.
  - Internal glue tabs fold inward against the inner surface of the adjacent body wall.
- **Interactive Three.js 3D Viewer**: Orbit, pan, zoom, and test folding animation with progress slider ($0 \to 1$).
- **Clean UI**: Streamlined interface with the 3D viewer directly below the PDF uploader, collapsed "Advanced" controls in the sidebar, and collapsed "Debug info" diagnostics.

---

## Known Limitations

- **Multi-Row Layouts**: Two-row cartons (e.g. British Tea, Nordqvist Moomin) are currently unsupported and will display a friendly informational notice detailing detected rows and lines.
- **Complex Bottom Closures**: Auto-locking crash bottoms and snap-lock bottoms are not supported in this stage.
- **Curved Folds**: Only orthogonal linear creases are supported. Curved crease folding is not supported.
- **Raster Scans**: Die-lines must be vector artwork (`PDF` drawings/paths). Scanned bitmap PDFs without vector paths cannot be parsed.
