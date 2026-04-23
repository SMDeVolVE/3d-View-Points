"""
Point Cloud Viewer - Streamlit + PyVista
Carica .xyz / .dxf / .dwg, ricostruisce la superficie 3D con shading
(look CAD/architettonico) e permette di selezionare un'area (rettangolo
o cerchio) dalla vista planare per ritagliare il modello 3D.
"""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyvista as pv
import streamlit as st

# CAD loaders opzionali
try:
    import ezdxf
    from ezdxf.addons import odafc
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

# PyVista lo usiamo solo per la ricostruzione geometrica (reconstruct_surface,
# delaunay_2d). Il rendering lo fa Plotly: niente trame, niente subprocess,
# niente export_html → niente crash su Streamlit Cloud.
pv.OFF_SCREEN = True


# ---------------------------------------------------------------------------
# CONFIG PAGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Point Cloud Viewer 3D",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { color: #FAFAFA; }
        .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# PARSER .XYZ
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_xyz(file_bytes: bytes) -> pd.DataFrame:
    """Parser robusto per .xyz: sniffing di delimitatore, header e RGB opzionale."""
    text = file_bytes.decode("utf-8", errors="ignore")
    sample_lines = [l for l in text.splitlines() if l.strip()][:20]
    if not sample_lines:
        raise ValueError("File vuoto o non leggibile.")

    # Delimitatore: candidati in ordine di priorità, sceglie quello più consistente
    candidates = [",", ";", "\t", " "]
    delim, best = " ", 0
    for c in candidates:
        counts = [len(l.split(c)) for l in sample_lines]
        if len(set(counts)) == 1 and counts[0] >= 3 and counts[0] > best:
            best, delim = counts[0], c
    sep = r"\s+" if delim == " " else delim

    # Header: se la prima riga non è numerica, trattala come header
    first = sample_lines[0].replace(",", " ").split()
    try:
        [float(t) for t in first[:3]]
        header = None
    except ValueError:
        header = 0

    df = pd.read_csv(
        io.StringIO(text), sep=sep, header=header,
        engine="python", comment="#", skip_blank_lines=True,
    )

    ncols = df.shape[1]
    if ncols < 3:
        raise ValueError(f"Formato non valido: {ncols} colonne (servono almeno 3).")

    if ncols >= 6:
        df = df.iloc[:, :6]
        df.columns = ["X", "Y", "Z", "R", "G", "B"]
        if df[["R", "G", "B"]].max().max() <= 1.0:
            df[["R", "G", "B"]] = (df[["R", "G", "B"]] * 255).astype(int)
        else:
            df[["R", "G", "B"]] = df[["R", "G", "B"]].astype(int).clip(0, 255)
    else:
        df = df.iloc[:, :3]
        df.columns = ["X", "Y", "Z"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# PARSER DXF / DWG (opzionali)
# ---------------------------------------------------------------------------
def _extract_points_from_doc(doc) -> pd.DataFrame:
    """Estrae coordinate 3D dalle entità del modelspace."""
    from ezdxf.colors import aci2rgb

    msp = doc.modelspace()
    rows = []
    for e in msp:
        dtype = e.dxftype()
        try:
            rgb = aci2rgb(e.dxf.color)
        except Exception:
            rgb = None

        if dtype == "POINT":
            p = e.dxf.location
            rows.append((p.x, p.y, p.z, rgb))
        elif dtype == "LINE":
            for p in (e.dxf.start, e.dxf.end):
                rows.append((p.x, p.y, p.z, rgb))
        elif dtype == "3DFACE":
            for attr in ("vtx0", "vtx1", "vtx2", "vtx3"):
                try:
                    v = getattr(e.dxf, attr)
                    rows.append((v.x, v.y, v.z, rgb))
                except AttributeError:
                    pass
        elif dtype == "INSERT":
            p = e.dxf.insert
            rows.append((p.x, p.y, p.z, rgb))

    if not rows:
        raise ValueError("Nessun punto trovato nel file CAD (POINT/LINE/3DFACE/INSERT).")

    xs, ys, zs, cols = zip(*rows)
    df = pd.DataFrame({"X": xs, "Y": ys, "Z": zs})
    if any(c is not None for c in cols):
        r = [c[0] if c else 200 for c in cols]
        g = [c[1] if c else 200 for c in cols]
        b = [c[2] if c else 200 for c in cols]
        df["R"], df["G"], df["B"] = r, g, b
    return df.dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_dxf(file_bytes: bytes) -> pd.DataFrame:
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf non installato: pip install ezdxf")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        doc = ezdxf.readfile(path)
    finally:
        try: os.unlink(path)
        except OSError: pass
    return _extract_points_from_doc(doc)


@st.cache_data(show_spinner=False)
def load_dwg(file_bytes: bytes) -> pd.DataFrame:
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf non installato: pip install ezdxf")
    tmpdir = Path(tempfile.mkdtemp(prefix="dwg_"))
    dwg = tmpdir / "input.dwg"
    dwg.write_bytes(file_bytes)
    try:
        doc = odafc.readfile(str(dwg))
    except odafc.ODAFCError as e:
        raise RuntimeError(
            "ODA File Converter mancante. Installa da "
            "https://www.opendesign.com/guestfiles/oda_file_converter\n"
            f"Dettaglio: {e}"
        )
    finally:
        try: dwg.unlink(); tmpdir.rmdir()
        except OSError: pass
    return _extract_points_from_doc(doc)


# ---------------------------------------------------------------------------
# VISTA PLANARE (Plotly) — selezione nativa via box/lasso di Streamlit
# ---------------------------------------------------------------------------
def build_topdown_plotly(df: pd.DataFrame, dragmode: str = "select",
                         circle: dict | None = None) -> go.Figure:
    """Scatter 2D (XY) con tool di selezione Plotly nativi."""
    fig = go.Figure(
        data=go.Scattergl(
            x=df["X"], y=df["Y"],
            mode="markers",
            marker=dict(
                size=3, color=df["Z"], colorscale="Viridis",
                opacity=0.8, line=dict(width=0),
            ),
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{marker.color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        dragmode=dragmode,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        xaxis=dict(
            scaleanchor="y", scaleratio=1,
            gridcolor="#2A2F3A", zerolinecolor="#3A4050",
        ),
        yaxis=dict(gridcolor="#2A2F3A", zerolinecolor="#3A4050"),
        showlegend=False,
    )
    # Overlay cerchio (modalità cerchio con slider)
    if circle is not None:
        cx, cy, r = circle["cx"], circle["cy"], circle["r"]
        fig.add_shape(
            type="circle",
            x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
            line=dict(color="#00E5FF", width=2),
            fillcolor="rgba(0, 229, 255, 0.15)",
        )
    return fig


def crop_by_selection(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    if sel["type"] == "rect":
        mask = (
            (df["X"].between(sel["x1"], sel["x2"])) &
            (df["Y"].between(sel["y1"], sel["y2"]))
        )
    elif sel["type"] == "circle":
        dx = df["X"] - sel["cx"]
        dy = df["Y"] - sel["cy"]
        mask = (dx * dx + dy * dy) <= sel["r"] ** 2
    else:
        return df
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# RICOSTRUZIONE SUPERFICIE
# ---------------------------------------------------------------------------
def build_cloud(df: pd.DataFrame) -> pv.PolyData:
    """PolyData con eventuale attributo RGB."""
    pts = df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)
    cloud = pv.PolyData(pts)
    if {"R", "G", "B"}.issubset(df.columns):
        cloud["RGB"] = df[["R", "G", "B"]].to_numpy(dtype=np.uint8)
    return cloud


def reconstruct(cloud: pv.PolyData, lod: int, method: str) -> pv.PolyData:
    """
    Ricostruzione superficie.
    - 'reconstruct': VTK SurfaceReconstructionFilter (buono per forme chiuse/organiche).
    - 'delaunay_2d': Delaunay 2.5D (ideale per edifici/terreno visti dall'alto).
    LOD 1..10 (10 = massimo dettaglio).
    """
    if cloud.n_points < 10:
        return cloud  # troppo pochi punti, fallback a nuvola

    bounds = np.array(cloud.bounds).reshape(3, 2)
    diag = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))

    if method == "delaunay_2d":
        # alpha limita i triangoli troppo grandi (buchi); cala al crescere del LOD
        alpha = diag / (10 * lod)
        return cloud.delaunay_2d(alpha=alpha)

    # reconstruct_surface: sample_spacing minore = più dettaglio
    sample_spacing = max(diag / (20 * lod), 1e-6)
    try:
        surf = cloud.reconstruct_surface(
            nbr_sz=20,
            sample_spacing=sample_spacing,
        )
    except Exception:
        surf = cloud.delaunay_2d()
    return surf


def _concave_hull_2d(xy: np.ndarray, alpha_factor: float = 3.0) -> np.ndarray:
    """
    Alpha-shape / concave hull 2D senza dipendenze esterne.
    alpha_factor basso → contorno più aderente (L-shape); alto → convex hull.
    Ritorna un array Nx2 di punti ordinati lungo il contorno.
    """
    from collections import Counter
    from scipy.spatial import ConvexHull, Delaunay

    if len(xy) < 4:
        return xy
    try:
        tri = Delaunay(xy)
    except Exception:
        return xy[ConvexHull(xy).vertices]

    simplices = tri.simplices
    v0, v1, v2 = xy[simplices[:, 0]], xy[simplices[:, 1]], xy[simplices[:, 2]]
    max_edge = np.maximum.reduce([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ])
    threshold = np.median(max_edge) * alpha_factor
    kept = simplices[max_edge <= threshold]
    if len(kept) == 0:
        return xy[ConvexHull(xy).vertices]

    # Edge che appaiono in UN SOLO triangolo = bordo
    edges = np.vstack([
        np.sort(kept[:, [0, 1]], axis=1),
        np.sort(kept[:, [1, 2]], axis=1),
        np.sort(kept[:, [2, 0]], axis=1),
    ])
    counts = Counter(map(tuple, edges.tolist()))
    boundary = [e for e, c in counts.items() if c == 1]
    if not boundary:
        return xy[ConvexHull(xy).vertices]

    # Stitch edges → polygon
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = boundary[0][0]
    loop = [start]
    current, prev = start, None
    safety = len(boundary) + 10
    while safety > 0:
        safety -= 1
        neigh = [n for n in adj.get(current, []) if n != prev]
        if not neigh:
            break
        nxt = neigh[0]
        if nxt == start and len(loop) > 2:
            break
        if nxt in loop:
            break
        loop.append(nxt)
        prev, current = current, nxt

    if len(loop) < 3:
        return xy[ConvexHull(xy).vertices]
    return xy[loop]


def extrude_rectangular_building(
    points: np.ndarray,
    floor_height: float = 3.0,
    window_w: float = 1.2,
    window_h: float = 1.5,
    window_spacing: float = 3.5,
    window_margin_side: float = 1.0,
    window_margin_top: float = 0.6,
    add_windows: bool = True,
) -> dict:
    """
    Ricostruzione rettangolare (Minimum Bounding Rectangle via PCA):
      1. PCA sui punti del corpo dell'edificio → orientamento principale
      2. Bounding rectangle orientato (robusto via percentili 2/98)
      3. Estrusione verticale → prisma rettangolare
      4. Griglia di finestre procedurali sui muri
    Ritorna dict: {walls, roof, floor, [windows]}.
    """
    if len(points) < 10:
        return {}

    z_sorted = np.sort(points[:, 2])
    z_floor = float(z_sorted[int(0.02 * (len(z_sorted) - 1))])
    z_roof = float(z_sorted[int(0.98 * (len(z_sorted) - 1))])
    if z_roof - z_floor < 1e-6:
        return {}

    # Usa punti del corpo (esclude terreno) per orientare il rettangolo
    z_thresh = z_floor + (z_roof - z_floor) * 0.35
    body = points[points[:, 2] > z_thresh]
    if len(body) < 20:
        body = points
    xy = body[:, :2].astype(np.float64)

    # PCA per trovare l'orientamento
    center_xy = xy.mean(axis=0)
    centered = xy - center_xy
    if len(centered) > 3000:
        idx = np.random.default_rng(0).choice(len(centered), 3000, replace=False)
        cov = np.cov(centered[idx].T)
    else:
        cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    axes = eigvecs  # 2x2: colonne = autovettori (assi principali)

    # Proietta sugli assi, prendi min/max robusti (2°/98° percentile)
    projected = centered @ axes
    pmin = np.percentile(projected, 2, axis=0)
    pmax = np.percentile(projected, 98, axis=0)

    # Protezione anti-degenere: se una dimensione è < 1% dell'altra,
    # la allarga in modo che il rettangolo abbia un rapporto sensato.
    dims = pmax - pmin
    if dims.max() > 0:
        min_ratio = 0.10
        target = dims.max() * min_ratio
        for k in range(2):
            if dims[k] < target:
                pad = (target - dims[k]) / 2
                pmin[k] -= pad
                pmax[k] += pad

    # 4 corner in coord principali → riporta in world
    corners_p = np.array([
        [pmin[0], pmin[1]],
        [pmax[0], pmin[1]],
        [pmax[0], pmax[1]],
        [pmin[0], pmax[1]],
    ])
    corners = (corners_p @ axes.T + center_xy).astype(np.float32)
    bcenter = corners.mean(axis=0)

    n = 4
    total_h = z_roof - z_floor
    n_floors = max(1, int(round(total_h / floor_height)))
    actual_floor_h = total_h / n_floors  # stira i piani per coprire esattamente

    # --- MURI: 4 quad con vertici duplicati (flat shading pulito) ---
    wall_v = np.zeros((4 * n, 3), dtype=np.float32)
    wall_f = np.zeros((2 * n, 3), dtype=np.int32)
    wall_meta = []  # (p0, p1, wdir, normal, wall_length) per il posizionamento finestre

    for i in range(n):
        j = (i + 1) % n
        base = 4 * i
        p0, p1 = corners[i], corners[j]
        wall_v[base + 0] = [p0[0], p0[1], z_floor]
        wall_v[base + 1] = [p1[0], p1[1], z_floor]
        wall_v[base + 2] = [p1[0], p1[1], z_roof]
        wall_v[base + 3] = [p0[0], p0[1], z_roof]
        wall_f[2 * i + 0] = [base, base + 1, base + 2]
        wall_f[2 * i + 1] = [base, base + 2, base + 3]

        wd = p1[:2] - p0[:2]
        wlen = float(np.linalg.norm(wd))
        wdir = wd / wlen if wlen > 0 else np.array([1.0, 0.0], dtype=np.float32)
        # Normale esterna (perpendicolare, orientata verso l'esterno)
        nrm = np.array([wdir[1], -wdir[0]], dtype=np.float32)
        edge_mid = (p0[:2] + p1[:2]) / 2
        if np.dot(nrm, edge_mid - bcenter[:2]) < 0:
            nrm = -nrm
        wall_meta.append((p0, p1, wdir, nrm, wlen))

    # --- TETTO: un singolo rettangolo (2 triangoli) ---
    roof_v = np.array([
        [corners[0, 0], corners[0, 1], z_roof],
        [corners[1, 0], corners[1, 1], z_roof],
        [corners[2, 0], corners[2, 1], z_roof],
        [corners[3, 0], corners[3, 1], z_roof],
    ], dtype=np.float32)
    roof_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    # --- PAVIMENTO (winding invertito per normale verso il basso) ---
    floor_v = np.array([
        [corners[0, 0], corners[0, 1], z_floor],
        [corners[1, 0], corners[1, 1], z_floor],
        [corners[2, 0], corners[2, 1], z_floor],
        [corners[3, 0], corners[3, 1], z_floor],
    ], dtype=np.float32)
    floor_f = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)

    result = {
        "walls": (wall_v, wall_f),
        "roof": (roof_v, roof_f),
        "floor": (floor_v, floor_f),
    }

    # --- FINESTRE: griglia procedurale piano × posizione ---
    if add_windows:
        win_v: list = []
        win_f: list = []
        offset = 0.04  # piccolo offset esterno per evitare z-fighting

        for (p0, p1, wdir, nrm, wlen) in wall_meta:
            available = wlen - 2 * window_margin_side
            if available < window_w:
                continue
            n_win = max(1, int(available / window_spacing))
            step = available / n_win

            for floor_idx in range(n_floors):
                f_z = z_floor + floor_idx * actual_floor_h
                win_z_b = f_z + (actual_floor_h - window_h) / 2
                win_z_t = win_z_b + window_h
                if win_z_t > z_roof - window_margin_top:
                    continue

                for wi in range(n_win):
                    t_along = window_margin_side + step * (wi + 0.5)
                    cx = p0[0] + wdir[0] * t_along + nrm[0] * offset
                    cy = p0[1] + wdir[1] * t_along + nrm[1] * offset
                    half = window_w / 2
                    pl_x, pl_y = cx - wdir[0] * half, cy - wdir[1] * half
                    pr_x, pr_y = cx + wdir[0] * half, cy + wdir[1] * half
                    base = len(win_v)
                    win_v.extend([
                        [pl_x, pl_y, win_z_b],
                        [pr_x, pr_y, win_z_b],
                        [pr_x, pr_y, win_z_t],
                        [pl_x, pl_y, win_z_t],
                    ])
                    win_f.extend([
                        [base, base + 1, base + 2],
                        [base, base + 2, base + 3],
                    ])

        if win_v:
            result["windows"] = (
                np.asarray(win_v, dtype=np.float32),
                np.asarray(win_f, dtype=np.int32),
            )

    return result


def extrude_building(points: np.ndarray, lod: int, use_concave: bool = True) -> dict:
    """
    Ricostruzione prismatica di un edificio:
      - ground/roof stimati dai percentili Z
      - footprint 2D dai punti della metà superiore (concave o convex hull)
      - estrusione verticale
    Ritorna dict con mesh separate: 'walls', 'roof', 'floor'.
    """
    if len(points) < 10:
        return {}

    z_sorted = np.sort(points[:, 2])
    z_floor = float(z_sorted[int(0.02 * (len(z_sorted) - 1))])
    z_roof = float(z_sorted[int(0.98 * (len(z_sorted) - 1))])
    if z_roof - z_floor < 1e-6:
        return {}

    # Usa la parte alta per il footprint (esclude terreno/vegetazione bassa)
    z_thresh = z_floor + (z_roof - z_floor) * 0.35
    body = points[points[:, 2] > z_thresh]
    if len(body) < 20:
        body = points

    xy = body[:, :2].astype(np.float32)
    # Downsample hull per velocità
    if len(xy) > 4000:
        idx = np.random.default_rng(0).choice(len(xy), 4000, replace=False)
        xy = xy[idx]

    # LOD: 1..10 → alpha_factor 6..1.5 (alto = conservativo/convex, basso = aderente)
    alpha_factor = max(1.5, 7.0 - 0.55 * lod)
    if use_concave:
        footprint = _concave_hull_2d(xy, alpha_factor=alpha_factor)
    else:
        from scipy.spatial import ConvexHull
        footprint = xy[ConvexHull(xy).vertices]

    n = len(footprint)
    if n < 3:
        return {}

    # Protezione anti-degenere: se la bbox del footprint è troppo sottile,
    # vuol dire che i punti erano quasi collineari → fallback a rettangolo PCA
    fp_w = float(footprint[:, 0].max() - footprint[:, 0].min())
    fp_h = float(footprint[:, 1].max() - footprint[:, 1].min())
    if min(fp_w, fp_h) < 0.05 * max(fp_w, fp_h, 1e-9):
        return extrude_rectangular_building(points, add_windows=False)

    # --- MESH MURI: un quad per ogni edge, vertici separati per flatshading netto
    wall_v = np.zeros((4 * n, 3), dtype=np.float32)
    wall_f = np.zeros((2 * n, 3), dtype=np.int32)
    for i in range(n):
        j = (i + 1) % n
        base = 4 * i
        wall_v[base + 0] = [footprint[i, 0], footprint[i, 1], z_floor]
        wall_v[base + 1] = [footprint[j, 0], footprint[j, 1], z_floor]
        wall_v[base + 2] = [footprint[j, 0], footprint[j, 1], z_roof]
        wall_v[base + 3] = [footprint[i, 0], footprint[i, 1], z_roof]
        wall_f[2 * i + 0] = [base, base + 1, base + 2]
        wall_f[2 * i + 1] = [base, base + 2, base + 3]

    # --- MESH TETTO: triangle-fan dal centroide
    cx, cy = float(footprint[:, 0].mean()), float(footprint[:, 1].mean())
    roof_v = np.zeros((n + 1, 3), dtype=np.float32)
    roof_v[:n, :2] = footprint
    roof_v[:n, 2] = z_roof
    roof_v[n] = [cx, cy, z_roof]
    roof_f = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        roof_f[i] = [i, (i + 1) % n, n]

    # --- MESH PAVIMENTO: triangle-fan invertito
    floor_v = np.zeros((n + 1, 3), dtype=np.float32)
    floor_v[:n, :2] = footprint
    floor_v[:n, 2] = z_floor
    floor_v[n] = [cx, cy, z_floor]
    floor_f = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        floor_f[i] = [(i + 1) % n, i, n]  # winding invertito

    return {
        "walls": (wall_v, wall_f),
        "roof": (roof_v, roof_f),
        "floor": (floor_v, floor_f),
    }


def reconstruct_mesh_arrays(
    points: np.ndarray,
    lod: int,
    method: str,
    rect_params: dict | None = None,
    max_points: int = 60_000,
    max_cells: int = 150_000,
) -> dict:
    """
    Ritorna un dict {nome_parte: (verts, faces)}.
    - 'building_rect': rettangolo PCA + finestre procedurali (incarto energia)
    - 'building_*':    parti separate (walls/roof/floor) per look CAD
    - 'reconstruct_surface' / 'delaunay_2d': unica mesh 'surface'
    """
    if len(points) > max_points:
        idx = np.random.default_rng(42).choice(len(points), max_points, replace=False)
        points = points[idx]

    # --- EDIFICIO RETTANGOLARE IDEALIZZATO (con finestre) ---
    if method == "building_rect":
        rp = rect_params or {}
        return extrude_rectangular_building(
            points,
            floor_height=rp.get("floor_height", 3.0),
            window_w=rp.get("window_w", 1.2),
            window_h=rp.get("window_h", 1.5),
            window_spacing=rp.get("window_spacing", 3.5),
            add_windows=rp.get("add_windows", True),
        )

    # --- ESTRUSIONE EDIFICIO (footprint reale) ---
    if method in ("building_convex", "building_concave"):
        return extrude_building(
            points, lod=lod, use_concave=(method == "building_concave")
        )

    # --- RICOSTRUZIONE SUPERFICIE (organica) ---
    cloud = pv.PolyData(points.astype(np.float32))
    surf = reconstruct(cloud, lod, method)
    try:
        surf = surf.triangulate()
    except Exception:
        pass
    if surf.n_cells > max_cells:
        try:
            ratio = 1.0 - (max_cells / surf.n_cells)
            surf = surf.decimate(ratio).triangulate()
        except Exception:
            pass

    verts = np.asarray(surf.points, dtype=np.float32)
    if surf.n_cells > 0 and surf.faces.size > 0:
        faces = surf.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    else:
        faces = np.zeros((0, 3), dtype=np.int32)
    return {"surface": (verts, faces)}


# Palette CAD architettonica (light theme come da riferimento)
_PART_COLORS = {
    "walls":   "#C9CCD1",   # muri grigio chiaro
    "roof":    "#6B5A3E",   # tetto marrone piatto
    "floor":   "#3D3D3D",   # pavimento scuro (raramente visibile)
    "windows": "#7DA9D1",   # finestre azzurro vetro
    "surface": "#B8C4D6",   # superficie organica (fallback)
}


def _mesh_trace(verts, faces, color, flatshading=True, show_edges=False):
    """Mesh3d trace con lighting architettonico + opzionale wireframe."""
    traces = [go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        flatshading=flatshading,
        lighting=dict(
            ambient=0.45, diffuse=0.85,
            specular=0.15, roughness=0.85, fresnel=0.05,
        ),
        lightposition=dict(x=10_000, y=10_000, z=20_000),
        hoverinfo="skip",
        showscale=False,
    )]
    if show_edges and len(faces) > 0:
        # Segmenti per ogni edge di ogni triangolo (con NaN come separatore)
        xs, ys, zs = [], [], []
        for tri in faces:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                xs += [verts[tri[a], 0], verts[tri[b], 0], None]
                ys += [verts[tri[a], 1], verts[tri[b], 1], None]
                zs += [verts[tri[a], 2], verts[tri[b], 2], None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#2B2B2B", width=2),
            hoverinfo="skip", showlegend=False,
        ))
    return traces


def build_3d_figure(mesh_dict: dict, raw_points: np.ndarray | None = None,
                    theme: str = "light", show_edges: bool = True) -> go.Figure:
    """
    Costruisce figura Plotly 3D con mesh separate per parte (walls/roof/...).
    theme: 'light' (stile CAD architettonico) o 'dark'.
    """
    fig = go.Figure()

    if theme == "light":
        bg, grid, text = "#F5F5F7", "#D0D0D4", "#1A1A1A"
    else:
        bg, grid, text = "#0E1117", "#2A2F3A", "#FAFAFA"

    # Mesh per ogni parte, con flatshading per spigoli netti
    any_mesh = False
    for name, (v, f) in mesh_dict.items():
        if len(f) == 0:
            continue
        any_mesh = True
        color = _PART_COLORS.get(name, "#B8C4D6")
        # Pavimento e finestre non mostrano gli edge (rumore visivo)
        edges = show_edges and name not in ("floor", "windows")
        for tr in _mesh_trace(v, f, color, flatshading=True, show_edges=edges):
            fig.add_trace(tr)

    # Fallback a scatter se la ricostruzione ha prodotto 0 facce
    if not any_mesh and mesh_dict:
        v, _ = next(iter(mesh_dict.values()))
        if len(v) > 0:
            fig.add_trace(go.Scatter3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                mode="markers",
                marker=dict(size=2, color=v[:, 2], colorscale="Viridis"),
                name="Punti", hoverinfo="skip",
            ))

    # Nuvola raw in sovraimpressione (opzionale)
    if raw_points is not None and len(raw_points) > 0:
        if len(raw_points) > 25_000:
            idx = np.random.default_rng(0).choice(len(raw_points), 25_000, replace=False)
            raw_points = raw_points[idx]
        fig.add_trace(go.Scatter3d(
            x=raw_points[:, 0], y=raw_points[:, 1], z=raw_points[:, 2],
            mode="markers",
            marker=dict(size=1.2, color="#0078D4", opacity=0.45),
            name="Nuvola", hoverinfo="skip",
        ))

    axis = dict(
        backgroundcolor=bg, gridcolor=grid,
        showbackground=True, zerolinecolor=grid,
        color=text, showspikes=False,
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis,
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9),
                        up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=text),
        margin=dict(l=0, r=0, t=0, b=0),
        height=680,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("current_file", None)
    ss.setdefault("cropped_df", None)
    ss.setdefault("last_selection", None)
    ss.setdefault("mesh_data", None)  # dict: {"verts": ndarray, "faces": ndarray, "points": ndarray|None}

_init_state()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🛰️ Point Cloud → CAD Viewer")
st.caption("Carica `.xyz` / `.dxf` / `.dwg`, seleziona un'area e ottieni il modello 3D ricostruito.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controlli")

    z_threshold = st.slider(
        "Soglia altezza (Z)", 0.0, 1.0, 0.0, 0.01,
        help="Frazione del range Z sotto cui i punti vengono esclusi (rimuove il terreno).",
    )
    lod = st.slider("Livello di Dettaglio", 1, 10, 5,
                    help="Più alto = superficie più fine (ma più lento).")
    _method_labels = {
        "building_rect":    "🏠 Edificio Rettangolare + Finestre",
        "building_concave": "🏛️ Estrusione Edificio (concave)",
        "building_convex":  "🏛️ Estrusione Edificio (convex)",
        "reconstruct_surface": "Reconstruct Surface (VTK)",
        "delaunay_2d": "Delaunay 2.5D (superficie)",
    }
    method = st.radio(
        "Metodo ricostruzione",
        options=list(_method_labels.keys()),
        format_func=lambda x: _method_labels[x],
        index=0,
        help=(
            "**Edificio Rettangolare + Finestre** → modello idealizzato con muri "
            "dritti, tetto piatto e griglia di finestre (per incarto energia).\n"
            "**Estrusione Edificio** → look CAD pulito con footprint reale.\n"
            "**Reconstruct Surface** → superficie organica continua.\n"
            "**Delaunay 2.5D** → superficie aderente ai punti, buona per terreni."
        ),
    )

    # Parametri specifici per edificio rettangolare (incarto energia)
    rect_params: dict = {}
    if method == "building_rect":
        st.markdown("**🏠 Parametri edificio**")
        rect_params["floor_height"] = st.slider(
            "Altezza piano (m)", 2.2, 5.0, 3.0, 0.1,
            help="Altezza di interpiano usata per posizionare le finestre.",
        )
        rect_params["add_windows"] = st.checkbox("Aggiungi finestre", value=True)
        if rect_params["add_windows"]:
            rect_params["window_w"] = st.slider("Larghezza finestra (m)", 0.6, 2.5, 1.2, 0.1)
            rect_params["window_h"] = st.slider("Altezza finestra (m)", 0.8, 2.4, 1.5, 0.1)
            rect_params["window_spacing"] = st.slider(
                "Passo orizzontale (m)", 2.0, 8.0, 3.5, 0.1,
                help="Distanza tipica tra finestre: l'algoritmo le distribuisce uniformi.",
            )

    theme = st.radio("Tema 3D", ["light", "dark"], horizontal=True,
                     format_func=lambda x: "Chiaro (CAD)" if x == "light" else "Scuro")
    show_edges = st.checkbox("Mostra spigoli (wireframe)", value=True)
    shape_mode = st.radio("Forma selezione", ["rect", "circle"],
                          format_func=lambda x: "Rettangolo" if x == "rect" else "Cerchio",
                          horizontal=True)
    show_points = st.checkbox("Sovrapponi nuvola di punti", value=False)

    st.markdown("---")
    st.markdown(
        "**Formati supportati**\n\n"
        "- `.xyz` / `.txt` / `.csv`\n"
        "- `.dxf` — POINT/LINE/3DFACE/INSERT\n"
        "- `.dwg` — richiede ODA File Converter"
    )
    if not EZDXF_AVAILABLE:
        st.warning("`ezdxf` non installato: DXF/DWG disabilitati.")

# Upload
uploaded = st.file_uploader(
    "Trascina qui il file (.xyz / .dxf / .dwg)",
    type=["xyz", "txt", "csv", "dxf", "dwg"],
)

if uploaded is None:
    st.info("👆 Carica un file per iniziare.")
    st.stop()

# Reset stato se il file è cambiato
file_id = getattr(uploaded, "file_id", uploaded.name)
if st.session_state.current_file != file_id:
    st.session_state.current_file = file_id
    st.session_state.cropped_df = None
    st.session_state.last_selection = None
    st.session_state.mesh_data = None

# Parsing
ext = Path(uploaded.name).suffix.lower()
try:
    with st.spinner(f"Parsing {ext}..."):
        if ext == ".dxf":
            df_full = load_dxf(uploaded.getvalue())
        elif ext == ".dwg":
            df_full = load_dwg(uploaded.getvalue())
        else:
            df_full = load_xyz(uploaded.getvalue())
except Exception as e:
    st.error(f"Errore nel parsing: {e}")
    st.stop()

# Filtro Z (frazione del range)
z_min, z_max = float(df_full["Z"].min()), float(df_full["Z"].max())
z_cut = z_min + z_threshold * (z_max - z_min)
df_filtered = df_full[df_full["Z"] >= z_cut].reset_index(drop=True)

if df_filtered.empty:
    st.error("Nessun punto sopra la soglia Z selezionata.")
    st.stop()

# Statistiche
c1, c2, c3, c4 = st.columns(4)
c1.metric("Punti totali", f"{len(df_full):,}")
c2.metric("Dopo filtro Z", f"{len(df_filtered):,}")
c3.metric("ΔZ", f"{df_filtered['Z'].max() - df_filtered['Z'].min():.2f}")
c4.metric("RGB", "Sì" if {"R","G","B"}.issubset(df_full.columns) else "No")

st.markdown("---")

# =========================================================================
# STEP 1 — VISTA PLANARE + SELEZIONE
# =========================================================================
left, right = st.columns([3, 2])

# Downsample per il rendering (WebGL regge bene 100k punti)
preview_src = df_filtered
if len(preview_src) > 100_000:
    preview_src = df_filtered.sample(100_000, random_state=42)

# Bound XY su dati filtrati (serve al cerchio)
x_min, x_max = float(df_filtered["X"].min()), float(df_filtered["X"].max())
y_min, y_max = float(df_filtered["Y"].min()), float(df_filtered["Y"].max())

selection: dict | None = None

with left:
    st.subheader("📐 Vista planare — seleziona l'area di interesse")

    if shape_mode == "rect":
        st.caption("Usa lo strumento **Box Select** / **Lasso** nella toolbar in alto a destra.")
        fig = build_topdown_plotly(preview_src, dragmode="select")
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["box", "lasso"],
            key=f"selector_{file_id}",
        )

        # Estrai bbox (rect) o poligono (lasso) dall'evento Streamlit
        sel_data = event.get("selection", {}) if isinstance(event, dict) else {}
        boxes = sel_data.get("box") or []
        lassos = sel_data.get("lasso") or []
        if boxes:
            b = boxes[0]
            xs = sorted(b["x"]); ys = sorted(b["y"])
            selection = {"type": "rect", "x1": xs[0], "x2": xs[1], "y1": ys[0], "y2": ys[1]}
        elif lassos:
            l = lassos[0]
            xs, ys = list(l["x"]), list(l["y"])
            selection = {
                "type": "rect",
                "x1": min(xs), "x2": max(xs),
                "y1": min(ys), "y2": max(ys),
            }

    else:  # cerchio con raggio modificabile via slider
        st.caption("Regola centro e raggio con gli slider nel pannello a destra.")
        # Placeholder, il plot viene ridisegnato dopo aver letto gli slider
        circle_placeholder = st.empty()

with right:
    st.subheader("🎯 Selezione")

    if shape_mode == "circle":
        cx = st.slider("Centro X", x_min, x_max, float((x_min + x_max) / 2))
        cy = st.slider("Centro Y", y_min, y_max, float((y_min + y_max) / 2))
        max_r = max(x_max - x_min, y_max - y_min) / 2
        r = st.slider("Raggio", max_r / 100, max_r, max_r / 4)
        selection = {"type": "circle", "cx": cx, "cy": cy, "r": r}

        # Ridisegna il plot con il cerchio sovrapposto
        with left:
            fig = build_topdown_plotly(preview_src, dragmode="pan", circle=selection)
            circle_placeholder.plotly_chart(fig, use_container_width=True, key=f"circle_{file_id}")

    if selection:
        if selection["type"] == "rect":
            st.code(
                f"Rettangolo\n"
                f"X: [{selection['x1']:.2f}, {selection['x2']:.2f}]\n"
                f"Y: [{selection['y1']:.2f}, {selection['y2']:.2f}]\n"
                f"Area: {(selection['x2']-selection['x1'])*(selection['y2']-selection['y1']):.2f}"
            )
        else:
            st.code(
                f"Cerchio\n"
                f"Centro: ({selection['cx']:.2f}, {selection['cy']:.2f})\n"
                f"Raggio: {selection['r']:.2f}\n"
                f"Area: {np.pi * selection['r']**2:.2f}"
            )
    else:
        st.info("Usa Box/Lasso sul grafico per selezionare un'area.")

    st.markdown("")
    confirm = st.button(
        "✅ Conferma Selezione e Genera 3D",
        type="primary",
        disabled=selection is None,
        use_container_width=True,
    )
    gen_full = st.button(
        "🏛️ Genera 3D (tutta la nuvola)",
        use_container_width=True,
        help="Genera il modello 3D senza ritagliare.",
    )
    if st.button("↩️ Reset", use_container_width=True):
        st.session_state.cropped_df = None
        st.session_state.mesh_data = None

    # Logica di generazione
    source_df = None
    if confirm and selection is not None:
        cropped = crop_by_selection(df_filtered, selection)
        if cropped.empty:
            st.error("Nessun punto nella selezione.")
        else:
            st.session_state.cropped_df = cropped
            source_df = cropped
            st.success(f"Ritagliati {len(cropped):,} punti.")
    elif gen_full:
        source_df = df_filtered

    if source_df is not None:
        with st.spinner(f"Ricostruzione superficie su {len(source_df):,} punti..."):
            try:
                pts = source_df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)
                mesh_parts = reconstruct_mesh_arrays(
                    points=pts, lod=lod, method=method,
                    rect_params=rect_params,
                )
                # Diagnostica: mesh quasi degenere?
                all_v = [v for v, _ in mesh_parts.values() if len(v) > 0]
                if all_v:
                    stacked = np.concatenate(all_v, axis=0)
                    extent = stacked.max(axis=0) - stacked.min(axis=0)
                    if extent.max() > 0 and extent.min() < 0.02 * extent.max():
                        st.warning(
                            "⚠️ Geometria molto sottile: la nuvola è anisotropa. "
                            "Prova il metodo **🏠 Edificio Rettangolare + Finestre** "
                            "per ottenere un volume coerente."
                        )
                st.session_state.mesh_data = {
                    "parts": mesh_parts,
                    "points": pts if show_points else None,
                }
            except Exception as e:
                st.error(f"Errore nella ricostruzione 3D: {e}")
                st.session_state.mesh_data = None

# =========================================================================
# STEP 2 — VISTA 3D RICOSTRUITA
# =========================================================================
st.markdown("---")
st.subheader("🏛️ Modello 3D ricostruito")

if st.session_state.mesh_data:
    md = st.session_state.mesh_data
    parts = md.get("parts", {})
    fig_3d = build_3d_figure(
        parts, raw_points=md.get("points"),
        theme=theme, show_edges=show_edges,
    )
    st.plotly_chart(fig_3d, use_container_width=True, theme=None)

    total_v = sum(len(v) for v, _ in parts.values())
    total_f = sum(len(f) for _, f in parts.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Parti mesh", len(parts))
    c2.metric("Vertici", f"{total_v:,}")
    c3.metric("Triangoli", f"{total_f:,}")

    df_3d = st.session_state.cropped_df if st.session_state.cropped_df is not None else df_filtered
    with st.expander("🔎 Anteprima dati"):
        st.dataframe(df_3d.head(200), use_container_width=True)
else:
    st.info(
        "👉 Seleziona un'area e clicca **Conferma Selezione e Genera 3D**, "
        "oppure **Genera 3D (tutta la nuvola)** per ricostruire il modello."
    )
