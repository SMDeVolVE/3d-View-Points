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

# LAS/LAZ loader opzionale
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

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
# PARSER .LAS / .LAZ
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_las(file_bytes: bytes, suffix: str = ".las") -> pd.DataFrame:
    """Parser per file LAS/LAZ (LiDAR). Estrae XYZ e RGB se presenti.

    I file .laz (compressi) richiedono il backend `laszip` o `lazrs`
    installato insieme a laspy: `pip install laspy[lazrs]`.
    """
    if not LASPY_AVAILABLE:
        raise RuntimeError("laspy non installato: pip install laspy[lazrs]")

    # laspy legge da path o da stream: usiamo un tempfile per compatibilità
    # massima con .laz che richiede backend nativo.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        try:
            las = laspy.read(path)
        except laspy.errors.LaspyException as e:
            # Tipico: file .laz senza backend di decompressione
            if suffix.lower() == ".laz":
                raise RuntimeError(
                    "File .laz: manca il backend di decompressione. "
                    "Installa con `pip install laspy[lazrs]` "
                    "oppure `pip install laspy[laszip]`.\n"
                    f"Dettaglio: {e}"
                ) from e
            raise
    finally:
        try: os.unlink(path)
        except OSError: pass

    # Coordinate scalate (laspy applica automaticamente scale/offset)
    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)
    zs = np.asarray(las.z, dtype=np.float64)

    # --- RICENTRATURA per coordinate georeferenziate (CRUCIALE) ---------------
    # I LAS aerei/topografici usano spesso UTM / CH1903+ con X,Y ~1e6-1e7.
    # In float32 (usato a valle) la risoluzione a 2.7e6 è ~0.25 m: punti
    # distanti 10 cm collassano sullo stesso valore → la nuvola diventa
    # un piano ("geometria anisotropa") e la ricostruzione fallisce.
    # Sottraiamo un offset intero per ogni asse che lo richiede: le
    # coordinate restano fisicamente in metri, solo traslate vicino a 0.
    offsets = {}
    PRECISION_THRESHOLD = 1e5  # sopra questo valore float32 perde cm
    for name, arr in (("X", xs), ("Y", ys), ("Z", zs)):
        if arr.size and np.nanmax(np.abs(arr)) > PRECISION_THRESHOLD:
            off = float(np.floor(np.nanmin(arr)))
            arr -= off
            offsets[name] = off

    df = pd.DataFrame({"X": xs, "Y": ys, "Z": zs})
    # Esponi gli offset applicati per uso/diagnostica a valle
    df.attrs["las_offsets"] = offsets
    # parse_crs richiede pyproj (opzionale): se manca, non bloccare il caricamento
    try:
        df.attrs["las_crs"] = las.header.parse_crs()
    except Exception:
        df.attrs["las_crs"] = None

    # RGB è presente nei Point Format 2, 3, 5, 7, 8, 10 (LAS 1.2+)
    has_rgb = all(hasattr(las, c) for c in ("red", "green", "blue"))
    if has_rgb:
        r = np.asarray(las.red, dtype=np.int64)
        g = np.asarray(las.green, dtype=np.int64)
        b = np.asarray(las.blue, dtype=np.int64)
        # Nei LAS l'RGB è spesso a 16 bit (0–65535): riportalo a 8 bit se serve
        maxv = int(max(r.max() if r.size else 0,
                       g.max() if g.size else 0,
                       b.max() if b.size else 0))
        if maxv > 255:
            r = (r >> 8).astype(np.int64)
            g = (g >> 8).astype(np.int64)
            b = (b >> 8).astype(np.int64)
        df["R"] = np.clip(r, 0, 255).astype(int)
        df["G"] = np.clip(g, 0, 255).astype(int)
        df["B"] = np.clip(b, 0, 255).astype(int)

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
                         circle: dict | None = None,
                         rect_rotated: dict | None = None) -> go.Figure:
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
    # Overlay rettangolo ruotato (modalità "rect_rotated")
    if rect_rotated is not None:
        cx, cy = rect_rotated["cx"], rect_rotated["cy"]
        w, h = rect_rotated["w"], rect_rotated["h"]
        theta = np.deg2rad(rect_rotated["angle_deg"])
        # 4 corner nel frame locale (senso orario)
        local = np.array([
            [-w / 2, -h / 2],
            [ w / 2, -h / 2],
            [ w / 2,  h / 2],
            [-w / 2,  h / 2],
            [-w / 2, -h / 2],  # chiudi
        ])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        world = local @ R.T + np.array([cx, cy])
        fig.add_trace(go.Scatter(
            x=world[:, 0], y=world[:, 1],
            mode="lines",
            line=dict(color="#00E5FF", width=2),
            fill="toself",
            fillcolor="rgba(0, 229, 255, 0.15)",
            hoverinfo="skip",
            showlegend=False,
        ))
    return fig


def crop_by_selection(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    if sel["type"] == "rect":
        mask = (
            (df["X"].between(sel["x1"], sel["x2"])) &
            (df["Y"].between(sel["y1"], sel["y2"]))
        )
    elif sel["type"] == "rect_rotated":
        # Ruota i punti nel frame locale del rettangolo, poi test bbox
        cx, cy = sel["cx"], sel["cy"]
        w, h = sel["w"], sel["h"]
        theta = np.deg2rad(sel["angle_deg"])
        dx = df["X"].to_numpy() - cx
        dy = df["Y"].to_numpy() - cy
        c, s = np.cos(-theta), np.sin(-theta)
        xr = dx * c - dy * s
        yr = dx * s + dy * c
        mask = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
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


def _rdp_simplify(poly: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker su poligono chiuso. Sceglie come "ancora" i due
    vertici più distanti (diametro del poligono), spezza in due polilinee
    aperte e le semplifica ricorsivamente. Più epsilon è alto, più vertici
    vengono eliminati.
    """
    n = len(poly)
    if n < 4 or epsilon <= 0:
        return poly

    # diametro: coppia di vertici più distanti
    d2 = np.sum((poly[:, None, :] - poly[None, :, :]) ** 2, axis=-1)
    i0, i1 = np.unravel_index(d2.argmax(), d2.shape)
    if i0 > i1:
        i0, i1 = i1, i0

    def rdp_open(pts: np.ndarray, eps: float) -> np.ndarray:
        if len(pts) < 3:
            return pts
        start, end = pts[0], pts[-1]
        seg = end - start
        L = np.linalg.norm(seg)
        if L < 1e-12:
            d = np.linalg.norm(pts - start, axis=1)
        else:
            n_hat = np.array([-seg[1], seg[0]]) / L
            d = np.abs((pts - start) @ n_hat)
        idx = int(np.argmax(d))
        if d[idx] > eps:
            left = rdp_open(pts[:idx + 1], eps)
            right = rdp_open(pts[idx:], eps)
            return np.vstack([left[:-1], right])
        return np.vstack([start[None, :], end[None, :]])

    pl1 = poly[i0:i1 + 1]
    pl2 = np.vstack([poly[i1:], poly[:i0 + 1]])
    s1 = rdp_open(pl1, epsilon)
    s2 = rdp_open(pl2, epsilon)
    result = np.vstack([s1[:-1], s2[:-1]])
    return result if len(result) >= 3 else poly


def _remove_collinear(poly: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Rimuove vertici che giacciono (quasi) sul segmento tra i vicini."""
    n = len(poly)
    if n < 4:
        return poly
    keep = []
    for i in range(n):
        p_prev = poly[(i - 1) % n]
        p = poly[i]
        p_next = poly[(i + 1) % n]
        a = p - p_prev
        b = p_next - p
        cross = abs(a[0] * b[1] - a[1] * b[0])
        if cross > tol:
            keep.append(p)
    if len(keep) < 3:
        return poly
    return np.asarray(keep)


def _concave_hull_2d(xy: np.ndarray, alpha_factor: float = 3.0,
                     simplify: float = 0.0) -> np.ndarray:
    """
    Alpha-shape / concave hull 2D senza dipendenze esterne.
    alpha_factor basso → contorno più aderente (L-shape); alto → convex hull.
    Se `simplify > 0`, applica RDP con quella tolleranza post-estrazione.
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
    result = xy[loop]

    # Semplificazione aggressiva: rimuove micro-spigoli e zig-zag
    if simplify > 0 and len(result) > 4:
        result = _rdp_simplify(np.asarray(result, dtype=np.float64), simplify)
        result = _remove_collinear(result)
    return np.asarray(result, dtype=np.float64)


def _orthogonalize_polygon(poly: np.ndarray) -> np.ndarray:
    """
    Snap-to-grid del perimetro: ruota al frame PCA dominante, forza edge
    alternati orizzontali/verticali, media le coordinate lungo ogni edge,
    poi ruota indietro. Output: poligono con tutti gli angoli a 90°.
    """
    n = len(poly)
    if n < 4:
        return poly.astype(np.float32)

    center = poly.mean(axis=0)
    centered = poly - center
    cov = np.cov(centered.T)
    _, vecs = np.linalg.eigh(cov)
    principal = vecs[:, -1]
    theta = np.arctan2(principal[1], principal[0])
    # riduci al primo quadrante: vogliamo un angolo di rotazione "piccolo"
    theta = (theta + np.pi / 4) % (np.pi / 2) - np.pi / 4

    c, s = np.cos(-theta), np.sin(-theta)
    R_fwd = np.array([[c, -s], [s, c]], dtype=np.float64)
    rotated = centered @ R_fwd.T

    # classifica ogni edge come orizzontale (y-costante) o verticale
    edge_h = np.zeros(n, dtype=bool)
    for i in range(n):
        dx = rotated[(i + 1) % n, 0] - rotated[i, 0]
        dy = rotated[(i + 1) % n, 1] - rotated[i, 1]
        edge_h[i] = abs(dx) >= abs(dy)

    # forza alternanza: se due edge adiacenti hanno stesso tipo, flippa il più corto
    for _pass in range(3):
        changed = False
        for i in range(n):
            if edge_h[i] == edge_h[(i + 1) % n]:
                li = np.hypot(rotated[(i + 1) % n, 0] - rotated[i, 0],
                              rotated[(i + 1) % n, 1] - rotated[i, 1])
                j = (i + 1) % n
                lj = np.hypot(rotated[(j + 1) % n, 0] - rotated[j, 0],
                              rotated[(j + 1) % n, 1] - rotated[j, 1])
                if lj <= li:
                    edge_h[j] = not edge_h[j]
                else:
                    edge_h[i] = not edge_h[i]
                changed = True
        if not changed:
            break

    snapped = rotated.copy()
    for i in range(n):
        j = (i + 1) % n
        if edge_h[i]:
            avg_y = (rotated[i, 1] + rotated[j, 1]) / 2
            snapped[i, 1] = avg_y
            snapped[j, 1] = avg_y
        else:
            avg_x = (rotated[i, 0] + rotated[j, 0]) / 2
            snapped[i, 0] = avg_x
            snapped[j, 0] = avg_x

    # rimuovi vertici duplicati/collineari
    cleaned = [snapped[0]]
    for k in range(1, n):
        if np.linalg.norm(snapped[k] - cleaned[-1]) > 1e-4:
            cleaned.append(snapped[k])
    cleaned = np.array(cleaned)
    # se primo e ultimo coincidono, eliminane uno
    if len(cleaned) > 3 and np.linalg.norm(cleaned[0] - cleaned[-1]) < 1e-4:
        cleaned = cleaned[:-1]
    if len(cleaned) < 4:
        return poly.astype(np.float32)

    c2, s2 = np.cos(theta), np.sin(theta)
    R_back = np.array([[c2, -s2], [s2, c2]], dtype=np.float64)
    out = cleaned @ R_back.T + center
    return out.astype(np.float32)


def _snap_alternating_hv(poly: np.ndarray) -> np.ndarray:
    """
    Dato un poligono in un frame già allineato, forza TUTTI gli edge a essere
    o perfettamente orizzontali o verticali, alternati. Ogni edge classificato:
      - |dx| >= |dy| → orizzontale (fissa y)
      - altrimenti  → verticale (fissa x)
    Poi l'alternanza viene forzata flippando gli edge "sbagliati" (i più corti).
    Infine ricostruisce i vertici come intersezione di linee consecutive.
    Input: (N, 2) float. Output: (M, 2) orthogonal, M <= N.
    """
    n = len(poly)
    if n < 4:
        return poly
    poly = np.asarray(poly, dtype=np.float64)

    # classifica ogni edge
    edge_h = np.zeros(n, dtype=bool)
    for i in range(n):
        j = (i + 1) % n
        dx = poly[j, 0] - poly[i, 0]
        dy = poly[j, 1] - poly[i, 1]
        edge_h[i] = abs(dx) >= abs(dy)

    # forza alternanza
    for _ in range(3):
        changed = False
        for i in range(n):
            if edge_h[i] == edge_h[(i + 1) % n]:
                j = (i + 1) % n
                li = np.hypot(poly[(i + 1) % n, 0] - poly[i, 0],
                              poly[(i + 1) % n, 1] - poly[i, 1])
                lj = np.hypot(poly[(j + 1) % n, 0] - poly[j, 0],
                              poly[(j + 1) % n, 1] - poly[j, 1])
                if lj <= li:
                    edge_h[j] = not edge_h[j]
                else:
                    edge_h[i] = not edge_h[i]
                changed = True
        if not changed:
            break

    # raggruppa edge consecutivi con stessa direzione → una sola linea
    groups: list[tuple[bool, list[int]]] = []
    cur_dir = edge_h[0]
    cur_edges = [0]
    for i in range(1, n):
        if edge_h[i] == cur_dir:
            cur_edges.append(i)
        else:
            groups.append((cur_dir, cur_edges))
            cur_dir = edge_h[i]
            cur_edges = [i]
    groups.append((cur_dir, cur_edges))
    # wraparound merge
    if len(groups) > 1 and groups[0][0] == groups[-1][0]:
        groups[0] = (groups[0][0], groups[-1][1] + groups[0][1])
        groups.pop()

    if len(groups) < 4:
        # degenere: fallback al bounding rect del poligono
        xm, ym = poly.min(axis=0)
        xM, yM = poly.max(axis=0)
        return np.array([[xm, ym], [xM, ym], [xM, yM], [xm, yM]])

    # per ogni gruppo calcola la coordinata costante (media pesata per lunghezza)
    lines = []
    for is_h, edges in groups:
        total_w = 0.0
        total_c = 0.0
        for e in edges:
            p0 = poly[e]
            p1 = poly[(e + 1) % n]
            w = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            c_val = (p0[1] + p1[1]) / 2 if is_h else (p0[0] + p1[0]) / 2
            total_w += w
            total_c += w * c_val
        coord = total_c / total_w if total_w > 1e-12 else (
            poly[edges[0], 1 if is_h else 0]
        )
        lines.append((is_h, coord))

    # intersect consecutive lines per ottenere i vertici
    ng = len(lines)
    vertices = []
    for k in range(ng):
        prev_h, prev_c = lines[(k - 1) % ng]
        cur_h, cur_c = lines[k]
        if prev_h == cur_h:
            continue
        if prev_h:   # prev: y=prev_c; cur: x=cur_c
            vertices.append([cur_c, prev_c])
        else:        # prev: x=prev_c; cur: y=cur_c
            vertices.append([prev_c, cur_c])

    if len(vertices) < 4:
        return poly
    return np.asarray(vertices, dtype=np.float64)


def _drop_short_edges(poly: np.ndarray, min_len: float) -> np.ndarray:
    """
    Rimuove iterativamente gli edge più corti di `min_len`: ad ogni passaggio
    collassa l'edge più corto fondendo i suoi due vertici nel punto medio.
    Efficace per ripulire residui di snap (edge di lunghezza ~epsilon).
    """
    if len(poly) < 4 or min_len <= 0:
        return poly
    poly = np.asarray(poly, dtype=np.float64).tolist()
    changed = True
    while changed and len(poly) > 4:
        changed = False
        # trova edge più corto
        best_i = -1
        best_len = np.inf
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            dx = poly[j][0] - poly[i][0]
            dy = poly[j][1] - poly[i][1]
            L = (dx * dx + dy * dy) ** 0.5
            if L < best_len:
                best_len = L
                best_i = i
        if best_len < min_len and best_i >= 0:
            j = (best_i + 1) % len(poly)
            mx = (poly[best_i][0] + poly[j][0]) / 2
            my = (poly[best_i][1] + poly[j][1]) / 2
            # sostituisci i due vertici con il punto medio
            if j > best_i:
                poly[best_i] = [mx, my]
                poly.pop(j)
            else:  # wrap: j=0, best_i=n-1
                poly[0] = [mx, my]
                poly.pop(best_i)
            changed = True
    return np.asarray(poly, dtype=np.float64)


def _mbr_angle(xy: np.ndarray, step_deg: float = 2.0) -> float:
    """
    Angolo di orientamento del Minimum Bounding Rectangle (rotating calipers
    discreto): cerca θ ∈ [0, 90°) che minimizza l'area del bbox allineato
    agli assi dopo rotazione. Più robusto del puro PCA per forme a L/T.
    """
    if len(xy) < 3:
        return 0.0
    c = xy - xy.mean(axis=0)
    best_area = np.inf
    best = 0.0
    # downsample per velocità
    if len(c) > 1500:
        idx = np.random.default_rng(0).choice(len(c), 1500, replace=False)
        c = c[idx]
    for deg in np.arange(0.0, 90.0, step_deg):
        t = np.deg2rad(deg)
        cs, sn = np.cos(-t), np.sin(-t)
        rot = c @ np.array([[cs, -sn], [sn, cs]]).T
        w = rot[:, 0].max() - rot[:, 0].min()
        h = rot[:, 1].max() - rot[:, 1].min()
        a = w * h
        if a < best_area:
            best_area = a
            best = t
    # riduci a [-π/4, π/4]
    while best > np.pi / 4:
        best -= np.pi / 2
    while best < -np.pi / 4:
        best += np.pi / 2
    return float(best)


def _orthogonal_footprint_raster(xy: np.ndarray, epsilon: float,
                                 pca_angle: float | None = None
                                 ) -> np.ndarray:
    """
    Genera un footprint ORTOGONALE con numero minimo di vertici:
      1. Stima angolo di MBR (rotating calipers, robusto anche per L-shape)
      2. Ruota al frame aligned
      3. Rasterizza con soglia di DENSITÀ (celle con <2 punti = vuote)
      4. Closing + fill_holes → smussa staircase da rumore
      5. Tiene solo la componente connessa più grande
      6. Estrae boundary pixel-accurate
      7. Collapse staircase: RDP aggressivo + re-snap a H/V
      8. Rimuove vertici collineari e quasi-coincidenti
      9. Ruota indietro

    Epsilon (metri) è la tolleranza di semplificazione: valori piccoli
    preservano dettagli, valori grandi collassano verso un rettangolo.
    """
    if len(xy) < 4:
        return xy.astype(np.float32)

    try:
        from scipy.ndimage import binary_closing, binary_fill_holes, label
    except Exception:
        return xy.astype(np.float32)

    xy = np.asarray(xy, dtype=np.float64)
    center = xy.mean(axis=0)
    centered = xy - center

    # angolo: MBR (più robusto di PCA per forme a L) oppure forzato
    if pca_angle is None:
        theta = _mbr_angle(centered, step_deg=1.5)
    else:
        theta = float(pca_angle)

    c, s = np.cos(-theta), np.sin(-theta)
    R_fwd = np.array([[c, -s], [s, c]])
    rot = centered @ R_fwd.T

    # dimensioni griglia (padding di 1 cella tutt'intorno → outline CCW pulito)
    x_min, y_min = rot.min(axis=0)
    x_max, y_max = rot.max(axis=0)
    eps = max(epsilon, 1e-3)
    nx = max(3, int(np.ceil((x_max - x_min) / eps)) + 3)
    ny = max(3, int(np.ceil((y_max - y_min) / eps)) + 3)
    if nx > 400 or ny > 400:
        eps = max((x_max - x_min) / 397, (y_max - y_min) / 397, eps)
        nx = int(np.ceil((x_max - x_min) / eps)) + 3
        ny = int(np.ceil((y_max - y_min) / eps)) + 3

    x0 = x_min - eps
    y0 = y_min - eps

    # rasterizza con count (non bool) → applica soglia di densità
    ix = np.clip(((rot[:, 0] - x0) / eps).astype(int), 0, nx - 1)
    iy = np.clip(((rot[:, 1] - y0) / eps).astype(int), 0, ny - 1)
    counts = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(counts, (iy, ix), 1)

    # Soglia densità robusta: relativa alla MEDIANA delle celle occupate
    # (non alla media su tutta la griglia). Così zone a densità bassa ma
    # consistente — es. base larga di un edificio multi-livello la cui torre
    # centrale è molto più densa — non vengono erroneamente tagliate.
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return xy.astype(np.float32)
    med = float(np.median(occupied))
    density_thr = max(0, int(med * 0.10))  # 10% mediana, minimo 0 (≥1 punto)
    grid = counts > density_thr

    # morfologia: chiude gap e smussa piccoli staircase
    grid = binary_closing(grid, iterations=3)
    grid = binary_fill_holes(grid)

    # componente connessa più grande
    labeled, nlab = label(grid)
    if nlab == 0:
        return xy.astype(np.float32)
    if nlab > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        grid = labeled == int(sizes.argmax())

    # Estrazione boundary come catena di mezze-edge sui bordi della griglia:
    # per ogni cella occupata, per ogni lato con vicino vuoto → emette una
    # half-edge con direzione tale che l'interno stia a SINISTRA (CCW esterno).
    ny, nx = grid.shape
    stride = nx + 1  # indice piatto del vertex grid (ny+1) x (nx+1)

    edges: dict[int, int] = {}

    def vidx(vy: int, vx: int) -> int:
        return vy * stride + vx

    # Iteriamo solo sulle celle occupate
    occ_y, occ_x = np.nonzero(grid)
    for iy_c, ix_c in zip(occ_y, occ_x):
        # bottom neighbor (iy-1) empty? → edge (iy, ix) → (iy, ix+1)  (verso +x, interno sopra)
        if iy_c == 0 or not grid[iy_c - 1, ix_c]:
            edges[vidx(iy_c, ix_c)] = vidx(iy_c, ix_c + 1)
        # top neighbor (iy+1) empty? → edge (iy+1, ix+1) → (iy+1, ix)  (verso -x, interno sotto)
        if iy_c == ny - 1 or not grid[iy_c + 1, ix_c]:
            edges[vidx(iy_c + 1, ix_c + 1)] = vidx(iy_c + 1, ix_c)
        # left neighbor (ix-1) empty? → edge (iy+1, ix) → (iy, ix)  (verso -y, interno a destra)
        if ix_c == 0 or not grid[iy_c, ix_c - 1]:
            edges[vidx(iy_c + 1, ix_c)] = vidx(iy_c, ix_c)
        # right neighbor (ix+1) empty? → edge (iy, ix+1) → (iy+1, ix+1)  (verso +y, interno a sinistra)
        if ix_c == nx - 1 or not grid[iy_c, ix_c + 1]:
            edges[vidx(iy_c, ix_c + 1)] = vidx(iy_c + 1, ix_c + 1)

    if not edges:
        return xy.astype(np.float32)

    # Traccia il loop partendo dall'arco più in basso-a-sinistra (scelta canonica)
    start = min(edges.keys(), key=lambda k: (k // stride, k % stride))
    loop = [start]
    current = start
    safety = len(edges) + 4
    while safety > 0:
        safety -= 1
        nxt = edges.get(current)
        if nxt is None or nxt == start:
            break
        loop.append(nxt)
        current = nxt

    # converti indici vertex → coordinate world (frame ruotato)
    poly_rot = np.empty((len(loop), 2), dtype=np.float64)
    for i, vi in enumerate(loop):
        vy = vi // stride
        vx = vi % stride
        poly_rot[i, 0] = x0 + vx * eps
        poly_rot[i, 1] = y0 + vy * eps

    # --- Cleanup aggressivo ---
    # pass 1: rimuovi collineari (pixels dritti lungo un muro)
    poly_rot = _remove_collinear(poly_rot, tol=eps * 0.1)

    # pass 2: RDP con tolleranza = epsilon → collassa staircase in diagonali
    if len(poly_rot) > 4 and eps > 0:
        rdp_tol = eps * 1.1  # leggermente > pixel per includere piccoli zig-zag
        poly_rot = _rdp_simplify(poly_rot, rdp_tol)

    # pass 3: snap delle diagonali residue a H/V (frame già aligned)
    #   Per ogni edge: se |dx| >= |dy| è orizzontale, altrimenti verticale.
    #   Forza alternanza H/V, poi intersect consecutive lines per ottenere vertici.
    if len(poly_rot) >= 4:
        poly_rot = _snap_alternating_hv(poly_rot)

    # pass 4: rimuovi collineari post-snap + vertici troppo vicini (< epsilon/2)
    poly_rot = _remove_collinear(poly_rot, tol=eps * 0.1)
    poly_rot = _drop_short_edges(poly_rot, min_len=eps * 0.5)

    if len(poly_rot) < 4:
        # fallback: bounding rect nel frame ruotato
        poly_rot = np.array([
            [x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max],
        ])

    # ruota indietro al frame world
    c2, s2 = np.cos(theta), np.sin(theta)
    R_back = np.array([[c2, -s2], [s2, c2]])
    out = poly_rot @ R_back.T + center
    return out.astype(np.float32)


def _segment_heights(points: np.ndarray, z_floor: float, z_roof: float,
                     max_blocks: int = 3, min_gap_frac: float = 0.08
                     ) -> list[tuple[float, np.ndarray]]:
    """
    Rileva quote di gronda distinte (piani sfalsati) e restituisce una
    lista di blocchi (z_top, xy_points) ordinata per z_top crescente.

    COMPORTAMENTO CUMULATIVO (fix multi_height):
      Per ogni gronda z_k, il blocco contiene TUTTI i punti che raggiungono
      almeno quella quota. Così:
        - blocco basso → footprint grande (tutto l'edificio)
        - blocco medio → footprint intermedio (base + torre)
        - blocco alto  → footprint piccolo (solo torre)
      I prismi risultano annidati e visivamente coerenti (torre su base),
      anziché frammenti disgiunti.
    """
    total_h = z_roof - z_floor
    if total_h < 1e-6 or len(points) < 30:
        body = points[points[:, 2] > z_floor + 0.35 * total_h]
        if len(body) < 10:
            body = points
        return [(z_roof, body[:, :2])]

    body = points[points[:, 2] > z_floor + 0.35 * total_h]
    if len(body) < 30:
        return [(z_roof, body[:, :2] if len(body) else points[:, :2])]

    z_vals = body[:, 2]
    bins = max(20, min(60, int(total_h * 3)))
    hist, edges = np.histogram(z_vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    try:
        from scipy.signal import find_peaks
        min_distance = max(2, int(bins * min_gap_frac))
        peaks, _ = find_peaks(
            hist,
            height=max(len(body) * 0.015, 3),
            distance=min_distance,
        )
    except Exception:
        peaks = np.array([], dtype=int)

    if len(peaks) == 0:
        return [(z_roof, body[:, :2])]

    # top-N picchi per altezza di histogram (più densi = più significativi)
    order = np.argsort(hist[peaks])[::-1][:max_blocks]
    peak_z = sorted(float(centers[peaks[i]]) for i in order)

    # Filtro: rimuovi picchi troppo ravvicinati verticalmente
    # (frequente artefatto: due "picchi" a 30 cm di distanza dallo stesso tetto piatto)
    min_dz = max(1.2, total_h * 0.10)
    filtered = [peak_z[0]] if peak_z else []
    for z in peak_z[1:]:
        if z - filtered[-1] >= min_dz:
            filtered.append(z)
    peak_z = filtered

    # CUMULATIVO: ogni blocco prende i punti con z >= (pz - tol)
    blocks = []
    tol = total_h * 0.03
    for pz in peak_z:
        mask = body[:, 2] >= pz - tol
        pts_xy = body[mask][:, :2]
        if len(pts_xy) >= 10:
            blocks.append((pz, pts_xy))

    if not blocks:
        blocks = [(z_roof, body[:, :2])]
    return blocks


def _inset_polygon(poly: np.ndarray, inset: float) -> np.ndarray:
    """
    Offset pseudo-interno: ogni vertice viene spinto verso il centroide
    di una quantità `inset`. Sufficiente per l'effetto "tetto rientrato".
    """
    if len(poly) < 3 or inset <= 0:
        return poly
    center = poly.mean(axis=0)
    direction = center - poly
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    unit = direction / np.maximum(norms, 1e-9)
    return (poly + unit * inset).astype(np.float32)


def _decimate_parts(parts: dict, ratio: float = 0.9, min_faces: int = 500) -> dict:
    """
    Decima ogni mesh con PyVista (riduzione % triangoli) prima di passare
    a Plotly. I pezzi già leggeri (muri prismatici) vengono saltati.
    """
    if ratio <= 0:
        return parts
    out = {}
    for name, value in parts.items():
        # Chiavi meta (es. "_windows_meta": list[dict]) pass-through
        if name.startswith("_"):
            out[name] = value
            continue
        v, f = value
        if len(f) < min_faces:
            out[name] = (v, f)
            continue
        try:
            faces_pv = np.hstack(
                [np.full((len(f), 1), 3, dtype=np.int64), f.astype(np.int64)]
            ).ravel()
            poly = pv.PolyData(v.astype(np.float32), faces_pv)
            dec = poly.decimate(ratio)
            nv = np.asarray(dec.points, dtype=np.float32)
            nf = dec.faces.reshape(-1, 4)[:, 1:].astype(np.int32) if dec.n_cells > 0 else np.zeros((0, 3), dtype=np.int32)
            out[name] = (nv, nf) if len(nf) > 0 else (v, f)
        except Exception:
            out[name] = (v, f)
    return out


def _window_box(p_center_xy: np.ndarray, wdir: np.ndarray, nrm: np.ndarray,
                z_bot: float, z_top: float, width: float, depth: float = 0.08
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Box sottile che sporge dalla parete: fronte + 4 lati, normale = `nrm`.
    Restituisce (vertici, facce) per un singolo davanzale/finestra 3D.
    """
    half = width / 2
    base_l = p_center_xy - wdir * half
    base_r = p_center_xy + wdir * half
    front_l = base_l + nrm * depth
    front_r = base_r + nrm * depth
    # 8 vertici: 4 sul muro (back) + 4 sulla faccia esterna (front)
    v = np.array([
        [base_l[0],  base_l[1],  z_bot],   # 0  back-bot-left
        [base_r[0],  base_r[1],  z_bot],   # 1  back-bot-right
        [base_r[0],  base_r[1],  z_top],   # 2  back-top-right
        [base_l[0],  base_l[1],  z_top],   # 3  back-top-left
        [front_l[0], front_l[1], z_bot],   # 4  front-bot-left
        [front_r[0], front_r[1], z_bot],   # 5  front-bot-right
        [front_r[0], front_r[1], z_top],   # 6  front-top-right
        [front_l[0], front_l[1], z_top],   # 7  front-top-left
    ], dtype=np.float32)
    # facce: fronte + 4 bordi (no retro, già coperto dal muro)
    f = np.array([
        [4, 5, 6], [4, 6, 7],      # front
        [0, 4, 7], [0, 7, 3],      # sinistra
        [5, 1, 2], [5, 2, 6],      # destra
        [3, 7, 6], [3, 6, 2],      # sopra
        [0, 1, 5], [0, 5, 4],      # sotto
    ], dtype=np.int32)
    return v, f


def extrude_rectangular_building(
    points: np.ndarray,
    floor_height: float = 3.0,
    window_w: float = 1.2,
    window_h: float = 1.5,
    window_spacing: float = 3.5,
    window_margin_side: float = 1.0,
    window_margin_top: float = 0.6,
    add_windows: bool = True,
    roof_inset: float = 0.1,
    windows_mode: str = "procedural",
    window_params: dict | None = None,
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

    # --- TETTO: rettangolo leggermente rientrato rispetto al muro ---
    roof_corners = _inset_polygon(corners.astype(np.float64), roof_inset) if roof_inset > 0 else corners
    roof_v = np.array([
        [roof_corners[0, 0], roof_corners[0, 1], z_roof],
        [roof_corners[1, 0], roof_corners[1, 1], z_roof],
        [roof_corners[2, 0], roof_corners[2, 1], z_roof],
        [roof_corners[3, 0], roof_corners[3, 1], z_roof],
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

    # --- FINESTRE ---
    if add_windows:
        if windows_mode == "detected":
            # Detection REALE dalla nuvola (RGB o voids)
            fp_xy = corners[:, :2].astype(np.float32)
            wp = window_params or {}
            win_v, win_f, win_meta = _windows_from_cloud(
                points, fp_xy, z_floor, z_roof,
                wall_thickness=wp.get("wall_thickness", 0.35),
                cell=wp.get("detect_cell", 0.10),
                min_w=wp.get("window_min_w", 0.4),
                max_w=wp.get("window_max_w", 4.0),
                min_h=wp.get("window_min_h", 0.4),
                max_h=wp.get("window_max_h", 3.0),
                min_wall_frac=wp.get("min_wall_frac", 0.25),
                depth=0.08,
            )
            if len(win_v) > 0:
                result["windows"] = (win_v, win_f)
                result["_windows_meta"] = win_meta
        else:
            # Procedurale: griglia regolare sulle 4 pareti
            all_v: list[np.ndarray] = []
            all_f: list[np.ndarray] = []
            v_offset = 0
            depth = 0.08

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
                        c_xy = p0[:2].astype(np.float64) + wdir.astype(np.float64) * t_along
                        v, f = _window_box(
                            c_xy, wdir.astype(np.float64), nrm.astype(np.float64),
                            win_z_b, win_z_t, width=window_w, depth=depth,
                        )
                        all_v.append(v)
                        all_f.append(f + v_offset)
                        v_offset += len(v)

            if all_v:
                result["windows"] = (
                    np.concatenate(all_v, axis=0).astype(np.float32),
                    np.concatenate(all_f, axis=0).astype(np.int32),
                )

    return result


def _prism_from_footprint(footprint: np.ndarray, z_bot: float, z_top: float,
                          roof_inset: float = 0.0
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce muri (quad per edge, vertici duplicati per flatshading netto)
    e tetto (fan dal centroide, con inset opzionale) di un blocco prismatico.
    Ritorna (wall_v, wall_f, roof_v, roof_f).
    """
    n = len(footprint)
    wall_v = np.zeros((4 * n, 3), dtype=np.float32)
    wall_f = np.zeros((2 * n, 3), dtype=np.int32)
    for i in range(n):
        j = (i + 1) % n
        base = 4 * i
        wall_v[base + 0] = [footprint[i, 0], footprint[i, 1], z_bot]
        wall_v[base + 1] = [footprint[j, 0], footprint[j, 1], z_bot]
        wall_v[base + 2] = [footprint[j, 0], footprint[j, 1], z_top]
        wall_v[base + 3] = [footprint[i, 0], footprint[i, 1], z_top]
        wall_f[2 * i + 0] = [base, base + 1, base + 2]
        wall_f[2 * i + 1] = [base, base + 2, base + 3]

    roof_poly = _inset_polygon(footprint, roof_inset) if roof_inset > 0 else footprint
    cx, cy = float(roof_poly[:, 0].mean()), float(roof_poly[:, 1].mean())
    roof_v = np.zeros((n + 1, 3), dtype=np.float32)
    roof_v[:n, :2] = roof_poly
    roof_v[:n, 2] = z_top
    roof_v[n] = [cx, cy, z_top]
    roof_f = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        roof_f[i] = [i, (i + 1) % n, n]

    return wall_v, wall_f, roof_v, roof_f


def _footprint_from_xy(xy: np.ndarray, lod: int, use_concave: bool,
                       orthogonalize: bool,
                       epsilon: float = 0.8,
                       pca_angle: float | None = None) -> np.ndarray | None:
    """
    Hull 2D con opzione di semplificazione aggressiva:
      - orthogonalize=True → usa _orthogonal_footprint_raster (raster+outline
        estratto con numero minimo di vertici: 4 per rettangolo, 6 per L, ...)
      - orthogonalize=False → concave/convex hull grezzo, con eventuale RDP
        se epsilon > 0 (collassa micro-spigoli).

    `epsilon` (metri) governa la tolleranza di semplificazione.
    `pca_angle` forza l'angolo PCA condiviso tra blocchi (per allineamento
    coerente tra piani sfalsati).
    """
    if len(xy) < 3:
        return None
    if len(xy) > 4000:
        idx = np.random.default_rng(0).choice(len(xy), 4000, replace=False)
        xy = xy[idx]

    if orthogonalize:
        fp = _orthogonal_footprint_raster(
            xy.astype(np.float64),
            epsilon=max(epsilon, 0.1),
            pca_angle=pca_angle,
        )
        if len(fp) < 4:
            return None
        return np.asarray(fp, dtype=np.float32)

    # modalità non-ortogonale: concave o convex hull "grezzo"
    alpha_factor = max(1.5, 7.0 - 0.55 * lod)
    if use_concave:
        fp = _concave_hull_2d(
            xy.astype(np.float64),
            alpha_factor=alpha_factor,
            simplify=max(epsilon, 0.0),
        )
    else:
        from scipy.spatial import ConvexHull
        fp = xy[ConvexHull(xy).vertices]
        if epsilon > 0 and len(fp) > 4:
            fp = _rdp_simplify(np.asarray(fp, dtype=np.float64), epsilon)
            fp = _remove_collinear(fp)
    if len(fp) < 3:
        return None
    return np.asarray(fp, dtype=np.float32)


def _dominant_pca_angle(xy: np.ndarray) -> float:
    """Angolo dell'asse principale PCA, ridotto a [-π/4, π/4]."""
    if len(xy) < 3:
        return 0.0
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    _, vecs = np.linalg.eigh(cov)
    theta = float(np.arctan2(vecs[1, -1], vecs[0, -1]))
    while theta > np.pi / 4:
        theta -= np.pi / 2
    while theta < -np.pi / 4:
        theta += np.pi / 2
    return theta


def extrude_building(points: np.ndarray, lod: int,
                     use_concave: bool = True,
                     orthogonalize: bool = False,
                     multi_height: bool = False,
                     roof_inset: float = 0.0,
                     add_windows: bool = False,
                     windows_mode: str = "procedural",
                     window_params: dict | None = None,
                     epsilon: float = 0.8,
                     roof_top_frac: float = 0.1) -> dict:
    """
    Ricostruzione "Incarto Energia" di un edificio:

      1. stima Z_floor / Z_roof (2°/98° percentile)
      2. FILTRO TETTO: usa solo i punti nel top `roof_top_frac` dell'altezza
         (default 10%) → esclude vegetazione/dettagli bassi che sporcano
         il perimetro
      3. footprint 2D aggressivamente semplificato:
         - orthogonalize=True → raster-based envelope con tolleranza `epsilon`,
           produce 4 vertici per rettangolo, 6 per L-shape, ...
         - orthogonalize=False → concave hull + RDP
      4. (opzionale) rileva quote di gronda multiple → blocchi sfalsati,
         allineati allo stesso angolo PCA per coerenza visiva
      5. estrusione verticale: SOLO muri + tetto piatto (niente pavimento,
         niente alette interne)
      6. tetto eventualmente rientrato rispetto al filo muro

    Ritorna dict con chiavi 'walls'/'roof' (o suffissate '_1', '_2', ...
    per blocchi multipli), più eventuali 'windows'.
    """
    if len(points) < 10:
        return {}

    z_sorted = np.sort(points[:, 2])
    z_floor = float(z_sorted[int(0.02 * (len(z_sorted) - 1))])
    z_roof = float(z_sorted[int(0.98 * (len(z_sorted) - 1))])
    total_h = z_roof - z_floor
    if total_h < 1e-6:
        return {}

    # --- FILTRO PUNTI TETTO: solo il top-N% dell'altezza
    #     Questo è il cuore dell'"incarto energia": il perimetro va misurato
    #     dove c'è solo l'edificio, non la vegetazione alla base.
    roof_cut = z_roof - max(roof_top_frac, 0.02) * total_h
    roof_pts = points[points[:, 2] >= roof_cut]
    if len(roof_pts) < 20:
        # fallback: allarga la banda per avere abbastanza punti
        for frac in (0.15, 0.25, 0.4, 0.6):
            roof_cut = z_roof - frac * total_h
            roof_pts = points[points[:, 2] >= roof_cut]
            if len(roof_pts) >= 20:
                break
    if len(roof_pts) < 20:
        roof_pts = points

    # angolo PCA unico, calcolato sui punti tetto: i blocchi multi-altezza
    # lo ereditano → rettangoli tutti allineati tra loro
    pca_angle = _dominant_pca_angle(roof_pts[:, :2].astype(np.float64))

    full_fp = _footprint_from_xy(
        roof_pts[:, :2].astype(np.float32),
        lod=lod, use_concave=use_concave, orthogonalize=orthogonalize,
        epsilon=epsilon, pca_angle=pca_angle,
    )
    if full_fp is None or len(full_fp) < 4:
        return extrude_rectangular_building(points, add_windows=False)

    fp_w = float(full_fp[:, 0].max() - full_fp[:, 0].min())
    fp_h = float(full_fp[:, 1].max() - full_fp[:, 1].min())
    if min(fp_w, fp_h) < 0.05 * max(fp_w, fp_h, 1e-9):
        return extrude_rectangular_building(points, add_windows=False)

    # blocchi per altezza (solo se richiesto)
    if multi_height:
        blocks_raw = _segment_heights(points, z_floor, z_roof, max_blocks=3)
        if len(blocks_raw) < 2:
            blocks_raw = [(z_roof, roof_pts[:, :2])]
    else:
        blocks_raw = [(z_roof, roof_pts[:, :2])]

    blocks: list[tuple[float, np.ndarray]] = []
    for (z_top, xy_block) in blocks_raw:
        fp = _footprint_from_xy(
            xy_block.astype(np.float32),
            lod=lod, use_concave=use_concave, orthogonalize=orthogonalize,
            epsilon=epsilon, pca_angle=pca_angle,  # stesso angolo per coerenza
        )
        if fp is None or len(fp) < 4:
            fp = full_fp
        blocks.append((z_top, fp))
    blocks.sort(key=lambda b: b[0])

    # --- Assembla mesh: SOLO muri + tetto (niente pavimento, niente alette) ---
    result: dict = {}
    single = len(blocks) == 1
    for i, (z_top, fp) in enumerate(blocks):
        wv, wf, rv, rf = _prism_from_footprint(
            fp, z_floor, z_top, roof_inset=roof_inset
        )
        if single:
            result["walls"] = (wv, wf)
            result["roof"] = (rv, rf)
        else:
            result[f"walls_{i+1}"] = (wv, wf)
            result[f"roof_{i+1}"] = (rv, rf)

    # Finestre (opzionale) — solo sulle pareti esterne principali del blocco più alto
    if add_windows and len(full_fp) >= 4:
        wp = window_params or {}
        z_top_highest = blocks[-1][0]

        if windows_mode == "detected":
            # detection REALE dalla nuvola (RGB se presente, sennò voids)
            win_v, win_f, win_meta = _windows_from_cloud(
                points, full_fp, z_floor, z_top_highest,
                wall_thickness=wp.get("wall_thickness", 0.35),
                cell=wp.get("detect_cell", 0.10),
                min_w=wp.get("window_min_w", 0.4),
                max_w=wp.get("window_max_w", 4.0),
                min_h=wp.get("window_min_h", 0.4),
                max_h=wp.get("window_max_h", 3.0),
                min_wall_frac=wp.get("min_wall_frac", 0.25),
                depth=wp.get("window_depth", 0.08),
            )
            if len(win_v) > 0:
                result["windows"] = (win_v, win_f)
                result["_windows_meta"] = win_meta  # per statistiche in UI
        else:
            win_v, win_f = _windows_on_footprint(
                full_fp, z_floor, z_top_highest,
                floor_height=wp.get("floor_height", 3.0),
                window_w=wp.get("window_w", 1.2),
                window_h=wp.get("window_h", 1.5),
                window_spacing=wp.get("window_spacing", 3.5),
                window_margin_side=wp.get("window_margin_side", 1.0),
                window_margin_top=wp.get("window_margin_top", 0.6),
                depth=wp.get("window_depth", 0.08),
                min_wall_frac=wp.get("min_wall_frac", 0.25),
            )
            if len(win_v) > 0:
                result["windows"] = (win_v, win_f)

    return result


def _windows_on_footprint(footprint: np.ndarray, z_floor: float, z_roof: float,
                          floor_height: float, window_w: float, window_h: float,
                          window_spacing: float, window_margin_side: float,
                          window_margin_top: float, depth: float,
                          min_wall_frac: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Distribuisce finestre 3D (box sottili) SOLO sulle pareti esterne principali
    (pareti la cui lunghezza supera `min_wall_frac` * edge più lungo).
    """
    n = len(footprint)
    total_h = z_roof - z_floor
    n_floors = max(1, int(round(total_h / floor_height)))
    actual_floor_h = total_h / n_floors

    centroid = footprint.mean(axis=0)

    # calcola lunghezze e filtra solo le principali
    edges = []
    for i in range(n):
        j = (i + 1) % n
        p0 = footprint[i].astype(np.float64)
        p1 = footprint[j].astype(np.float64)
        wd = p1 - p0
        wlen = float(np.linalg.norm(wd))
        if wlen < 1e-3:
            continue
        wdir = wd / wlen
        nrm = np.array([wdir[1], -wdir[0]])
        if np.dot(nrm, (p0 + p1) / 2 - centroid) < 0:
            nrm = -nrm
        edges.append((p0, p1, wdir, nrm, wlen))

    if not edges:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    max_len = max(e[4] for e in edges)
    main_edges = [e for e in edges if e[4] >= max_len * min_wall_frac]

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    v_offset = 0
    for (p0, p1, wdir, nrm, wlen) in main_edges:
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
                c_xy = p0 + wdir * t_along
                v, f = _window_box(c_xy, wdir, nrm, win_z_b, win_z_t,
                                   width=window_w, depth=depth)
                all_v.append(v)
                all_f.append(f + v_offset)
                v_offset += len(v)

    if not all_v:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    return (
        np.concatenate(all_v, axis=0).astype(np.float32),
        np.concatenate(all_f, axis=0).astype(np.int32),
    )


def _windows_from_cloud(points: np.ndarray,
                        footprint: np.ndarray,
                        z_floor: float, z_roof: float,
                        wall_thickness: float = 0.35,
                        cell: float = 0.10,
                        min_w: float = 0.4, max_w: float = 4.0,
                        min_h: float = 0.4, max_h: float = 3.0,
                        min_wall_frac: float = 0.20,
                        depth: float = 0.08,
                        use_rgb: bool | None = None,
                        ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Detection finestre REALI dalla nuvola, parete per parete.

    points: (N,3) solo XYZ, oppure (N,6) con XYZ+RGB (R,G,B in 0..255).
    Strategia per ogni parete del footprint:
      1. seleziona i punti vicini al piano parete (|t_out| <= wall_thickness)
      2. proietta in coord locali (u lungo parete, v altezza)
      3. rasterizza a `cell` (default 10 cm)
      4. classifica ogni cella come "finestra" se:
           a) RGB: blu dominante  O  molto scura  O  chiara+neutra (riflesso cielo)
           b) void: densità molto inferiore alla mediana della parete,
              lontano dai bordi
      5. morphology (close→open) + connected components
      6. bounding box di ogni cluster → rettangolo finestra
      7. filtra dimensioni plausibili

    Ritorna (vertices, faces, window_list) dove window_list contiene metadati
    (posizione, orientamento, dimensioni) utili per statistiche/incarto energia.
    """
    has_rgb = (points.ndim == 2 and points.shape[1] >= 6)
    if use_rgb is None:
        use_rgb = has_rgb

    n_edges = len(footprint)
    if n_edges < 3 or len(points) < 100:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32), [])

    try:
        from scipy.ndimage import binary_opening, binary_closing, label
        scipy_ok = True
    except Exception:
        scipy_ok = False

    centroid = footprint.mean(axis=0)
    edge_lengths = [
        float(np.linalg.norm(footprint[(i + 1) % n_edges] - footprint[i]))
        for i in range(n_edges)
    ]
    max_len = max(edge_lengths) if edge_lengths else 0.0

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    meta: list[dict] = []
    v_offset = 0

    pts_xy = points[:, :2].astype(np.float64)
    pts_z = points[:, 2]

    for i in range(n_edges):
        j = (i + 1) % n_edges
        p0 = footprint[i].astype(np.float64)
        p1 = footprint[j].astype(np.float64)
        wd = p1 - p0
        wlen = float(np.linalg.norm(wd))
        if wlen < 1.0 or wlen < max_len * min_wall_frac:
            continue
        wdir = wd / wlen
        nrm = np.array([wdir[1], -wdir[0]])
        # normale orientata verso esterno
        if np.dot(nrm, (p0 + p1) / 2 - centroid) < 0:
            nrm = -nrm

        to_p = pts_xy - p0
        u = to_p @ wdir
        t_out = to_p @ nrm

        # Stima l'offset della parete REALE rispetto al footprint estruso
        # (il raster ha padding di ~1 cella → la parete vera è traslata di
        # qualche cm. Centriamo la banda sulla mediana dei t_out plausibili.)
        search = 1.5 * wall_thickness + 1.0
        pre_mask = (
            (u >= 0) & (u <= wlen) &
            (np.abs(t_out) <= search) &
            (pts_z >= z_floor) & (pts_z <= z_roof)
        )
        if pre_mask.sum() < 50:
            continue
        t_offset = float(np.median(t_out[pre_mask]))

        mask = (
            (u >= 0) & (u <= wlen) &
            (np.abs(t_out - t_offset) <= wall_thickness) &
            (pts_z >= z_floor) & (pts_z <= z_roof)
        )
        n_wall = int(mask.sum())
        if n_wall < 100:
            continue

        u_sel = u[mask]
        v_sel = pts_z[mask] - z_floor
        h_wall = z_roof - z_floor

        nu = max(10, int(wlen / cell))
        nv = max(10, int(h_wall / cell))
        iu = np.clip((u_sel / cell).astype(int), 0, nu - 1)
        iv = np.clip((v_sel / cell).astype(int), 0, nv - 1)

        density = np.zeros((nv, nu), dtype=np.int32)
        np.add.at(density, (iv, iu), 1)

        # -- classificazione per colore (se RGB presente)
        win_color = np.zeros((nv, nu), dtype=bool)
        if use_rgb and has_rgb:
            rgb_sel = points[mask, 3:6].astype(np.float32)
            r_sum = np.zeros((nv, nu), dtype=np.float32)
            g_sum = np.zeros((nv, nu), dtype=np.float32)
            b_sum = np.zeros((nv, nu), dtype=np.float32)
            np.add.at(r_sum, (iv, iu), rgb_sel[:, 0])
            np.add.at(g_sum, (iv, iu), rgb_sel[:, 1])
            np.add.at(b_sum, (iv, iu), rgb_sel[:, 2])
            d_safe = np.maximum(density, 1)
            r_m = r_sum / d_safe
            g_m = g_sum / d_safe
            b_m = b_sum / d_safe
            lum = 0.299 * r_m + 0.587 * g_m + 0.114 * b_m
            is_blue = (b_m > r_m + 10) & (b_m > g_m) & (b_m > 60)
            is_dark = lum < 55
            is_skylike = (lum > 195) & (np.abs(r_m - g_m) < 20) & (np.abs(g_m - b_m) < 25)
            win_color = (is_blue | is_dark | is_skylike) & (density > 0)

        # -- voids: cella occupata ma densità <30% della mediana, lontano dai bordi
        win_void = np.zeros((nv, nu), dtype=bool)
        occupied = density[density > 0]
        if occupied.size > 5:
            med_d = float(np.median(occupied))
            low_d = (density > 0) & (density < med_d * 0.30)
            border = np.zeros((nv, nu), dtype=bool)
            if nv > 6 and nu > 6:
                border[3:-3, 3:-3] = True
            win_void = low_d & border

        win_mask = win_color | win_void
        if not win_mask.any():
            continue

        if scipy_ok:
            win_mask = binary_closing(win_mask, iterations=2)
            win_mask = binary_opening(win_mask, iterations=1)
            labeled, n_cc = label(win_mask)
        else:
            labeled = win_mask.astype(np.int32)
            n_cc = 1

        # orientamento cardinale (utile per incarto energia)
        theta = float(np.degrees(np.arctan2(nrm[1], nrm[0])))
        compass_idx = int(((theta + 360 + 22.5) % 360) // 45)
        compass = ["E", "NE", "N", "NO", "O", "SO", "S", "SE"][compass_idx]

        for cc_id in range(1, n_cc + 1):
            ys, xs = np.where(labeled == cc_id)
            if len(xs) < 4:
                continue
            u_lo = float(xs.min()) * cell
            u_hi = float(xs.max() + 1) * cell
            v_lo = float(ys.min()) * cell
            v_hi = float(ys.max() + 1) * cell
            w = u_hi - u_lo
            h = v_hi - v_lo
            if w < min_w or h < min_h or w > max_w or h > max_h:
                continue
            # scarta cluster con aspect ratio grottesco (strisce < 1:10)
            if min(w, h) / max(w, h) < 0.10:
                continue
            u_c = (u_lo + u_hi) / 2
            c_xy = p0 + wdir * u_c
            z_b = z_floor + v_lo
            z_t = z_floor + v_hi
            vb, fb = _window_box(c_xy, wdir, nrm, z_b, z_t, width=w, depth=depth)
            all_v.append(vb)
            all_f.append(fb + v_offset)
            v_offset += len(vb)
            meta.append({
                "wall_index": i,
                "compass": compass,
                "center_xy": (float(c_xy[0]), float(c_xy[1])),
                "z_bot": float(z_b), "z_top": float(z_t),
                "width": float(w), "height": float(h),
                "area": float(w * h),
            })

    if not all_v:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32), [])
    return (
        np.concatenate(all_v, axis=0).astype(np.float32),
        np.concatenate(all_f, axis=0).astype(np.int32),
        meta,
    )


def extrude_facade_2d(
    points: np.ndarray,
    facade_thickness: float = 0.05,
    window_detect: bool = True,
    plane_band: float = 0.30,
    cell: float = 0.10,
    min_w: float = 0.4, max_w: float = 4.0,
    min_h: float = 0.4, max_h: float = 3.0,
    use_rgb: bool | None = None,
) -> dict:
    """
    Ricostruzione 2.5D per nuvole intrinsecamente PLANARI (facciate singole,
    sezioni, profili esportati da CAD).

    Differisce dai metodi volumetrici: invece di estrudere un footprint
    verticalmente, assume che la nuvola GIACCIA su un piano. Rileva il
    piano principale (asse con estensione minima = normale), proietta i
    punti nel sistema locale (u, v) e produce:

      - `facciata`  : pannello rettangolare sottile sul piano della nuvola
      - `finestre`  : rettangoli delle finestre rilevate (recessati nel muro)

    Strategia detection finestre (riutilizza la logica di _windows_from_cloud
    ma per UN solo piano):
      1. seleziona punti entro ±plane_band dal piano mediano
      2. rasterizza (u, v) a `cell` metri
      3. classifica celle finestra: RGB (blu/scuro/sky) + voids
         (densità < 30% mediana, lontano dai bordi)
      4. morphology close→open + connected components
      5. filtra per dimensioni plausibili e aspect ratio

    Ritorna dict parti + `_facade_meta` + `_windows_meta` per UI/export.
    """
    pts_all = np.asarray(points, dtype=np.float64)
    pts_xyz = pts_all[:, :3]
    has_rgb = pts_all.ndim == 2 and pts_all.shape[1] >= 6
    if use_rgb is None:
        use_rgb = has_rgb

    n = len(pts_xyz)
    if n < 20:
        return {"_facade_meta": {"status": "empty", "reason": "too few points"}}

    mins = pts_xyz.min(0)
    maxs = pts_xyz.max(0)
    ext = maxs - mins
    if ext.max() < 1e-6:
        return {"_facade_meta": {"status": "empty", "reason": "zero extent"}}

    # Asse "piatto" = normale al piano della facciata
    flat_axis = int(np.argmin(ext))
    # (u, v) sono i due assi non piatti. Convenzione: se la normale è
    # orizzontale (X o Y), manteniamo v = Z (altezza reale). Se la normale
    # è Z (piano orizzontale), u = X, v = Y.
    if flat_axis == 2:
        u_axis, v_axis = 0, 1
    else:
        u_axis = 1 if flat_axis == 0 else 0
        v_axis = 2

    # ---- Validazione: la nuvola è davvero un PIANO, non una LINEA? ----
    # Se il secondo asse più piccolo è anch'esso < 10 cm, la nuvola è
    # unidimensionale (profilo, traccia, sezione estratta troppo stretta)
    # e non può rappresentare una facciata.
    sorted_ext = np.sort(ext)      # [min, mid, max]
    MIN_PLANE_WIDTH = 0.10         # 10 cm minimo su entrambi gli assi del piano
    if sorted_ext[1] < MIN_PLANE_WIDTH:
        return {"_facade_meta": {
            "status": "not_a_plane",
            "reason": (
                f"La nuvola è lineare (1D): due assi su tre sono < {MIN_PLANE_WIDTH*100:.0f} cm "
                f"(extent X={ext[0]:.3f}, Y={ext[1]:.3f}, Z={ext[2]:.3f} m). "
                "Servono punti distribuiti su un piano (larghezza E altezza), "
                "non lungo una singola retta."
            ),
            "extent_x": float(ext[0]), "extent_y": float(ext[1]),
            "extent_z": float(ext[2]),
        }}

    u_all = pts_xyz[:, u_axis] - mins[u_axis]
    v_all = pts_xyz[:, v_axis] - mins[v_axis]
    w_all = pts_xyz[:, flat_axis]
    w_center = float(np.median(w_all))

    u_len = float(ext[u_axis])
    v_len = float(ext[v_axis])

    def uv_to_xyz(uv_arr: np.ndarray, w_off: float) -> np.ndarray:
        """(u,v) locali → (x,y,z) mondo, a profondità w_center + w_off."""
        m = len(uv_arr)
        out = np.zeros((m, 3), dtype=np.float64)
        out[:, u_axis] = uv_arr[:, 0] + mins[u_axis]
        out[:, v_axis] = uv_arr[:, 1] + mins[v_axis]
        out[:, flat_axis] = w_center + w_off
        return out

    # ---- PANNELLO FACCIATA: scatola sottile centrata sul piano ----
    half = facade_thickness / 2.0
    rect_uv = np.array([[0, 0], [u_len, 0], [u_len, v_len], [0, v_len]],
                       dtype=np.float64)
    front = uv_to_xyz(rect_uv, -half)
    back = uv_to_xyz(rect_uv, +half)
    V_fac = np.concatenate([front, back], axis=0)   # 8 vertici (0-3 fronte, 4-7 retro)
    F_fac = np.array([
        [0, 1, 2], [0, 2, 3],          # faccia frontale
        [4, 6, 5], [4, 7, 6],          # faccia posteriore
        [0, 4, 5], [0, 5, 1],          # bordo inferiore
        [1, 5, 6], [1, 6, 2],          # bordo destro
        [2, 6, 7], [2, 7, 3],          # bordo superiore
        [3, 7, 4], [3, 4, 0],          # bordo sinistro
    ], dtype=np.int32)

    # ---- DETECTION FINESTRE sul piano ----
    V_win_list: list[np.ndarray] = []
    F_win_list: list[np.ndarray] = []
    v_off = 0
    meta: list[dict] = []

    if window_detect and n >= 100:
        band = np.abs(w_all - w_center) <= max(plane_band, ext[flat_axis] * 1.2)
        if int(band.sum()) >= 100:
            u_sel = u_all[band]
            v_sel = v_all[band]
            nu = max(10, int(u_len / cell))
            nv = max(10, int(v_len / cell))
            iu = np.clip((u_sel / cell).astype(int), 0, nu - 1)
            iv = np.clip((v_sel / cell).astype(int), 0, nv - 1)

            density = np.zeros((nv, nu), dtype=np.int32)
            np.add.at(density, (iv, iu), 1)

            # --- classificazione colore (RGB) ---
            win_color = np.zeros((nv, nu), dtype=bool)
            if use_rgb and has_rgb:
                rgb_sel = pts_all[band, 3:6].astype(np.float32)
                r_s = np.zeros((nv, nu), dtype=np.float32)
                g_s = np.zeros((nv, nu), dtype=np.float32)
                b_s = np.zeros((nv, nu), dtype=np.float32)
                np.add.at(r_s, (iv, iu), rgb_sel[:, 0])
                np.add.at(g_s, (iv, iu), rgb_sel[:, 1])
                np.add.at(b_s, (iv, iu), rgb_sel[:, 2])
                d_safe = np.maximum(density, 1)
                r_m = r_s / d_safe; g_m = g_s / d_safe; b_m = b_s / d_safe
                lum = 0.299 * r_m + 0.587 * g_m + 0.114 * b_m
                is_blue = (b_m > r_m + 10) & (b_m > g_m) & (b_m > 60)
                is_dark = lum < 55
                is_skylike = (lum > 195) & (np.abs(r_m - g_m) < 20) & (np.abs(g_m - b_m) < 25)
                win_color = (is_blue | is_dark | is_skylike) & (density > 0)

            # --- voids: cella a bassa densità non ai bordi ---
            win_void = np.zeros((nv, nu), dtype=bool)
            occupied = density[density > 0]
            if occupied.size > 5:
                med_d = float(np.median(occupied))
                low_d = (density > 0) & (density < med_d * 0.30)
                border = np.zeros((nv, nu), dtype=bool)
                if nv > 6 and nu > 6:
                    border[3:-3, 3:-3] = True
                win_void = low_d & border

            win_mask = win_color | win_void
            if win_mask.any():
                try:
                    from scipy.ndimage import binary_opening, binary_closing, label
                    win_mask = binary_closing(win_mask, iterations=2)
                    win_mask = binary_opening(win_mask, iterations=1)
                    labeled, n_cc = label(win_mask)
                except ImportError:
                    labeled = win_mask.astype(np.int32)
                    n_cc = 1

                recess = 0.02   # 2 cm recesso del vetro dentro la facciata
                depth = 0.04    # profondità 4 cm scatola finestra
                for cc in range(1, n_cc + 1):
                    ys, xs = np.where(labeled == cc)
                    if len(xs) < 4:
                        continue
                    u_lo = float(xs.min()) * cell
                    u_hi = float(xs.max() + 1) * cell
                    v_lo = float(ys.min()) * cell
                    v_hi = float(ys.max() + 1) * cell
                    w = u_hi - u_lo
                    h = v_hi - v_lo
                    if w < min_w or h < min_h or w > max_w or h > max_h:
                        continue
                    if min(w, h) / max(w, h) < 0.10:
                        continue

                    win_uv = np.array([[u_lo, v_lo], [u_hi, v_lo],
                                       [u_hi, v_hi], [u_lo, v_hi]],
                                      dtype=np.float64)
                    f_front = uv_to_xyz(win_uv, -half - recess)
                    f_back = uv_to_xyz(win_uv, -half - recess - depth)
                    V_win_list.append(np.concatenate([f_front, f_back], axis=0))
                    F_win_list.append(np.array([
                        [0, 1, 2], [0, 2, 3],
                        [4, 6, 5], [4, 7, 6],
                        [0, 4, 5], [0, 5, 1],
                        [1, 5, 6], [1, 6, 2],
                        [2, 6, 7], [2, 7, 3],
                        [3, 7, 4], [3, 4, 0],
                    ], dtype=np.int32) + v_off)
                    v_off += 8
                    meta.append({
                        "u_lo": u_lo, "u_hi": u_hi,
                        "v_lo": v_lo, "v_hi": v_hi,
                        "width": float(w), "height": float(h),
                        "area": float(w * h),
                        "compass": "—",
                    })

    parts: dict = {
        "facciata": (V_fac.astype(np.float32), F_fac.astype(np.int32)),
    }
    if V_win_list:
        parts["finestre"] = (
            np.concatenate(V_win_list, axis=0).astype(np.float32),
            np.concatenate(F_win_list, axis=0).astype(np.int32),
        )
    parts["_facade_meta"] = {
        "u_axis": ["X", "Y", "Z"][u_axis],
        "v_axis": ["X", "Y", "Z"][v_axis],
        "normal_axis": ["X", "Y", "Z"][flat_axis],
        "u_size": u_len,
        "v_size": v_len,
        "real_thickness": float(ext[flat_axis]),
        "n_windows": len(meta),
        "total_window_area": float(sum(m["area"] for m in meta)),
        "facade_area": float(u_len * v_len),
    }
    parts["_windows_meta"] = meta
    return parts


def reconstruct_mesh_arrays(
    points: np.ndarray,
    lod: int,
    method: str,
    rect_params: dict | None = None,
    decimate_ratio: float = 0.9,
    max_points: int = 60_000,
    max_cells: int = 150_000,
) -> dict:
    """
    Ritorna un dict {nome_parte: (verts, faces)} con mesh CAD pronte per Plotly.

    Metodi disponibili:
      - 'building_rect':     rettangolo MBR (PCA) + finestre procedurali 3D
      - 'building_squared':  concave hull → orthogonalize 90° + piani sfalsati
      - 'building_concave':  concave hull grezzo (profilo reale, angoli liberi)
      - 'building_convex':   bounding convex hull
      - 'reconstruct_surface'/'delaunay_2d': superficie organica unica

    `decimate_ratio` applica PyVista.decimate(ratio) PRIMA di inviare a Plotly.
    Per le mesh prismatiche l'effetto è nullo (già leggere); per le superfici
    organiche riduce drasticamente il conteggio triangoli evitando crash.
    """
    if len(points) > max_points:
        idx = np.random.default_rng(42).choice(len(points), max_points, replace=False)
        points = points[idx]

    rp = rect_params or {}

    # --- EDIFICIO RETTANGOLARE IDEALIZZATO (con finestre 3D e tetto rientrato) ---
    if method == "building_rect":
        parts = extrude_rectangular_building(
            points,
            floor_height=rp.get("floor_height", 3.0),
            window_w=rp.get("window_w", 1.2),
            window_h=rp.get("window_h", 1.5),
            window_spacing=rp.get("window_spacing", 3.5),
            add_windows=rp.get("add_windows", True),
            roof_inset=rp.get("roof_inset", 0.1),
            windows_mode=rp.get("windows_mode", "procedural"),
            window_params=rp,
        )

    # --- SQUARED CONCAVE HULL (profilo reale squadrato a 90° + multi-altezze) ---
    elif method == "building_squared":
        parts = extrude_building(
            points, lod=lod, use_concave=True,
            orthogonalize=True,
            multi_height=rp.get("multi_height", False),
            roof_inset=rp.get("roof_inset", 0.08),
            add_windows=rp.get("add_windows", False),
            windows_mode=rp.get("windows_mode", "procedural"),
            window_params=rp,
            epsilon=rp.get("epsilon", 0.8),
            roof_top_frac=rp.get("roof_top_frac", 0.10),
        )

    # --- FACCIATA 2D (nuvole planari / sezioni) -------------------------------
    elif method == "facade_2d":
        parts = extrude_facade_2d(
            points,
            facade_thickness=rp.get("facade_thickness", 0.05),
            window_detect=rp.get("facade_detect_windows", True),
            plane_band=rp.get("facade_plane_band", 0.30),
            cell=rp.get("facade_cell", 0.10),
            min_w=rp.get("facade_min_w", 0.4),
            max_w=rp.get("facade_max_w", 4.0),
            min_h=rp.get("facade_min_h", 0.4),
            max_h=rp.get("facade_max_h", 3.0),
        )

    # --- ESTRUSIONE EDIFICIO (footprint concave o convex, no snap) ---
    elif method in ("building_convex", "building_concave"):
        parts = extrude_building(
            points, lod=lod,
            use_concave=(method == "building_concave"),
            orthogonalize=False,
            multi_height=False,
            roof_inset=rp.get("roof_inset", 0.0),
            add_windows=rp.get("add_windows", False),
            windows_mode=rp.get("windows_mode", "procedural"),
            window_params=rp,
            epsilon=rp.get("epsilon", 0.5),
            roof_top_frac=rp.get("roof_top_frac", 0.10),
        )

    # --- RICOSTRUZIONE SUPERFICIE (organica) ---
    else:
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
        parts = {"surface": (verts, faces)}

    # Decimazione finale: riduce il carico su Plotly (evita crash a molti triangoli)
    if decimate_ratio and decimate_ratio > 0:
        parts = _decimate_parts(parts, ratio=decimate_ratio)
    return parts


# Palette CAD architettonica (look edificio reale)
_PART_COLORS = {
    "walls":   "#D1D1D1",   # muri grigio chiaro
    "roof":    "#5D4037",   # tetto marrone scuro
    "floor":   "#3D3D3D",   # pavimento scuro (raramente visibile)
    "windows": "#7DA9D1",   # finestre vetro azzurro riflettente
    "surface": "#B8C4D6",   # superficie organica (fallback)
}
# Prefisso → colore (permette varianti "walls_1", "roof_2" per blocchi multipli)
_PART_COLORS_PREFIXES = {
    "walls": "#D1D1D1",
    "roof":  "#5D4037",
    "floor": "#3D3D3D",
    "windows": "#7DA9D1",
}


def _color_for(name: str) -> str:
    if name in _PART_COLORS:
        return _PART_COLORS[name]
    for pfx, col in _PART_COLORS_PREFIXES.items():
        if name.startswith(pfx):
            return col
    return "#B8C4D6"


def _silhouette_edges(verts: np.ndarray, faces: np.ndarray) -> list[tuple[int, int]]:
    """
    Estrae gli edge "di contorno": quelli condivisi da facce con normali
    diverse (angolo > soglia). Evita di disegnare le diagonali interne
    dei quad → look CAD pulito senza rumore.
    """
    if len(faces) == 0:
        return []
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nrm = n / np.maximum(nn, 1e-9)

    from collections import defaultdict
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (int(tri[a]), int(tri[b])) if tri[a] < tri[b] else (int(tri[b]), int(tri[a]))
            edge_faces[key].append(fi)

    out: list[tuple[int, int]] = []
    cos_thresh = np.cos(np.deg2rad(20))  # angolo minimo per considerare "spigolo"
    for key, fs in edge_faces.items():
        if len(fs) == 1:
            out.append(key)
        else:
            dot = float(np.dot(nrm[fs[0]], nrm[fs[1]]))
            if dot < cos_thresh:
                out.append(key)
    return out


def _mesh_trace(verts, faces, color, flatshading=True, show_edges=False,
                opacity: float = 1.0, edge_color: str = "#111111",
                edge_width: float = 1.0):
    """Mesh3d trace con lighting architettonico + wireframe dei soli spigoli."""
    traces = [go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        flatshading=flatshading,
        opacity=opacity,
        lighting=dict(
            ambient=0.45, diffuse=0.85,
            specular=0.15, roughness=0.85, fresnel=0.05,
        ),
        lightposition=dict(x=10_000, y=10_000, z=20_000),
        hoverinfo="skip",
        showscale=False,
    )]
    if show_edges and len(faces) > 0:
        edges = _silhouette_edges(verts, faces)
        xs, ys, zs = [], [], []
        for a, b in edges:
            xs += [float(verts[a, 0]), float(verts[b, 0]), None]
            ys += [float(verts[a, 1]), float(verts[b, 1]), None]
            zs += [float(verts[a, 2]), float(verts[b, 2]), None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=edge_color, width=edge_width),
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
    for name, value in mesh_dict.items():
        if name.startswith("_"):  # chiavi meta (es. "_windows_meta")
            continue
        v, f = value
        if len(f) == 0:
            continue
        any_mesh = True
        color = _color_for(name)
        # Pavimento e finestre non mostrano gli edge (rumore visivo)
        is_floor = name.startswith("floor")
        is_window = name.startswith("windows")
        edges = show_edges and not is_floor and not is_window
        edge_w = 1.0 if not name.startswith("roof") else 1.2
        for tr in _mesh_trace(
            v, f, color,
            flatshading=True, show_edges=edges,
            edge_color="#111111", edge_width=edge_w,
        ):
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
# EXPORT HELPERS  (Sketchup/AutoCAD/Web/PNG — attacchabili a un incarto)
# ---------------------------------------------------------------------------

def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def mesh_to_obj(parts: dict, mtl_name: str | None = None) -> str:
    """Wavefront OBJ multi-gruppo (un gruppo/material per parte mesh)."""
    lines = ["# Point Cloud CAD Viewer — export OBJ"]
    if mtl_name:
        lines.append(f"mtllib {mtl_name}")
    v_offset = 1
    for name, value in parts.items():
        if name.startswith("_"):
            continue
        verts, faces = value
        if len(verts) == 0 or len(faces) == 0:
            continue
        lines.append(f"o {name}")
        lines.append(f"g {name}")
        if mtl_name:
            lines.append(f"usemtl {name}")
        for v in verts:
            lines.append(f"v {float(v[0]):.4f} {float(v[1]):.4f} {float(v[2]):.4f}")
        for tri in faces:
            a = int(tri[0]) + v_offset
            b = int(tri[1]) + v_offset
            c = int(tri[2]) + v_offset
            lines.append(f"f {a} {b} {c}")
        v_offset += len(verts)
    return "\n".join(lines) + "\n"


def mesh_to_mtl(parts: dict) -> str:
    """MTL che replica la palette CAD usata nel viewer."""
    lines = ["# Point Cloud CAD Viewer — materials"]
    for name in parts.keys():
        if name.startswith("_"):
            continue
        r, g, b = _hex_to_rgb01(_color_for(name))
        lines += [
            f"newmtl {name}",
            f"Kd {r:.3f} {g:.3f} {b:.3f}",
            f"Ka {r*0.35:.3f} {g*0.35:.3f} {b*0.35:.3f}",
            "Ks 0.050 0.050 0.050",
            "Ns 16.0",
            "illum 2",
            "",
        ]
    return "\n".join(lines) + "\n"


def mesh_to_obj_zip(parts: dict, base: str) -> bytes:
    """Zip contenente {base}.obj + {base}.mtl, pronto per Sketchup/Blender."""
    import io as _io
    import zipfile as _zip
    obj_text = mesh_to_obj(parts, mtl_name=f"{base}.mtl")
    mtl_text = mesh_to_mtl(parts)
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w", _zip.ZIP_DEFLATED) as z:
        z.writestr(f"{base}.obj", obj_text)
        z.writestr(f"{base}.mtl", mtl_text)
    return buf.getvalue()


def mesh_to_dxf_bytes(parts: dict) -> bytes:
    """DXF 3D con un layer per parte (walls/roof/windows/...), entità 3DFACE
    per massima compatibilità AutoCAD/BricsCAD/LibreCAD/Sketchup."""
    if not EZDXF_AVAILABLE:
        raise RuntimeError("ezdxf non installato")
    import io as _io
    import ezdxf as _ez
    doc = _ez.new("R2018", setup=True)
    msp = doc.modelspace()
    aci_by_prefix = {
        "walls":   8,    # grigio
        "roof":    32,   # marrone
        "windows": 5,    # blu
        "floor":   251,  # grigio scuro
        "surface": 9,
    }
    for name, value in parts.items():
        if name.startswith("_"):
            continue
        verts, faces = value
        if len(verts) == 0 or len(faces) == 0:
            continue
        pref = next((p for p in aci_by_prefix if name.startswith(p)), None)
        col = aci_by_prefix.get(pref, 7)
        if name not in doc.layers:
            doc.layers.add(name=name, color=col)
        for tri in faces:
            p0 = [float(x) for x in verts[int(tri[0])]]
            p1 = [float(x) for x in verts[int(tri[1])]]
            p2 = [float(x) for x in verts[int(tri[2])]]
            msp.add_3dface([p0, p1, p2, p2], dxfattribs={"layer": name})
    buf = _io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def fig_to_html_bytes(fig: go.Figure, title: str = "Modello 3D") -> bytes:
    """HTML self-contained (Plotly via CDN) — apribile ovunque, 3D interattivo."""
    html = fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
    )
    # Inietta un <title> pulito
    html = html.replace("<head>", f"<head><title>{title}</title>", 1)
    return html.encode("utf-8")


def fig_to_png_bytes(fig: go.Figure, width: int = 1600, height: int = 1000) -> bytes | None:
    """PNG ad alta risoluzione via kaleido (se disponibile)."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


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

    # ── 3 scelte principali: tipo di dato → metodo ─────────────────────────
    _method_labels = {
        "building_squared": "🏠 Edificio (scansione dall'alto)",
        "facade_2d":        "🪟 Facciata 2D (scansione di un muro)",
        "delaunay_2d":      "🏞️ Superficie libera (terreno / organica)",
    }
    method = st.radio(
        "Cosa vuoi ricostruire?",
        options=list(_method_labels.keys()),
        format_func=lambda x: _method_labels[x],
        index=0,
    )

    # ── Parametri essenziali (quelli che cambiano davvero il risultato) ───
    rect_params: dict = {}
    if method == "building_squared":
        rect_params["floor_height"] = st.slider(
            "Altezza piano (m)", 2.2, 5.0, 3.0, 0.1,
            help="Serve a distribuire le finestre in verticale.",
        )
        rect_params["add_windows"] = st.checkbox("Aggiungi finestre", value=True)
        if rect_params["add_windows"]:
            rect_params["windows_mode"] = "detected"   # default intelligente

    # Facciata 2D e Superficie libera non richiedono parametri visibili.

    # ── Opzioni avanzate (espanse solo se servono, collassate di default) ──
    with st.expander("⚙️ Opzioni avanzate", expanded=False):

        z_threshold = st.slider(
            "Soglia altezza (Z)", 0.0, 1.0, 0.0, 0.01,
            help="Esclude i punti sotto questa frazione del range Z (per togliere il terreno).",
        )
        show_points = st.checkbox("Sovrapponi nuvola di punti", value=False)

        # -- Avanzate specifiche per metodo --
        if method == "building_squared":
            st.markdown("**Edificio – dettagli geometrici**")
            rect_params["roof_inset"] = st.slider(
                "Rientro tetto (m)", 0.0, 0.5, 0.08, 0.02,
            )
            rect_params["epsilon"] = st.slider(
                "Semplificazione sagoma (m)", 0.2, 3.0, 0.8, 0.1,
                help="Più alta = meno vertici (rettangolo → 4, L-shape → 6).",
            )
            rect_params["roof_top_frac"] = st.slider(
                "Filtro tetto (% altezza)", 0.05, 1.00, 0.25, 0.05,
                help="Usa solo i punti nel top-N% per stimare la sagoma.",
            )
            rect_params["multi_height"] = st.checkbox(
                "Rileva piani sfalsati (torre/attico)", value=False,
            )
            if rect_params.get("add_windows"):
                _win_mode_labels = {
                    "detected":   "🔍 Rilevate dalla nuvola (reali)",
                    "procedural": "📐 Procedurali (griglia uniforme)",
                }
                rect_params["windows_mode"] = st.radio(
                    "Modalità finestre",
                    options=list(_win_mode_labels.keys()),
                    format_func=lambda x: _win_mode_labels[x],
                    index=0,
                )
                if rect_params["windows_mode"] == "detected":
                    rect_params["wall_thickness"] = st.slider(
                        "Banda parete (m)", 0.10, 1.00, 0.35, 0.05,
                    )
                    rect_params["detect_cell"] = st.slider(
                        "Risoluzione detection (m)", 0.05, 0.30, 0.10, 0.01,
                    )
                else:
                    rect_params["window_w"] = st.slider("Larghezza finestra (m)", 0.6, 2.5, 1.2, 0.1)
                    rect_params["window_h"] = st.slider("Altezza finestra (m)", 0.8, 2.4, 1.5, 0.1)
                    rect_params["window_spacing"] = st.slider("Passo orizzontale (m)", 2.0, 8.0, 3.5, 0.1)

        elif method == "facade_2d":
            st.markdown("**Facciata 2D – dettagli detection**")
            rect_params["facade_thickness"] = st.slider(
                "Spessore pannello (m)", 0.02, 0.50, 0.05, 0.01,
            )
            rect_params["facade_detect_windows"] = st.checkbox(
                "Rileva finestre sul piano", value=True,
            )
            if rect_params["facade_detect_windows"]:
                rect_params["facade_plane_band"] = st.slider(
                    "Banda piano (m)", 0.05, 1.00, 0.30, 0.05,
                )
                rect_params["facade_cell"] = st.slider(
                    "Risoluzione detection (m)", 0.05, 0.30, 0.10, 0.01,
                )

        # -- Rendering (raramente si cambiano) --
        st.markdown("**Rendering**")
        lod = st.slider("Livello di dettaglio mesh", 1, 10, 5)
        decimate_pct = st.slider("Decimazione mesh (%)", 0, 95, 90, 5)
        theme = st.radio("Tema 3D", ["light", "dark"], horizontal=True,
                         format_func=lambda x: "Chiaro (CAD)" if x == "light" else "Scuro",
                         index=1)
        show_edges = st.checkbox("Mostra spigoli (wireframe)", value=True)

        # -- Selezione --
        shape_mode = st.radio(
            "Forma selezione", ["rect", "rect_rotated", "circle"],
            format_func=lambda x: {
                "rect": "Rettangolo",
                "rect_rotated": "Rettangolo ruotabile",
                "circle": "Cerchio",
            }[x],
            horizontal=True,
        )

        # -- Formati --
        st.markdown("**Formati supportati**")
        st.caption(
            "`.xyz` / `.txt` / `.csv` · `.las` / `.laz` (LiDAR, RGB opz.) · "
            "`.dxf` · `.dwg` (richiede ODA File Converter)"
        )
        if not EZDXF_AVAILABLE:
            st.warning("`ezdxf` non installato: DXF/DWG disabilitati.")
        if not LASPY_AVAILABLE:
            st.warning("`laspy` non installato: LAS/LAZ disabilitati.")

    decimate_ratio = decimate_pct / 100.0

# Upload
_accepted = ["xyz", "txt", "csv", "dxf", "dwg"]
if LASPY_AVAILABLE:
    _accepted += ["las", "laz"]
uploaded = st.file_uploader(
    "Trascina qui il file (.xyz / .las / .laz / .dxf / .dwg)",
    type=_accepted,
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
        elif ext in (".las", ".laz"):
            df_full = load_las(uploaded.getvalue(), suffix=ext)
        else:
            df_full = load_xyz(uploaded.getvalue())
except Exception as e:
    st.error(f"Errore nel parsing: {e}")
    st.stop()

# Info ricentramento LAS (se applicato, per preservare precisione float32)
_las_off = df_full.attrs.get("las_offsets", {}) if hasattr(df_full, "attrs") else {}
if _las_off:
    _fmt = ", ".join(f"{k} −{v:,.0f}".replace(",", "'") for k, v in _las_off.items())
    st.info(
        f"🛰️ File LAS georeferenziato: coordinate traslate per preservare "
        f"precisione ({_fmt} m). Il modello resta in scala reale; "
        f"per riportarlo nel CRS originale somma gli offset in export."
    )

# --- DIAGNOSTICA bounding box (sempre visibile) ------------------------------
st.caption(f"🛠️ build 2026-04-24b · facade 2D + simplified UI")
with st.expander("🔍 Diagnostica nuvola (bounding box)", expanded=True):
    _bx = float(df_full["X"].max() - df_full["X"].min())
    _by = float(df_full["Y"].max() - df_full["Y"].min())
    _bz = float(df_full["Z"].max() - df_full["Z"].min())
    _maxd = max(_bx, _by, _bz) or 1.0
    _mind = min(_bx, _by, _bz)
    _ratio = _mind / _maxd
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Extent X", f"{_bx:.2f} m")
    d2.metric("Extent Y", f"{_by:.2f} m")
    d3.metric("Extent Z", f"{_bz:.2f} m")
    d4.metric("Ratio min/max", f"{_ratio:.4f}",
              help="Se < 0.02 → nuvola quasi-planare (facciata singola, tetto, ecc.)")
    if _ratio < 0.02:
        # Quale asse è "schiacciato"?
        _axes = {"X": _bx, "Y": _by, "Z": _bz}
        _flat = min(_axes, key=_axes.get)
        st.warning(
            f"⚠️ Asse **{_flat}** quasi-piatto ({_axes[_flat]:.3f} m vs "
            f"{_maxd:.1f} m sugli altri). Significa che la nuvola caricata è "
            f"una **singola facciata / piano** (non un edificio completo dall'alto). "
            f"Per estrudere un volume 3D servono punti distribuiti anche sul "
            f"terzo asse."
        )
    st.caption(
        f"Range grezzi: X [{df_full['X'].min():.2f}, {df_full['X'].max():.2f}] · "
        f"Y [{df_full['Y'].min():.2f}, {df_full['Y'].max():.2f}] · "
        f"Z [{df_full['Z'].min():.2f}, {df_full['Z'].max():.2f}]"
    )

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

    elif shape_mode == "rect_rotated":
        st.caption("Regola centro, dimensioni e rotazione con gli slider a destra per allineare il rettangolo all'edificio.")
        rotrect_placeholder = st.empty()
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

    elif shape_mode == "rect_rotated":
        span_x, span_y = x_max - x_min, y_max - y_min
        cx = st.slider("Centro X", x_min, x_max, float((x_min + x_max) / 2), key="rr_cx")
        cy = st.slider("Centro Y", y_min, y_max, float((y_min + y_max) / 2), key="rr_cy")
        w = st.slider("Larghezza", span_x / 100, span_x, span_x / 3, key="rr_w")
        h = st.slider("Altezza",   span_y / 100, span_y, span_y / 3, key="rr_h")
        angle_deg = st.slider("Rotazione (°)", -180.0, 180.0, 0.0, 1.0, key="rr_ang",
                              help="Ruota il rettangolo per allinearlo all'edificio.")
        selection = {
            "type": "rect_rotated",
            "cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg,
        }
        with left:
            fig = build_topdown_plotly(preview_src, dragmode="pan", rect_rotated=selection)
            rotrect_placeholder.plotly_chart(fig, use_container_width=True, key=f"rotrect_{file_id}")

    if selection:
        t = selection["type"]
        if t == "rect":
            st.code(
                f"Rettangolo\n"
                f"X: [{selection['x1']:.2f}, {selection['x2']:.2f}]\n"
                f"Y: [{selection['y1']:.2f}, {selection['y2']:.2f}]\n"
                f"Area: {(selection['x2']-selection['x1'])*(selection['y2']-selection['y1']):.2f}"
            )
        elif t == "rect_rotated":
            st.code(
                f"Rettangolo ruotato\n"
                f"Centro: ({selection['cx']:.2f}, {selection['cy']:.2f})\n"
                f"W×H: {selection['w']:.2f} × {selection['h']:.2f}\n"
                f"Angolo: {selection['angle_deg']:.1f}°\n"
                f"Area: {selection['w']*selection['h']:.2f}"
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
                has_rgb = {"R", "G", "B"}.issubset(source_df.columns)
                cols = ["X", "Y", "Z", "R", "G", "B"] if has_rgb else ["X", "Y", "Z"]
                pts = source_df[cols].to_numpy(dtype=np.float32)
                mesh_parts = reconstruct_mesh_arrays(
                    points=pts, lod=lod, method=method,
                    rect_params=rect_params,
                    decimate_ratio=decimate_ratio,
                )
                # Diagnostica: mesh quasi degenere?
                all_v = [v for v, _ in (
                    (k, val) for k, val in mesh_parts.items() if not k.startswith("_")
                ) if len(v) > 0]
                # rigenera (helper per iterare senza le chiavi meta)
                _mesh_items = [(k, val) for k, val in mesh_parts.items() if not k.startswith("_")]
                all_v = [val[0] for _, val in _mesh_items if len(val[0]) > 0]
                if all_v:
                    stacked = np.concatenate(all_v, axis=0)
                    extent = stacked.max(axis=0) - stacked.min(axis=0)
                    if extent.max() > 0 and extent.min() < 0.02 * extent.max():
                        if method == "facade_2d":
                            # Atteso: il metodo 2D produce una mesh "piatta"
                            pass
                        else:
                            st.warning(
                                "⚠️ Geometria molto sottile: la nuvola è "
                                "intrinsecamente planare. Per questo tipo di dati "
                                "usa il metodo **🪟 Facciata 2D (scansione di un muro)** "
                                "— rileva il piano e le finestre sul piano, "
                                "senza tentare un'estrusione volumetrica."
                            )
                st.session_state.mesh_data = {
                    "parts": mesh_parts,
                    "points": pts[:, :3] if show_points else None,
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

    mesh_items = [(k, v) for k, v in parts.items() if not k.startswith("_")]
    total_v = sum(len(v) for _, (v, _) in (((k, val) for k, val in mesh_items)))
    total_f = sum(len(f) for _, (_, f) in (((k, val) for k, val in mesh_items)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Parti mesh", len(mesh_items))
    c2.metric("Vertici", f"{total_v:,}")
    c3.metric("Triangoli", f"{total_f:,}")

    # ─── Facciata 2D: info piano rilevato ──────────────────────────────────
    fac_meta = parts.get("_facade_meta")
    if fac_meta and fac_meta.get("status") == "not_a_plane":
        st.error(
            f"❌ **Impossibile ricostruire: la nuvola non è un piano, è una linea.**\n\n"
            f"{fac_meta['reason']}\n\n"
            "**Cause tipiche**:\n"
            "• File esportato come *profilo / sezione / traccia* invece della nuvola completa\n"
            "• Selezione troppo stretta sulla vista top-down\n"
            "• File originato da un trasduttore 1D (livella laser, profilometro)\n\n"
            "**Cosa fare**: ricarica il LAS originale (non sezionato) oppure carica un altro file. "
            "Per modellare un edificio serve una nuvola 3D (drone aereo) o una facciata scansionata "
            "a terra con spessore di almeno 10 cm."
        )
    elif fac_meta and fac_meta.get("u_size"):
        st.info(
            f"🪟 **Facciata 2D rilevata** · piano **{fac_meta['u_axis']}-"
            f"{fac_meta['v_axis']}** (normale = asse **{fac_meta['normal_axis']}**) · "
            f"dimensioni **{fac_meta['u_size']:.2f} × {fac_meta['v_size']:.2f} m** "
            f"({fac_meta['facade_area']:.1f} m²) · spessore reale nuvola "
            f"{fac_meta['real_thickness']*100:.1f} cm"
        )

    # ─── Stato modalità finestre + statistiche se detected ──────────────────
    win_meta = parts.get("_windows_meta")
    if win_meta is not None and len(win_meta) > 0:
        st.success(
            f"🔍 **Finestre rilevate dalla nuvola**: {len(win_meta)} aperture identificate."
        )
        # Modalità facciata 2D: nessun compass, mostra tabella semplice dimensioni
        if fac_meta and all(w.get("compass", "—") == "—" for w in win_meta):
            tot_area = sum(w["area"] for w in win_meta)
            tot_fac = fac_meta.get("facade_area", 0) or 1.0
            st.caption(
                f"Superficie vetrata totale: **{tot_area:.2f} m²** "
                f"({tot_area/tot_fac*100:.1f}% della facciata)"
            )
            rows_f = [
                {
                    "N°": i + 1,
                    "Larghezza (m)": round(w["width"], 2),
                    "Altezza (m)": round(w["height"], 2),
                    "Area (m²)": round(w["area"], 2),
                    "u (m)": round((w["u_lo"] + w["u_hi"]) / 2, 2),
                    "v (m)": round((w["v_lo"] + w["v_hi"]) / 2, 2),
                }
                for i, w in enumerate(win_meta)
            ]
            st.markdown("**Finestre rilevate sul piano**")
            st.dataframe(pd.DataFrame(rows_f), use_container_width=True, hide_index=True)
        else:
            by_compass: dict[str, list[dict]] = {}
            for w in win_meta:
                by_compass.setdefault(w["compass"], []).append(w)
            if by_compass:
                rows = []
                for d in ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]:
                    ws = by_compass.get(d, [])
                    if not ws:
                        continue
                    tot_area = sum(w["area"] for w in ws)
                    rows.append({
                        "Orientamento": d,
                        "N° finestre": len(ws),
                        "Area totale (m²)": round(tot_area, 2),
                        "Larghezza media (m)": round(sum(w["width"] for w in ws) / len(ws), 2),
                        "Altezza media (m)": round(sum(w["height"] for w in ws) / len(ws), 2),
                    })
                if rows:
                    st.markdown("**Ripartizione finestre per orientamento**")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif win_meta is not None and len(win_meta) == 0 and method == "facade_2d":
        st.warning(
            "🔍 Nessuna finestra rilevata sul piano. Prova ad abbassare **Min L/Min H**, "
            "aumentare **Banda piano**, o verifica che la nuvola abbia sufficienti "
            "variazioni di densità/colore (voids = aree senza punti dove c'è il vetro)."
        )
    elif rect_params.get("windows_mode") == "procedural" and rect_params.get("add_windows"):
        st.warning(
            "⚠️ **Finestre procedurali**: pattern uniforme non derivato dal dato reale. "
            "Se hai una nuvola con colori RGB, seleziona *\"Rilevate dalla nuvola\"* "
            "per ottenere posizioni reali."
        )

    # ─── EXPORT ──────────────────────────────────────────────────────────────
    st.markdown("#### 📥 Esporta per l'incarto energia")
    st.caption(
        "Scarica il modello direttamente allegabile all'incarto — niente SketchUp, "
        "niente rielaborazione esterna."
    )
    base = Path(uploaded.name).stem or "modello"
    ec1, ec2, ec3, ec4 = st.columns(4)

    try:
        obj_zip = mesh_to_obj_zip(parts, base)
        ec1.download_button(
            "⬇ OBJ + MTL (zip)",
            data=obj_zip, file_name=f"{base}_obj.zip",
            mime="application/zip", use_container_width=True,
            help="Apribile in SketchUp / Blender / Rhino con colori CAD preservati.",
        )
    except Exception as e:
        ec1.button("OBJ non disponibile", disabled=True, use_container_width=True)
        ec1.caption(f"Errore: {e}")

    if EZDXF_AVAILABLE:
        try:
            dxf_bytes = mesh_to_dxf_bytes(parts)
            ec2.download_button(
                "⬇ DXF 3D (AutoCAD)",
                data=dxf_bytes, file_name=f"{base}.dxf",
                mime="application/dxf", use_container_width=True,
                help="3DFACE + layer per parte (walls/roof/windows).",
            )
        except Exception as e:
            ec2.button("DXF non disponibile", disabled=True, use_container_width=True)
            ec2.caption(f"Errore: {e}")
    else:
        ec2.button("DXF — manca ezdxf", disabled=True, use_container_width=True)

    try:
        html_bytes = fig_to_html_bytes(fig_3d, title=f"Modello 3D — {base}")
        ec3.download_button(
            "⬇ HTML interattivo",
            data=html_bytes, file_name=f"{base}.html",
            mime="text/html", use_container_width=True,
            help="Pagina autonoma con 3D ruotabile — allegabile a mail/incarto.",
        )
    except Exception as e:
        ec3.button("HTML non disponibile", disabled=True, use_container_width=True)
        ec3.caption(f"Errore: {e}")

    png_bytes = fig_to_png_bytes(fig_3d)
    if png_bytes:
        ec4.download_button(
            "⬇ PNG (alta risoluzione)",
            data=png_bytes, file_name=f"{base}.png",
            mime="image/png", use_container_width=True,
            help="Immagine 1600×1000 @2x — incollabile direttamente nel documento.",
        )
    else:
        ec4.button("PNG — installa kaleido", disabled=True, use_container_width=True)
        ec4.caption("`pip install kaleido` per l'export statico PNG.")

    df_3d = st.session_state.cropped_df if st.session_state.cropped_df is not None else df_filtered
    with st.expander("🔎 Anteprima dati"):
        st.dataframe(df_3d.head(200), use_container_width=True)
else:
    st.info(
        "👉 Seleziona un'area e clicca **Conferma Selezione e Genera 3D**, "
        "oppure **Genera 3D (tutta la nuvola)** per ricostruire il modello."
    )
