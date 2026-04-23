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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st
from PIL import Image
from stpyvista import stpyvista

# --- Compat shim per streamlit-drawable-canvas su Streamlit >= 1.40 ---
# image_to_url è stata spostata da streamlit.elements.image a
# streamlit.elements.lib.image_utils. Il componente non è aggiornato, quindi
# ricolleghiamo il simbolo nel namespace originale prima dell'import.
try:
    import streamlit.elements.image as _st_image  # type: ignore
    if not hasattr(_st_image, "image_to_url"):
        try:
            from streamlit.elements.lib.image_utils import image_to_url as _img2url
        except ImportError:
            from streamlit.elements.lib.image_utils import _image_to_url as _img2url
        _st_image.image_to_url = _img2url
except Exception:
    pass

from streamlit_drawable_canvas import st_canvas  # noqa: E402

# CAD loaders opzionali
try:
    import ezdxf
    from ezdxf.addons import odafc
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

# PyVista in modalità headless (server Streamlit)
pv.OFF_SCREEN = True
if os.name != "nt":
    try:
        pv.start_xvfb()
    except Exception:
        pass


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
# VISTA PLANARE (PNG) — sfondo per il canvas di selezione
# ---------------------------------------------------------------------------
def render_topdown_preview(df: pd.DataFrame, target_size: int = 600):
    """
    Rende una vista dall'alto (scatter XY) come PIL Image.
    Ritorna (img, (x_min, x_max, y_min, y_max), (img_w, img_h)).
    L'aspect ratio del plot riflette quello del mondo reale → mapping pixel↔mondo esatto.
    """
    x_min, x_max = float(df["X"].min()), float(df["X"].max())
    y_min, y_max = float(df["Y"].min()), float(df["Y"].max())
    dx, dy = x_max - x_min, y_max - y_min
    if dx <= 0 or dy <= 0:
        raise ValueError("Estensione XY nulla.")

    # Dimensioni finali preservando aspect
    if dx >= dy:
        img_w = target_size
        img_h = max(100, int(target_size * dy / dx))
    else:
        img_h = target_size
        img_w = max(100, int(target_size * dx / dy))

    fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0E1117")
    ax.scatter(df["X"], df["Y"], s=0.4, c=df["Z"], cmap="viridis", linewidths=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", facecolor="#0E1117")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((img_w, img_h))
    return img, (x_min, x_max, y_min, y_max), (img_w, img_h)


# ---------------------------------------------------------------------------
# MAPPING CANVAS → COORDINATE MONDO
# ---------------------------------------------------------------------------
def canvas_obj_to_world(obj: dict, bounds, img_dims) -> dict | None:
    """
    Converte un oggetto disegnato (rect o circle) da pixel canvas a coord mondo.
    Canvas: origine top-left, Y verso il basso. Mondo: Y verso l'alto.
    """
    x_min, x_max, y_min, y_max = bounds
    img_w, img_h = img_dims
    sx = (x_max - x_min) / img_w
    sy = (y_max - y_min) / img_h
    t = obj.get("type")

    if t == "rect":
        left = obj["left"]; top = obj["top"]
        w = obj["width"] * obj.get("scaleX", 1)
        h = obj["height"] * obj.get("scaleY", 1)
        x1 = x_min + left * sx
        x2 = x_min + (left + w) * sx
        # flip Y
        y1 = y_max - top * sy
        y2 = y_max - (top + h) * sy
        return {
            "type": "rect",
            "x1": min(x1, x2), "x2": max(x1, x2),
            "y1": min(y1, y2), "y2": max(y1, y2),
        }

    if t == "circle":
        r_px = obj["radius"] * obj.get("scaleX", 1)
        cx_px = obj["left"] + r_px
        cy_px = obj["top"] + r_px
        cx = x_min + cx_px * sx
        cy = y_max - cy_px * sy
        r = r_px * (sx + sy) / 2
        return {"type": "circle", "cx": cx, "cy": cy, "r": r}

    return None


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


def build_plotter(df: pd.DataFrame, lod: int, method: str, show_points: bool) -> pv.Plotter:
    """Plotter PyVista con shading e illuminazione per look CAD."""
    cloud = build_cloud(df)
    surf = reconstruct(cloud, lod, method)

    pl = pv.Plotter(window_size=[900, 650], off_screen=True)
    pl.set_background("#0E1117")

    if surf.n_cells > 0:
        pl.add_mesh(
            surf,
            color="#B8C4D6",          # grigio-azzurro architettonico
            smooth_shading=True,
            specular=0.4,
            specular_power=20,
            ambient=0.25,
            diffuse=0.75,
            show_edges=False,
            lighting=True,
        )
    else:
        # fallback a nuvola se la ricostruzione fallisce
        pl.add_mesh(cloud, color="#B8C4D6", point_size=3, render_points_as_spheres=True)

    if show_points:
        pl.add_mesh(cloud, color="#00E5FF", point_size=1.5,
                    render_points_as_spheres=True, opacity=0.6)

    pl.enable_eye_dome_lighting()  # depth cue senza costo
    pl.add_light(pv.Light(position=(1, 1, 1), intensity=0.6, light_type="scene light"))
    pl.view_isometric()
    return pl


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("current_file", None)
    ss.setdefault("cropped_df", None)
    ss.setdefault("last_selection", None)

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
    method = st.radio(
        "Metodo ricostruzione",
        options=["reconstruct_surface", "delaunay_2d"],
        format_func=lambda x: "Reconstruct Surface (VTK)" if x == "reconstruct_surface"
                              else "Delaunay 2.5D (edifici)",
        index=0,
    )
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

with left:
    st.subheader("📐 Vista planare — disegna l'area di interesse")

    # Downsample per la preview (rendering matplotlib veloce)
    preview_src = df_filtered
    if len(preview_src) > 80_000:
        preview_src = preview_src.sample(80_000, random_state=42)

    img, bounds, img_dims = render_topdown_preview(preview_src, target_size=600)

    canvas = st_canvas(
        fill_color="rgba(0, 229, 255, 0.15)",
        stroke_width=2,
        stroke_color="#00E5FF",
        background_color="#0E1117",
        background_image=img,
        update_streamlit=True,
        width=img_dims[0],
        height=img_dims[1],
        drawing_mode=shape_mode,
        key=f"canvas_{file_id}_{shape_mode}",
    )

with right:
    st.subheader("🎯 Selezione")

    selection = None
    if canvas.json_data is not None and canvas.json_data.get("objects"):
        # Prendi l'ultima forma disegnata
        last = canvas.json_data["objects"][-1]
        selection = canvas_obj_to_world(last, bounds, img_dims)
        st.session_state.last_selection = selection

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

        if selection["type"] == "circle":
            scale = st.slider("Scala raggio", 0.1, 3.0, 1.0, 0.05)
            selection = {**selection, "r": selection["r"] * scale}
    else:
        st.info("Disegna un rettangolo o un cerchio sulla vista planare.")

    st.markdown("")
    confirm = st.button(
        "✅ Conferma Selezione e Ritaglia 3D",
        type="primary",
        disabled=selection is None,
        use_container_width=True,
    )
    if st.button("↩️ Reset vista 3D", use_container_width=True):
        st.session_state.cropped_df = None

    if confirm and selection is not None:
        cropped = crop_by_selection(df_filtered, selection)
        if cropped.empty:
            st.error("Nessun punto nella selezione.")
        else:
            st.session_state.cropped_df = cropped
            st.success(f"Ritagliati {len(cropped):,} punti.")

# =========================================================================
# STEP 2 — VISTA 3D RICOSTRUITA
# =========================================================================
st.markdown("---")
st.subheader("🏛️ Modello 3D ricostruito")

df_3d = st.session_state.cropped_df if st.session_state.cropped_df is not None else df_filtered

if len(df_3d) > 200_000:
    st.info(f"Downsample automatico da {len(df_3d):,} a 200.000 punti per la ricostruzione.")
    df_3d = df_3d.sample(200_000, random_state=42).reset_index(drop=True)

with st.spinner("Ricostruzione superficie..."):
    plotter = build_plotter(df_3d, lod=lod, method=method, show_points=show_points)

stpyvista(plotter, key=f"pv_{file_id}_{len(df_3d)}_{lod}_{method}")

with st.expander("🔎 Anteprima dati ritagliati"):
    st.dataframe(df_3d.head(200), use_container_width=True)
