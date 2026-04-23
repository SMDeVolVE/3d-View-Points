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
import streamlit.components.v1 as components

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


def generate_3d_html(
    points: np.ndarray,
    rgb: np.ndarray | None,
    lod: int,
    method: str,
    show_points: bool,
    max_points: int = 60_000,
    max_cells: int = 120_000,
) -> str:
    """
    Pipeline completo: PolyData → superficie → HTML standalone VTK.js.
    - Downsample input a `max_points` per tenere la RAM sotto controllo.
    - Decimation automatica se la mesh supera `max_cells` celle.
    """
    # Downsample duro dei punti in ingresso (protegge da OOM)
    if len(points) > max_points:
        idx = np.random.default_rng(42).choice(len(points), max_points, replace=False)
        points = points[idx]
        if rgb is not None:
            rgb = rgb[idx]

    cloud = pv.PolyData(points.astype(np.float32))
    if rgb is not None:
        cloud["RGB"] = rgb.astype(np.uint8)

    surf = reconstruct(cloud, lod, method)

    # Decima la mesh se è troppo densa per l'export HTML
    if surf.n_cells > max_cells:
        try:
            ratio = 1.0 - (max_cells / surf.n_cells)
            surf = surf.decimate(ratio)
        except Exception:
            pass

    pl = pv.Plotter(window_size=[900, 650], off_screen=True)
    pl.set_background("#0E1117")

    if surf.n_cells > 0:
        pl.add_mesh(
            surf,
            color="#B8C4D6",
            smooth_shading=True,
            specular=0.4, specular_power=20,
            ambient=0.25, diffuse=0.75,
            show_edges=False, lighting=True,
        )
    else:
        pl.add_mesh(cloud, color="#B8C4D6", point_size=3, render_points_as_spheres=True)

    if show_points:
        pl.add_mesh(cloud, color="#00E5FF", point_size=1.5,
                    render_points_as_spheres=True, opacity=0.6)

    pl.enable_eye_dome_lighting()
    pl.add_light(pv.Light(position=(1, 1, 1), intensity=0.6, light_type="scene light"))
    pl.view_isometric()

    # Export HTML (VTK.js) senza subprocess
    tmp_path = None
    html_str = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            tmp_path = f.name
        pl.export_html(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as fh:
            html_str = fh.read()
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass
        try: pl.close()
        except Exception: pass

    return html_str


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("current_file", None)
    ss.setdefault("cropped_df", None)
    ss.setdefault("last_selection", None)
    ss.setdefault("mesh_html", None)

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
    st.session_state.mesh_html = None

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
        st.session_state.mesh_html = None

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
                has_rgb = {"R", "G", "B"}.issubset(source_df.columns)
                rgb = source_df[["R", "G", "B"]].to_numpy(dtype=np.uint8) if has_rgb else None
                html = generate_3d_html(
                    points=pts, rgb=rgb,
                    lod=lod, method=method, show_points=show_points,
                )
                st.session_state.mesh_html = html
            except Exception as e:
                st.error(f"Errore nella ricostruzione 3D: {e}")
                st.session_state.mesh_html = None

# =========================================================================
# STEP 2 — VISTA 3D RICOSTRUITA
# =========================================================================
st.markdown("---")
st.subheader("🏛️ Modello 3D ricostruito")

if st.session_state.mesh_html:
    components.html(st.session_state.mesh_html, height=670, scrolling=False)
    df_3d = st.session_state.cropped_df if st.session_state.cropped_df is not None else df_filtered
    with st.expander("🔎 Anteprima dati"):
        st.dataframe(df_3d.head(200), use_container_width=True)
else:
    st.info(
        "👉 Seleziona un'area e clicca **Conferma Selezione e Genera 3D**, "
        "oppure **Genera 3D (tutta la nuvola)** per ricostruire il modello."
    )
