"""
Point Cloud Viewer - Streamlit Web App
Carica un file .xyz / .dxf / .dwg (nuvola di punti) e visualizza
uno schema 3D interattivo dell'edificio.
"""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ezdxf è opzionale: se manca, i formati CAD vengono disabilitati con messaggio chiaro
try:
    import ezdxf
    from ezdxf.addons import odafc
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


# ---------------------------------------------------------------------------
# CONFIG PAGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Point Cloud Viewer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Stile minimalista dark
st.markdown(
    """
    <style>
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { color: #FAFAFA; }
        .metric-card {
            background-color: #1E222A;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #2A2F3A;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# PARSING FILE .XYZ
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_xyz(file_bytes: bytes) -> pd.DataFrame:
    """
    Parser robusto per file .xyz.
    - Rileva automaticamente il delimitatore (spazio, tab, virgola, punto e virgola)
    - Rileva automaticamente la presenza di un header
    - Supporta 3 colonne (X Y Z) o 6 colonne (X Y Z R G B)
    """
    # Decodifica e prendi un campione per lo sniffing
    text = file_bytes.decode("utf-8", errors="ignore")
    sample_lines = [l for l in text.splitlines() if l.strip()][:20]
    if not sample_lines:
        raise ValueError("Il file è vuoto o non leggibile.")

    # Rileva delimitatore contando gli split su più righe
    candidates = [",", ";", "\t", " "]
    delim = " "
    best_score = 0
    for c in candidates:
        counts = [len(l.split(c)) for l in sample_lines]
        # Score = numero di colonne medio, purché consistente
        if len(set(counts)) == 1 and counts[0] >= 3 and counts[0] > best_score:
            best_score = counts[0]
            delim = c
    # Per spazi multipli usiamo whitespace regex
    sep = r"\s+" if delim == " " else delim

    # Rileva header: prova a convertire la prima riga in float
    first_tokens = sample_lines[0].replace(",", " ").split()
    try:
        [float(t) for t in first_tokens[:3]]
        header = None
    except ValueError:
        header = 0

    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        header=header,
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )

    # Normalizza nomi colonne
    ncols = df.shape[1]
    if ncols < 3:
        raise ValueError(f"Formato non valido: trovate {ncols} colonne, attese almeno 3.")

    if ncols >= 6:
        df = df.iloc[:, :6]
        df.columns = ["X", "Y", "Z", "R", "G", "B"]
        # Se i valori RGB sembrano normalizzati [0,1] portali a [0,255]
        if df[["R", "G", "B"]].max().max() <= 1.0:
            df[["R", "G", "B"]] = (df[["R", "G", "B"]] * 255).astype(int)
        else:
            df[["R", "G", "B"]] = df[["R", "G", "B"]].astype(int).clip(0, 255)
    else:
        df = df.iloc[:, :3]
        df.columns = ["X", "Y", "Z"]

    # Forza numerico e rimuovi NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# PARSING FILE DXF / DWG
# ---------------------------------------------------------------------------
# Indice AutoCAD → RGB per i 256 colori ACI standard (sottoinsieme comune).
# ezdxf espone già la conversione via aci2rgb, quindi deleghiamo a lui.
def _aci_to_rgb(aci: int):
    try:
        from ezdxf.colors import aci2rgb
        return aci2rgb(aci)
    except Exception:
        return None


def _extract_points_from_doc(doc) -> pd.DataFrame:
    """
    Estrae coordinate 3D da un documento ezdxf.
    Priorità: entità POINT (standard per nuvole). Fallback: vertici di entità 3D
    comuni (3DFACE, MESH, LINE endpoints, INSERT positions).
    Colore: true_color se presente, altrimenti color index ACI.
    """
    msp = doc.modelspace()
    rows = []

    for e in msp:
        dtype = e.dxftype()

        # True color (0xRRGGBB) se impostato
        rgb = None
        try:
            if e.dxf.hasattr("true_color"):
                tc = e.dxf.true_color
                rgb = ((tc >> 16) & 0xFF, (tc >> 8) & 0xFF, tc & 0xFF)
        except Exception:
            rgb = None
        if rgb is None:
            try:
                rgb = _aci_to_rgb(e.dxf.color)
            except Exception:
                rgb = None

        if dtype == "POINT":
            p = e.dxf.location
            rows.append((p.x, p.y, p.z, rgb))
        elif dtype == "LINE":
            s, en = e.dxf.start, e.dxf.end
            rows.append((s.x, s.y, s.z, rgb))
            rows.append((en.x, en.y, en.z, rgb))
        elif dtype == "3DFACE":
            for attr in ("vtx0", "vtx1", "vtx2", "vtx3"):
                try:
                    v = getattr(e.dxf, attr)
                    rows.append((v.x, v.y, v.z, rgb))
                except AttributeError:
                    pass
        elif dtype == "MESH":
            try:
                for v in e.vertices:
                    rows.append((v[0], v[1], v[2], rgb))
            except Exception:
                pass
        elif dtype == "INSERT":
            p = e.dxf.insert
            rows.append((p.x, p.y, p.z, rgb))

    if not rows:
        raise ValueError(
            "Nessun punto trovato nel file CAD. "
            "Il file deve contenere entità POINT, LINE, 3DFACE, MESH o INSERT nel modelspace."
        )

    xs, ys, zs, cols = zip(*rows)
    df = pd.DataFrame({"X": xs, "Y": ys, "Z": zs})

    # Se almeno una entità ha colore valido, aggiungiamo le colonne RGB
    if any(c is not None for c in cols):
        r, g, b = [], [], []
        for c in cols:
            if c is None:
                r.append(200); g.append(200); b.append(200)  # grigio default
            else:
                r.append(int(c[0])); g.append(int(c[1])); b.append(int(c[2]))
        df["R"], df["G"], df["B"] = r, g, b

    return df.dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_dxf(file_bytes: bytes) -> pd.DataFrame:
    """Carica un file .dxf (ASCII o binary) e ritorna il DataFrame X Y Z [R G B]."""
    if not EZDXF_AVAILABLE:
        raise RuntimeError("La libreria ezdxf non è installata. Esegui: pip install ezdxf")

    # ezdxf preferisce un path su disco per essere robusto con DXF binari
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return _extract_points_from_doc(doc)


@st.cache_data(show_spinner=False)
def load_dwg(file_bytes: bytes) -> pd.DataFrame:
    """
    Carica un file .dwg convertendolo prima in DXF tramite ODA File Converter.
    Richiede che ODA File Converter sia installato sul sistema.
    """
    if not EZDXF_AVAILABLE:
        raise RuntimeError("La libreria ezdxf non è installata. Esegui: pip install ezdxf")

    tmpdir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
    dwg_path = tmpdir / "input.dwg"
    dwg_path.write_bytes(file_bytes)

    try:
        # odafc.readfile converte il DWG in DXF in un file temporaneo e lo legge
        doc = odafc.readfile(str(dwg_path))
    except odafc.ODAFCError as e:
        raise RuntimeError(
            "ODA File Converter non trovato o errore di conversione.\n\n"
            "Per abilitare il supporto DWG installa **ODA File Converter** (gratuito):\n"
            "https://www.opendesign.com/guestfiles/oda_file_converter\n\n"
            f"Dettaglio tecnico: {e}"
        )
    finally:
        try:
            dwg_path.unlink()
            tmpdir.rmdir()
        except OSError:
            pass

    return _extract_points_from_doc(doc)


# ---------------------------------------------------------------------------
# PLOT 3D
# ---------------------------------------------------------------------------
def build_scatter(df: pd.DataFrame, point_size: float, opacity: float) -> go.Figure:
    """Costruisce lo scatter plot 3D con coloring RGB o basato su Z."""
    has_rgb = {"R", "G", "B"}.issubset(df.columns)

    if has_rgb:
        colors = [f"rgb({r},{g},{b})" for r, g, b in zip(df["R"], df["G"], df["B"])]
        marker = dict(size=point_size, color=colors, opacity=opacity)
    else:
        marker = dict(
            size=point_size,
            color=df["Z"],
            colorscale="Viridis",
            opacity=opacity,
            colorbar=dict(title="Altezza (Z)", tickfont=dict(color="#FAFAFA")),
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["X"], y=df["Y"], z=df["Z"],
                mode="markers",
                marker=marker,
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            )
        ]
    )

    # Layout dark mode professionale
    axis_style = dict(
        backgroundcolor="#0E1117",
        gridcolor="#2A2F3A",
        showbackground=True,
        zerolinecolor="#3A4050",
        color="#FAFAFA",
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis_style, yaxis=axis_style, zaxis=axis_style,
            aspectmode="data",
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
    )
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🛰️ Point Cloud Viewer")
st.caption("Carica un file `.xyz`, `.dxf` o `.dwg` e visualizza la nuvola di punti 3D in tempo reale.")

# Sidebar: controlli visuali
with st.sidebar:
    st.header("⚙️ Controlli")
    point_size = st.slider("Dimensione punti", 0.5, 5.0, 1.5, 0.1)
    opacity = st.slider("Opacità", 0.1, 1.0, 0.8, 0.05)
    max_points = st.slider(
        "Downsample (max punti)",
        min_value=10_000, max_value=1_000_000,
        value=200_000, step=10_000,
        help="Per nuvole molto grandi, riduce i punti per mantenere la fluidità.",
    )
    st.markdown("---")
    st.markdown(
        "**Formati supportati**\n\n"
        "- `.xyz` / `.txt` / `.csv` — X Y Z [R G B], delimitatore auto\n"
        "- `.dxf` — entità POINT / LINE / 3DFACE / MESH / INSERT\n"
        "- `.dwg` — richiede **ODA File Converter** installato"
    )
    if not EZDXF_AVAILABLE:
        st.warning("`ezdxf` non installato: DXF/DWG disabilitati.")

# Area drag & drop
uploaded = st.file_uploader(
    "Trascina qui il tuo file (.xyz / .dxf / .dwg)",
    type=["xyz", "txt", "csv", "dxf", "dwg"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("👆 Carica un file per iniziare.")
    st.stop()

# Parsing: dispatch in base all'estensione
ext = Path(uploaded.name).suffix.lower()
try:
    with st.spinner(f"Parsing del file {ext}..."):
        if ext == ".dxf":
            df = load_dxf(uploaded.getvalue())
        elif ext == ".dwg":
            df = load_dwg(uploaded.getvalue())
        else:
            df = load_xyz(uploaded.getvalue())
except Exception as e:
    st.error(f"Errore nel parsing: {e}")
    st.stop()

# Downsampling (solo per la visualizzazione)
total_points = len(df)
if total_points > max_points:
    df_plot = df.sample(n=max_points, random_state=42).reset_index(drop=True)
    st.warning(f"Visualizzando {max_points:,} punti su {total_points:,} totali.")
else:
    df_plot = df

# Statistiche
has_rgb = {"R", "G", "B"}.issubset(df.columns)
dx = df["X"].max() - df["X"].min()
dy = df["Y"].max() - df["Y"].min()
dz = df["Z"].max() - df["Z"].min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Punti totali", f"{total_points:,}")
c2.metric("Area X×Y", f"{dx:.2f} × {dy:.2f}")
c3.metric("Altezza (ΔZ)", f"{dz:.2f}")
c4.metric("Colori RGB", "Sì" if has_rgb else "No (fallback su Z)")

# Plot
fig = build_scatter(df_plot, point_size=point_size, opacity=opacity)
st.plotly_chart(fig, use_container_width=True, theme=None)

# Anteprima dati
with st.expander("🔎 Anteprima dati"):
    st.dataframe(df.head(100), use_container_width=True)
