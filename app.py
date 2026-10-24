import sys
import streamlit as st

# Si se ejecuta directamente con 'python app.py', lanzar Streamlit automáticamente
if __name__ == "__main__" and (not hasattr(st, "runtime") or not st.runtime.exists()):
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO PROFESIONAL
# ==========================================
st.set_page_config(
    page_title="PetroPy Suite - Análisis Petrofísico",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS modernos
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        padding: 16px 20px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #cbd5e1;
        margin-top: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 6px 6px 0 0;
        padding: 0 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado principal
st.title("🛢️ PetroPy Suite™ - Plataforma de Análisis Petrofísico")
st.caption("Evaluación Cuantitativa de Formaciones, Registros de Pozos (Triple Combo), Saturación de Archie y Discriminación de Zonas Productoras (Net Pay).")

# ==========================================
# SIDEBAR - CONTROLES Y PARÁMETROS
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/oil-rig.png", width=70)
st.sidebar.title("Configuración Petrofísica")

# Selección de Fuente de Datos
st.sidebar.subheader("1. Origen de Datos")
data_source = st.sidebar.radio(
    "Seleccione entrada:",
    ["Simulador Geológico Multizona", "Cargar Archivo CSV propio"],
    index=0
)

uploaded_file = None
if data_source == "Cargar Archivo CSV propio":
    uploaded_file = st.sidebar.file_uploader("Subir CSV de registros", type=["csv"])
    st.sidebar.info("Columnas requeridas: DEPTH, GR, RES, RHOB, NPHI")

max_depth = 5000
depth_step = 0.5
if data_source == "Simulador Geológico Multizona":
    max_depth = st.sidebar.slider("Profundidad Total (ft)", 2000, 10000, 5000, step=250)
    depth_step = 0.5

# Parámetros Litológicos
st.sidebar.subheader("2. Parámetros de Arcillosidad (Vsh)")
vsh_method = st.sidebar.selectbox(
    "Modelo de Volumen de Arcilla:",
    ["Larionov (Rocas Terciarias)", "Larionov (Pre-Terciario)", "Stieber", "Lineal"],
    index=0
)
col_gr1, col_gr2 = st.sidebar.columns(2)
gr_clean = col_gr1.number_input("GR Limpio (API)", value=25.0, step=5.0)
gr_shale = col_gr2.number_input("GR Lutita (API)", value=125.0, step=5.0)

# Parámetros de Porosidad
st.sidebar.subheader("3. Matriz y Fluidos")
col_rho1, col_rho2 = st.sidebar.columns(2)
rho_matrix = col_rho1.selectbox("Matriz (g/cc)", [2.65, 2.71, 2.87], index=0, format_func=lambda x: f"{x} ({'Arenisca' if x==2.65 else 'Caliza' if x==2.71 else 'Dolomita'})")
rho_fluid = col_rho2.selectbox("Fluido (g/cc)", [1.0, 1.1], index=0, format_func=lambda x: f"{x} ({'Agua Dulce' if x==1.0 else 'Lodo Salado'})")

# Parámetros de Archie (Saturación de Agua)
st.sidebar.subheader("4. Parámetros de Archie (Sw)")
col_a1, col_a2 = st.sidebar.columns(2)
a_param = col_a1.number_input("Factor Tortuosidad (a)", value=1.0, step=0.1)
m_param = col_a2.number_input("Exp. Cementación (m)", value=2.0, step=0.1)
col_a3, col_a4 = st.sidebar.columns(2)
n_param = col_a3.number_input("Exp. Saturación (n)", value=2.0, step=0.1)
rw_param = col_a4.number_input("Resistividad Rw (Ω·m)", value=0.045, format="%.3f", step=0.005)

# Criterios de Corte (Cutoffs)
st.sidebar.subheader("5. Criterios de Corte (Cutoffs)")
vsh_cutoff = st.sidebar.slider("Corte Vsh Máximo (Vsh Cutoff)", 0.10, 0.60, 0.35, step=0.05)
phi_cutoff = st.sidebar.slider("Corte Porosidad Mínima (Phi Cutoff)", 0.04, 0.25, 0.10, step=0.01)
sw_cutoff = st.sidebar.slider("Corte Saturación Sw Máxima (Sw Cutoff)", 0.30, 0.80, 0.55, step=0.05)

# ==========================================
# GENERADOR / CARGADOR DE DATOS PETROFÍSICOS
# ==========================================
@st.cache_data
def generate_geological_well_logs(total_depth: int, step: float) -> pd.DataFrame:
    """Genera un pozo sintético realista con múltiples zonas geológicas bien calibradas."""
    np.random.seed(42)
    depths = np.arange(0, total_depth + step, step)
    n = len(depths)

    # Base geológica: caminata aleatoria suave
    trend = np.cumsum(np.random.randn(n))
    w = int(80 / step)
    base_geology = np.convolve(trend, np.ones(w)/w, mode='same')
    norm_base = (base_geology - base_geology.min()) / (base_geology.max() - base_geology.min() + 1e-6)

    # Configuración de capas geológicas típicas
    gr = 30 + norm_base * 90 + np.random.normal(0, 3, n)
    rhob = 2.45 + (gr / 150.0) * 0.22 + np.random.normal(0, 0.02, n)
    nphi = 0.18 + (gr / 150.0) * 0.22 + np.random.normal(0, 0.015, n)
    log_rt = 0.5 + (1.0 - (gr / 150.0)) * 1.5 + np.random.normal(0, 0.1, n)
    caliper = 8.5 + np.random.normal(0, 0.05, n)

    # Inserción de Zonas Reservorio Clave:
    # 1. Zona Gas (aprox 2150 - 2400 ft si la profundidad lo permite)
    gas_start, gas_end = 2150, 2400
    if total_depth >= gas_end:
        mask_gas = (depths >= gas_start) & (depths <= gas_end)
        gr[mask_gas] = 22 + np.random.normal(0, 3, mask_gas.sum())
        rhob[mask_gas] = 2.18 + np.random.normal(0, 0.025, mask_gas.sum()) # Baja densidad
        nphi[mask_gas] = 0.11 + np.random.normal(0, 0.015, mask_gas.sum()) # Fuerte crossover gas
        log_rt[mask_gas] = 2.2 + np.random.normal(0, 0.15, mask_gas.sum()) # Alta resistividad

    # 2. Zona Petróleo (aprox 2500 - 2850 ft)
    oil_start, oil_end = 2500, 2850
    if total_depth >= oil_end:
        mask_oil = (depths >= oil_start) & (depths <= oil_end)
        gr[mask_oil] = 28 + np.random.normal(0, 4, mask_oil.sum())
        rhob[mask_oil] = 2.28 + np.random.normal(0, 0.02, mask_oil.sum())
        nphi[mask_oil] = 0.24 + np.random.normal(0, 0.015, mask_oil.sum())
        log_rt[mask_oil] = 1.6 + np.random.normal(0, 0.12, mask_oil.sum())

    # 3. Zona Acuífero / Agua Salada (aprox 2950 - 3300 ft)
    wtr_start, wtr_end = 2950, 3300
    if total_depth >= wtr_end:
        mask_wtr = (depths >= wtr_start) & (depths <= wtr_end)
        gr[mask_wtr] = 30 + np.random.normal(0, 3, mask_wtr.sum())
        rhob[mask_wtr] = 2.32 + np.random.normal(0, 0.02, mask_wtr.sum())
        nphi[mask_wtr] = 0.25 + np.random.normal(0, 0.015, mask_wtr.sum())
        log_rt[mask_wtr] = 0.2 + np.random.normal(0, 0.08, mask_wtr.sum()) # Muy baja resistividad

    # Clip y conversión
    gr = np.clip(gr, 10, 160)
    rhob = np.clip(rhob, 1.95, 2.95)
    nphi = np.clip(nphi, 0.02, 0.50)
    rt = np.clip(10 ** log_rt, 0.2, 2000.0)
    rxo = np.clip(rt * np.random.uniform(0.7, 1.3, n), 0.2, 1000.0)

    return pd.DataFrame({
        "DEPTH": depths,
        "GR": gr,
        "RES": rt,
        "RXO": rxo,
        "RHOB": rhob,
        "NPHI": nphi,
        "CALI": caliper
    })

# Carga o simulación
if data_source == "Cargar Archivo CSV propio" and uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        # Validar columnas
        req_cols = ["DEPTH", "GR", "RES", "RHOB", "NPHI"]
        missing = [c for c in req_cols if c not in raw_df.columns]
        if missing:
            st.error(f"Faltan las siguientes columnas en el archivo subido: {missing}")
            st.stop()
        df = raw_df.copy()
        if "RXO" not in df.columns:
            df["RXO"] = df["RES"] * 0.9
        if "CALI" not in df.columns:
            df["CALI"] = 8.5
        max_depth = float(df["DEPTH"].max())
        depth_step = float(np.diff(df["DEPTH"]).mean())
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        st.stop()
else:
    df = generate_geological_well_logs(max_depth, depth_step)

# ==========================================
# CÁLCULOS PETROFÍSICOS AVANZADOS
# ==========================================
# 1. Índice de Gamma Ray (IGR)
igr = (df["GR"] - gr_clean) / (gr_shale - gr_clean + 1e-6)
igr = np.clip(igr, 0.0, 1.0)

# 2. Volumen de Arcilla (Vsh) según el modelo elegido
if vsh_method == "Larionov (Rocas Terciarias)":
    vsh = 0.083 * (2**(3.7 * igr) - 1.0)
elif vsh_method == "Larionov (Pre-Terciario)":
    vsh = 0.33 * (2**(2.0 * igr) - 1.0)
elif vsh_method == "Stieber":
    vsh = igr / (3.0 - 2.0 * igr + 1e-6)
else: # Lineal
    vsh = igr
vsh = np.clip(vsh, 0.0, 1.0)
df["VSH"] = vsh

# 3. Porosidad Densidad (PhiD)
phid = (rho_matrix - df["RHOB"]) / (rho_matrix - rho_fluid)
phid = np.clip(phid, 0.0, 0.50)
df["PHID"] = phid

# 4. Porosidad Combinada Neutrón-Densidad (PhiND) y Porosidad Efectiva (PhiE)
# Fórmula estándar de combinación cuadrática
phind = np.sqrt(np.clip((df["NPHI"]**2 + phid**2) / 2.0, 0.0, 1.0))
# Porosidad efectiva corregida por volumen de arcilla
phie = phind * (1.0 - df["VSH"])
phie = np.clip(phie, 0.005, 0.45)
df["PHIND"] = phind
df["PHIE"] = phie

# 5. Saturación de Agua de Archie (Sw)
# Sw = ((a * Rw) / (PhiE^m * Rt))^(1/n)
f_factor = a_param / (df["PHIE"]**m_param + 1e-6)
ro = f_factor * rw_param
sw_raw = (ro / (df["RES"] + 1e-6))**(1.0 / n_param)
sw = np.clip(sw_raw, 0.05, 1.0)
df["SW"] = sw
df["SH"] = 1.0 - sw # Saturación de hidrocarburo

# 6. Estimación de Permeabilidad de Timur (k en mD)
# K = 0.136 * (PhiE^4.4 / Swirr^2)
swirr = np.clip(sw, 0.12, 1.0)
perm_timur = 0.136 * (df["PHIE"]**4.4) / (swirr**2 + 1e-6) * 10000.0
perm_timur = np.clip(perm_timur, 0.01, 5000.0)
df["PERM"] = perm_timur

# 7. Discriminación de Zonas y Flags de Pay
df["FLAG_NET_SAND"] = df["VSH"] <= vsh_cutoff
df["FLAG_NET_RES"] = df["FLAG_NET_SAND"] & (df["PHIE"] >= phi_cutoff)
df["FLAG_NET_PAY"] = df["FLAG_NET_RES"] & (df["SW"] <= sw_cutoff)

# ==========================================
# KPIs Y EVALUACIÓN DE RESERVORIO
# ==========================================
gross_thickness = float(df["DEPTH"].max() - df["DEPTH"].min())
net_sand_ft = float(df["FLAG_NET_SAND"].sum() * depth_step)
net_res_ft = float(df["FLAG_NET_RES"].sum() * depth_step)
net_pay_ft = float(df["FLAG_NET_PAY"].sum() * depth_step)
ntg = (net_pay_ft / gross_thickness * 100) if gross_thickness > 0 else 0.0

# Promedios ponderados en la zona de Net Pay
if df["FLAG_NET_PAY"].any():
    pay_df = df[df["FLAG_NET_PAY"]]
    avg_phi_pay = float(pay_df["PHIE"].mean() * 100)
    avg_sw_pay = float(pay_df["SW"].mean() * 100)
    avg_sh_pay = 100.0 - avg_sw_pay
    avg_perm_pay = float(pay_df["PERM"].mean())
    kh = float((pay_df["PERM"] * depth_step).sum())
    phih = float((pay_df["PHIE"] * depth_step).sum())
else:
    avg_phi_pay = 0.0
    avg_sw_pay = 0.0
    avg_sh_pay = 0.0
    avg_perm_pay = 0.0
    kh = 0.0
    phih = 0.0

# Renderizado de Tarjetas Métricas
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Espesor Bruto</div>
        <div class="metric-value">{gross_thickness:,.0f} <span style="font-size:1rem;color:#94a3b8">ft</span></div>
        <div class="metric-sub">Profundidad Total</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Arena Neta (Net Sand)</div>
        <div class="metric-value">{net_sand_ft:,.1f} <span style="font-size:1rem;color:#94a3b8">ft</span></div>
        <div class="metric-sub">Vsh ≤ {vsh_cutoff*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Espesor Pay (Net Pay)</div>
        <div class="metric-value">{net_pay_ft:,.1f} <span style="font-size:1rem;color:#10b981">ft</span></div>
        <div class="metric-sub">NTG: {ntg:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Porosidad Pay (Ø)</div>
        <div class="metric-value">{avg_phi_pay:.1f} <span style="font-size:1rem;color:#38bdf8">%</span></div>
        <div class="metric-sub">Promedio en Pay Zone</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Saturación HC (Sh)</div>
        <div class="metric-value">{avg_sh_pay:.1f} <span style="font-size:1rem;color:#f59e0b">%</span></div>
        <div class="metric-sub">Sw prom: {avg_sw_pay:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# PESTAÑAS DE VISUALIZACIÓN Y ANÁLISIS
# ==========================================
tab_log, tab_crossplots, tab_summary, tab_data = st.tabs([
    "📊 Registro de Pozos (Composite Log)",
    "📈 Gráficos Cruzados (Crossplots)",
    "📑 Resumen de Reservorio & Zonas",
    "💾 Tabla de Datos y Descargas"
])

# ------------------------------------------
# TAB 1: REGISTRO DE POZOS COMPOSITE PLOT
# ------------------------------------------
with tab_log:
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        depth_min_plot = st.number_input("Profundidad Inicio (ft)", min_value=float(df["DEPTH"].min()), max_value=float(df["DEPTH"].max()-100), value=float(df["DEPTH"].min()), step=100.0)
        depth_max_plot = st.number_input("Profundidad Fin (ft)", min_value=float(df["DEPTH"].min()+100), max_value=float(df["DEPTH"].max()), value=float(df["DEPTH"].max()), step=100.0)
    
    with col_ctrl2:
        st.info("💡 **Guía de Interpretación:**\n"
                "- **Track 1:** Verde = Gamma Ray (API). Amarillo = Arenas limpias. Gris = Lutitas/Arcillas.\n"
                "- **Track 2:** Azul = Resistividad profunda en escala logarítmica (Ω·m). Picos altos indican hidrocarburos o formaciones compactas.\n"
                "- **Track 3:** Rojo = Porosidad Neutrón (v/v), Negro punteado = Densidad (g/cc). Sombreado amarillo = Efecto de cruce por gas.\n"
                "- **Track 4:** Cyan = Saturación de Agua ($S_w$). Verde = Saturación de Hidrocarburos ($S_h$).\n"
                "- **Track 5:** Barras de discriminación: Amarillo = Net Sand, Verde Esmeralda = **Net Pay** (Zona productora comercial).")

    plot_df = df[(df["DEPTH"] >= depth_min_plot) & (df["DEPTH"] <= depth_max_plot)]

    # Construcción de figura Matplotlib con 5 tracks
    fig = plt.figure(figsize=(15, 14))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1.2, 1.2, 1.2, 1.2, 0.6], wspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)
    ax4 = fig.add_subplot(gs[3], sharey=ax1)
    ax5 = fig.add_subplot(gs[4], sharey=ax1)

    # TRACK 1: Gamma Ray & Litología
    ax1.plot(plot_df["GR"], plot_df["DEPTH"], color="green", lw=1.0, label="GR")
    gr_cutoff_val = gr_clean + vsh_cutoff * (gr_shale - gr_clean)
    ax1.axvline(gr_cutoff_val, color="red", linestyle="--", lw=1.2, label=f"Cutoff ({gr_cutoff_val:.0f} API)")
    ax1.fill_betweenx(plot_df["DEPTH"], gr_cutoff_val, plot_df["GR"], where=plot_df["GR"] >= gr_cutoff_val, color="#8d8d8d", alpha=0.4, label="Lutita")
    ax1.fill_betweenx(plot_df["DEPTH"], plot_df["GR"], gr_cutoff_val, where=plot_df["GR"] < gr_cutoff_val, color="#fde047", alpha=0.5, label="Arena")
    ax1.set_xlim(0, 150)
    ax1.set_xlabel("Gamma Ray (API)", color="green", fontweight="bold")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.tick_params(axis="x", colors="green")
    ax1.set_ylabel("Profundidad Medida MD (ft)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8)

    # TRACK 2: Resistividad Logarítmica
    ax2.plot(plot_df["RES"], plot_df["DEPTH"], color="navy", lw=1.2, label="RT (Prof)")
    ax2.plot(plot_df["RXO"], plot_df["DEPTH"], color="skyblue", lw=0.9, linestyle="--", label="RXO (Som)")
    ax2.set_xscale("log")
    ax2.set_xlim(0.2, 2000)
    ax2.set_xlabel("Resistividad (Ω·m)", color="navy", fontweight="bold")
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.tick_params(axis="x", colors="navy")
    ax2.grid(True, which="major", linestyle="-", alpha=0.4)
    ax2.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax2.legend(loc="lower left", fontsize=8)

    # TRACK 3: Neutrón - Densidad & Crossover de Gas
    ax3_rhob = ax3.twiny()
    l1 = ax3.plot(plot_df["NPHI"], plot_df["DEPTH"], color="red", lw=1.1, label="NPHI (v/v)")
    l2 = ax3_rhob.plot(plot_df["RHOB"], plot_df["DEPTH"], color="black", lw=1.0, linestyle="--", label="RHOB (g/cc)")
    ax3.set_xlim(0.45, -0.15) # Escala estándar inversa para NPHI
    ax3_rhob.set_xlim(1.95, 2.95) # Escala compatible para RHOB
    ax3.set_xlabel("NPHI (v/v)", color="red", fontweight="bold")
    ax3_rhob.set_xlabel("RHOB (g/cc)", color="black", fontweight="bold")
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position("top")
    ax3_rhob.xaxis.tick_top()
    ax3_rhob.xaxis.set_label_position("top")
    ax3_rhob.spines['top'].set_position(('outward', 30))
    ax3.tick_params(axis="x", colors="red")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Sombreado de Crossover de Gas (cuando NPHI está a la derecha de RHOB en escalas compatibles)
    rhob_scaled = (plot_df["RHOB"] - 1.95) / (2.95 - 1.95)
    nphi_scaled = (0.45 - plot_df["NPHI"]) / (0.45 - (-0.15))
    ax3.fill_betweenx(plot_df["DEPTH"], plot_df["NPHI"], -0.15, where=(nphi_scaled > rhob_scaled), color="#facc15", alpha=0.5, label="Efecto Gas")

    # TRACK 4: Saturación de Fluidos (Sw y Sh)
    ax4.plot(plot_df["SW"] * 100, plot_df["DEPTH"], color="deepskyblue", lw=1.1, label="Sw (%)")
    ax4.axvline(sw_cutoff * 100, color="orange", linestyle="--", lw=1.0, label=f"Cutoff Sw ({sw_cutoff*100:.0f}%)")
    ax4.fill_betweenx(plot_df["DEPTH"], 0, plot_df["SW"] * 100, color="lightskyblue", alpha=0.4, label="Agua")
    ax4.fill_betweenx(plot_df["DEPTH"], plot_df["SW"] * 100, 100, color="#10b981", alpha=0.45, label="Hidrocarburo")
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("Saturación Sw (%)", color="deepskyblue", fontweight="bold")
    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position("top")
    ax4.tick_params(axis="x", colors="deepskyblue")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower left", fontsize=8)

    # TRACK 5: Pay Flags
    pay_flag_plot = np.where(plot_df["FLAG_NET_PAY"], 3, np.where(plot_df["FLAG_NET_RES"], 2, np.where(plot_df["FLAG_NET_SAND"], 1, 0)))
    ax5.fill_betweenx(plot_df["DEPTH"], 0, 1, where=(pay_flag_plot == 1), color="#fef08a", label="Net Sand")
    ax5.fill_betweenx(plot_df["DEPTH"], 0, 1, where=(pay_flag_plot == 2), color="#fb923c", label="Net Res")
    ax5.fill_betweenx(plot_df["DEPTH"], 0, 1, where=(pay_flag_plot == 3), color="#10b981", label="Net Pay")
    ax5.set_xlim(0, 1)
    ax5.set_xlabel("Pay Flags", color="darkgreen", fontweight="bold")
    ax5.xaxis.tick_top()
    ax5.xaxis.set_label_position("top")
    ax5.set_xticks([])
    ax5.legend(loc="lower center", fontsize=8)

    # Invertir eje Y compartido para profundidad
    ax1.set_ylim(depth_max_plot, depth_min_plot)

    # Ocultar etiquetas Y en los tracks 2, 3, 4, 5
    for ax in [ax2, ax3, ax4, ax5]:
        plt.setp(ax.get_yticklabels(), visible=False)

    st.pyplot(fig, use_container_width=True)

# ------------------------------------------
# TAB 2: GRÁFICOS CRUZADOS (CROSSPLOTS)
# ------------------------------------------
with tab_crossplots:
    col_cp1, col_cp2 = st.columns(2)

    with col_cp1:
        st.subheader("📈 Gráfico de Pickett (log Rt vs log Phi)")
        st.caption("Identificación de Rw, exponente de cementación m y líneas de saturación de agua.")

        fig_pickett, ax_p = plt.subplots(figsize=(6, 5))
        valid_mask = (df["PHIE"] > 0.02) & (df["RES"] > 0.1)
        scatter = ax_p.scatter(
            df.loc[valid_mask, "PHIE"],
            df.loc[valid_mask, "RES"],
            c=df.loc[valid_mask, "SW"] * 100,
            cmap="viridis_r",
            s=12,
            alpha=0.65
        )
        cbar = plt.colorbar(scatter, ax=ax_p)
        cbar.set_label("Saturación de Agua Sw (%)", fontsize=9)

        # Líneas teóricas de Sw (100%, 50%, 25%, 10%)
        phi_range = np.linspace(0.03, 0.40, 100)
        for sw_line, ls, col in [(1.0, "-", "blue"), (0.50, "--", "teal"), (0.25, "-.", "forestgreen"), (0.10, ":", "darkgreen")]:
            rt_line = (a_param * rw_param) / ((phi_range**m_param) * (sw_line**n_param))
            ax_p.plot(phi_range, rt_line, linestyle=ls, color=col, lw=1.2, label=f"Sw = {int(sw_line*100)}%")

        ax_p.set_xscale("log")
        ax_p.set_yscale("log")
        ax_p.set_xlim(0.02, 0.45)
        ax_p.set_ylim(0.2, 2000)
        ax_p.set_xlabel("Porosidad Efectiva Øe (v/v)", fontweight="bold")
        ax_p.set_ylabel("Resistividad Verdadera Rt (Ω·m)", fontweight="bold")
        ax_p.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_p.legend(fontsize=8, loc="lower left")
        st.pyplot(fig_pickett)

    with col_cp2:
        st.subheader("🔬 Crossplot Neutrón - Densidad (NPHI vs RHOB)")
        st.caption("Identificación litológica de matriz mineralógica y separación de fluidos ligeros (gas).")

        fig_nd, ax_nd = plt.subplots(figsize=(6, 5))
        scatter_nd = ax_nd.scatter(
            df["NPHI"],
            df["RHOB"],
            c=df["GR"],
            cmap="YlOrBr",
            s=12,
            alpha=0.65
        )
        cbar_nd = plt.colorbar(scatter_nd, ax=ax_nd)
        cbar_nd.set_label("Gamma Ray (API)", fontsize=9)

        # Líneas de referencia mineralógica ideales
        phi_points = np.array([0.0, 0.10, 0.20, 0.30, 0.40])
        # Arenisca (rho_ma = 2.65, fluid = 1.0)
        rho_ss = 2.65 - phi_points * (2.65 - 1.0)
        ax_nd.plot(phi_points, rho_ss, color="goldenrod", lw=1.5, marker="o", markersize=3, label="Arenisca (Sandstone)")
        # Caliza (rho_ma = 2.71)
        rho_ls = 2.71 - phi_points * (2.71 - 1.0)
        ax_nd.plot(phi_points, rho_ls, color="blue", lw=1.5, marker="s", markersize=3, label="Caliza (Limestone)")
        # Dolomita (rho_ma = 2.87)
        rho_dol = 2.87 - phi_points * (2.87 - 1.0)
        ax_nd.plot(phi_points, rho_dol, color="purple", lw=1.5, marker="^", markersize=3, label="Dolomita (Dolomite)")

        ax_nd.set_xlim(-0.05, 0.50)
        ax_nd.set_ylim(3.0, 1.9) # Eje invertido para RHOB
        ax_nd.set_xlabel("Neutrón NPHI (v/v)", fontweight="bold")
        ax_nd.set_ylabel("Densidad RHOB (g/cc)", fontweight="bold")
        ax_nd.grid(True, linestyle=":", alpha=0.5)
        ax_nd.legend(fontsize=8, loc="upper right")
        st.pyplot(fig_nd)

    # Buckles Plot
    st.subheader("📊 Gráfico de Buckles (Porosidad vs Saturación de Agua)")
    st.caption("Evaluación de agua irreducible (BVW = PhiE * Sw constante indica reservorio en Sw irreducible libre de agua).")
    fig_b, ax_b = plt.subplots(figsize=(8, 4))
    bvw = df["PHIE"] * df["SW"]
    scatter_b = ax_b.scatter(df["SW"] * 100, df["PHIE"] * 100, c=df["VSH"], cmap="coolwarm", s=10, alpha=0.6)
    cb_b = plt.colorbar(scatter_b, ax=ax_b)
    cb_b.set_label("Volumen de Arcilla Vsh", fontsize=9)
    ax_b.set_xlabel("Saturación de Agua Sw (%)", fontweight="bold")
    ax_b.set_ylabel("Porosidad Efectiva Øe (%)", fontweight="bold")
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 45)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig_b)

# ------------------------------------------
# TAB 3: RESUMEN DE RESERVORIO & ZONAS
# ------------------------------------------
with tab_summary:
    st.subheader("📋 Resumen Ejecutivo y Potencial de Reservorio")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("### Parámetros Estructurales y Espesores")
        summary_table = pd.DataFrame({
            "Parámetro": [
                "Profundidad Inicial",
                "Profundidad Final",
                "Espesor Bruto (Gross)",
                "Espesor Arena Neta (Net Sand)",
                "Espesor Reservorio Neto (Net Res)",
                "Espesor Productor Neto (Net Pay)",
                "Relación Net-to-Gross (NTG)"
            ],
            "Valor": [
                f"{df['DEPTH'].min():.1f} ft",
                f"{df['DEPTH'].max():.1f} ft",
                f"{gross_thickness:,.1f} ft",
                f"{net_sand_ft:,.1f} ft",
                f"{net_res_ft:,.1f} ft",
                f"{net_pay_ft:,.1f} ft",
                f"{ntg:.2f} %"
            ]
        })
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    with col_s2:
        st.write("### Propiedades Promedio en Zona de Pay")
        props_table = pd.DataFrame({
            "Propiedad": [
                "Porosidad Efectiva Promedio (Øe)",
                "Saturación de Agua Promedio (Sw)",
                "Saturación de Hidrocarburo (Sh)",
                "Permeabilidad Promedio Estimada (K)",
                "Capacidad de Almacenamiento (Ø·h)",
                "Capacidad de Flujo (K·h)"
            ],
            "Valor": [
                f"{avg_phi_pay:.2f} %",
                f"{avg_sw_pay:.2f} %",
                f"{avg_sh_pay:.2f} %",
                f"{avg_perm_pay:,.1f} mD",
                f"{phih:,.2f} ft",
                f"{kh:,.1f} mD·ft"
            ]
        })
        st.dataframe(props_table, use_container_width=True, hide_index=True)

    # Identificación de Zonas Continuas de Pay
    st.write("### Intervalos Continuos de Pay Identificados")
    pay_diff = np.diff(np.concatenate(([0], df["FLAG_NET_PAY"].astype(int), [0])))
    starts = np.where(pay_diff == 1)[0]
    ends = np.where(pay_diff == -1)[0]

    if len(starts) > 0:
        intervals = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            top_d = df["DEPTH"].iloc[s]
            bot_d = df["DEPTH"].iloc[min(e, len(df)-1)]
            thick = bot_d - top_d
            if thick >= 2.0: # Ignorar paquetes menores a 2 ft
                sub_df = df.iloc[s:e]
                intervals.append({
                    "Zona": f"Intervalo #{len(intervals)+1}",
                    "Tope (ft)": f"{top_d:.1f}",
                    "Base (ft)": f"{bot_d:.1f}",
                    "Espesor (ft)": f"{thick:.1f}",
                    "Øe Prom (%)": f"{sub_df['PHIE'].mean()*100:.1f}",
                    "Sw Prom (%)": f"{sub_df['SW'].mean()*100:.1f}",
                    "Sh Prom (%)": f"{(1.0 - sub_df['SW'].mean())*100:.1f}",
                    "K Prom (mD)": f"{sub_df['PERM'].mean():.1f}"
                })
        if intervals:
            st.dataframe(pd.DataFrame(intervals), use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron paquetes de pay continuos mayores a 2 ft con los cutoffs actuales.")
    else:
        st.warning("No se identificaron intervalos productores netos con los criterios de corte seleccionados.")

# ------------------------------------------
# TAB 4: TABLA DE DATOS Y DESCARGAS
# ------------------------------------------
with tab_data:
    st.subheader("💾 Vista de Curvas Calculadas y Exportación")
    st.caption("Inspeccione o descargue la tabla completa con todas las curvas petrofísicas calculadas.")

    export_df = df[[
        "DEPTH", "GR", "RES", "RXO", "RHOB", "NPHI", "VSH", "PHID", "PHIND", "PHIE", "SW", "SH", "PERM", "FLAG_NET_SAND", "FLAG_NET_RES", "FLAG_NET_PAY"
    ]].copy()

    # Formatear columnas decimales para vista previa clara
    st.dataframe(export_df.head(100), use_container_width=True)

    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Base de Datos Completa (CSV)",
        data=csv_data,
        file_name="analisis_petrofisico_resultados.csv",
        mime="text/csv"
    )
