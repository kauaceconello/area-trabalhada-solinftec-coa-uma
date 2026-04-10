# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC)
# Desenvolvido por Kauã Ceconello

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.geometry import LineString
from shapely.ops import unary_union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib as mpl

import zipfile
import tempfile
import os
import pytz
from datetime import datetime


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Área Trabalhada – Solinftec",
    layout="wide"
)

st.title("📍 Área Trabalhada – Solinftec")

st.markdown(
    "Aplicação para cálculo e visualização da **área trabalhada** com base em "
    "dados operacionais da **Solinftec** e base cartográfica da Usina Monte Alegre."
)

st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%;
        height: 3.2em;
        font-size: 1.2em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# UPLOAD
# =========================
uploaded_zips = st.file_uploader(
    "📦 Upload dos ZIPs contendo o CSV da Solinftec",
    type=["zip"],
    accept_multiple_files=True
)

uploaded_gpkg = st.file_uploader(
    "🗺️ Upload da base cartográfica (GPKG)",
    type=["gpkg"]
)

GERAR = st.button("▶️ Gerar mapa")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Parâmetros")

TIPOS_MAPA = st.sidebar.multiselect(
    "Selecione os mapas",
    ["Área trabalhada", "Velocidade", "RPM", "Heatmap"],
    default=["Área trabalhada"]
)

TEMPO_MAX_SEG = 60

MULTIPLICADOR_BUFFER = st.sidebar.number_input(
    "Tamanho do Buffer",
    min_value=1.0,
    max_value=10.0,
    value=2.5,
    step=0.1
)

AREA_MIN_HA = st.sidebar.number_input(
    "Área mínima trabalhada (ha)",
    min_value=0.0,
    value=0.50,
    step=0.1
)

# aparece SOMENTE se necessário
if "Velocidade" in TIPOS_MAPA:
    VEL_MIN = st.sidebar.number_input("Velocidade mínima (termômetro)", value=0.0)
    VEL_MAX = st.sidebar.number_input("Velocidade máxima (termômetro)", value=20.0)

if "RPM" in TIPOS_MAPA:
    RPM_MIN = st.sidebar.number_input("RPM mínimo (termômetro)", value=0.0)
    RPM_MAX = st.sidebar.number_input("RPM máxima (termômetro)", value=2500.0)

MOSTRAR_TALHOES = st.sidebar.checkbox(
    "📊 Mostrar área por Gleba / Talhão",
    value=False
)


# =========================
# CORES
# =========================
COR_TRABALHADA = "#62b27f"
COR_NAO_TRAB = "#f6b1b3"
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9


# =========================
# TERMÔMETRO DINÂMICO
# =========================
def plot_termometro(ax, valor, vmin, vmax, titulo):
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(0, 1)

    cmap = mpl.cm.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    ax.barh(0.5, vmax - vmin, left=vmin, color="#e0e0e0")
    ax.barh(0.5, valor - vmin, left=vmin, color=cmap(norm(valor)))

    ax.axvline(valor, color="black", linewidth=2)
    ax.set_yticks([])
    ax.set_title(titulo, fontsize=10)


# =========================
# LINE HEAT (VELOCIDADE / RPM)
# =========================
def add_colored_line(ax, gdf, value_col, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)

    for _, grupo in gdf.groupby("cd_equipamento"):
        grupo = grupo.sort_values("dt_hr_local_inicial")

        if len(grupo) < 2:
            continue

        coords = list(zip(grupo.geometry.x, grupo.geometry.y))
        values = grupo[value_col].fillna(method="ffill").values

        segments = []
        seg_colors = []

        for i in range(len(coords) - 1):
            segments.append([coords[i], coords[i + 1]])
            seg_colors.append(values[i])

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(np.nanmin(values), np.nanmax(values)),
            linewidth=2.5
        )
        lc.set_array(np.array(seg_colors))
        ax.add_collection(lc)


# =========================
# PROCESSAMENTO
# =========================
if uploaded_zips and uploaded_gpkg and GERAR:

    with tempfile.TemporaryDirectory() as tmpdir:

        dfs = []

        for uploaded_zip in uploaded_zips:
            zip_path = os.path.join(tmpdir, uploaded_zip.name)

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
            if not csv_files:
                continue

            csv_path = os.path.join(tmpdir, csv_files[0])
            dfs.append(pd.read_csv(csv_path, sep=";", encoding="latin1", engine="python"))

        df = pd.concat(dfs, ignore_index=True)

        # =========================
        # TRATAMENTO
        # =========================
        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")

        df["vl_velocidade"] = pd.to_numeric(df.get("vl_velocidade", np.nan), errors="coerce")
        df["vl_rpm"] = pd.to_numeric(df.get("vl_rpm", np.nan), errors="coerce")

        df = df[(df["cd_estado"] == "E") & (df["cd_operacao_parada"] == -1)].copy()
        df["cd_fazenda"] = df["cd_fazenda"].astype(str)

        # =========================
        # GPKG
        # =========================
        gpkg_path = os.path.join(tmpdir, "base.gpkg")
        with open(gpkg_path, "wb") as f:
            f.write(uploaded_gpkg.read())

        base = gpd.read_file(gpkg_path)
        base["FAZENDA"] = base["FAZENDA"].astype(str)

        # =========================
        # LOOP
        # =========================
        for FAZENDA_ID in df["cd_fazenda"].dropna().unique():

            df_faz = df[df["cd_fazenda"] == FAZENDA_ID].copy()
            base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

            if df_faz.empty:
                continue

            nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]

            gdf_pts = gpd.GeoDataFrame(
                df_faz,
                geometry=gpd.points_from_xy(df_faz["vl_longitude_inicial"], df_faz["vl_latitude_inicial"]),
                crs="EPSG:4326"
            )

            base_fazenda = base_fazenda.to_crs(31983)
            gdf_pts = gdf_pts.to_crs(31983)

            geom_fazenda = unary_union(base_fazenda.geometry)

            # =========================
            # LINHAS (ÁREA)
            # =========================
            linhas = []
            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha = []
                last = None

                for _, row in grupo.iterrows():
                    if last is None:
                        linha = [row.geometry]
                    else:
                        if (row["dt_hr_local_inicial"] - last).total_seconds() <= TEMPO_MAX_SEG:
                            linha.append(row.geometry)
                        else:
                            if len(linha) >= 2:
                                linhas.append(LineString(linha))
                            linha = [row.geometry]
                    last = row["dt_hr_local_inicial"]

                if len(linha) >= 2:
                    linhas.append(LineString(linha))

            gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_fazenda.crs)

            largura = df_faz["vl_largura_implemento"].mean()
            buffer = gdf_linhas.buffer(largura * MULTIPLICADOR_BUFFER / 2)

            area_trab = unary_union(buffer).intersection(geom_fazenda)

            area_total = base_fazenda.area.sum() / 10000
            area_trab_ha = area_trab.area / 10000

            if area_trab_ha < AREA_MIN_HA:
                continue

            # =========================
            # MÉTRICAS
            # =========================
            vel_min = df_faz["vl_velocidade"].min()
            vel_max = df_faz["vl_velocidade"].max()
            vel_med = df_faz["vl_velocidade"].mean()

            rpm_min = df_faz["vl_rpm"].min()
            rpm_max = df_faz["vl_rpm"].max()
            rpm_med = df_faz["vl_rpm"].mean()

            dt_min = df_faz["dt_hr_local_inicial"].min()
            dt_max = df_faz["dt_hr_local_inicial"].max()

            hora = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M")

            # =========================
            # PLOT
            # =========================
            with st.expander(f"🗺️ {nome_fazenda}", expanded=False):

                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                plt.subplots_adjust(bottom=0.25)

                # ================= AREA =================
                if "Área trabalhada" in TIPOS_MAPA:

                    base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB)
                    gpd.GeoSeries(area_trab, crs=base_fazenda.crs).plot(ax=ax, color=COR_TRABALHADA)

                # ================= VELOCIDADE =================
                if "Velocidade" in TIPOS_MAPA:
                    add_colored_line(ax, gdf_pts, "vl_velocidade")

                    ax2 = fig.add_axes([0.25, 0.10, 0.5, 0.04])
                    plot_termometro(ax2, vel_med, VEL_MIN, VEL_MAX, "Velocidade")

                # ================= RPM =================
                if "RPM" in TIPOS_MAPA:
                    add_colored_line(ax, gdf_pts, "vl_rpm")

                    ax2 = fig.add_axes([0.25, 0.06, 0.5, 0.04])
                    plot_termometro(ax2, rpm_med, RPM_MIN, RPM_MAX, "RPM")

                # ================= HEATMAP =================
                if "Heatmap" in TIPOS_MAPA:
                    ax.hexbin(
                        gdf_pts.geometry.x,
                        gdf_pts.geometry.y,
                        gridsize=50,
                        cmap="inferno",
                        alpha=0.7
                    )

                ax.axis("off")

                # ================= LEGEND =================
                ax.legend(handles=[
                    mpatches.Patch(color=COR_TRABALHADA, label="Trabalhado"),
                    mpatches.Patch(color=COR_NAO_TRAB, label="Não trabalhado"),
                ])

                # ================= RESUMO =================
                fig.text(
                    0.82, 0.5,
                    f"Fazenda: {FAZENDA_ID}\n"
                    f"Área: {area_trab_ha:.2f} ha\n"
                    f"Vel: {vel_min:.1f}/{vel_med:.1f}/{vel_max:.1f}\n"
                    f"RPM: {rpm_min:.0f}/{rpm_med:.0f}/{rpm_max:.0f}"
                )

                # ================= RODAPÉ =================
                fig.text(0.5, 0.05,
                         "Solinftec • Desenvolvido por Kauã Ceconello",
                         ha="center")

                st.pyplot(fig)

else:
    st.info("Envie os arquivos e clique em gerar.")
