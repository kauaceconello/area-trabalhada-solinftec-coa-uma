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
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import zipfile
import tempfile
import os
import pytz
from datetime import datetime


# CONFIG STREAMLIT
st.set_page_config(
    page_title="Área Trabalhada – Solinftec",
    layout="wide"
)

st.title("📍 Área Trabalhada – Solinftec")

st.markdown(
    "Aplicação para cálculo e visualização da **área trabalhada** com base em "
    "dados operacionais da **Solinftec** e base cartográfica da Usina Monte Alegre."
)

# UPLOAD
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
# SIDEBAR CONTROLES
# =========================
st.sidebar.header("⚙️ Parâmetros")

TEMPO_MAX_SEG = 60

MULTIPLICADOR_BUFFER = st.sidebar.number_input(
    "Tamanho do Buffer",
    1.0, 10.0, 2.5, 0.1
)

AREA_MIN_HA = st.sidebar.number_input(
    "Área mínima trabalhada (ha)",
    0.0, 50.0, 0.5, 0.1
)

MOSTRAR_TALHOES = st.sidebar.checkbox(
    "📊 Mostrar área por Gleba / Talhão",
    False
)

TIPO_MAPA = st.sidebar.multiselect(
    "🗺️ Tipo de mapa",
    ["Área Trabalhada", "Velocidade", "RPM"],
    default=["Área Trabalhada"]
)

# caixas separadas
if "Velocidade" in TIPO_MAPA:
    st.sidebar.subheader("🚜 Velocidade (km/h)")
    VEL_MIN = st.sidebar.number_input("Velocidade mínima", 0.0, 50.0, 0.0)
    VEL_MAX = st.sidebar.number_input("Velocidade máxima", 0.0, 50.0, 20.0)

if "RPM" in TIPO_MAPA:
    st.sidebar.subheader("⚙️ RPM")
    RPM_MIN = st.sidebar.number_input("RPM mínimo", 0.0, 5000.0, 800.0)
    RPM_MAX = st.sidebar.number_input("RPM máximo", 0.0, 5000.0, 2500.0)


COR_TRABALHADA = "#62b27f"
COR_NAO_TRAB = "#f6b1b3"
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9


# =========================
# FUNÇÃO HEATMAP LINHA
# =========================
def add_colored_line(ax, gdf, value_col, cmap="viridis", vmin=None, vmax=None):

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap)

    for _, grupo in gdf.groupby("cd_equipamento"):
        grupo = grupo.sort_values("dt_hr_local_inicial")

        coords = list(grupo.geometry)
        values = grupo[value_col].fillna(method="ffill").values

        if len(coords) < 2:
            continue

        for i in range(len(coords) - 1):
            seg = [coords[i], coords[i + 1]]
            val = values[i]

            ax.plot(
                [seg[0].x, seg[1].x],
                [seg[0].y, seg[1].y],
                color=cmap(norm(val)),
                linewidth=2
            )


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

            df_temp = pd.read_csv(os.path.join(tmpdir, csv_files[0]),
                                   sep=";", encoding="latin1", engine="python")

            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)

        # tratamento
        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")
        df["vl_largura_implemento"] = pd.to_numeric(df["vl_largura_implemento"], errors="coerce")

        df = df[(df["cd_estado"] == "E") & (df["cd_operacao_parada"] == -1)].copy()
        df["cd_fazenda"] = df["cd_fazenda"].astype(str)

        gpkg_path = os.path.join(tmpdir, "base.gpkg")
        with open(gpkg_path, "wb") as f:
            f.write(uploaded_gpkg.read())

        base = gpd.read_file(gpkg_path)
        base["FAZENDA"] = base["FAZENDA"].astype(str)

        if "TALHAO" in base.columns:
            base["TALHAO"] = base["TALHAO"].astype(str)

        if "GLEBA" in base.columns:
            base["GLEBA"] = base["GLEBA"].astype(str)

        # =========================
        # LOOP
        # =========================
        for FAZENDA_ID in df["cd_fazenda"].dropna().unique():

            df_faz = df[df["cd_fazenda"] == FAZENDA_ID].copy()
            base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

            if df_faz.empty:
                continue

            gdf_pts = gpd.GeoDataFrame(
                df_faz,
                geometry=gpd.points_from_xy(df_faz["vl_longitude_inicial"],
                                            df_faz["vl_latitude_inicial"]),
                crs="EPSG:4326"
            )

            base_fazenda = base_fazenda.to_crs(31983)
            gdf_pts = gdf_pts.to_crs(31983)

            geom_fazenda = unary_union(base_fazenda.geometry)

            linhas = []
            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")
                pts = list(grupo.geometry)

                if len(pts) > 1:
                    linhas.append(LineString(pts))

            gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_fazenda.crs)

            largura = df_faz["vl_largura_implemento"].mean()
            buffer = gdf_linhas.buffer(largura * MULTIPLICADOR_BUFFER / 2)

            area_trabalhada = unary_union(buffer).intersection(geom_fazenda)

            area_total = round(base_fazenda.geometry.area.sum() / 10000, 2)
            area_trab = round(area_trabalhada.area / 10000, 2)

            if area_trab < AREA_MIN_HA:
                continue

            dt_min = df_faz["dt_hr_local_inicial"].min()
            dt_max = df_faz["dt_hr_local_inicial"].max()

            periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M")
            periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M")

            # =========================
            # RESUMO MÉTRICAS
            # =========================
            vel_min = df_faz.get("vl_velocidade", pd.Series()).min()
            vel_max = df_faz.get("vl_velocidade", pd.Series()).max()
            vel_med = df_faz.get("vl_velocidade", pd.Series()).mean()

            rpm_min = df_faz.get("vl_rpm", pd.Series()).min()
            rpm_max = df_faz.get("vl_rpm", pd.Series()).max()
            rpm_med = df_faz.get("vl_rpm", pd.Series()).mean()

            # =========================
            # PLOT
            # =========================
            with st.expander(f"Mapa {FAZENDA_ID}", expanded=False):

                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

                base_fazenda.plot(ax=ax, color=COR_NAO_TRAB, edgecolor="black")

                if "Área Trabalhada" in TIPO_MAPA:
                    gpd.GeoSeries(area_trabalhada).plot(ax=ax, color=COR_TRABALHADA)

                if "Velocidade" in TIPO_MAPA:
                    add_colored_line(ax, gdf_pts, "vl_velocidade",
                                     vmin=VEL_MIN, vmax=VEL_MAX)

                if "RPM" in TIPO_MAPA:
                    add_colored_line(ax, gdf_pts, "vl_rpm",
                                     vmin=RPM_MIN, vmax=RPM_MAX)

                ax.axis("off")

                pos = ax.get_position()

                fig.text(
                    pos.x1 + 0.02, 0.5,
                    f"Fazenda {FAZENDA_ID}\n"
                    f"Área: {area_trab} ha\n"
                    f"Vel min/max/méd: {vel_min:.1f}/{vel_max:.1f}/{vel_med:.1f}\n"
                    f"RPM min/max/méd: {rpm_min:.0f}/{rpm_max:.0f}/{rpm_med:.0f}\n"
                    f"{periodo_ini} até {periodo_fim}",
                    bbox=dict(boxstyle="round", facecolor=COR_CAIXA)
                )

                brasilia = pytz.timezone("America/Sao_Paulo")
                hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

                fig.text(0.5, 0.02,
                         f"Gerado em {hora} • Solinftec",
                         ha="center", color=COR_RODAPE)

                st.pyplot(fig)

else:
    st.info("Envie arquivos e gere o mapa.")
