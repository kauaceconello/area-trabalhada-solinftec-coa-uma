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

import zipfile
import tempfile
import os
import pytz
from datetime import datetime


# =========================
# CONFIG STREAMLIT
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

TEMPO_MAX_SEG = 60

TIPO_MAPA = st.sidebar.selectbox(
    "Tipo de mapa",
    ["Área trabalhada", "Velocidade", "RPM", "Heatmap"]
)

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

VEL_MIN = st.sidebar.number_input("Velocidade mínima (termômetro)", value=0.0)
VEL_MAX = st.sidebar.number_input("Velocidade máxima (termômetro)", value=20.0)

RPM_MIN = st.sidebar.number_input("RPM mínimo (termômetro)", value=0.0)
RPM_MAX = st.sidebar.number_input("RPM máximo (termômetro)", value=2500.0)

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
# FUNÇÃO TERMÔMETRO HORIZONTAL
# =========================
def plot_termometro(ax, valor, vmin, vmax, titulo):
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(0, 1)

    ax.barh(0.5, vmax - vmin, left=vmin, color="#e0e0e0")
    ax.barh(0.5, valor - vmin, left=vmin, color="#4c78a8")

    ax.set_yticks([])
    ax.set_title(titulo, fontsize=10)
    ax.axvline(valor, color="black", linewidth=2)


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

            csv_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".csv")]
            if not csv_files:
                st.error(f"❌ Nenhum CSV encontrado no ZIP {uploaded_zip.name}")
                continue

            csv_path = os.path.join(tmpdir, csv_files[0])
            df_temp = pd.read_csv(csv_path, sep=";", encoding="latin1", engine="python")

            dfs.append(df_temp)

        if not dfs:
            st.error("❌ Nenhum dado válido encontrado nos ZIPs.")
            st.stop()

        df = pd.concat(dfs, ignore_index=True)

        # =========================
        # TRATAMENTO
        # =========================
        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")
        df["vl_largura_implemento"] = pd.to_numeric(df["vl_largura_implemento"], errors="coerce")

        if "vl_velocidade" in df.columns:
            df["vl_velocidade"] = pd.to_numeric(df["vl_velocidade"], errors="coerce")
        else:
            df["vl_velocidade"] = np.nan

        if "vl_rpm" in df.columns:
            df["vl_rpm"] = pd.to_numeric(df["vl_rpm"], errors="coerce")
        else:
            df["vl_rpm"] = np.nan

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

            if df_faz.empty or base_fazenda.empty:
                continue

            nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]

            gdf_pts = gpd.GeoDataFrame(
                df_faz,
                geometry=gpd.points_from_xy(df_faz["vl_longitude_inicial"], df_faz["vl_latitude_inicial"]),
                crs="EPSG:4326"
            )

            base_fazenda = base_fazenda.to_crs(epsg=31983)
            gdf_pts = gdf_pts.to_crs(epsg=31983)

            geom_fazenda = unary_union(base_fazenda.geometry)

            linhas = []
            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha_atual = []
                ultimo_tempo = None

                for _, row in grupo.iterrows():
                    if ultimo_tempo is None:
                        linha_atual = [row.geometry]
                    else:
                        delta = (row["dt_hr_local_inicial"] - ultimo_tempo).total_seconds()
                        if delta <= TEMPO_MAX_SEG:
                            linha_atual.append(row.geometry)
                        else:
                            if len(linha_atual) >= 2:
                                linhas.append(LineString(linha_atual))
                            linha_atual = [row.geometry]
                    ultimo_tempo = row["dt_hr_local_inicial"]

                if len(linha_atual) >= 2:
                    linhas.append(LineString(linha_atual))

            if not linhas:
                continue

            gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_fazenda.crs)

            largura_media = df_faz["vl_largura_implemento"].dropna().mean()
            largura_final = largura_media * MULTIPLICADOR_BUFFER

            buffer_linhas = gdf_linhas.buffer(largura_final / 2)
            area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
            area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

            area_total_ha = round(base_fazenda.geometry.area.sum() / 10000, 2)
            area_trab_ha = round(area_trabalhada.area / 10000, 2)

            if area_trab_ha < AREA_MIN_HA:
                continue

            pct_trab = round(area_trab_ha / area_total_ha * 100, 1)

            dt_min = df_faz["dt_hr_local_inicial"].min()
            dt_max = df_faz["dt_hr_local_inicial"].max()

            # =========================
            # MÉTRICAS NOVAS
            # =========================
            vel_min = df_faz["vl_velocidade"].min()
            vel_max = df_faz["vl_velocidade"].max()
            vel_med = df_faz["vl_velocidade"].mean()

            rpm_min = df_faz["vl_rpm"].min()
            rpm_max = df_faz["vl_rpm"].max()
            rpm_med = df_faz["vl_rpm"].mean()

            periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M")
            periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M")

            brasilia = pytz.timezone("America/Sao_Paulo")
            hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

            # =========================
            # FIGURA
            # =========================
            with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):

                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                plt.subplots_adjust(left=0.15, right=0.85, bottom=0.25, top=0.88)

                base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB, edgecolor="black", linewidth=1.2)
                gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(ax=ax, color=COR_TRABALHADA, alpha=0.9)
                base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.2)

                ax.axis("off")

                pos = ax.get_position()
                centro_mapa = (pos.x0 + pos.x1) / 2
                base_y = pos.y0

                ax.legend(
                    handles=[
                        mpatches.Patch(color=COR_TRABALHADA, label="Área trabalhada"),
                        mpatches.Patch(color=COR_NAO_TRAB, label="Área não trabalhada"),
                    ],
                    loc="lower center",
                    bbox_to_anchor=(centro_mapa, base_y - 0.35),
                    ncol=2
                )

                # TERMÔMETRO
                if TIPO_MAPA in ["Velocidade", "RPM"]:
                    ax2 = fig.add_axes([0.25, 0.12, 0.5, 0.05])

                    if TIPO_MAPA == "Velocidade":
                        plot_termometro(ax2, vel_med, VEL_MIN, VEL_MAX, "Velocidade média")
                    else:
                        plot_termometro(ax2, rpm_med, RPM_MIN, RPM_MAX, "RPM médio")

                # RESUMO
                fig.text(
                    pos.x1 + 0.02,
                    0.50,
                    f"Fazenda: {FAZENDA_ID} – {nome_fazenda}\n\n"
                    f"Área total: {area_total_ha} ha\n"
                    f"Trabalhada: {area_trab_ha} ha ({pct_trab}%)\n\n"
                    f"Período: {periodo_ini} até {periodo_fim}\n\n"
                    f"Vel: {vel_min:.1f}/{vel_max:.1f}/{vel_med:.1f}\n"
                    f"RPM: {rpm_min:.0f}/{rpm_max:.0f}/{rpm_med:.0f}"
                )

                # RODAPÉ (MANTIDO)
                fig.text(centro_mapa, base_y - 0.11,
                         "⚠️ Os resultados dependem da qualidade dos dados.",
                         ha="center", fontsize=10, color=COR_RODAPE)

                fig.text(centro_mapa, base_y - 0.14,
                         "Relatório baseado em Solinftec.",
                         ha="center", fontsize=10, color=COR_RODAPE)

                fig.text(centro_mapa, base_y - 0.17,
                         f"Desenvolvido por Kauã Ceconello • Gerado em {hora}",
                         ha="center", fontsize=10, color=COR_RODAPE)

                st.pyplot(fig)

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
