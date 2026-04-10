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

st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 3.2em;
    font-size: 1.2em;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

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

# SIDEBAR
st.sidebar.header("⚙️ Parâmetros")

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

MOSTRAR_TALHOES = st.sidebar.checkbox(
    "📊 Mostrar área por Gleba / Talhão",
    value=False
)

# 🔥 NOVO: seleção de mapas
TIPO_MAPA = st.sidebar.multiselect(
    "🗺️ Tipo de mapa",
    ["Área trabalhada", "Velocidade", "RPM"],
    default=["Área trabalhada"]
)

COR_TRABALHADA = "#62b27f"
COR_NAO_TRAB = "#f6b1b3"
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9


# PROCESSAMENTO
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
            st.stop()

        df = pd.concat(dfs, ignore_index=True)

        # tratamento base
        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")
        df["vl_largura_implemento"] = pd.to_numeric(df["vl_largura_implemento"], errors="coerce")

        # 🔥 NOVO CAMPOS
        if "vl_velocidade" in df.columns:
            df["vl_velocidade"] = pd.to_numeric(df["vl_velocidade"], errors="coerce")
        else:
            df["vl_velocidade"] = np.nan

        if "vl_rpm" in df.columns:
            df["vl_rpm"] = pd.to_numeric(df["vl_rpm"], errors="coerce")
        else:
            df["vl_rpm"] = np.nan

        df = df[
            (df["cd_estado"] == "E") &
            (df["cd_operacao_parada"] == -1)
        ].copy()

        df["cd_fazenda"] = df["cd_fazenda"].astype(str)

        # GPKG
        gpkg_path = os.path.join(tmpdir, "base.gpkg")
        with open(gpkg_path, "wb") as f:
            f.write(uploaded_gpkg.read())

        base = gpd.read_file(gpkg_path)
        base["FAZENDA"] = base["FAZENDA"].astype(str)

        if "TALHAO" in base.columns:
            base["TALHAO"] = base["TALHAO"].astype(str)
        if "GLEBA" in base.columns:
            base["GLEBA"] = base["GLEBA"].astype(str)

        # LOOP
        for FAZENDA_ID in df["cd_fazenda"].dropna().unique():

            df_faz = df[df["cd_fazenda"] == FAZENDA_ID].copy()
            base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

            if df_faz.empty or base_fazenda.empty:
                continue

            nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]

            gdf_pts = gpd.GeoDataFrame(
                df_faz,
                geometry=gpd.points_from_xy(
                    df_faz["vl_longitude_inicial"],
                    df_faz["vl_latitude_inicial"]
                ),
                crs="EPSG:4326"
            )

            base_fazenda = base_fazenda.to_crs(31983)
            gdf_pts = gdf_pts.to_crs(31983)

            geom_fazenda = unary_union(base_fazenda.geometry)

            linhas = []
            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha_atual = []
                ultimo = None

                for _, row in grupo.iterrows():
                    if ultimo is None:
                        linha_atual = [row.geometry]
                    else:
                        delta = (row["dt_hr_local_inicial"] - ultimo).total_seconds()
                        if delta <= TEMPO_MAX_SEG:
                            linha_atual.append(row.geometry)
                        else:
                            if len(linha_atual) >= 2:
                                linhas.append(LineString(linha_atual))
                            linha_atual = [row.geometry]
                    ultimo = row["dt_hr_local_inicial"]

                if len(linha_atual) >= 2:
                    linhas.append(LineString(linha_atual))

            if not linhas:
                continue

            gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_fazenda.crs)

            largura_media = df_faz["vl_largura_implemento"].mean()
            largura_final = largura_media * MULTIPLICADOR_BUFFER

            buffer_linhas = gdf_linhas.buffer(largura_final / 2)

            area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
            area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

            area_total_ha = base_fazenda.geometry.area.sum() / 10000
            area_trab_ha = area_trabalhada.area / 10000
            area_nao_ha = area_nao_trabalhada.area / 10000

            if area_trab_ha < AREA_MIN_HA:
                continue

            dt_min = df_faz["dt_hr_local_inicial"].min()
            dt_max = df_faz["dt_hr_local_inicial"].max()

            # stats velocidade / rpm
            vel_min = df_faz["vl_velocidade"].min()
            vel_max = df_faz["vl_velocidade"].max()
            vel_med = df_faz["vl_velocidade"].mean()

            rpm_min = df_faz["vl_rpm"].min()
            rpm_max = df_faz["vl_rpm"].max()
            rpm_med = df_faz["vl_rpm"].mean()

            def rodape(fig, centro_mapa, base_y):
                brasilia = pytz.timezone("America/Sao_Paulo")
                hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

                fig.text(centro_mapa, base_y - 0.11,
                         "⚠️ Dados sujeitos à qualidade operacional.",
                         ha="center", fontsize=10, color=COR_RODAPE)

                fig.text(centro_mapa, base_y - 0.14,
                         "Relatório baseado em Solinftec.",
                         ha="center", fontsize=10, color=COR_RODAPE)

                fig.text(centro_mapa, base_y - 0.17,
                         f"Desenvolvido por Kauã Ceconello • {hora}",
                         ha="center", fontsize=10, color=COR_RODAPE)

            # =======================
            # MAPA ÁREA TRABALHADA
            # =======================
            if "Área trabalhada" in TIPO_MAPA:

                with st.expander(f"🗺️ Área – {nome_fazenda}"):

                    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.25)

                    base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB)
                    gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(ax=ax, color=COR_TRABALHADA)

                    base_fazenda.boundary.plot(ax=ax, color="black")

                    ax.axis("off")

                    pos = ax.get_position()
                    centro = (pos.x0 + pos.x1) / 2
                    base_y = pos.y0

                    ax.legend(handles=[
                        mpatches.Patch(color=COR_TRABALHADA, label="Trabalhada"),
                        mpatches.Patch(color=COR_NAO_TRAB, label="Não trabalhada")
                    ], loc="lower center")

                    fig.text(pos.x1, 0.5,
                             f"Fazenda {FAZENDA_ID}\nÁrea: {area_trab_ha:.2f} ha",
                             bbox=dict(facecolor=COR_CAIXA))

                    rodape(fig, centro, base_y)

                    st.pyplot(fig)

            # =======================
            # HEATMAP VELOCIDADE
            # =======================
            if "Velocidade" in TIPO_MAPA:

                with st.expander(f"🔥 Velocidade – {nome_fazenda}"):

                    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

                    ax.hexbin(
                        df_faz["vl_longitude_inicial"],
                        df_faz["vl_latitude_inicial"],
                        gridsize=50,
                        cmap="Reds",
                        mincnt=1
                    )

                    ax.set_title(
                        f"Velocidade\nMin {vel_min:.1f} | Max {vel_max:.1f} | Méd {vel_med:.1f}"
                    )

                    ax.axis("off")

                    rodape(fig, 0.5, 0.1)
                    st.pyplot(fig)

            # =======================
            # HEATMAP RPM
            # =======================
            if "RPM" in TIPO_MAPA:

                with st.expander(f"⚙️ RPM – {nome_fazenda}"):

                    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

                    ax.hexbin(
                        df_faz["vl_longitude_inicial"],
                        df_faz["vl_latitude_inicial"],
                        gridsize=50,
                        cmap="Blues",
                        mincnt=1
                    )

                    ax.set_title(
                        f"RPM\nMin {rpm_min:.0f} | Max {rpm_max:.0f} | Méd {rpm_med:.0f}"
                    )

                    ax.axis("off")

                    rodape(fig, 0.5, 0.1)
                    st.pyplot(fig)

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
