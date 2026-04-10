# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC) - VERSÃO OTIMIZADA
# Desenvolvido por Kauã Ceconello

import streamlit as st
import pandas as pd
import geopandas as gpd

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
# CONFIG
# =========================
st.set_page_config(page_title="Área Trabalhada – Solinftec", layout="wide")

st.title("📍 Área Trabalhada – Solinftec")

st.markdown(
    "Aplicação para cálculo e visualização da **área trabalhada** com base em dados da Solinftec."
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


# =========================
# UPLOAD
# =========================
uploaded_zips = st.file_uploader(
    "📦 Upload dos ZIPs",
    type=["zip"],
    accept_multiple_files=True
)

uploaded_gpkg = st.file_uploader(
    "🗺️ Upload GPKG",
    type=["gpkg"]
)

GERAR = st.button("▶️ Gerar mapa")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Parâmetros")

TEMPO_MAX_SEG = 60

MULTIPLICADOR_BUFFER = st.sidebar.number_input(
    "Tamanho do Buffer", 1.0, 10.0, 2.5, 0.1
)

AREA_MIN_HA = st.sidebar.number_input(
    "Área mínima (ha)", 0.0, 0.5, 0.5, 0.1
)

MOSTRAR_TALHOES = st.sidebar.checkbox(
    "📊 Mostrar área por Talhão",
    value=False
)


# =========================
# PROCESSAMENTO OTIMIZADO
# =========================
if uploaded_zips and uploaded_gpkg and GERAR:

    with tempfile.TemporaryDirectory() as tmpdir:

        # =========================
        # LEITURA ZIPs (OTIMIZADO)
        # =========================
        dfs = []

        for zfile in uploaded_zips:

            zip_path = os.path.join(tmpdir, zfile.name)

            with open(zip_path, "wb") as f:
                f.write(zfile.read())

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
            if not csv_files:
                continue

            df_temp = pd.read_csv(
                os.path.join(tmpdir, csv_files[0]),
                sep=";",
                encoding="latin1",
                usecols=[
                    "dt_hr_local_inicial",
                    "vl_latitude_inicial",
                    "vl_longitude_inicial",
                    "vl_largura_implemento",
                    "cd_estado",
                    "cd_operacao_parada",
                    "cd_fazenda",
                    "cd_equipamento"
                ]
            )

            dfs.append(df_temp)

        if not dfs:
            st.error("❌ Nenhum dado válido encontrado.")
            st.stop()

        df = pd.concat(dfs, ignore_index=True)

        # =========================
        # TRATAMENTO VETORIZADO
        # =========================
        df = df[
            (df["cd_estado"] == "E") &
            (df["cd_operacao_parada"] == -1)
        ].copy()

        df["cd_fazenda"] = df["cd_fazenda"].astype(str)

        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")

        df.dropna(subset=["vl_latitude_inicial", "vl_longitude_inicial"], inplace=True)


        # =========================
        # GPKG
        # =========================
        gpkg_path = os.path.join(tmpdir, "base.gpkg")

        with open(gpkg_path, "wb") as f:
            f.write(uploaded_gpkg.read())

        base = gpd.read_file(gpkg_path)
        base["FAZENDA"] = base["FAZENDA"].astype(str)


        # =========================
        # LOOP FAZENDAS
        # =========================
        for faz_id, df_faz in df.groupby("cd_fazenda"):

            base_faz = base[base["FAZENDA"] == faz_id]

            if base_faz.empty:
                continue

            nome_fazenda = base_faz["PROPRIEDADE"].iloc[0]

            # -------------------------
            # GEO DATA
            # -------------------------
            gdf_pts = gpd.GeoDataFrame(
                df_faz,
                geometry=gpd.points_from_xy(
                    df_faz["vl_longitude_inicial"],
                    df_faz["vl_latitude_inicial"]
                ),
                crs="EPSG:4326"
            )

            base_faz = base_faz.to_crs(31983)
            gdf_pts = gdf_pts.to_crs(31983)

            geom_fazenda = base_faz.unary_union

            # =========================
            # LINHAS (OTIMIZADO - itertuples)
            # =========================
            linhas = []

            for _, grupo in gdf_pts.groupby("cd_equipamento"):

                grupo = grupo.sort_values("dt_hr_local_inicial")

                coords = []
                prev_time = None

                for row in grupo.itertuples():

                    if prev_time is None:
                        coords = [row.geometry]
                    else:
                        delta = (row.dt_hr_local_inicial - prev_time).total_seconds()

                        if delta <= TEMPO_MAX_SEG:
                            coords.append(row.geometry)
                        else:
                            if len(coords) >= 2:
                                linhas.append(LineString(coords))
                            coords = [row.geometry]

                    prev_time = row.dt_hr_local_inicial

                if len(coords) >= 2:
                    linhas.append(LineString(coords))

            if not linhas:
                continue

            gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_faz.crs)

            # =========================
            # BUFFER OTIMIZADO (1 operação só)
            # =========================
            largura = df_faz["vl_largura_implemento"].mean()

            if pd.isna(largura):
                continue

            largura *= MULTIPLICADOR_BUFFER

            area_trabalhada = unary_union(gdf_linhas.geometry).buffer(largura / 2)
            area_trabalhada = area_trabalhada.intersection(geom_fazenda)
            area_nao = geom_fazenda.difference(area_trabalhada)

            # =========================
            # ÁREAS
            # =========================
            area_total = base_faz.geometry.area.sum() / 10000
            area_trab = area_trabalhada.area / 10000

            if area_trab < AREA_MIN_HA:
                continue

            pct = (area_trab / area_total) * 100

            # =========================
            # MAPA
            # =========================
            with st.expander(f"🗺️ {nome_fazenda}"):

                fig, ax = plt.subplots(figsize=(22, 9))

                base_faz.plot(ax=ax, color="#f6b1b3", edgecolor="black")
                gpd.GeoSeries(area_trabalhada).plot(ax=ax, color="#62b27f", alpha=0.8)

                base_faz.boundary.plot(ax=ax, color="black")

                # labels talhão (simples e rápido)
                if "TALHAO" in base_faz.columns:
                    for r in base_faz.itertuples():
                        if r.geometry.is_empty:
                            continue
                        c = r.geometry.centroid
                        ax.text(c.x, c.y, str(r.TALHAO), fontsize=7)

                ax.axis("off")

                ax.set_title(
                    f"{faz_id} - {nome_fazenda} | {area_trab:.2f} ha ({pct:.1f}%)"
                )

                st.pyplot(fig)

else:
    st.info("⬆️ Envie os arquivos e clique em Gerar mapa")
