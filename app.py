# =========================================================
# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC)
# Desenvolvido por Kauã Ceconello
# =========================================================

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

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(page_title="Área Trabalhada – Solinftec", layout="wide")

st.title("📍 Área Trabalhada – Solinftec")

st.markdown(
    "Aplicação para cálculo e visualização da **área trabalhada** com base em "
    "dados operacionais da **Solinftec** e base cartográfica da fazenda."
)

# =========================================================
# UPLOAD DE ARQUIVOS
# =========================================================
uploaded_zip = st.file_uploader(
    "📦 Upload do arquivo ZIP contendo o CSV da Solinftec", type=["zip"]
)

uploaded_gpkg = st.file_uploader(
    "🗺️ Upload da base cartográfica da fazenda (GPKG)", type=["gpkg"]
)

# =========================================================
# PARÂMETROS
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

TEMPO_MAX_SEG = st.sidebar.number_input(
    "Tempo máximo entre pontos (segundos)", 5, 300, 60, 5
)

LARGURA_IMPLEMENTO = st.sidebar.number_input(
    "Largura do implemento (metros)", 1.0, 30.0, 6.0, 0.5
)

COR_TRABALHADA = "#61b27f"
COR_NAO_TRAB = "#f8cfc6"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9

# =========================================================
# PROCESSAMENTO
# =========================================================
if uploaded_zip and uploaded_gpkg:

    with tempfile.TemporaryDirectory() as tmpdir:

        # -------------------------
        # Extrai ZIP
        # -------------------------
        zip_path = os.path.join(tmpdir, "dados.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)

        csv_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".csv")]

        if not csv_files:
            st.error("❌ Nenhum CSV encontrado dentro do ZIP.")
            st.stop()

        csv_path = os.path.join(tmpdir, csv_files[0])

        df = pd.read_csv(csv_path, sep=";", encoding="latin1", engine="python")

        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")

        df = df.dropna(subset=[
            "cd_fazenda", "dt_hr_local_inicial",
            "vl_latitude_inicial", "vl_longitude_inicial"
        ])

        if df.empty:
            st.error("❌ CSV sem dados válidos.")
            st.stop()

        # -------------------------
        # Leitura GPKG
        # -------------------------
        gpkg_path = os.path.join(tmpdir, "base.gpkg")
        with open(gpkg_path, "wb") as f:
            f.write(uploaded_gpkg.read())

        base = gpd.read_file(gpkg_path)

        # =========================================================
        # LOOP POR FAZENDA
        # =========================================================
        for FAZENDA_ID in sorted(df["cd_fazenda"].unique()):

            st.subheader(f"🗺️ Fazenda {FAZENDA_ID}")

            df_faz = df[
                (df["cd_fazenda"] == FAZENDA_ID) &
                (df["cd_estado"] == "E") &
                (df["cd_operacao_parada"] == -1)
            ].copy()

            if df_faz.empty:
                st.warning("⚠️ Sem dados produtivos.")
                continue

            base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

            if base_fazenda.empty:
                st.warning("⚠️ Fazenda não encontrada no GPKG.")
                continue

            nome_fazenda = base_fazenda.get("PROPRIEDADE", [""])[0]

            # -------------------------
            # Geo
            # -------------------------
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

            # -------------------------
            # Linhas
            # -------------------------
            linhas = []

            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha = []
                ultimo = None

                for _, row in grupo.iterrows():
                    if ultimo is None or (
                        (row["dt_hr_local_inicial"] - ultimo).total_seconds() <= TEMPO_MAX_SEG
                    ):
                        linha.append(row.geometry)
                    else:
                        if len(linha) >= 2:
                            linhas.append(LineString(linha))
                        linha = [row.geometry]

                    ultimo = row["dt_hr_local_inicial"]

                if len(linha) >= 2:
                    linhas.append(LineString(linha))

            if not linhas:
                st.warning("⚠️ Nenhuma linha válida.")
                continue

            buffer_linhas = gpd.GeoSeries(linhas, crs=base_fazenda.crs).buffer(LARGURA_IMPLEMENTO / 2)
            area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
            area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

            # -------------------------
            # Estatísticas
            # -------------------------
            area_total_ha = base_fazenda.area.sum() / 10000
            area_trab_ha = area_trabalhada.area / 10000
            area_nao_ha = area_total_ha - area_trab_ha

            pct_trab = area_trab_ha / area_total_ha * 100 if area_total_ha else 0

            # -------------------------
            # Plot
            # -------------------------
            fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

            minx, miny, maxx, maxy = geom_fazenda.bounds
            dx, dy = maxx - minx, maxy - miny
            pad = 0.15

            ax.set_xlim(minx - dx * pad, maxx + dx * pad)
            ax.set_ylim(miny - dy * pad, maxy + dy * pad)
            ax.set_aspect("equal")

            base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB, edgecolor="black")
            gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(
                ax=ax, color=COR_TRABALHADA, alpha=0.9
            )
            base_fazenda.boundary.plot(ax=ax, color="black")

            ax.legend(
                handles=[
                    mpatches.Patch(color=COR_TRABALHADA, label="Área trabalhada"),
                    mpatches.Patch(color=COR_NAO_TRAB, label="Área não trabalhada"),
                    mpatches.Patch(facecolor="none", edgecolor="black", label="Limite da fazenda"),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.12),
                ncol=3,
                fontsize=13
            )

            resumo = (
                f"Fazenda: {FAZENDA_ID}\n"
                f"Área total: {area_total_ha:,.2f} ha\n"
                f"Área trabalhada: {area_trab_ha:,.2f} ha ({pct_trab:.1f}%)\n"
                f"Largura do implemento: {LARGURA_IMPLEMENTO:.1f} m"
            )

            ax.text(
                0.02, 0.98, resumo,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f8ff", edgecolor="black")
            )

            ax.axis("off")

            agora = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M")
            fig.text(
                0.5, 0.03,
                f"Relatório Solinftec • Desenvolvido por Kauã Ceconello • {agora}",
                ha="center", fontsize=10, color=COR_RODAPE
            )

            st.pyplot(fig)

else:
    st.info("⬆️ Envie o ZIP com o CSV e o GPKG.")
