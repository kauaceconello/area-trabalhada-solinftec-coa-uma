# =========================================================
# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC)
# Desenvolvido por Kauã Ceconello
# =========================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.geometry import Point, LineString
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
st.set_page_config(
    page_title="Área Trabalhada – Solinftec",
    layout="wide"
)

st.title("📍 Área Trabalhada – Solinftec")

st.markdown(
    "Aplicação para cálculo e visualização da **área trabalhada** com base em "
    "dados operacionais da **Solinftec** e base cartográfica da fazenda."
)

# =========================================================
# UPLOAD DE ARQUIVOS
# =========================================================
uploaded_zip = st.file_uploader(
    "📦 Upload do arquivo ZIP contendo o CSV da Solinftec",
    type=["zip"]
)

uploaded_gpkg = st.file_uploader(
    "🗺️ Upload da base cartográfica da fazenda (GPKG)",
    type=["gpkg"]
)

# =========================================================
# PARÂMETROS INTERATIVOS
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

TEMPO_MAX_SEG = st.sidebar.number_input(
    "Tempo máximo entre pontos (segundos)",
    min_value=5,
    max_value=300,
    value=60,
    step=5
)

LARGURA_IMPLEMENTO = st.sidebar.number_input(
    "Largura do implemento (metros)",
    min_value=1.0,
    max_value=30.0,
    value=6.0,
    step=0.5
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

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        csv_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".csv")]

        if not csv_files:
            st.error("❌ Nenhum arquivo CSV encontrado dentro do ZIP.")
            st.stop()

        csv_path = os.path.join(tmpdir, csv_files[0])

        # -------------------------
        # Leitura CSV
        # -------------------------
        df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="latin1",
            engine="python"
        )

        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")

        df = df.dropna(subset=[
            "cd_fazenda",
            "dt_hr_local_inicial",
            "vl_latitude_inicial",
            "vl_longitude_inicial"
        ])

        if df.empty:
            st.error("❌ O CSV não possui dados válidos.")
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

            st.subheader(f"🗺️ Mapa – Fazenda {FAZENDA_ID}")

            df_faz = df[
                (df["cd_fazenda"] == FAZENDA_ID) &
                (df["cd_estado"] == "E") &
                (df["cd_operacao_parada"] == -1)
            ].copy()

            if df_faz.empty:
                st.warning(f"⚠️ Fazenda {FAZENDA_ID}: sem dados produtivos.")
                continue

            base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

            if base_fazenda.empty:
                st.warning(f"⚠️ Fazenda {FAZENDA_ID} não encontrada no GPKG.")
                continue

            nome_fazenda = (
                base_fazenda["PROPRIEDADE"].iloc[0]
                if "PROPRIEDADE" in base_fazenda.columns
                else ""
            )

            # -------------------------
            # GeoDataFrame pontos
            # -------------------------
            gdf_pts = gpd.GeoDataFrame(
                df_faz,
                geometry=gpd.points_from_xy(
                    df_faz["vl_longitude_inicial"],
                    df_faz["vl_latitude_inicial"]
                ),
                crs="EPSG:4326"
            )

            base_fazenda = base_fazenda.to_crs(epsg=31983)
            gdf_pts = gdf_pts.to_crs(epsg=31983)

            geom_fazenda = unary_union(base_fazenda.geometry)

            # -------------------------
            # Construção das linhas
            # -------------------------
            linhas = []

            for equipamento, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha = []
                ultimo_tempo = None

                for _, row in grupo.iterrows():
                    if ultimo_tempo is None:
                        linha = [row.geometry]
                    else:
                        delta = (row["dt_hr_local_inicial"] - ultimo_tempo).total_seconds()
                        if delta <= TEMPO_MAX_SEG:
                            linha.append(row.geometry)
                        else:
                            if len(linha) >= 2:
                                linhas.append(LineString(linha))
                            linha = [row.geometry]

                    ultimo_tempo = row["dt_hr_local_inicial"]

                if len(linha) >= 2:
                    linhas.append(LineString(linha))

            if not linhas:
                st.warning("⚠️ Nenhuma linha válida para gerar área.")
                continue

            gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_fazenda.crs)

            buffer_linhas = gdf_linhas.buffer(LARGURA_IMPLEMENTO / 2)
            area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
            area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

            # -------------------------
            # Plot
            # -------------------------
            fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

            # -------------------------------------------------
            # EXTENSÃO DO MAPA (corrigida para evitar "mapa vazio")
            # -------------------------------------------------
geom_ref = area_trabalhada if not area_trabalhada.is_empty else geom_fazenda

minx, miny, maxx, maxy = geom_ref.bounds

dx = maxx - minx
dy = maxy - miny
pad = 0.15  # 15% de margem real

ax.set_xlim(minx - dx * pad, maxx + dx * pad)
ax.set_ylim(miny - dy * pad, maxy + dy * pad)
ax.set_aspect("equal")

            base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB, edgecolor="black", linewidth=1.2)
            gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(
                ax=ax, color=COR_TRABALHADA, alpha=0.9
            )
            base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.2)

            legenda = [
                mpatches.Patch(color=COR_TRABALHADA, label="Área trabalhada"),
                mpatches.Patch(color=COR_NAO_TRAB, label="Área não trabalhada"),
                mpatches.Patch(facecolor="none", edgecolor="black", label="Limite da fazenda"),
            ]

            ax.legend(
                handles=legenda,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.12),
                ncol=3,
                frameon=True,
                fontsize=13,
                handlelength=1.2,
                labelspacing=0.6,
                borderpad=0.8
            )

            brasilia = pytz.timezone("America/Sao_Paulo")
            agora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

            fig.suptitle(
                f"Área trabalhada – Fazenda {FAZENDA_ID} – {nome_fazenda}",
                fontsize=16
            )

            fig.text(
                0.5,
                0.03,
                "Relatório elaborado com base em dados da Solinftec. "
                f"Desenvolvido por Kauã Ceconello • Gerado em {agora}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=COR_RODAPE
            )

            ax.axis("off")
            plt.subplots_adjust(bottom=0.20)

            st.pyplot(fig)

else:
    st.info("⬆️ Envie o ZIP com o CSV e o arquivo GPKG para gerar os mapas.")
