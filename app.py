# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC)
# Desenvolvido por Kauã Ceconello

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

# CONFIG STREAMLIT
st.set_page_config(
    page_title="Área Trabalhada – Solinftec",
    layout="wide"
)

st.title("📍 Área Trabalhada – Solinftec")

st.markdown(
    "Aplicação para cálculo e visualização da **área trabalhada** com base em "
    "dados operacionais da **Solinftec** e base cartográfica da fazenda."
)

# UPLOAD DE ARQUIVOS
uploaded_zip = st.file_uploader(
    "📦 Upload do arquivo ZIP contendo o CSV da Solinftec",
    type=["zip"]
)

uploaded_gpkg = st.file_uploader(
    "🗺️ Upload da base cartográfica da fazenda (GPKG)",
    type=["gpkg"]
)

# PARÂMETROS INTERATIVOS
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
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9

# PROCESSAMENTO
if uploaded_zip and uploaded_gpkg:

    with tempfile.TemporaryDirectory() as tmpdir:

        # Extrai ZIP
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

        # Leitura CSV
        df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="latin1",
            engine="python"
        )

        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")

        if df.empty:
            st.error("❌ O CSV está vazio.")
            st.stop()

        # Identificação da fazenda
        fazendas_csv = df["cd_fazenda"].dropna().unique()

        if len(fazendas_csv) == 0:
            st.error("❌ Nenhuma fazenda encontrada no CSV.")
            st.stop()

        if len(fazendas_csv) > 1:
            st.warning(f"⚠️ Mais de uma fazenda detectada no CSV: {fazendas_csv}")

        FAZENDA_ID = int(fazendas_csv[0])

        # Filtros essenciais
        df = df[
            (df["cd_fazenda"] == FAZENDA_ID) &
            (df["cd_estado"] == "E") &
            (df["cd_operacao_parada"] == -1)
        ].copy()

        if df.empty:
            st.error("❌ Nenhum dado produtivo após os filtros.")
            st.stop()

        # GeoDataFrame pontos
        gdf_pts = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df["vl_longitude_inicial"],
                df["vl_latitude_inicial"]
            ),
            crs="EPSG:4326"
        )

        # Base cartográfica
        gpkg_path = os.path.join(tmpdir, "base.gpkg")
        with open(gpkg_path, "wb") as f:
            f.write(uploaded_gpkg.read())

        base = gpd.read_file(gpkg_path)
        base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

        if base_fazenda.empty:
            st.error("❌ Fazenda não encontrada na base cartográfica.")
            st.stop()

        nome_fazenda = (
            base_fazenda["PROPRIEDADE"].iloc[0]
            if "PROPRIEDADE" in base_fazenda.columns
            else ""
        )

        geom_fazenda = unary_union(base_fazenda.geometry)

        # Projeção métrica
        base_fazenda = base_fazenda.to_crs(epsg=31983)
        gdf_pts = gdf_pts.to_crs(epsg=31983)
        geom_fazenda = gpd.GeoSeries([geom_fazenda], crs=base_fazenda.crs).iloc[0]

        # Construção das linhas
        linhas = []

        for equipamento, grupo in gdf_pts.groupby("cd_equipamento"):
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

        gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_fazenda.crs)

        # Buffer + áreas
        buffer_linhas = gdf_linhas.buffer(LARGURA_IMPLEMENTO / 2)
        area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
        area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

        # Estatísticas
        area_total_ha = base_fazenda.geometry.area.sum() / 10000
        area_trab_ha = area_trabalhada.area / 10000 if not area_trabalhada.is_empty else 0
        area_nao_ha = area_nao_trabalhada.area / 10000

        pct_trab = area_trab_ha / area_total_ha * 100 if area_total_ha else 0
        pct_nao = area_nao_ha / area_total_ha * 100 if area_total_ha else 0

        dt_min = df["dt_hr_local_inicial"].min()
        dt_max = df["dt_hr_local_inicial"].max()

        periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M") if pd.notna(dt_min) else "—"
        periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M") if pd.notna(dt_max) else "—"

        # PLOT
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

        minx, miny, maxx, maxy = base_fazenda.total_bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        w, h = (maxx - minx), (maxy - miny)
        pad = 0.10

        ax.set_xlim(cx - w*(1+pad)/2, cx + w*(1+pad)/2)
        ax.set_ylim(cy - h*(1+pad)/2, cy + h*(1+pad)/2)
        ax.set_aspect("equal")

        base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB, edgecolor="black", linewidth=1.2)
        gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(ax=ax, color=COR_TRABALHADA, alpha=0.9)
        base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.2)

        ax.legend(
            handles=[
                mpatches.Patch(color=COR_TRABALHADA, label="Área trabalhada"),
                mpatches.Patch(color=COR_NAO_TRAB, label="Área não trabalhada"),
                mpatches.Patch(facecolor="none", edgecolor="black", label="Limites da fazenda"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=3,
            frameon=True,
            fontsize=13
        )

        brasilia = pytz.timezone("America/Sao_Paulo")
        agora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

        fig.text(
            0.5,
            0.02,
            "Relatório elaborado com base em dados extraídos do sistema Solinftec. "
            "Podem ocorrer divergências decorrentes de inconsistências operacionais ou geoespaciais.\n"
            f"Desenvolvido por Kauã Ceconello • Gerado em {agora}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COR_RODAPE
        )

        fig.suptitle(
            f"Área trabalhada – Fazenda {FAZENDA_ID} – {nome_fazenda}",
            fontsize=15
        )

        ax.axis("off")
        plt.subplots_adjust(bottom=0.15)

        st.pyplot(fig)

else:
    st.info("⬆️ Envie o ZIP com o CSV e o arquivo GPKG para gerar o mapa.")
