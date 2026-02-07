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
# BOTÃO MAIOR (CSS)
# =========================================================
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

# =========================================================
# UPLOAD DE ARQUIVOS
# =========================================================
uploaded_zip = st.file_uploader(
    "📦 Upload do ZIP contendo o CSV da Solinftec",
    type=["zip"]
)

uploaded_gpkg = st.file_uploader(
    "🗺️ Upload da base cartográfica (GPKG)",
    type=["gpkg"]
)

GERAR = st.button("▶️ Gerar mapa")

# =========================================================
# SIDEBAR – PARÂMETROS
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

TEMPO_MAX_SEG = 60

LARGURA_IMPLEMENTO = st.sidebar.number_input(
    "Largura do implemento (metros)",
    min_value=1.0,
    max_value=30.0,
    value=6.0,
    step=0.5
)

# =========================================================
# CORES E FIGURA
# =========================================================
COR_TRABALHADA = "#61b27f"
COR_NAO_TRAB = "#f8cfc6"
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9

# =========================================================
# PROCESSAMENTO
# =========================================================
if uploaded_zip and uploaded_gpkg and GERAR:

    with tempfile.TemporaryDirectory() as tmpdir:

        # -------------------------
        # Extrai ZIP
        # -------------------------
        zip_path = os.path.join(tmpdir, "dados.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        csv_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".csv")]
        if not csv_files:
            st.error("❌ Nenhum CSV encontrado dentro do ZIP.")
            st.stop()

        csv_path = os.path.join(tmpdir, csv_files[0])

        # -------------------------
        # Leitura CSV
        # -------------------------
        df = pd.read_csv(csv_path, sep=";", encoding="latin1", engine="python")

        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")

        df = df[
            (df["cd_estado"] == "E") &
            (df["cd_operacao_parada"] == -1)
        ].copy()

        # 🔧 CORREÇÃO CRÍTICA: garantir mesmo tipo
        df["cd_fazenda"] = df["cd_fazenda"].astype(str)

        # -------------------------
        # GPKG
        # -------------------------
        gpkg_path = os.path.join(tmpdir, "base.gpkg")
        with open(gpkg_path, "wb") as f:
            f.write(uploaded_gpkg.read())

        base = gpd.read_file(gpkg_path)
        base["FAZENDA"] = base["FAZENDA"].astype(str)

        # =========================================================
        # LOOP POR FAZENDA
        # =========================================================
        for FAZENDA_ID in df["cd_fazenda"].dropna().unique():

            with st.expander(f"🗺️ Mapa – Fazenda {FAZENDA_ID}", expanded=True):

                df_faz = df[df["cd_fazenda"] == FAZENDA_ID].copy()
                base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

                if df_faz.empty or base_fazenda.empty:
                    st.warning("Dados insuficientes para esta fazenda.")
                    continue

                nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]

                # -------------------------
                # Projeção
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

                # 🔧 união correta da fazenda
                geom_fazenda = unary_union(base_fazenda.geometry)

                # -------------------------
                # Linhas
                # -------------------------
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

                gdf_linhas = gpd.GeoDataFrame(geometry=linhas, crs=base_fazenda.crs)

                buffer_linhas = gdf_linhas.buffer(LARGURA_IMPLEMENTO / 2)
                area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
                area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

                # -------------------------
                # Estatísticas
                # -------------------------
                area_total_ha = base_fazenda.geometry.area.sum() / 10000
                area_trab_ha = area_trabalhada.area / 10000
                area_nao_ha = area_nao_trabalhada.area / 10000

                pct_trab = area_trab_ha / area_total_ha * 100
                pct_nao = area_nao_ha / area_total_ha * 100

                dt_min = df_faz["dt_hr_local_inicial"].min()
                dt_max = df_faz["dt_hr_local_inicial"].max()

                periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M")
                periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M")

                # =========================================================
                # PLOT
                # =========================================================
                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

                base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB, edgecolor="black", linewidth=1.2)
                gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(
                    ax=ax, color=COR_TRABALHADA, alpha=0.9
                )
                base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.2)

                # LEGENDA
                ax.legend(
                    handles=[
                        mpatches.Patch(color=COR_TRABALHADA, label="Área trabalhada"),
                        mpatches.Patch(color=COR_NAO_TRAB, label="Área não trabalhada"),
                        mpatches.Patch(facecolor="none", edgecolor="black", label="Limites da fazenda"),
                    ],
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.20),
                    ncol=3,
                    frameon=True,
                    fontsize=13
                )

                # RESUMO LATERAL
                pos = ax.get_position()
                fig.text(
                    pos.x1 + 0.01,
                    0.5,
                    f"Resumo da operação\n\n"
                    f"Fazenda: {FAZENDA_ID} – {nome_fazenda}\n\n"
                    f"Área total: {area_total_ha:.2f} ha\n"
                    f"Trabalhada: {area_trab_ha:.2f} ha ({pct_trab:.1f}%)\n"
                    f"Não trabalhada: {area_nao_ha:.2f} ha ({pct_nao:.1f}%)\n\n"
                    f"Período:\n{periodo_ini} até {periodo_fim}",
                    fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.8", facecolor=COR_CAIXA, edgecolor="black")
                )

                fig.suptitle(
                    f"Área trabalhada – Fazenda {FAZENDA_ID} – {nome_fazenda}",
                    fontsize=15
                )

                brasilia = pytz.timezone("America/Sao_Paulo")
                hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

                # DISCLAIMER
                fig.text(
                    0.5,
                    0.08,
                    "⚠️ Os resultados apresentados dependem da qualidade dos dados operacionais e geoespaciais fornecidos.",
                    ha="center",
                    fontsize=9,
                    color=COR_RODAPE
                )

                # RODAPÉ
                fig.text(
                    0.5,
                    0.045,
                    "Relatório elaborado com base em dados da Solinftec.",
                    ha="center",
                    fontsize=10,
                    color=COR_RODAPE
                )

                fig.text(
                    0.5,
                    0.025,
                    f"Desenvolvido por Kauã Ceconello • Gerado em {hora}",
                    ha="center",
                    fontsize=10,
                    color=COR_RODAPE
                )

                plt.subplots_adjust(left=0.05, right=0.90, bottom=0.32)
                ax.axis("off")

                st.pyplot(fig)

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
