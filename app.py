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
import io

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
uploaded_zip = st.file_uploader("📦 Upload ZIP (CSV Solinftec)", type=["zip"])
uploaded_gpkg = st.file_uploader("🗺️ Upload GPKG (base cartográfica)", type=["gpkg"])
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
    "📊 Mostrar área por Gleba/Talhão",
    value=False
)

COR_TRABALHADA = "#62b27f"
COR_NAO_TRAB = "#f6b1b3"
COR_CAIXA = "#f1f8ff"

FIG_WIDTH = 25
FIG_HEIGHT = 9

# =========================
# PROCESSAMENTO
# =========================
if uploaded_zip and uploaded_gpkg and GERAR:

    with tempfile.TemporaryDirectory() as tmpdir:

        zip_path = os.path.join(tmpdir, "dados.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        csv_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".csv")]
        if not csv_files:
            st.error("❌ Nenhum CSV encontrado.")
            st.stop()

        df = pd.read_csv(os.path.join(tmpdir, csv_files[0]),
                         sep=";", encoding="latin1", engine="python")

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
        # LOOP FAZENDAS
        # =========================
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

            # =========================
            # LINHAS
            # =========================
            linhas = []
            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha_atual = []
                ultimo = None

                for _, row in grupo.iterrows():
                    if ultimo is None:
                        linha_atual = [row.geometry]
                    else:
                        dt = (row["dt_hr_local_inicial"] - ultimo).total_seconds()
                        if dt <= TEMPO_MAX_SEG:
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

            largura = df_faz["vl_largura_implemento"].dropna().mean()
            if pd.isna(largura):
                continue

            buffer_linhas = gdf_linhas.buffer((largura * MULTIPLICADOR_BUFFER) / 2)

            area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
            area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

            area_total_ha = round(base_fazenda.geometry.area.sum() / 10000, 2)
            area_trab_ha = round(area_trabalhada.area / 10000, 2)
            area_nao_ha = round(area_nao_trabalhada.area / 10000, 2)

            if area_trab_ha < AREA_MIN_HA:
                continue

            # =========================
            # TALHÕES
            # =========================
            df_talhoes = None

            if MOSTRAR_TALHOES and "TALHAO" in base_fazenda.columns and "GLEBA" in base_fazenda.columns:

                intersec = gpd.overlay(
                    base_fazenda,
                    gpd.GeoDataFrame(geometry=[area_trabalhada], crs=base_fazenda.crs),
                    how="intersection"
                )

                if not intersec.empty:
                    intersec["area_trab_ha"] = (intersec.geometry.area / 10000).round(2)
                    trab = intersec.groupby(["GLEBA", "TALHAO"])["area_trab_ha"].sum().reset_index()
                else:
                    trab = pd.DataFrame(columns=["GLEBA", "TALHAO", "area_trab_ha"])

                total = base_fazenda.copy()
                total["area_total_ha"] = (total.geometry.area / 10000).round(2)
                total = total[["GLEBA", "TALHAO", "area_total_ha"]]

                df_talhoes = total.merge(trab, on=["GLEBA", "TALHAO"], how="left")
                df_talhoes["area_trab_ha"] = df_talhoes["area_trab_ha"].fillna(0).round(2)

            # =========================
            # MAPA
            # =========================
            with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):

                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

                base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB, edgecolor="black")
                gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(ax=ax, color=COR_TRABALHADA)

                ax.axis("off")

                st.pyplot(fig)

                # =========================
                # TALHÕES UI MELHORADO
                # =========================
                if df_talhoes is not None:

                    st.markdown("### 🌾 Gleba / Talhão")

                    df_view = df_talhoes.rename(columns={
                        "GLEBA": "Gleba",
                        "TALHAO": "Talhão",
                        "area_total_ha": "Área Total (ha)",
                        "area_trab_ha": "Área Trabalhada (ha)"
                    })

                    st.dataframe(df_view, use_container_width=True, hide_index=True)

                    csv = df_view.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "📋 Copiar tabela (CSV)",
                        data=csv,
                        file_name=f"talhoes_{FAZENDA_ID}.csv",
                        mime="text/csv"
                    )

                # =========================
                # PDF MAPA
                # =========================
                pdf_buffer = io.BytesIO()
                fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
                pdf_buffer.seek(0)

                st.download_button(
                    "📄 Exportar mapa em PDF",
                    data=pdf_buffer,
                    file_name=f"mapa_{FAZENDA_ID}.pdf",
                    mime="application/pdf"
                )

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
