import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import LineString
import zipfile
import io
import pytz

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(layout="wide")
st.title("Mapas de Área Trabalhada por Fazenda")

# =========================================================
# UPLOAD DE ARQUIVOS
# =========================================================
zip_file = st.file_uploader("Upload do ZIP contendo o CSV", type=["zip"])
gpkg_file = st.file_uploader("Upload da Base Cartográfica (GPKG)", type=["gpkg"])

if not zip_file or not gpkg_file:
    st.warning("Faça upload do ZIP com o CSV e do arquivo GPKG.")
    st.stop()

# =========================================================
# LEITURA DO CSV A PARTIR DO ZIP
# =========================================================
with zipfile.ZipFile(zip_file) as z:
    csv_name = [f for f in z.namelist() if f.endswith(".csv")][0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f)

# =========================================================
# LEITURA DO GPKG
# =========================================================
fazendas_gdf = gpd.read_file(gpkg_file)

# =========================================================
# PADRONIZAÇÕES
# =========================================================
st.write("Colunas disponíveis no dataframe:")
st.write(df.columns.tolist())

if "dt_hr_local_inicial" in df.columns:
    df["dt_hr_local_inicial"] = pd.to_datetime(
        df["dt_hr_local_inicial"], errors="coerce"
    )
else:
    st.warning("Coluna 'dt_hr_local_inicial' não encontrada no arquivo.")

tz_brasilia = pytz.timezone("America/Sao_Paulo")
df["dt_hr_local_inicial"] = df["dt_hr_local_inicial"].dt.tz_localize(
    tz_brasilia, nonexistent="NaT", ambiguous="NaT"
)

# =========================================================
# CORES
# =========================================================
COR_TRABALHADA = "#61b27f"
COR_NAO_TRABALHADA = "#f6b1b2"
COR_RESUMO = "#f1f8ff"

# =========================================================
# FAZENDAS PRESENTES NO CSV
# =========================================================
fazendas_csv = df["cd_fazenda"].dropna().unique()

# =========================================================
# LOOP POR FAZENDA
# =========================================================
for cod_fazenda in fazendas_csv:

    st.subheader(f"Mapa – Fazenda {cod_fazenda}")

    df_faz = df[df["cd_fazenda"] == cod_fazenda]

    faz_gdf = fazendas_gdf[fazendas_gdf["FAZENDA"] == cod_fazenda]

    if faz_gdf.empty:
        st.error(f"Fazenda {cod_fazenda} não encontrada no GPKG.")
        continue

    faz_geom = faz_gdf.geometry.iloc[0]

    # =========================================================
    # LINHAS DE TRABALHO
    # =========================================================
    linhas = []
    for _, row in df_faz.iterrows():
        try:
            linhas.append(
                LineString(
                    [(row["longitude"], row["latitude"]),
                     (row["longitude_fim"], row["latitude_fim"])]
                )
            )
        except:
            pass

    if not linhas:
        st.warning("Sem linhas válidas para esta fazenda.")
        continue

    linhas_union = unary_union(linhas)

    # =========================================================
    # BUFFER (parametrizável no futuro)
    # =========================================================
    largura_buffer = 5  # metros
    area_trabalhada = gpd.GeoSeries(linhas_union).buffer(largura_buffer)

    # limitar ao contorno da fazenda
    area_trabalhada = area_trabalhada.intersection(faz_geom)

    # área não trabalhada
    area_nao_trabalhada = faz_geom.difference(unary_union(area_trabalhada))

    # =========================================================
    # FIGURA
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 10))

    gpd.GeoSeries(area_nao_trabalhada).plot(
        ax=ax,
        color=COR_NAO_TRABALHADA,
        edgecolor="black",
        linewidth=1,
        label="Área não trabalhada"
    )

    gpd.GeoSeries(area_trabalhada).plot(
        ax=ax,
        color=COR_TRABALHADA,
        edgecolor="black",
        linewidth=1,
        label="Área trabalhada"
    )

    gpd.GeoSeries(faz_geom).boundary.plot(
        ax=ax,
        color="black",
        linewidth=2,
        label="Limites da fazenda"
    )

    ax.set_axis_off()

    # =========================================================
    # LEGENDA AJUSTADA
    # =========================================================
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        fontsize=11,
        borderpad=0.6,
        handletextpad=0.8
    )

    # =========================================================
    # RESUMO DA OPERAÇÃO
    # =========================================================
    periodo_ini = df_faz["dt_hr_local_inicial"].min()
    periodo_fim = df_faz["dt_hr_local_inicial"].max()

    if pd.notna(periodo_ini) and pd.notna(periodo_fim):
        periodo_txt = (
            f"{periodo_ini.strftime('%d/%m/%Y %H:%M')} "
            f"a {periodo_fim.strftime('%d/%m/%Y %H:%M')}"
        )
    else:
        periodo_txt = "Período indisponível"

    resumo = (
        f"Fazenda: {cod_fazenda}\n"
        f"Período: {periodo_txt}\n"
        f"Largura do buffer: {largura_buffer} m"
    )

    ax.text(
        0.02, 0.98,
        resumo,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=COR_RESUMO,
            edgecolor="black"
        )
    )

    st.pyplot(fig)
