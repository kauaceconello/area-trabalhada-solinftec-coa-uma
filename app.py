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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

import zipfile
import tempfile
import os
import pytz
from datetime import datetime

# =========================
# FUNÇÕES AUXILIARES
# =========================
def criar_cmap_discreto(tipo: str, vmin: float, vmax: float):
    """
    Cria colormap discreto com colorbar estilo 'termômetro',
    mantendo valores acima do máximo na cor 'over' (vermelho).
    """
    if tipo == "rpm":
        # Azul escuro -> azul -> azul claro -> verde claro -> verde ideal -> amarelo -> laranja
        cores = [
            "#08306b",  # azul escuro
            "#2171b5",  # azul
            "#6baed6",  # azul claro
            "#74c476",  # verde claro
            "#31a354",  # verde ideal
            "#fed976",  # amarelo
            "#fd8d3c",  # laranja
        ]
        cor_over = "#e31a1c"  # vermelho para > máximo
    else:
        # Velocidade: parecido, mas com leve diferença visual
        cores = [
            "#084081",  # azul escuro
            "#0868ac",  # azul
            "#43a2ca",  # azul claro
            "#7bccc4",  # verde água
            "#41ab5d",  # verde
            "#fed976",  # amarelo
            "#f16913",  # laranja
        ]
        cor_over = "#cb181d"  # vermelho para > máximo

    cmap = ListedColormap(cores)
    cmap.set_under(cores[0])
    cmap.set_over(cor_over)

    # N bins internos = número de cores
    limites = np.linspace(vmin, vmax, len(cores) + 1)
    norm = BoundaryNorm(limites, cmap.N, clip=False)

    return cmap, norm, limites


def desenhar_base_mapa(ax, base_fazenda, mostrar_talhoes=True, facecolor="#f7f7f7"):
    """
    Desenha a base da fazenda e, se existir, os rótulos dos talhões.
    """
    base_fazenda.plot(ax=ax, facecolor=facecolor, edgecolor="black", linewidth=1.2)
    base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.2)

    if mostrar_talhoes and "TALHAO" in base_fazenda.columns:
        for _, row in base_fazenda.iterrows():
            if row.geometry.is_empty:
                continue
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(row["TALHAO"]),
                fontsize=7,
                ha="center",
                va="center",
                color="black",
                weight="bold"
            )

    ax.axis("off")


def adicionar_footer(fig, centro_mapa, base_y, cor_rodape):
    """
    Adiciona o rodapé institucional do relatório.
    """
    brasilia = pytz.timezone("America/Sao_Paulo")
    hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

    fig.text(
        centro_mapa,
        base_y - 0.11,
        "⚠️ Os resultados apresentados dependem da qualidade dos dados operacionais e geoespaciais fornecidos.",
        ha="center",
        fontsize=10,
        color=cor_rodape
    )

    fig.text(
        centro_mapa,
        base_y - 0.14,
        "Relatório elaborado com base em dados da Solinftec.",
        ha="center",
        fontsize=10,
        color=cor_rodape
    )

    fig.text(
        centro_mapa,
        base_y - 0.17,
        f"Desenvolvido por Kauã Ceconello • Gerado em {hora}",
        ha="center",
        fontsize=10,
        color=cor_rodape
    )


def adicionar_resumo(fig, x, y, texto, cor_caixa):
    """
    Adiciona a caixa de resumo lateral.
    """
    fig.text(
        x,
        y,
        texto,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.8", facecolor=cor_caixa, edgecolor="black")
    )


def adicionar_colorbar(fig, pos_ax, cmap, norm, limites, titulo):
    """
    Cria um 'termômetro' vertical ao lado do mapa, com régua e indicação do range.
    """
    # eixo da colorbar (entre o mapa e a caixa de resumo)
    cax = fig.add_axes([pos_ax.x1 + 0.01, pos_ax.y0 + 0.10, 0.015, pos_ax.height * 0.62])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        cax=cax,
        boundaries=limites,
        ticks=limites,
        spacing="proportional",
        extend="max"
    )

    cb.set_label(titulo, fontsize=10)
    cb.ax.tick_params(labelsize=8)

    # Formatação dos ticks
    if "Velocidade" in titulo:
        cb.ax.set_yticklabels([f"{v:.1f}" for v in limites])
    else:
        cb.ax.set_yticklabels([f"{int(round(v))}" for v in limites])


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

# Novos controles
st.sidebar.markdown("### 🗺️ Tipos de mapa")
MAPA_AREA = st.sidebar.checkbox("Área trabalhada", value=True)
MAPA_RPM = st.sidebar.checkbox("Mapa de RPM", value=False)
MAPA_VEL = st.sidebar.checkbox("Mapa de Velocidade", value=False)

st.sidebar.markdown("### ⚙️ Parâmetros RPM")
RPM_MIN = st.sidebar.number_input(
    "RPM mínimo",
    min_value=0,
    max_value=10000,
    value=1200,
    step=100
)
RPM_MAX = st.sidebar.number_input(
    "RPM máximo",
    min_value=0,
    max_value=10000,
    value=2000,
    step=100
)

st.sidebar.markdown("### ⚙️ Parâmetros Velocidade (km/h)")
VEL_MIN = st.sidebar.number_input(
    "Velocidade mínima",
    min_value=0.0,
    max_value=100.0,
    value=4.0,
    step=0.5
)
VEL_MAX = st.sidebar.number_input(
    "Velocidade máxima",
    min_value=0.0,
    max_value=100.0,
    value=8.0,
    step=0.5
)

COR_TRABALHADA = "#62b27f"
COR_NAO_TRAB = "#f6b1b3"
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9

# Validações básicas
if RPM_MAX <= RPM_MIN:
    st.sidebar.error("⚠️ O RPM máximo deve ser maior que o RPM mínimo.")
if VEL_MAX <= VEL_MIN:
    st.sidebar.error("⚠️ A velocidade máxima deve ser maior que a velocidade mínima.")
if not (MAPA_AREA or MAPA_RPM or MAPA_VEL):
    st.sidebar.warning("Selecione pelo menos um tipo de mapa.")

# =========================
# PROCESSAMENTO
# =========================
if uploaded_zips and uploaded_gpkg and GERAR:

    if RPM_MAX <= RPM_MIN or VEL_MAX <= VEL_MIN:
        st.error("❌ Ajuste os parâmetros mínimos e máximos na sidebar antes de gerar os mapas.")
        st.stop()

    if not (MAPA_AREA or MAPA_RPM or MAPA_VEL):
        st.error("❌ Selecione pelo menos um tipo de mapa na sidebar.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:

        # =========================
        # LEITURA DE MÚLTIPLOS ZIPS
        # =========================
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
        # TRATAMENTO ORIGINAL + NOVAS COLUNAS
        # =========================
        df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
        df["vl_latitude_inicial"] = pd.to_numeric(df["vl_latitude_inicial"], errors="coerce")
        df["vl_longitude_inicial"] = pd.to_numeric(df["vl_longitude_inicial"], errors="coerce")
        df["vl_largura_implemento"] = pd.to_numeric(df["vl_largura_implemento"], errors="coerce")
        df["vl_rpm"] = pd.to_numeric(df["vl_rpm"], errors="coerce")
        df["vl_velocidade"] = pd.to_numeric(df["vl_velocidade"], errors="coerce")

        df = df[
            (df["cd_estado"] == "E") &
            (df["cd_operacao_parada"] == -1)
        ].copy()

        df["cd_fazenda"] = df["cd_fazenda"].astype(str)

        # Remove coordenadas inválidas
        df = df.dropna(subset=["vl_latitude_inicial", "vl_longitude_inicial", "dt_hr_local_inicial"])

        if df.empty:
            st.error("❌ Os dados filtrados não possuem coordenadas e datas válidas para gerar os mapas.")
            st.stop()

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

        # Colormaps (montados após leitura da sidebar)
        rpm_cmap, rpm_norm, rpm_limites = criar_cmap_discreto("rpm", RPM_MIN, RPM_MAX)
        vel_cmap, vel_norm, vel_limites = criar_cmap_discreto("vel", VEL_MIN, VEL_MAX)

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

            base_fazenda = base_fazenda.to_crs(epsg=31983)
            gdf_pts = gdf_pts.to_crs(epsg=31983)

            geom_fazenda = unary_union(base_fazenda.geometry)

            # =========================
            # CRIAÇÃO DAS LINHAS + MÉDIA RPM/VEL POR TRECHO
            # =========================
            linhas = []
            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha_atual = []
                rpm_atual = []
                vel_atual = []
                ultimo_tempo = None

                for _, row in grupo.iterrows():
                    if ultimo_tempo is None:
                        linha_atual = [row.geometry]
                        rpm_atual = [row["vl_rpm"]]
                        vel_atual = [row["vl_velocidade"]]
                    else:
                        delta = (row["dt_hr_local_inicial"] - ultimo_tempo).total_seconds()
                        if delta <= TEMPO_MAX_SEG:
                            linha_atual.append(row.geometry)
                            rpm_atual.append(row["vl_rpm"])
                            vel_atual.append(row["vl_velocidade"])
                        else:
                            if len(linha_atual) >= 2:
                                linhas.append({
                                    "geometry": LineString(linha_atual),
                                    "rpm_medio": float(np.nanmean(rpm_atual)) if len(rpm_atual) else np.nan,
                                    "vel_media": float(np.nanmean(vel_atual)) if len(vel_atual) else np.nan
                                })
                            linha_atual = [row.geometry]
                            rpm_atual = [row["vl_rpm"]]
                            vel_atual = [row["vl_velocidade"]]
                    ultimo_tempo = row["dt_hr_local_inicial"]

                if len(linha_atual) >= 2:
                    linhas.append({
                        "geometry": LineString(linha_atual),
                        "rpm_medio": float(np.nanmean(rpm_atual)) if len(rpm_atual) else np.nan,
                        "vel_media": float(np.nanmean(vel_atual)) if len(vel_atual) else np.nan
                    })

            if not linhas:
                continue

            gdf_linhas = gpd.GeoDataFrame(linhas, crs=base_fazenda.crs)

            # =========================
            # CÁLCULO DE ÁREA (PRESERVADO)
            # =========================
            largura_media = df_faz["vl_largura_implemento"].dropna().mean()
            if pd.isna(largura_media):
                continue

            largura_final = largura_media * MULTIPLICADOR_BUFFER

            buffer_linhas = gdf_linhas.geometry.buffer(largura_final / 2)

            area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
            area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

            area_total_ha = round(base_fazenda.geometry.area.sum() / 10000, 2)
            area_trab_ha = round(area_trabalhada.area / 10000, 2)
            area_nao_ha = round(area_nao_trabalhada.area / 10000, 2)

            if area_trab_ha < AREA_MIN_HA:
                continue

            pct_trab = round(area_trab_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0
            pct_nao = round(area_nao_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0

            dt_min = df_faz["dt_hr_local_inicial"].min()
            dt_max = df_faz["dt_hr_local_inicial"].max()

            periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M")
            periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M")

            # =========================
            # TABELA GLEBA / TALHÃO (PRESERVADA)
            # =========================
            df_talhoes = None

            if MOSTRAR_TALHOES and "TALHAO" in base_fazenda.columns and "GLEBA" in base_fazenda.columns:

                base_tmp = base_fazenda.copy()

                base_tmp["Área total (ha)"] = base_tmp.geometry.area / 10000

                total = base_tmp[["GLEBA", "TALHAO", "Área total (ha)"]]

                intersec = gpd.overlay(
                    base_tmp,
                    gpd.GeoDataFrame(geometry=[area_trabalhada], crs=base_tmp.crs),
                    how="intersection"
                )

                if not intersec.empty:
                    intersec["Área trabalhada (ha)"] = intersec.geometry.area / 10000
                    trab = intersec.groupby(["GLEBA", "TALHAO"])["Área trabalhada (ha)"].sum().reset_index()
                else:
                    trab = pd.DataFrame(columns=["GLEBA", "TALHAO", "Área trabalhada (ha)"])

                df_talhoes = total.merge(trab, on=["GLEBA", "TALHAO"], how="left")

                df_talhoes["Área trabalhada (ha)"] = df_talhoes["Área trabalhada (ha)"].fillna(0)

                df_talhoes = df_talhoes.rename(columns={
                    "GLEBA": "Gleba",
                    "TALHAO": "Talhão"
                })

                df_talhoes = df_talhoes[
                    ["Gleba", "Talhão", "Área total (ha)", "Área trabalhada (ha)"]
                ]

                total_row = pd.DataFrame({
                    "Gleba": ["TOTAL"],
                    "Talhão": [""],
                    "Área total (ha)": [df_talhoes["Área total (ha)"].sum().round(2)],
                    "Área trabalhada (ha)": [df_talhoes["Área trabalhada (ha)"].sum().round(2)]
                })

                df_talhoes = pd.concat([df_talhoes, total_row], ignore_index=True)

            # =========================
            # ESTATÍSTICAS RPM / VELOCIDADE
            # =========================
            rpm_validos = df_faz["vl_rpm"].dropna()
            vel_validos = df_faz["vl_velocidade"].dropna()

            rpm_min_real = round(rpm_validos.min(), 0) if not rpm_validos.empty else np.nan
            rpm_max_real = round(rpm_validos.max(), 0) if not rpm_validos.empty else np.nan
            rpm_med_real = round(rpm_validos.mean(), 0) if not rpm_validos.empty else np.nan

            vel_min_real = round(vel_validos.min(), 1) if not vel_validos.empty else np.nan
            vel_max_real = round(vel_validos.max(), 1) if not vel_validos.empty else np.nan
            vel_med_real = round(vel_validos.mean(), 1) if not vel_validos.empty else np.nan

            # =========================
            # EXPANDER COM 1, 2 OU 3 MAPAS
            # =========================
            with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):

                # -----------------------------------
                # 1) MAPA DE ÁREA TRABALHADA
                # -----------------------------------
                if MAPA_AREA:
                    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.15, right=0.83, bottom=0.25, top=0.88)

                    # base + área trabalhada
                    base_fazenda.plot(ax=ax, facecolor=COR_NAO_TRAB, edgecolor="black", linewidth=1.2)
                    gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(
                        ax=ax, color=COR_TRABALHADA, alpha=0.9
                    )
                    base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.2)

                    if "TALHAO" in base_fazenda.columns:
                        for _, row in base_fazenda.iterrows():
                            if row.geometry.is_empty:
                                continue
                            centroid = row.geometry.centroid
                            ax.text(
                                centroid.x,
                                centroid.y,
                                str(row["TALHAO"]),
                                fontsize=7,
                                ha="center",
                                va="center",
                                color="black",
                                weight="bold"
                            )

                    ax.axis("off")

                    pos = ax.get_position()
                    centro_mapa = (pos.x0 + pos.x1) / 2
                    base_y = pos.y0

                    ax.legend(
                        handles=[
                            mpatches.Patch(color=COR_TRABALHADA, label="Área trabalhada"),
                            mpatches.Patch(color=COR_NAO_TRAB, label="Área não trabalhada"),
                            mpatches.Patch(facecolor="none", edgecolor="black", label="Limites da fazenda"),
                        ],
                        loc="lower center",
                        bbox_to_anchor=(centro_mapa, base_y - 0.35),
                        ncol=3,
                        frameon=True,
                        fontsize=13
                    )

                    resumo_area = (
                        f"Resumo da operação\n\n"
                        f"Fazenda: {FAZENDA_ID} – {nome_fazenda}\n\n"
                        f"Área total: {area_total_ha} ha\n"
                        f"Trabalhada: {area_trab_ha} ha ({pct_trab}%)\n"
                        f"Não trabalhada: {area_nao_ha} ha ({pct_nao}%)\n\n"
                        f"Período:\n{periodo_ini} até {periodo_fim}"
                    )

                    adicionar_resumo(fig, pos.x1 + 0.02, 0.50, resumo_area, COR_CAIXA)
                    adicionar_footer(fig, centro_mapa, base_y, COR_RODAPE)

                    st.pyplot(fig)

                # -----------------------------------
                # 2) MAPA DE RPM
                # -----------------------------------
                if MAPA_RPM:
                    fig_rpm, ax_rpm = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.15, right=0.80, bottom=0.25, top=0.88)

                    desenhar_base_mapa(ax_rpm, base_fazenda, mostrar_talhoes=True, facecolor="#f7f7f7")

                    gdf_plot_rpm = gdf_linhas.dropna(subset=["rpm_medio"]).copy()

                    if not gdf_plot_rpm.empty:
                        gdf_plot_rpm.plot(
                            ax=ax_rpm,
                            column="rpm_medio",
                            cmap=rpm_cmap,
                            norm=rpm_norm,
                            linewidth=3.0
                        )

                    pos_rpm = ax_rpm.get_position()
                    centro_mapa_rpm = (pos_rpm.x0 + pos_rpm.x1) / 2
                    base_y_rpm = pos_rpm.y0

                    adicionar_colorbar(
                        fig_rpm,
                        pos_rpm,
                        rpm_cmap,
                        rpm_norm,
                        rpm_limites,
                        "RPM"
                    )

                    resumo_rpm = (
                        f"Resumo de RPM\n\n"
                        f"Fazenda: {FAZENDA_ID} – {nome_fazenda}\n\n"
                        f"RPM mínimo trabalhado: {int(rpm_min_real) if not pd.isna(rpm_min_real) else '-'}\n"
                        f"RPM médio trabalhado: {int(rpm_med_real) if not pd.isna(rpm_med_real) else '-'}\n"
                        f"RPM máximo trabalhado: {int(rpm_max_real) if not pd.isna(rpm_max_real) else '-'}\n\n"
                        f"Faixa exibida:\n{RPM_MIN} até {RPM_MAX}+\n\n"
                        f"Período:\n{periodo_ini} até {periodo_fim}"
                    )

                    adicionar_resumo(fig_rpm, pos_rpm.x1 + 0.05, 0.47, resumo_rpm, COR_CAIXA)
                    adicionar_footer(fig_rpm, centro_mapa_rpm, base_y_rpm, COR_RODAPE)

                    st.pyplot(fig_rpm)

                # -----------------------------------
                # 3) MAPA DE VELOCIDADE
                # -----------------------------------
                if MAPA_VEL:
                    fig_vel, ax_vel = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.15, right=0.80, bottom=0.25, top=0.88)

                    desenhar_base_mapa(ax_vel, base_fazenda, mostrar_talhoes=True, facecolor="#f7f7f7")

                    gdf_plot_vel = gdf_linhas.dropna(subset=["vel_media"]).copy()

                    if not gdf_plot_vel.empty:
                        gdf_plot_vel.plot(
                            ax=ax_vel,
                            column="vel_media",
                            cmap=vel_cmap,
                            norm=vel_norm,
                            linewidth=3.0
                        )

                    pos_vel = ax_vel.get_position()
                    centro_mapa_vel = (pos_vel.x0 + pos_vel.x1) / 2
                    base_y_vel = pos_vel.y0

                    adicionar_colorbar(
                        fig_vel,
                        pos_vel,
                        vel_cmap,
                        vel_norm,
                        vel_limites,
                        "Velocidade (km/h)"
                    )

                    resumo_vel = (
                        f"Resumo de Velocidade\n\n"
                        f"Fazenda: {FAZENDA_ID} – {nome_fazenda}\n\n"
                        f"Velocidade mínima trabalhada: {vel_min_real if not pd.isna(vel_min_real) else '-'} km/h\n"
                        f"Velocidade média trabalhada: {vel_med_real if not pd.isna(vel_med_real) else '-'} km/h\n"
                        f"Velocidade máxima trabalhada: {vel_max_real if not pd.isna(vel_max_real) else '-'} km/h\n\n"
                        f"Faixa exibida:\n{VEL_MIN:.1f} até {VEL_MAX:.1f}+\n\n"
                        f"Período:\n{periodo_ini} até {periodo_fim}"
                    )

                    adicionar_resumo(fig_vel, pos_vel.x1 + 0.05, 0.47, resumo_vel, COR_CAIXA)
                    adicionar_footer(fig_vel, centro_mapa_vel, base_y_vel, COR_RODAPE)

                    st.pyplot(fig_vel)

                # -----------------------------------
                # TABELA DE TALHÕES (PRESERVADA)
                # -----------------------------------
                if df_talhoes is not None:
                    st.markdown("### 🌾 Área por Gleba / Talhão")
                    st.dataframe(df_talhoes, use_container_width=True, hide_index=True)

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
