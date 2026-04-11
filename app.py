# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC)
# Desenvolvido por Kauã Ceconello

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.cm import ScalarMappable

import zipfile
import tempfile
import os
import pytz
from datetime import datetime

# =========================
# FUNÇÕES AUXILIARES
# =========================
def arredondar_para_baixo(valor, base):
    return np.floor(valor / base) * base


def arredondar_para_cima(valor, base):
    return np.ceil(valor / base) * base


def formatar_numero(valor, casas=0):
    if pd.isna(valor):
        return "-"
    if casas == 0:
        return f"{int(round(valor))}"
    return f"{valor:.{casas}f}".replace(".", ",")


def gerar_faixas(vmin, vmax, passo, casas=0):
    """
    Gera faixas arredondadas:
    exemplo RPM -> 1700 a 1800, 1800 a 1900, ...
    e uma faixa final '2000+'
    """
    inicio = arredondar_para_baixo(vmin, passo)
    fim = arredondar_para_cima(vmax, passo)

    limites = np.arange(inicio, fim + passo, passo)

    faixas = []
    for i in range(len(limites) - 1):
        a = limites[i]
        b = limites[i + 1]
        if casas == 0:
            label = f"{int(a)} a {int(b)}"
        else:
            label = f"{a:.{casas}f} a {b:.{casas}f}".replace(".", ",")
        faixas.append((a, b, label))

    # faixa overflow
    if casas == 0:
        label_over = f"{int(fim)}+"
    else:
        label_over = f"{fim:.{casas}f}+".replace(".", ",")
    faixas.append((fim, np.inf, label_over))

    return faixas


def criar_cmap_smooth(tipo="rpm"):
    """
    Colormaps mais suaves / smooth
    """
    if tipo == "rpm":
        cores = [
            "#08306b",  # azul escuro
            "#2171b5",  # azul
            "#6baed6",  # azul claro
            "#74c476",  # verde claro
            "#31a354",  # verde ideal
            "#fed976",  # amarelo
            "#fd8d3c",  # laranja
            "#e31a1c",  # vermelho
        ]
    else:
        cores = [
            "#08519c",  # azul escuro
            "#3182bd",  # azul
            "#6baed6",  # azul claro
            "#66c2a4",  # verde água
            "#2ca25f",  # verde
            "#d9ef8b",  # amarelo esverdeado
            "#fdae61",  # laranja
            "#d73027",  # vermelho
        ]

    cmap = LinearSegmentedColormap.from_list(f"custom_{tipo}", cores, N=256)
    cmap.set_under(cores[0])
    cmap.set_over(cores[-1])
    return cmap


def adicionar_geometrias_clipadas(geom_clip, rpm_medio, vel_media, largura_media_linha, duracao_seg, linhas_saida):
    """
    Adiciona geometrias LineString ao array final, já clipadas no limite da fazenda.
    """
    if geom_clip.is_empty:
        return

    if geom_clip.geom_type == "LineString":
        if geom_clip.length > 0:
            linhas_saida.append({
                "geometry": geom_clip,
                "rpm_medio": rpm_medio,
                "vel_media": vel_media,
                "largura_media_linha": largura_media_linha,
                "duracao_seg": duracao_seg
            })

    elif geom_clip.geom_type == "MultiLineString":
        for parte in geom_clip.geoms:
            if parte.length > 0:
                linhas_saida.append({
                    "geometry": parte,
                    "rpm_medio": rpm_medio,
                    "vel_media": vel_media,
                    "largura_media_linha": largura_media_linha,
                    "duracao_seg": duracao_seg
                })


def calcular_largura_plot(gdf_linhas, largura_ref):
    """
    Converte largura do implemento em espessura visual da linha no matplotlib.
    """
    if pd.isna(largura_ref) or largura_ref <= 0:
        largura_ref = 1.0

    escala = gdf_linhas["largura_media_linha"].fillna(largura_ref) / largura_ref
    lw = 1.5 + (escala * 2.0)
    lw = lw.clip(lower=1.2, upper=6.0)
    return lw


def plotar_linhas_heatmap(ax, gdf_plot, coluna_valor, cmap, norm):
    """
    Plota cada linha individualmente para permitir largura variável
    baseada em vl_largura_implemento.
    """
    for _, row in gdf_plot.iterrows():
        valor = row[coluna_valor]
        if pd.isna(valor):
            continue

        cor = cmap(norm(valor))
        gpd.GeoSeries([row.geometry], crs=gdf_plot.crs).plot(
            ax=ax,
            color=[cor],
            linewidth=float(row["plot_lw"]),
            alpha=0.98
        )


def calcular_percentual_tempo_por_faixa(gdf_linhas, coluna_valor, faixas, cmap, norm, casas=0):
    """
    Calcula % do tempo trabalhado por faixa.
    Usa a duração da linha como peso.
    """
    dados = gdf_linhas.dropna(subset=[coluna_valor]).copy()

    if dados.empty:
        return pd.DataFrame(columns=["faixa", "percentual", "cor"])

    if "duracao_seg" not in dados.columns:
        dados["duracao_seg"] = 0

    total_dur = dados["duracao_seg"].fillna(0).sum()

    # fallback: se não houver duração válida, usa quantidade de linhas
    usar_qtd = total_dur <= 0

    linhas_resumo = []

    for a, b, label in faixas:
        if np.isinf(b):
            mask = dados[coluna_valor] >= a
            valor_cor = a + 0.0001
        else:
            mask = (dados[coluna_valor] >= a) & (dados[coluna_valor] < b)
            valor_cor = (a + b) / 2

        subset = dados[mask]

        if usar_qtd:
            percentual = (len(subset) / len(dados) * 100) if len(dados) > 0 else 0
        else:
            percentual = (subset["duracao_seg"].fillna(0).sum() / total_dur * 100) if total_dur > 0 else 0

        cor = to_hex(cmap(norm(valor_cor)))

        linhas_resumo.append({
            "faixa": label,
            "percentual": percentual,
            "cor": cor
        })

    return pd.DataFrame(linhas_resumo)


def desenhar_legenda_horizontal(fig, pos, df_legenda, titulo):
    """
    Desenha uma legenda horizontal estilo termômetro + % do tempo por faixa,
    ocupando a faixa onde antes ficava a legenda do mapa de área.
    """
    # eixo auxiliar embaixo do mapa
    leg_ax = fig.add_axes([pos.x0 + 0.02, pos.y0 - 0.18, pos.width * 0.96, 0.11])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    n = max(len(df_legenda), 1)
    margem = 0.01
    largura_total = 1 - 2 * margem
    largura_bloco = largura_total / n

    # título
    leg_ax.text(
        0.0, 0.98,
        f"{titulo} — Faixas e % do tempo trabalhado",
        fontsize=10,
        fontweight="bold",
        va="top"
    )

    # barra
    y_barra = 0.48
    h_barra = 0.22

    for i, row in df_legenda.reset_index(drop=True).iterrows():
        x = margem + i * largura_bloco
        w = largura_bloco * 0.96

        rect = mpatches.Rectangle(
            (x, y_barra),
            w,
            h_barra,
            facecolor=row["cor"],
            edgecolor="white",
            linewidth=1.0
        )
        leg_ax.add_patch(rect)

        # faixa
        leg_ax.text(
            x + w / 2,
            y_barra + h_barra + 0.10,
            row["faixa"],
            ha="center",
            va="bottom",
            fontsize=7
        )

        # percentual
        leg_ax.text(
            x + w / 2,
            y_barra - 0.08,
            f"{row['percentual']:.1f}%",
            ha="center",
            va="top",
            fontsize=7
        )

    # linha de base
    leg_ax.plot([margem, 1 - margem], [y_barra, y_barra], color="#888", linewidth=0.8)
    leg_ax.plot([margem, 1 - margem], [y_barra + h_barra, y_barra + h_barra], color="#888", linewidth=0.8)


def desenhar_base_mapa(ax, base_fazenda, mostrar_talhoes=True, facecolor="#f7f7f7"):
    """
    Desenha a base da fazenda + limites + rótulos dos talhões
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


def adicionar_resumo(fig, x, y, texto, cor_caixa):
    fig.text(
        x,
        y,
        texto,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.8", facecolor=cor_caixa, edgecolor="black")
    )


def adicionar_footer(fig, centro_mapa, base_y, cor_rodape):
    brasilia = pytz.timezone("America/Sao_Paulo")
    hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

    fig.text(
        centro_mapa,
        base_y - 0.23,
        "⚠️ Os resultados apresentados dependem da qualidade dos dados operacionais e geoespaciais fornecidos.",
        ha="center",
        fontsize=10,
        color=cor_rodape
    )

    fig.text(
        centro_mapa,
        base_y - 0.26,
        "Relatório elaborado com base em dados da Solinftec.",
        ha="center",
        fontsize=10,
        color=cor_rodape
    )

    fig.text(
        centro_mapa,
        base_y - 0.29,
        f"Desenvolvido por Kauã Ceconello • Gerado em {hora}",
        ha="center",
        fontsize=10,
        color=cor_rodape
    )


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

# Passos das faixas arredondadas
RPM_PASSO = 100
VEL_PASSO = 1.0

COR_TRABALHADA = "#62b27f"
COR_NAO_TRAB = "#f6b1b3"
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9

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
        # (melhoria: cada zip extrai em pasta própria, sem conflito)
        # =========================
        dfs = []

        for i, uploaded_zip in enumerate(uploaded_zips):
            zip_path = os.path.join(tmpdir, uploaded_zip.name)

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            extract_dir = os.path.join(tmpdir, f"zip_extraido_{i}")
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            csv_files = []
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(".csv"):
                        csv_files.append(os.path.join(root, file))

            if not csv_files:
                st.error(f"❌ Nenhum CSV encontrado no ZIP {uploaded_zip.name}")
                continue

            # Se houver mais de um CSV, concatena todos
            for csv_path in csv_files:
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

        # Remove coordenadas e data inválidas
        df = df.dropna(subset=["dt_hr_local_inicial", "vl_latitude_inicial", "vl_longitude_inicial"])

        if df.empty:
            st.error("❌ Nenhum ponto válido encontrado após o tratamento dos dados.")
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

        # Colormaps smooth
        rpm_cmap = criar_cmap_smooth("rpm")
        vel_cmap = criar_cmap_smooth("vel")

        rpm_norm = Normalize(vmin=RPM_MIN, vmax=RPM_MAX, clip=False)
        vel_norm = Normalize(vmin=VEL_MIN, vmax=VEL_MAX, clip=False)

        rpm_faixas = gerar_faixas(RPM_MIN, RPM_MAX, RPM_PASSO, casas=0)
        vel_faixas = gerar_faixas(VEL_MIN, VEL_MAX, VEL_PASSO, casas=1)

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
            # CRIAÇÃO DAS LINHAS + MÉDIAS RPM/VEL + LARGURA + DURAÇÃO
            # agora clipadas no limite da fazenda
            # =========================
            linhas = []

            for _, grupo in gdf_pts.groupby("cd_equipamento"):
                grupo = grupo.sort_values("dt_hr_local_inicial")

                linha_atual = []
                rpm_atual = []
                vel_atual = []
                largura_atual = []
                ultimo_tempo = None
                tempo_inicio_seg = None

                for _, row in grupo.iterrows():
                    tempo_atual = row["dt_hr_local_inicial"]

                    if ultimo_tempo is None:
                        linha_atual = [row.geometry]
                        rpm_atual = [row["vl_rpm"]]
                        vel_atual = [row["vl_velocidade"]]
                        largura_atual = [row["vl_largura_implemento"]]
                        tempo_inicio_seg = tempo_atual
                    else:
                        delta = (tempo_atual - ultimo_tempo).total_seconds()

                        if delta <= TEMPO_MAX_SEG:
                            linha_atual.append(row.geometry)
                            rpm_atual.append(row["vl_rpm"])
                            vel_atual.append(row["vl_velocidade"])
                            largura_atual.append(row["vl_largura_implemento"])
                        else:
                            if len(linha_atual) >= 2:
                                geom_linha = LineString(linha_atual)
                                geom_clip = geom_linha.intersection(geom_fazenda)

                                rpm_medio = float(np.nanmean(rpm_atual)) if len(rpm_atual) else np.nan
                                vel_media = float(np.nanmean(vel_atual)) if len(vel_atual) else np.nan
                                largura_media_linha = float(np.nanmean(largura_atual)) if len(largura_atual) else np.nan
                                duracao_seg = (ultimo_tempo - tempo_inicio_seg).total_seconds() if tempo_inicio_seg is not None else np.nan

                                adicionar_geometrias_clipadas(
                                    geom_clip,
                                    rpm_medio,
                                    vel_media,
                                    largura_media_linha,
                                    duracao_seg,
                                    linhas
                                )

                            # reinicia segmento
                            linha_atual = [row.geometry]
                            rpm_atual = [row["vl_rpm"]]
                            vel_atual = [row["vl_velocidade"]]
                            largura_atual = [row["vl_largura_implemento"]]
                            tempo_inicio_seg = tempo_atual

                    ultimo_tempo = tempo_atual

                # fecha último segmento
                if len(linha_atual) >= 2:
                    geom_linha = LineString(linha_atual)
                    geom_clip = geom_linha.intersection(geom_fazenda)

                    rpm_medio = float(np.nanmean(rpm_atual)) if len(rpm_atual) else np.nan
                    vel_media = float(np.nanmean(vel_atual)) if len(vel_atual) else np.nan
                    largura_media_linha = float(np.nanmean(largura_atual)) if len(largura_atual) else np.nan
                    duracao_seg = (ultimo_tempo - tempo_inicio_seg).total_seconds() if tempo_inicio_seg is not None else np.nan

                    adicionar_geometrias_clipadas(
                        geom_clip,
                        rpm_medio,
                        vel_media,
                        largura_media_linha,
                        duracao_seg,
                        linhas
                    )

            if not linhas:
                continue

            gdf_linhas = gpd.GeoDataFrame(linhas, crs=base_fazenda.crs)

            # espessura visual das linhas baseada no vl_largura_implemento
            largura_ref = df_faz["vl_largura_implemento"].dropna().mean()
            gdf_linhas["plot_lw"] = calcular_largura_plot(gdf_linhas, largura_ref)

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

            # Legendas horizontais (% do tempo por faixa)
            df_leg_rpm = calcular_percentual_tempo_por_faixa(
                gdf_linhas,
                "rpm_medio",
                rpm_faixas,
                rpm_cmap,
                rpm_norm,
                casas=0
            )

            df_leg_vel = calcular_percentual_tempo_por_faixa(
                gdf_linhas,
                "vel_media",
                vel_faixas,
                vel_cmap,
                vel_norm,
                casas=1
            )

            # =========================
            # EXPANDER COM 1, 2 OU 3 MAPAS
            # =========================
            with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):

                # -----------------------------------
                # 1) MAPA DE ÁREA TRABALHADA
                # -----------------------------------
                if MAPA_AREA:
                    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.33, top=0.88)

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

                    # legenda original preservada
                    ax.legend(
                        handles=[
                            mpatches.Patch(color=COR_TRABALHADA, label="Área trabalhada"),
                            mpatches.Patch(color=COR_NAO_TRAB, label="Área não trabalhada"),
                            mpatches.Patch(facecolor="none", edgecolor="black", label="Limites da fazenda"),
                        ],
                        loc="lower center",
                        bbox_to_anchor=(centro_mapa, base_y - 0.40),
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
                    plt.subplots_adjust(left=0.15, right=0.83, bottom=0.33, top=0.88)

                    desenhar_base_mapa(
                        ax_rpm,
                        base_fazenda,
                        mostrar_talhoes=True,
                        facecolor="#f7f7f7"
                    )

                    gdf_plot_rpm = gdf_linhas.dropna(subset=["rpm_medio"]).copy()

                    if not gdf_plot_rpm.empty:
                        plotar_linhas_heatmap(
                            ax_rpm,
                            gdf_plot_rpm,
                            "rpm_medio",
                            rpm_cmap,
                            rpm_norm
                        )

                    pos_rpm = ax_rpm.get_position()
                    centro_mapa_rpm = (pos_rpm.x0 + pos_rpm.x1) / 2
                    base_y_rpm = pos_rpm.y0

                    # legenda horizontal no lugar da antiga legenda
                    desenhar_legenda_horizontal(
                        fig_rpm,
                        pos_rpm,
                        df_leg_rpm,
                        "RPM"
                    )

                    resumo_rpm = (
                        f"Resumo de RPM\n\n"
                        f"Fazenda: {FAZENDA_ID} – {nome_fazenda}\n\n"
                        f"RPM mínimo trabalhado: {formatar_numero(rpm_min_real, 0)}\n"
                        f"RPM médio trabalhado: {formatar_numero(rpm_med_real, 0)}\n"
                        f"RPM máximo trabalhado: {formatar_numero(rpm_max_real, 0)}\n\n"
                        f"Faixa exibida:\n{int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))} até {int(arredondar_para_cima(RPM_MAX, RPM_PASSO))}+\n\n"
                        f"Período:\n{periodo_ini} até {periodo_fim}"
                    )

                    adicionar_resumo(fig_rpm, pos_rpm.x1 + 0.03, 0.47, resumo_rpm, COR_CAIXA)
                    adicionar_footer(fig_rpm, centro_mapa_rpm, base_y_rpm, COR_RODAPE)

                    st.pyplot(fig_rpm)

                # -----------------------------------
                # 3) MAPA DE VELOCIDADE
                # -----------------------------------
                if MAPA_VEL:
                    fig_vel, ax_vel = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.15, right=0.83, bottom=0.33, top=0.88)

                    desenhar_base_mapa(
                        ax_vel,
                        base_fazenda,
                        mostrar_talhoes=True,
                        facecolor="#f7f7f7"
                    )

                    gdf_plot_vel = gdf_linhas.dropna(subset=["vel_media"]).copy()

                    if not gdf_plot_vel.empty:
                        plotar_linhas_heatmap(
                            ax_vel,
                            gdf_plot_vel,
                            "vel_media",
                            vel_cmap,
                            vel_norm
                        )

                    pos_vel = ax_vel.get_position()
                    centro_mapa_vel = (pos_vel.x0 + pos_vel.x1) / 2
                    base_y_vel = pos_vel.y0

                    # legenda horizontal no lugar da antiga legenda
                    desenhar_legenda_horizontal(
                        fig_vel,
                        pos_vel,
                        df_leg_vel,
                        "Velocidade (km/h)"
                    )

                    resumo_vel = (
                        f"Resumo de Velocidade\n\n"
                        f"Fazenda: {FAZENDA_ID} – {nome_fazenda}\n\n"
                        f"Velocidade mínima trabalhada: {formatar_numero(vel_min_real, 1)} km/h\n"
                        f"Velocidade média trabalhada: {formatar_numero(vel_med_real, 1)} km/h\n"
                        f"Velocidade máxima trabalhada: {formatar_numero(vel_max_real, 1)} km/h\n\n"
                        f"Faixa exibida:\n{formatar_numero(arredondar_para_baixo(VEL_MIN, VEL_PASSO), 1)} até {formatar_numero(arredondar_para_cima(VEL_MAX, VEL_PASSO), 1)}+\n\n"
                        f"Período:\n{periodo_ini} até {periodo_fim}"
                    )

                    adicionar_resumo(fig_vel, pos_vel.x1 + 0.03, 0.47, resumo_vel, COR_CAIXA)
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
