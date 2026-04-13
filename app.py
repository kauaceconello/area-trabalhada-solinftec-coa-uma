# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC)
# Desenvolvido por Kauã Ceconello

import io
import os
import zipfile
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pytz

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import FancyBboxPatch

from shapely.geometry import LineString
from shapely.ops import unary_union


# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Área Trabalhada – Solinftec",
    layout="wide"
)

if "mapas_gerados" not in st.session_state:
    st.session_state["mapas_gerados"] = False


# =========================================================
# ESTILO GLOBAL
# =========================================================
st.markdown(
    """
    <style>
    /* =====================================================
       FUNDO GERAL - REMOVE O BRANCO
       ===================================================== */
    html, body, [data-testid="stApp"] {
        background: linear-gradient(180deg, #020617 0%, #0B1020 50%, #0F172A 100%) !important;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    section.main {
        background: transparent !important;
    }

    section.main > div {
        background: transparent !important;
    }

    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* =====================================================
       TIPOGRAFIA GLOBAL
       ===================================================== */
    * {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }

    p, span, small {
        color: #CBD5E1;
    }

    .stMarkdown p {
        color: #CBD5E1 !important;
    }

    .stCaption {
        color: #94A3B8 !important;
    }

    /* =====================================================
       HERO / TOPO
       ===================================================== */
    .hero-card {
        background: linear-gradient(135deg, #0B1020 0%, #111827 100%);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 14px;
        box-shadow: 0 16px 36px rgba(0,0,0,0.32);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 8px;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 0.96rem;
        color: #CBD5E1;
        line-height: 1.5;
        margin-bottom: 0;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0B1020 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 8px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.30);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #F8FAFC !important;
    }

    /* =====================================================
       INPUTS / PARÂMETROS
       ===================================================== */
    label {
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
    }

    input, textarea {
        background-color: #020617 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #64748B !important;
    }

    [data-testid="stNumberInput"] input {
        background-color: #020617 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    [data-testid="stNumberInput"] button {
        background: rgba(255,255,255,0.04) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: #F8FAFC !important;
    }

    [data-baseweb="select"] > div {
        background-color: #020617 !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    [data-baseweb="select"] * {
        color: #F8FAFC !important;
    }

    [data-testid="stCheckbox"] label {
        color: #F8FAFC !important;
    }

    /* =====================================================
       BOTÕES
       ===================================================== */
    div.stButton > button {
        width: 100%;
        height: 3.05em;
        font-size: 1.0em;
        font-weight: 700;
        border-radius: 12px;
        border: 1px solid #2563EB;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
        transition: all 0.18s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.45);
    }

    /* =====================================================
       CARD DE ENVIO
       ===================================================== */
    .upload-hero {
        background: linear-gradient(135deg, #0B1020 0%, #111827 100%);
        border: 1px solid rgba(37, 99, 235, 0.22);
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.28);
    }

    .upload-hero-title {
        font-size: 1.04rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
    }

    .upload-hero-text {
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.45;
    }

    /* =====================================================
       UPLOADERS
       ===================================================== */
    [data-testid="stFileUploader"] {
        background: linear-gradient(180deg, #0B1020 0%, #111827 100%) !important;
        border: 1px solid rgba(96, 165, 250, 0.22) !important;
        border-radius: 16px !important;
        padding: 10px !important;
        box-shadow: 0 10px 28px rgba(0,0,0,0.28);
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #E5E7EB !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px dashed rgba(96,165,250,0.45) !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #DCE7F5 !important;
        fill: #93C5FD !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
    }

    [data-testid="stFileUploaderFile"] * {
        color: #F8FAFC !important;
        fill: #93C5FD !important;
    }

    [data-testid="stFileUploader"] button[kind="secondary"] {
        background: rgba(255,255,255,0.05) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(96,165,250,0.35) !important;
    }

    [data-testid="stFileUploader"] svg {
        fill: #93C5FD !important;
    }

    /* =====================================================
       EXPANDERS / ALERTAS / DATAFRAME
       ===================================================== */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.28);
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: #F8FAFC !important;
    }

    [data-testid="stDataFrame"] {
        background: #020617 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    [data-testid="stInfo"],
    [data-testid="stWarning"],
    [data-testid="stError"],
    [data-testid="stSuccess"] {
        background: rgba(255,255,255,0.06) !important;
        color: #E5E7EB !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO / TOPO DA PÁGINA
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📍 Área Trabalhada – Solinftec</div>
        <div class="hero-subtitle">
            Aplicação para cálculo e visualização da área trabalhada com base em dados operacionais da Solinftec
            e base cartográfica da Usina Monte Alegre.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def sidebar_container():
    """
    Compatibilidade com versões do Streamlit.
    """
    try:
        return st.sidebar.container(border=True)
    except TypeError:
        return st.sidebar.container()


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


def validar_colunas(df, colunas_obrigatorias):
    return [c for c in colunas_obrigatorias if c not in df.columns]


def gerar_faixas(vmin, vmax, passo, casas=0):
    """
    Gera faixas arredondadas incluindo:
    - abaixo do mínimo
    - faixas internas
    - acima do máximo
    """
    inicio = arredondar_para_baixo(vmin, passo)
    fim = arredondar_para_cima(vmax, passo)

    edges = np.arange(inicio, fim + passo, passo)
    faixas = []

    # Abaixo do mínimo
    if casas == 0:
        label_under = f"< {int(inicio)}"
    else:
        label_under = f"< {inicio:.{casas}f}".replace(".", ",")
    faixas.append((-np.inf, inicio, label_under))

    # Faixas internas
    for i in range(len(edges) - 1):
        a = edges[i]
        b = edges[i + 1]

        if casas == 0:
            label = f"{int(a)} a {int(b)}"
        else:
            label = f"{a:.{casas}f} a {b:.{casas}f}".replace(".", ",")

        faixas.append((a, b, label))

    # Acima do máximo
    if casas == 0:
        label_over = f"{int(fim)}+"
    else:
        label_over = f"{fim:.{casas}f}+".replace(".", ",")
    faixas.append((fim, np.inf, label_over))

    return faixas


def criar_cmap_suave(tipo="rpm"):
    """
    Paletas vivas e harmônicas.
    """
    if tipo == "rpm":
        cores = [
            "#0B4F8A",
            "#1F78B4",
            "#2D9CDB",
            "#1FBBA6",
            "#20B15A",
            "#8FD14F",
            "#F2C94C",
            "#F2994A",
            "#E05A47",
        ]
    else:
        cores = [
            "#0A5E8A",
            "#1479C9",
            "#16A5C8",
            "#18B7B2",
            "#1DBE6B",
            "#86D44E",
            "#DCEB46",
            "#F4C542",
            "#F59E32",
            "#E1594F",
        ]
    return LinearSegmentedColormap.from_list(f"cmap_{tipo}", cores, N=256)


def amostrar_cores_classes(cmap, n_classes):
    if n_classes <= 1:
        return [to_hex(cmap(0.55))]
    pontos = np.linspace(0.10, 0.98, n_classes)
    return [to_hex(cmap(x)) for x in pontos]


def classificar_valor(valor, faixas):
    if pd.isna(valor):
        return None

    for a, b, label in faixas:
        if np.isneginf(a):
            if valor < b:
                return label
        elif np.isinf(b):
            if valor >= a:
                return label
        else:
            if a <= valor < b:
                return label
    return None


def figura_para_pdf_bytes(fig):
    """
    Exporta a figura em PDF vetorial.
    """
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="pdf",
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    buffer.seek(0)
    return buffer.getvalue()


def ler_csv_robusto(csv_path):
    tentativas = [
        {"sep": ";", "encoding": "latin1"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
    ]

    ultimo_erro = None
    for cfg in tentativas:
        try:
            return pd.read_csv(csv_path, engine="python", **cfg)
        except Exception as e:
            ultimo_erro = e

    raise ultimo_erro


def ler_csvs_de_zip(uploaded_zip, tmpdir, idx_zip):
    zip_path = os.path.join(tmpdir, uploaded_zip.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())

    extract_dir = os.path.join(tmpdir, f"zip_extraido_{idx_zip}")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    csv_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    return csv_files


def adicionar_segmento_clipado(
    linhas_saida,
    pontos,
    rpms,
    vels,
    larguras,
    t_inicio,
    t_fim,
    geom_fazenda
):
    """
    Cria a linha do segmento, clipa na fazenda e, se virar MultiLineString,
    rateia a duração pelos comprimentos clipados.
    """
    if len(pontos) < 2:
        return

    try:
        linha = LineString(pontos)
    except Exception:
        return

    if linha.is_empty or linha.length == 0:
        return

    linha_clip = linha.intersection(geom_fazenda)

    if linha_clip.is_empty:
        return

    rpm_medio = float(np.nanmean(rpms)) if len(rpms) else np.nan
    vel_media = float(np.nanmean(vels)) if len(vels) else np.nan
    largura_media = float(np.nanmean(larguras)) if len(larguras) else np.nan
    duracao_seg = (
        (t_fim - t_inicio).total_seconds()
        if t_inicio is not None and t_fim is not None else np.nan
    )

    geoms = []
    if linha_clip.geom_type == "LineString":
        geoms = [linha_clip]
    elif linha_clip.geom_type == "MultiLineString":
        geoms = [g for g in linha_clip.geoms if not g.is_empty and g.length > 0]

    comprimento_total_clipado = sum(g.length for g in geoms) if geoms else 0

    for geom in geoms:
        if geom.is_empty or geom.length == 0:
            continue

        if pd.notna(duracao_seg) and comprimento_total_clipado > 0:
            duracao_rateada = duracao_seg * (geom.length / comprimento_total_clipado)
        else:
            duracao_rateada = np.nan

        linhas_saida.append({
            "geometry": geom,
            "rpm_medio": rpm_medio,
            "vel_media": vel_media,
            "largura_media": largura_media,
            "duracao_seg": duracao_rateada
        })


def criar_poligonos_display(gdf_linhas, geom_fazenda):
    """
    Cria a faixa visual dos mapas temáticos (RPM / Velocidade)
    usando EXATAMENTE a largura informada no CSV, sem multiplicador extra.
    """
    registros = []

    for _, row in gdf_linhas.iterrows():
        largura = row["largura_media"]

        if pd.isna(largura) or largura <= 0:
            continue

        try:
            geom_disp = row.geometry.buffer(
                largura / 2.0,
                cap_style=2,
                join_style=2,
                quad_segs=1
            ).intersection(geom_fazenda)
        except Exception:
            continue

        if geom_disp.is_empty:
            continue

        registros.append({
            "geometry": geom_disp,
            "rpm_medio": row["rpm_medio"],
            "vel_media": row["vel_media"],
            "largura_media": row["largura_media"],
            "duracao_seg": row["duracao_seg"]
        })

    if not registros:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf_linhas.crs)

    return gpd.GeoDataFrame(registros, geometry="geometry", crs=gdf_linhas.crs)


def calcular_legenda_percentual(gdf_display, coluna_classe, faixas, mapa_cores):
    if gdf_display is None or gdf_display.empty:
        return pd.DataFrame(columns=["cor", "inicio", "fim", "faixa", "percentual"])

    if coluna_classe not in gdf_display.columns:
        return pd.DataFrame(columns=["cor", "inicio", "fim", "faixa", "percentual"])

    dados = gdf_display.dropna(subset=[coluna_classe]).copy()

    if dados.empty:
        return pd.DataFrame(columns=["cor", "inicio", "fim", "faixa", "percentual"])

    total_tempo = dados["duracao_seg"].fillna(0).sum()
    usar_contagem = total_tempo <= 0

    linhas = []
    for a, b, label in faixas:
        subset = dados[dados[coluna_classe] == label]

        if usar_contagem:
            percentual = (len(subset) / len(dados) * 100) if len(dados) > 0 else 0
        else:
            percentual = (
                subset["duracao_seg"].fillna(0).sum() / total_tempo * 100
                if total_tempo > 0 else 0
            )

        inicio_txt = "-" if np.isneginf(a) else a
        fim_txt = "+" if np.isinf(b) else b

        linhas.append({
            "cor": mapa_cores.get(label, "#cccccc"),
            "inicio": inicio_txt,
            "fim": fim_txt,
            "faixa": label,
            "percentual": percentual
        })

    return pd.DataFrame(linhas)


# =========================================================
# FUNÇÕES DE PLOT
# =========================================================
def desenhar_base_mapa(
    ax,
    base_fazenda,
    facecolor="#FFFFFF",
    mostrar_talhoes=True,
    margem_rel_x=0.020,
    margem_rel_y=0.030
):
    """
    Mapa com zoom melhor distribuído e leitura mais limpa.
    """
    ax.set_facecolor("#F8FAFC")

    base_fazenda.plot(
        ax=ax,
        facecolor=facecolor,
        edgecolor="#334155",
        linewidth=1.0,
        zorder=1
    )

    base_fazenda.boundary.plot(
        ax=ax,
        color="#0F172A",
        linewidth=1.1,
        zorder=3
    )

    if mostrar_talhoes and "TALHAO" in base_fazenda.columns:
        for _, row in base_fazenda.iterrows():
            if row.geometry.is_empty:
                continue

            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(row["TALHAO"]),
                fontsize=7.8,
                ha="center",
                va="center",
                color="#0F172A",
                weight="bold",
                zorder=4,
                bbox=dict(
                    boxstyle="round,pad=0.14",
                    facecolor=(1, 1, 1, 0.55),
                    edgecolor="none"
                )
            )

    minx, miny, maxx, maxy = base_fazenda.total_bounds
    dx = maxx - minx
    dy = maxy - miny

    margem_x = max(dx * margem_rel_x, 0.35)
    margem_y = max(dy * margem_rel_y, 0.70)

    ax.set_xlim(minx - margem_x, maxx + margem_x)
    ax.set_ylim(miny - margem_y, maxy + margem_y)
    ax.set_aspect("equal")
    ax.axis("off")


def plotar_mapa_classes(ax, base_fazenda, gdf_plot, coluna_classe, mapa_cores, mostrar_talhoes=True):
    """
    Plota os polígonos temáticos SEM dissolve,
    preservando os trechos/faixas individualmente.
    """
    desenhar_base_mapa(
        ax,
        base_fazenda,
        facecolor="#FFFFFF",
        mostrar_talhoes=mostrar_talhoes,
        margem_rel_x=0.020,
        margem_rel_y=0.030
    )

    if gdf_plot is None or gdf_plot.empty:
        return

    gdf_tmp = gdf_plot[[coluna_classe, "geometry"]].dropna(subset=[coluna_classe]).copy()
    if gdf_tmp.empty:
        return

    for classe, cor in mapa_cores.items():
        sub = gdf_tmp[gdf_tmp[coluna_classe] == classe]
        if not sub.empty:
            sub.plot(
                ax=ax,
                color=cor,
                edgecolor="none",
                alpha=1.0,
                zorder=2
            )


def adicionar_header_topo(fig, titulo_mapa, fazenda_id, nome_fazenda, periodo_ini, periodo_fim):
    """
    Cabeçalho superior com aparência mais executiva.
    """
    ax_header = fig.add_axes([0.025, 0.905, 0.95, 0.08])
    ax_header.axis("off")

    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#FFFFFF",
        edgecolor="#D8E1EB",
        linewidth=1.0
    )
    ax_header.add_patch(card)

    ax_header.text(
        0.03, 0.62,
        titulo_mapa,
        fontsize=16,
        weight="bold",
        color="#0F172A",
        ha="left",
        va="center"
    )

    ax_header.text(
        0.03, 0.26,
        f"Fazenda {fazenda_id} • {nome_fazenda}",
        fontsize=10.2,
        color="#475569",
        ha="left",
        va="center"
    )

    ax_header.text(
        0.97, 0.50,
        f"Período: {periodo_ini} até {periodo_fim}",
        fontsize=9.6,
        color="#64748B",
        ha="right",
        va="center"
    )


def adicionar_footer(fig, cor_rodape="#64748B"):
    """
    Rodapé mais discreto e elegante.
    """
    brasilia = pytz.timezone("America/Sao_Paulo")
    hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

    fig.text(
        0.50,
        0.030,
        "Relatório elaborado com base em dados da Solinftec • Resultados dependem da qualidade dos dados operacionais e geoespaciais.",
        ha="center",
        fontsize=8.8,
        color=cor_rodape
    )

    fig.text(
        0.50,
        0.012,
        f"Desenvolvido por Kauã Ceconello • Gerado em {hora}",
        ha="center",
        fontsize=8.6,
        color=cor_rodape
    )


def desenhar_box_legenda_tematica(
    fig,
    titulo_box,
    faixa_exibida_txt,
    media_txt,
    df_legenda,
    reserve_pos=(0.71, 0.16, 0.25, 0.68)
):
    """
    Card lateral da legenda com barras de percentual.
    """
    rx, ry, rw, rh = reserve_pos

    ax_box = fig.add_axes([rx, ry, rw, rh])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.axis("off")

    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        facecolor="#FFFFFF",
        edgecolor="#D8E1EB",
        linewidth=1.0
    )
    ax_box.add_patch(card)

    ax_box.text(
        0.07, 0.945,
        titulo_box,
        fontsize=12,
        weight="bold",
        color="#0F172A",
        ha="left",
        va="center"
    )

    ax_box.text(
        0.07, 0.895,
        f"Faixa exibida: {faixa_exibida_txt}",
        fontsize=8.8,
        color="#64748B",
        ha="left",
        va="center"
    )

    chip = FancyBboxPatch(
        (0.07, 0.805),
        0.86,
        0.072,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#EFF6FF",
        edgecolor="#BFDBFE",
        linewidth=0.8
    )
    ax_box.add_patch(chip)

    ax_box.text(
        0.50, 0.841,
        media_txt,
        fontsize=10.0,
        color="#1D4ED8",
        weight="bold",
        ha="center",
        va="center"
    )

    ax_box.plot([0.07, 0.93], [0.755, 0.755], color="#E2E8F0", linewidth=1)

    if df_legenda.empty:
        ax_box.text(
            0.50, 0.55,
            "Sem dados válidos para exibir.",
            fontsize=9.2,
            color="#64748B",
            ha="center"
        )
        return

    topo = 0.695
    base = 0.08
    n = len(df_legenda)
    row_h = (topo - base) / max(n, 1)

    for i, row in df_legenda.reset_index(drop=True).iterrows():
        y = topo - i * row_h

        faixa_txt = row["faixa"]
        percentual = row["percentual"]
        pct_txt = f"{percentual:.1f}%".replace(".", ",")

        ax_box.add_patch(
            FancyBboxPatch(
                (0.07, y - 0.020),
                0.025,
                0.025,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor=row["cor"],
                edgecolor="none"
            )
        )

        ax_box.text(
            0.11, y - 0.007,
            faixa_txt,
            fontsize=8.9,
            color="#0F172A",
            ha="left",
            va="center"
        )

        ax_box.add_patch(
            FancyBboxPatch(
                (0.57, y - 0.020),
                0.25,
                0.025,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor="#E2E8F0",
                edgecolor="none"
            )
        )

        largura_barra = 0.25 * max(0, min(percentual, 100)) / 100
        ax_box.add_patch(
            FancyBboxPatch(
                (0.57, y - 0.020),
                largura_barra,
                0.025,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor=row["cor"],
                edgecolor="none"
            )
        )

        ax_box.text(
            0.86, y - 0.007,
            pct_txt,
            fontsize=8.9,
            color="#334155",
            ha="center",
            va="center",
            weight="bold"
        )

        if i < n - 1:
            ax_box.plot(
                [0.07, 0.93],
                [y - 0.045, y - 0.045],
                color="#F1F5F9",
                linewidth=0.8
            )


def criar_figura_tematica(
    base_fazenda,
    gdf_display,
    coluna_classe,
    mapa_cores,
    df_legenda,
    titulo_mapa,
    titulo_box,
    faixa_exibida_txt,
    media_txt,
    periodo_ini,
    periodo_fim,
    fazenda_id,
    nome_fazenda
):
    """
    Figura temática com mapa + card lateral de legenda.
    """
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")

    moldura = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="#D8E1EB",
        linewidth=1.0,
        zorder=0
    )
    fig.add_artist(moldura)

    adicionar_header_topo(
        fig,
        titulo_mapa=titulo_mapa,
        fazenda_id=fazenda_id,
        nome_fazenda=nome_fazenda,
        periodo_ini=periodo_ini,
        periodo_fim=periodo_fim
    )

    painel_mapa = FancyBboxPatch(
        (0.03, 0.10),
        0.64,
        0.78,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        transform=fig.transFigure,
        facecolor="#FFFFFF",
        edgecolor="#D8E1EB",
        linewidth=1.0,
        zorder=0
    )
    fig.add_artist(painel_mapa)

    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])

    if gdf_display is not None and not gdf_display.empty:
        plotar_mapa_classes(
            ax=ax,
            base_fazenda=base_fazenda,
            gdf_plot=gdf_display.dropna(subset=[coluna_classe]).copy(),
            coluna_classe=coluna_classe,
            mapa_cores=mapa_cores,
            mostrar_talhoes=True
        )
    else:
        desenhar_base_mapa(
            ax=ax,
            base_fazenda=base_fazenda,
            facecolor="#FFFFFF",
            mostrar_talhoes=True
        )

    desenhar_box_legenda_tematica(
        fig=fig,
        titulo_box=titulo_box,
        faixa_exibida_txt=faixa_exibida_txt,
        media_txt=media_txt,
        df_legenda=df_legenda,
        reserve_pos=(0.71, 0.16, 0.25, 0.68)
    )

    adicionar_footer(fig, "#64748B")
    return fig


# =========================================================
# MODELO VISUAL PARA ÁREA TRABALHADA
# =========================================================
def criar_figura_area(
    base_fazenda,
    area_trabalhada,
    area_total_ha,
    area_trab_ha,
    area_nao_ha,
    pct_trab,
    pct_nao,
    periodo_ini,
    periodo_fim,
    fazenda_id,
    nome_fazenda,
    mostrar_talhoes,
    cor_trabalhada,
    cor_nao_trab
):
    """
    Mapa de área trabalhada com visual mais executivo.
    """
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")

    moldura = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="#D8E1EB",
        linewidth=1.0,
        zorder=0
    )
    fig.add_artist(moldura)

    adicionar_header_topo(
        fig,
        titulo_mapa="Mapa de Área Trabalhada",
        fazenda_id=fazenda_id,
        nome_fazenda=nome_fazenda,
        periodo_ini=periodo_ini,
        periodo_fim=periodo_fim
    )

    painel_mapa = FancyBboxPatch(
        (0.03, 0.10),
        0.64,
        0.78,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        transform=fig.transFigure,
        facecolor="#FFFFFF",
        edgecolor="#D8E1EB",
        linewidth=1.0,
        zorder=0
    )
    fig.add_artist(painel_mapa)

    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])

    base_fazenda.plot(
        ax=ax,
        facecolor=cor_nao_trab,
        edgecolor="#334155",
        linewidth=1.0,
        zorder=1
    )

    if not area_trabalhada.is_empty:
        gpd.GeoSeries([area_trabalhada], crs=base_fazenda.crs).plot(
            ax=ax,
            color=cor_trabalhada,
            alpha=0.88,
            zorder=2
        )

    base_fazenda.boundary.plot(
        ax=ax,
        color="#0F172A",
        linewidth=1.1,
        zorder=3
    )

    if mostrar_talhoes and "TALHAO" in base_fazenda.columns:
        for _, row in base_fazenda.iterrows():
            if row.geometry.is_empty:
                continue
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(row["TALHAO"]),
                fontsize=7.8,
                ha="center",
                va="center",
                color="#0F172A",
                weight="bold",
                zorder=4,
                bbox=dict(
                    boxstyle="round,pad=0.14",
                    facecolor=(1, 1, 1, 0.55),
                    edgecolor="none"
                )
            )

    minx, miny, maxx, maxy = base_fazenda.total_bounds
    dx = maxx - minx
    dy = maxy - miny
    margem_x = max(dx * 0.020, 0.35)
    margem_y = max(dy * 0.030, 0.70)

    ax.set_xlim(minx - margem_x, maxx + margem_x)
    ax.set_ylim(miny - margem_y, maxy + margem_y)
    ax.set_aspect("equal")
    ax.axis("off")

    resumo_ax = fig.add_axes([0.71, 0.23, 0.25, 0.48])
    resumo_ax.set_xlim(0, 1)
    resumo_ax.set_ylim(0, 1)
    resumo_ax.axis("off")

    resumo_box = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        facecolor="#FFFFFF",
        edgecolor="#D8E1EB",
        linewidth=1.0
    )
    resumo_ax.add_patch(resumo_box)

    resumo_ax.text(
        0.08, 0.92,
        "Resumo da Operação",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0F172A"
    )

    resumo_ax.text(0.08, 0.77, "Área total", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.77, f"{area_total_ha} ha", fontsize=10.2, color="#0F172A", ha="right", weight="bold")

    resumo_ax.text(0.08, 0.62, "Trabalhada", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.62, f"{area_trab_ha} ha ({pct_trab}%)", fontsize=10.0, color="#16A34A", ha="right", weight="bold")

    resumo_ax.text(0.08, 0.47, "Não trabalhada", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.47, f"{area_nao_ha} ha ({pct_nao}%)", fontsize=10.0, color="#475569", ha="right", weight="bold")

    resumo_ax.add_patch(
        FancyBboxPatch(
            (0.08, 0.28),
            0.84,
            0.06,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor="#E5E7EB",
            edgecolor="none"
        )
    )

    resumo_ax.add_patch(
        FancyBboxPatch(
            (0.08, 0.28),
            0.84 * min(max(pct_trab / 100, 0), 1),
            0.06,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor=cor_trabalhada,
            edgecolor="none"
        )
    )

    resumo_ax.text(
        0.50, 0.20,
        f"Cobertura operacional: {pct_trab}%",
        fontsize=9.6,
        color="#0F172A",
        ha="center",
        weight="bold"
    )

    leg_ax = fig.add_axes([0.06, 0.09, 0.58, 0.05])
    leg_ax.axis("off")

    handles = [
        mpatches.Patch(color=cor_trabalhada, label="Área trabalhada"),
        mpatches.Patch(color=cor_nao_trab, label="Área não trabalhada"),
        mpatches.Patch(facecolor="white", edgecolor="#0F172A", label="Limites da fazenda")
    ]

    leg_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=9.8
    )

    adicionar_footer(fig, "#64748B")
    return fig


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

with sidebar_container():
    st.markdown("### 🗺️ Tipos de mapa")
    MAPA_AREA = st.checkbox("Área trabalhada", value=True, key="mapa_area_chk")
    MAPA_RPM = st.checkbox("Mapa de RPM", value=True, key="mapa_rpm_chk")
    MAPA_VEL = st.checkbox("Mapa de Velocidade", value=True, key="mapa_vel_chk")

with sidebar_container():
    MULTIPLICADOR_BUFFER = st.number_input(
        "Tamanho do Buffer",
        min_value=1.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        key="buffer_input"
    )

    AREA_MIN_HA = st.number_input(
        "Área mínima trabalhada (ha)",
        min_value=0.0,
        value=0.50,
        step=0.1,
        key="area_min_input"
    )

    if not (MAPA_RPM or MAPA_VEL):
        MOSTRAR_TALHOES = st.checkbox(
            "📊 Mostrar área por Gleba / Talhão",
            value=False,
            key="mostrar_talhoes_chk"
        )
    else:
        MOSTRAR_TALHOES = False

RPM_MIN = 1200
RPM_MAX = 2000
RPM_PASSO = 100
if MAPA_RPM:
    with sidebar_container():
        st.markdown("### ⚙️ Parâmetros RPM")
        RPM_MIN = st.number_input(
            "RPM mínimo",
            min_value=0,
            max_value=10000,
            value=1200,
            step=100,
            key="rpm_min_input"
        )
        RPM_MAX = st.number_input(
            "RPM máximo",
            min_value=0,
            max_value=10000,
            value=2000,
            step=100,
            key="rpm_max_input"
        )
        RPM_PASSO = st.number_input(
            "Passo das faixas RPM",
            min_value=50,
            max_value=1000,
            value=100,
            step=50,
            key="rpm_passo_input"
        )

VEL_MIN = 4.0
VEL_MAX = 8.0
VEL_PASSO = 1.0
if MAPA_VEL:
    with sidebar_container():
        st.markdown("### ⚙️ Parâmetros Velocidade (km/h)")
        VEL_MIN = st.number_input(
            "Velocidade mínima",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.5,
            key="vel_min_input"
        )
        VEL_MAX = st.number_input(
            "Velocidade máxima",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.5,
            key="vel_max_input"
        )
        VEL_PASSO = st.number_input(
            "Passo das faixas Velocidade",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key="vel_passo_input"
        )

TEMPO_MAX_SEG = 60

COR_TRABALHADA = "#22C55E"
COR_NAO_TRAB = "#E5E7EB"

if MAPA_RPM and RPM_MAX <= RPM_MIN:
    st.sidebar.error("⚠️ O RPM máximo deve ser maior que o RPM mínimo.")
if MAPA_VEL and VEL_MAX <= VEL_MIN:
    st.sidebar.error("⚠️ A velocidade máxima deve ser maior que a velocidade mínima.")
if not (MAPA_AREA or MAPA_RPM or MAPA_VEL):
    st.sidebar.warning("Selecione pelo menos um tipo de mapa.")


# =========================================================
# CARD DE UPLOAD
# =========================================================
st.markdown(
    """
    <div class="upload-hero">
        <div class="upload-hero-title">📂 Envio dos arquivos</div>
        <div class="upload-hero-text">
            Faça o upload dos ZIPs com os CSVs operacionais e da base cartográfica em GPKG para gerar os mapas.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD
# =========================================================
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

if GERAR:
    st.session_state["mapas_gerados"] = True

if not uploaded_zips or not uploaded_gpkg:
    st.session_state["mapas_gerados"] = False

mapas_selecionados = []
if MAPA_AREA:
    mapas_selecionados.append("Área trabalhada")
if MAPA_RPM:
    mapas_selecionados.append("RPM")
if MAPA_VEL:
    mapas_selecionados.append("Velocidade")

if mapas_selecionados:
    st.caption("✅ Mapas selecionados para geração: " + ", ".join(mapas_selecionados))
else:
    st.caption("⚠️ Nenhum mapa selecionado.")


# =========================================================
# PROCESSAMENTO
# =========================================================
if uploaded_zips and uploaded_gpkg and st.session_state.get("mapas_gerados", False):

    if MAPA_RPM and RPM_MAX <= RPM_MIN:
        st.error("❌ Ajuste os parâmetros mínimos e máximos de RPM na sidebar antes de gerar os mapas.")
        st.stop()

    if MAPA_VEL and VEL_MAX <= VEL_MIN:
        st.error("❌ Ajuste os parâmetros mínimos e máximos de velocidade na sidebar antes de gerar os mapas.")
        st.stop()

    if not (MAPA_AREA or MAPA_RPM or MAPA_VEL):
        st.error("❌ Selecione pelo menos um tipo de mapa na sidebar.")
        st.stop()

    with st.spinner("Processando arquivos e gerando mapas..."):
        with tempfile.TemporaryDirectory() as tmpdir:

            dfs = []

            for i, uploaded_zip in enumerate(uploaded_zips):
                csv_files = ler_csvs_de_zip(uploaded_zip, tmpdir, i)

                if not csv_files:
                    st.error(f"❌ Nenhum CSV encontrado no ZIP {uploaded_zip.name}")
                    continue

                for csv_path in csv_files:
                    try:
                        df_temp = ler_csv_robusto(csv_path)
                        dfs.append(df_temp)
                    except Exception as e:
                        st.error(f"❌ Erro ao ler CSV {os.path.basename(csv_path)}: {e}")

            if not dfs:
                st.error("❌ Nenhum dado válido encontrado nos ZIPs.")
                st.stop()

            df = pd.concat(dfs, ignore_index=True)

            # =========================================================
            # VALIDAÇÃO DE COLUNAS OBRIGATÓRIAS
            # =========================================================
            # Sempre obrigatórias
            colunas_csv_obrigatorias = [
                "dt_hr_local_inicial",
                "vl_latitude_inicial",
                "vl_longitude_inicial",
                "vl_largura_implemento",
                "cd_estado",
                "cd_operacao_parada",
                "cd_fazenda",
                "cd_equipamento"
            ]

            # Obrigatórias apenas conforme mapas selecionados
            if MAPA_RPM:
                colunas_csv_obrigatorias.append("vl_rpm")

            if MAPA_VEL:
                colunas_csv_obrigatorias.append("vl_velocidade")

            faltantes_csv = validar_colunas(df, colunas_csv_obrigatorias)
            if faltantes_csv:
                st.error("❌ O(s) CSV(s) não possuem as colunas obrigatórias para os mapas selecionados: " + ", ".join(faltantes_csv))
                st.stop()

            # Se as colunas opcionais não existirem, cria com NaN
            if "vl_rpm" not in df.columns:
                df["vl_rpm"] = np.nan

            if "vl_velocidade" not in df.columns:
                df["vl_velocidade"] = np.nan

            # Conversões
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
            df["cd_equipamento"] = df["cd_equipamento"].astype(str)

            df = df.dropna(subset=[
                "dt_hr_local_inicial",
                "vl_latitude_inicial",
                "vl_longitude_inicial"
            ])

            if df.empty:
                st.error("❌ Nenhum ponto válido encontrado após o tratamento dos dados.")
                st.stop()

            gpkg_path = os.path.join(tmpdir, "base.gpkg")
            with open(gpkg_path, "wb") as f:
                f.write(uploaded_gpkg.read())

            base = gpd.read_file(gpkg_path)

            colunas_gpkg_obrigatorias = ["FAZENDA", "PROPRIEDADE", "geometry"]
            faltantes_gpkg = validar_colunas(base, colunas_gpkg_obrigatorias)
            if faltantes_gpkg:
                st.error("❌ O GPKG não possui as colunas obrigatórias: " + ", ".join(faltantes_gpkg))
                st.stop()

            base["FAZENDA"] = base["FAZENDA"].astype(str)

            if "TALHAO" in base.columns:
                base["TALHAO"] = base["TALHAO"].astype(str)
            if "GLEBA" in base.columns:
                base["GLEBA"] = base["GLEBA"].astype(str)

            rpm_faixas = gerar_faixas(RPM_MIN, RPM_MAX, RPM_PASSO, casas=0) if MAPA_RPM else []
            vel_faixas = gerar_faixas(VEL_MIN, VEL_MAX, VEL_PASSO, casas=1) if MAPA_VEL else []

            rpm_cmap = criar_cmap_suave("rpm")
            vel_cmap = criar_cmap_suave("vel")

            rpm_labels = [f[2] for f in rpm_faixas] if MAPA_RPM else []
            vel_labels = [f[2] for f in vel_faixas] if MAPA_VEL else []

            rpm_cores = dict(zip(rpm_labels, amostrar_cores_classes(rpm_cmap, len(rpm_labels)))) if MAPA_RPM else {}
            vel_cores = dict(zip(vel_labels, amostrar_cores_classes(vel_cmap, len(vel_labels)))) if MAPA_VEL else {}

            mapas_gerados_total = 0
            motivos_sem_mapa = []

            for FAZENDA_ID in df["cd_fazenda"].dropna().unique():

                df_faz = df[df["cd_fazenda"] == FAZENDA_ID].copy()
                base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()

                if df_faz.empty or base_fazenda.empty:
                    motivos_sem_mapa.append(f"Fazenda {FAZENDA_ID}: sem correspondência entre CSV e base cartográfica.")
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

                linhas = []

                for _, grupo in gdf_pts.groupby("cd_equipamento"):
                    grupo = grupo.sort_values("dt_hr_local_inicial")

                    linha_atual = []
                    rpm_atual = []
                    vel_atual = []
                    larguras_atuais = []
                    tempo_inicio = None
                    ultimo_tempo = None

                    for _, row in grupo.iterrows():
                        tempo = row["dt_hr_local_inicial"]

                        if ultimo_tempo is None:
                            linha_atual = [row.geometry]
                            rpm_atual = [row["vl_rpm"]]
                            vel_atual = [row["vl_velocidade"]]
                            larguras_atuais = [row["vl_largura_implemento"]]
                            tempo_inicio = tempo
                        else:
                            delta = (tempo - ultimo_tempo).total_seconds()

                            # quebra SOMENTE por tempo
                            if delta <= TEMPO_MAX_SEG:
                                linha_atual.append(row.geometry)
                                rpm_atual.append(row["vl_rpm"])
                                vel_atual.append(row["vl_velocidade"])
                                larguras_atuais.append(row["vl_largura_implemento"])
                            else:
                                adicionar_segmento_clipado(
                                    linhas_saida=linhas,
                                    pontos=linha_atual,
                                    rpms=rpm_atual,
                                    vels=vel_atual,
                                    larguras=larguras_atuais,
                                    t_inicio=tempo_inicio,
                                    t_fim=ultimo_tempo,
                                    geom_fazenda=geom_fazenda
                                )

                                linha_atual = [row.geometry]
                                rpm_atual = [row["vl_rpm"]]
                                vel_atual = [row["vl_velocidade"]]
                                larguras_atuais = [row["vl_largura_implemento"]]
                                tempo_inicio = tempo

                        ultimo_tempo = tempo

                    adicionar_segmento_clipado(
                        linhas_saida=linhas,
                        pontos=linha_atual,
                        rpms=rpm_atual,
                        vels=vel_atual,
                        larguras=larguras_atuais,
                        t_inicio=tempo_inicio,
                        t_fim=ultimo_tempo,
                        geom_fazenda=geom_fazenda
                    )

                if not linhas:
                    motivos_sem_mapa.append(f"Fazenda {FAZENDA_ID}: não foi possível formar linhas operacionais.")
                    continue

                gdf_linhas = gpd.GeoDataFrame(linhas, geometry="geometry", crs=base_fazenda.crs)

                largura_media = df_faz["vl_largura_implemento"].dropna().mean()
                if pd.isna(largura_media) or largura_media <= 0:
                    motivos_sem_mapa.append(f"Fazenda {FAZENDA_ID}: sem largura válida de implemento.")
                    continue

                # Mapa de área trabalhada continua com multiplicador
                largura_final = largura_media * MULTIPLICADOR_BUFFER

                buffer_linhas = gdf_linhas.geometry.buffer(largura_final / 2.0)
                area_trabalhada = unary_union(buffer_linhas).intersection(geom_fazenda)
                area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)

                area_total_ha = round(geom_fazenda.area / 10000, 2)
                area_trab_ha = round(area_trabalhada.area / 10000, 2)
                area_nao_ha = round(area_nao_trabalhada.area / 10000, 2)

                if area_trab_ha < AREA_MIN_HA:
                    motivos_sem_mapa.append(
                        f"Fazenda {FAZENDA_ID}: área trabalhada abaixo do mínimo configurado ({AREA_MIN_HA:.2f} ha)."
                    )
                    continue

                pct_trab = round(area_trab_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0
                pct_nao = round(area_nao_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0

                dt_min = df_faz["dt_hr_local_inicial"].min()
                dt_max = df_faz["dt_hr_local_inicial"].max()

                periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M")
                periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M")

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

                rpm_validos = df_faz["vl_rpm"].dropna()
                vel_validos = df_faz["vl_velocidade"].dropna()

                rpm_med_real = round(rpm_validos.mean(), 0) if not rpm_validos.empty else np.nan
                vel_med_real = round(vel_validos.mean(), 1) if not vel_validos.empty else np.nan

                gdf_display = None
                if MAPA_RPM or MAPA_VEL:
                    # mapas temáticos usam EXATAMENTE a largura do CSV, sem multiplicador
                    gdf_display = criar_poligonos_display(gdf_linhas, geom_fazenda)

                    if gdf_display is not None and not gdf_display.empty:
                        if MAPA_RPM:
                            gdf_display["classe_rpm"] = gdf_display["rpm_medio"].apply(
                                lambda x: classificar_valor(x, rpm_faixas)
                            )
                        if MAPA_VEL:
                            gdf_display["classe_vel"] = gdf_display["vel_media"].apply(
                                lambda x: classificar_valor(x, vel_faixas)
                            )

                df_leg_rpm = pd.DataFrame(columns=["cor", "inicio", "fim", "faixa", "percentual"])
                df_leg_vel = pd.DataFrame(columns=["cor", "inicio", "fim", "faixa", "percentual"])

                if gdf_display is not None and not gdf_display.empty:
                    if MAPA_RPM:
                        df_leg_rpm = calcular_legenda_percentual(
                            gdf_display,
                            "classe_rpm",
                            rpm_faixas,
                            rpm_cores
                        )

                    if MAPA_VEL:
                        df_leg_vel = calcular_legenda_percentual(
                            gdf_display,
                            "classe_vel",
                            vel_faixas,
                            vel_cores
                        )

                with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):

                    if gdf_display is not None and not gdf_display.empty:
                        col_debug1, col_debug2 = st.columns(2)
                        with col_debug1:
                            if MAPA_RPM and "classe_rpm" in gdf_display.columns:
                                st.caption(f"Trechos RPM sem classe: {int(gdf_display['classe_rpm'].isna().sum())}")
                        with col_debug2:
                            if MAPA_VEL and "classe_vel" in gdf_display.columns:
                                st.caption(f"Trechos Velocidade sem classe: {int(gdf_display['classe_vel'].isna().sum())}")

                    if MAPA_AREA:
                        fig_area = criar_figura_area(
                            base_fazenda=base_fazenda,
                            area_trabalhada=area_trabalhada,
                            area_total_ha=area_total_ha,
                            area_trab_ha=area_trab_ha,
                            area_nao_ha=area_nao_ha,
                            pct_trab=pct_trab,
                            pct_nao=pct_nao,
                            periodo_ini=periodo_ini,
                            periodo_fim=periodo_fim,
                            fazenda_id=FAZENDA_ID,
                            nome_fazenda=nome_fazenda,
                            mostrar_talhoes=MOSTRAR_TALHOES,
                            cor_trabalhada=COR_TRABALHADA,
                            cor_nao_trab=COR_NAO_TRAB
                        )

                        st.pyplot(fig_area)
                        mapas_gerados_total += 1
                        pdf_area = figura_para_pdf_bytes(fig_area)
                        st.download_button(
                            "⬇️ Baixar PDF vetorial – Área Trabalhada",
                            data=pdf_area,
                            file_name=f"mapa_area_{FAZENDA_ID}.pdf",
                            mime="application/pdf",
                            key=f"pdf_area_{FAZENDA_ID}"
                        )
                        plt.close(fig_area)

                    if MAPA_RPM:
                        faixa_rpm_ini = int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))
                        faixa_rpm_fim = int(arredondar_para_cima(RPM_MAX, RPM_PASSO))

                        fig_rpm = criar_figura_tematica(
                            base_fazenda=base_fazenda,
                            gdf_display=gdf_display,
                            coluna_classe="classe_rpm",
                            mapa_cores=rpm_cores,
                            df_legenda=df_leg_rpm,
                            titulo_mapa="Mapa de RPM",
                            titulo_box="Legenda de RPM",
                            faixa_exibida_txt=f"< {faixa_rpm_ini} | {faixa_rpm_ini} até {faixa_rpm_fim}+",
                            media_txt=f"RPM médio: {formatar_numero(rpm_med_real, 0)}",
                            periodo_ini=periodo_ini,
                            periodo_fim=periodo_fim,
                            fazenda_id=FAZENDA_ID,
                            nome_fazenda=nome_fazenda
                        )

                        st.pyplot(fig_rpm)
                        mapas_gerados_total += 1
                        pdf_rpm = figura_para_pdf_bytes(fig_rpm)
                        st.download_button(
                            "⬇️ Baixar PDF vetorial – RPM",
                            data=pdf_rpm,
                            file_name=f"mapa_rpm_{FAZENDA_ID}.pdf",
                            mime="application/pdf",
                            key=f"pdf_rpm_{FAZENDA_ID}"
                        )
                        plt.close(fig_rpm)

                    if MAPA_VEL:
                        faixa_vel_ini = arredondar_para_baixo(VEL_MIN, VEL_PASSO)
                        faixa_vel_fim = arredondar_para_cima(VEL_MAX, VEL_PASSO)

                        fig_vel = criar_figura_tematica(
                            base_fazenda=base_fazenda,
                            gdf_display=gdf_display,
                            coluna_classe="classe_vel",
                            mapa_cores=vel_cores,
                            df_legenda=df_leg_vel,
                            titulo_mapa="Mapa de Velocidade",
                            titulo_box="Legenda de Velocidade",
                            faixa_exibida_txt=f"< {formatar_numero(faixa_vel_ini, 1)} | {formatar_numero(faixa_vel_ini, 1)} até {formatar_numero(faixa_vel_fim, 1)}+ km/h",
                            media_txt=f"Vel. média: {formatar_numero(vel_med_real, 1)} km/h",
                            periodo_ini=periodo_ini,
                            periodo_fim=periodo_fim,
                            fazenda_id=FAZENDA_ID,
                            nome_fazenda=nome_fazenda
                        )

                        st.pyplot(fig_vel)
                        mapas_gerados_total += 1
                        pdf_vel = figura_para_pdf_bytes(fig_vel)
                        st.download_button(
                            "⬇️ Baixar PDF vetorial – Velocidade",
                            data=pdf_vel,
                            file_name=f"mapa_velocidade_{FAZENDA_ID}.pdf",
                            mime="application/pdf",
                            key=f"pdf_vel_{FAZENDA_ID}"
                        )
                        plt.close(fig_vel)

                    if df_talhoes is not None:
                        st.markdown("### 🌾 Área por Gleba / Talhão")
                        st.dataframe(df_talhoes, use_container_width=True, hide_index=True)

            if mapas_gerados_total == 0:
                st.warning("⚠️ Não foi possível gerar nenhum mapa com os dados enviados.")

                if motivos_sem_mapa:
                    with st.expander("Ver detalhes", expanded=False):
                        for motivo in sorted(set(motivos_sem_mapa)):
                            st.write(f"- {motivo}")
                else:
                    st.info("Verifique se os dados possuem correspondência com a base cartográfica e se a área trabalhada atende ao mínimo configurado.")

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
