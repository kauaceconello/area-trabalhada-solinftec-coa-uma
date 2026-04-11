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
    div.stButton > button {
        width: 100%;
        height: 3.2em;
        font-size: 1.15em;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171923 0%, #1d2130 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #f3f4f6 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        animation: popIn 0.18s ease-out;
    }

    @keyframes popIn {
        0% {
            transform: scale(0.985);
            opacity: 0.0;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.025);
        padding: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }

    section[data-testid="stSidebar"] .stNumberInput input {
        background-color: #0b1020;
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📍 Área Trabalhada – Solinftec")

st.markdown(
    "Aplicação para cálculo e visualização da **área trabalhada** com base em "
    "dados operacionais da **Solinftec** e base cartográfica da Usina Monte Alegre."
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


def gerar_faixas(vmin, vmax, passo, casas=0):
    """
    Gera faixas arredondadas.
    """
    inicio = arredondar_para_baixo(vmin, passo)
    fim = arredondar_para_cima(vmax, passo)

    edges = np.arange(inicio, fim + passo, passo)
    faixas = []

    for i in range(len(edges) - 1):
        a = edges[i]
        b = edges[i + 1]
        if casas == 0:
            label = f"{int(a)} a {int(b)}"
        else:
            label = f"{a:.{casas}f} a {b:.{casas}f}".replace(".", ",")
        faixas.append((a, b, label))

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
        if np.isinf(b):
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
    if len(pontos) < 2:
        return

    linha = LineString(pontos)
    linha_clip = linha.intersection(geom_fazenda)

    if linha_clip.is_empty:
        return

    rpm_medio = float(np.nanmean(rpms)) if len(rpms) else np.nan
    vel_media = float(np.nanmean(vels)) if len(vels) else np.nan
    largura_media = float(np.nanmean(larguras)) if len(larguras) else np.nan
    duracao_seg = (t_fim - t_inicio).total_seconds() if t_inicio is not None and t_fim is not None else np.nan

    geoms = []
    if linha_clip.geom_type == "LineString":
        geoms = [linha_clip]
    elif linha_clip.geom_type == "MultiLineString":
        geoms = list(linha_clip.geoms)

    for geom in geoms:
        if geom.is_empty or geom.length == 0:
            continue

        linhas_saida.append({
            "geometry": geom,
            "rpm_medio": rpm_medio,
            "vel_media": vel_media,
            "largura_media": largura_media,
            "duracao_seg": duracao_seg
        })


def criar_poligonos_display(gdf_linhas, geom_fazenda):
    """
    Cria a faixa real da operação usando apenas a largura do implemento (em metros),
    sem multiplicador extra.
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

        fim_txt = "+" if np.isinf(b) else b

        linhas.append({
            "cor": mapa_cores.get(label, "#cccccc"),
            "inicio": a,
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
    facecolor="#F8FAFC",
    mostrar_talhoes=True,
    margem_rel_x=0.005,
    margem_rel_y=0.014
):
    """
    Mapa com zoom mais próximo e melhor leitura.
    """
    ax.set_facecolor("#EEF2F7")

    base_fazenda.plot(ax=ax, facecolor=facecolor, edgecolor="#1F2937", linewidth=1.15, zorder=1)
    base_fazenda.boundary.plot(ax=ax, color="#111827", linewidth=1.15, zorder=3)

    if mostrar_talhoes and "TALHAO" in base_fazenda.columns:
        for _, row in base_fazenda.iterrows():
            if row.geometry.is_empty:
                continue
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(row["TALHAO"]),
                fontsize=8.1,
                ha="center",
                va="center",
                color="#111827",
                weight="bold",
                zorder=4
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
    desenhar_base_mapa(
        ax,
        base_fazenda,
        facecolor="#F8FAFC",
        mostrar_talhoes=mostrar_talhoes,
        margem_rel_x=0.005,
        margem_rel_y=0.014
    )

    if gdf_plot.empty:
        return

    gdf_tmp = gdf_plot[[coluna_classe, "geometry"]].dropna(subset=[coluna_classe]).copy()
    if gdf_tmp.empty:
        return

    gdf_diss = gdf_tmp.dissolve(by=coluna_classe, as_index=False)

    classes_na_ordem = [c for c in mapa_cores.keys() if c in gdf_diss[coluna_classe].dropna().unique()]

    for classe in classes_na_ordem:
        sub = gdf_diss[gdf_diss[coluna_classe] == classe]
        if not sub.empty:
            sub.plot(
                ax=ax,
                color=mapa_cores[classe],
                edgecolor="none",
                alpha=1.0,
                zorder=2
            )


def adicionar_header_topo(fig, titulo_mapa, fazenda_id, nome_fazenda, periodo_ini, periodo_fim):
    """
    Cabeçalho superior centralizado, com melhor espaçamento entre as linhas.
    """
    header = FancyBboxPatch(
        (0.02, 0.902),
        0.96,
        0.090,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=fig.transFigure,
        facecolor=(0.965, 0.972, 0.980, 0.97),
        edgecolor="#CBD5E1",
        linewidth=0.9,
        zorder=0
    )
    fig.add_artist(header)

    fig.text(
        0.50,
        0.962,
        titulo_mapa,
        fontsize=15.5,
        weight="bold",
        color="#0F172A",
        ha="center",
        va="center"
    )

    fig.text(
        0.50,
        0.936,
        f"Fazenda: {fazenda_id} – {nome_fazenda}",
        fontsize=10.4,
        color="#334155",
        ha="center",
        va="center"
    )

    fig.text(
        0.50,
        0.912,
        f"Período: {periodo_ini} até {periodo_fim}",
        fontsize=9.4,
        color="#64748B",
        ha="center",
        va="center"
    )


def adicionar_footer(fig, cor_rodape="#2F3B4A"):
    """
    Rodapé mais escuro e centralizado.
    """
    brasilia = pytz.timezone("America/Sao_Paulo")
    hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")

    fig.text(
        0.50,
        0.060,
        "⚠️ Os resultados apresentados dependem da qualidade dos dados operacionais e geoespaciais fornecidos.",
        ha="center",
        fontsize=9.7,
        color=cor_rodape
    )

    fig.text(
        0.50,
        0.038,
        "Relatório elaborado com base em dados da Solinftec.",
        ha="center",
        fontsize=9.7,
        color=cor_rodape
    )

    fig.text(
        0.50,
        0.016,
        f"Desenvolvido por Kauã Ceconello • Gerado em {hora}",
        ha="center",
        fontsize=9.7,
        color=cor_rodape
    )


def desenhar_box_legenda_tematica(
    fig,
    titulo_box,
    faixa_exibida_txt,
    media_txt,
    df_legenda,
    casas=0,
    reserve_pos=(0.75, 0.19, 0.21, 0.60)
):
    """
    Apenas uma caixa de legenda, centralizada no espaço da direita.
    """
    rx, ry, rw, rh = reserve_pos

    n = max(len(df_legenda), 1)

    box_h = min(0.20 + (n * 0.042), rh * 0.94)
    box_w = rw * 0.92

    # centraliza a caixa no espaço reservado
    box_x = rx + (rw - box_w) / 2
    box_y = ry + (rh - box_h) / 2

    ax_box = fig.add_axes([box_x, box_y, box_w, box_h])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.axis("off")

    cor_fundo = (0.87, 0.89, 0.92, 0.96)
    cor_borda = "#7B8794"
    cor_sombra = "#A5B4C2"
    cor_titulo = "#0F172A"
    cor_sec = "#475569"
    cor_linha = "#D7DEE7"
    cor_texto = "#0F172A"
    cor_destaque = "#047857"

    sombra = FancyBboxPatch(
        (0.010, -0.010),
        1,
        1,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        facecolor=cor_sombra,
        edgecolor="none",
        alpha=0.10,
        zorder=0
    )
    ax_box.add_patch(sombra)

    caixa = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        facecolor=cor_fundo,
        edgecolor=cor_borda,
        linewidth=1.25,
        zorder=1
    )
    ax_box.add_patch(caixa)

    ax_box.text(
        0.50, 0.93,
        titulo_box,
        fontsize=11.6,
        weight="bold",
        color=cor_titulo,
        va="top",
        ha="center",
        zorder=2
    )
    ax_box.text(
        0.50, 0.865,
        f"Faixa exibida: {faixa_exibida_txt}",
        fontsize=8.7,
        color=cor_sec,
        va="top",
        ha="center",
        zorder=2
    )
    ax_box.text(
        0.50, 0.805,
        media_txt,
        fontsize=9.5,
        color=cor_destaque,
        va="top",
        ha="center",
        weight="bold",
        zorder=2
    )

    ax_box.plot([0.08, 0.94], [0.74, 0.74], color=cor_borda, linewidth=0.9, zorder=2)

    ax_box.text(0.32, 0.705, "Início", fontsize=8.5, color=cor_sec, va="bottom", ha="center", zorder=2)
    ax_box.text(0.62, 0.705, "Fim", fontsize=8.5, color=cor_sec, va="bottom", ha="center", zorder=2)
    ax_box.text(0.86, 0.705, "%", fontsize=8.5, color=cor_sec, va="bottom", ha="center", zorder=2)

    if df_legenda.empty:
        ax_box.text(0.50, 0.58, "Sem dados válidos para exibir.", fontsize=8.6, color=cor_sec,
                    ha="center", zorder=2)
        return

    topo = 0.67
    base = 0.10
    row_h = (topo - base) / max(n, 1)

    if n <= 7:
        fonte = 8.8
        size_bolinha = 64
    elif n <= 10:
        fonte = 8.2
        size_bolinha = 56
    else:
        fonte = 7.5
        size_bolinha = 50

    for i, row in df_legenda.reset_index(drop=True).iterrows():
        y_row = topo - (i + 0.5) * row_h

        if i > 0:
            y_sep = topo - i * row_h
            ax_box.plot([0.08, 0.94], [y_sep, y_sep], color=cor_linha, linewidth=0.65, zorder=2)

        ax_box.scatter(
            [0.16], [y_row],
            s=size_bolinha,
            color=row["cor"],
            edgecolors="none",
            zorder=3
        )

        inicio = formatar_numero(row["inicio"], casas)
        fim = row["fim"]
        if fim != "+":
            fim = formatar_numero(fim, casas)

        percentual = f'{row["percentual"]:.1f}%'.replace(".", ",")

        ax_box.text(0.32, y_row, inicio, fontsize=fonte, va="center", ha="center", color=cor_texto, zorder=3)
        ax_box.text(0.62, y_row, str(fim), fontsize=fonte, va="center", ha="center", color=cor_texto, zorder=3)
        ax_box.text(0.86, y_row, percentual, fontsize=fonte, va="center", ha="center", color=cor_texto, zorder=3)


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
    nome_fazenda,
    casas
):
    """
    Figura temática com mapa + uma caixa de legenda centralizada à direita.
    """
    fig = plt.figure(figsize=(14.0, 8.6))
    fig.patch.set_facecolor("#E9EDF3")

    moldura = FancyBboxPatch(
        (0.015, 0.01),
        0.97,
        0.97,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="#CAD4E0",
        linewidth=1.0,
        zorder=0
    )
    fig.add_artist(moldura)

    painel_mapa = FancyBboxPatch(
        (0.03, 0.11),
        0.69,
        0.79,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=(0.96, 0.97, 0.99, 0.58),
        edgecolor="#D4DCE6",
        linewidth=0.9,
        zorder=0
    )
    fig.add_artist(painel_mapa)

    ax = fig.add_axes([0.11, 0.18, 0.43, 0.62])

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
            ax,
            base_fazenda,
            facecolor="#F8FAFC",
            mostrar_talhoes=True,
            margem_rel_x=0.005,
            margem_rel_y=0.014
        )

    adicionar_header_topo(
        fig,
        titulo_mapa=titulo_mapa,
        fazenda_id=fazenda_id,
        nome_fazenda=nome_fazenda,
        periodo_ini=periodo_ini,
        periodo_fim=periodo_fim
    )

    desenhar_box_legenda_tematica(
        fig=fig,
        titulo_box=titulo_box,
        faixa_exibida_txt=faixa_exibida_txt,
        media_txt=media_txt,
        df_legenda=df_legenda,
        casas=casas,
        reserve_pos=(0.75, 0.19, 0.21, 0.60)
    )

    adicionar_footer(fig, "#2F3B4A")

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
    Mesmo modelo visual para o mapa de área trabalhada,
    com resumo e legenda centralizados.
    """
    fig = plt.figure(figsize=(14.0, 8.6))
    fig.patch.set_facecolor("#E9EDF3")

    moldura = FancyBboxPatch(
        (0.015, 0.01),
        0.97,
        0.97,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="#CAD4E0",
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
        (0.03, 0.11),
        0.69,
        0.79,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=(0.96, 0.97, 0.99, 0.58),
        edgecolor="#D4DCE6",
        linewidth=0.9,
        zorder=0
    )
    fig.add_artist(painel_mapa)

    ax = fig.add_axes([0.11, 0.18, 0.45, 0.62])

    base_fazenda.plot(ax=ax, facecolor=cor_nao_trab, edgecolor="black", linewidth=1.1, zorder=1)
    gpd.GeoSeries(area_trabalhada, crs=base_fazenda.crs).plot(
        ax=ax, color=cor_trabalhada, alpha=0.9, zorder=2
    )
    base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.15, zorder=3)

    if mostrar_talhoes and "TALHAO" in base_fazenda.columns:
        for _, row in base_fazenda.iterrows():
            if row.geometry.is_empty:
                continue
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(row["TALHAO"]),
                fontsize=8.0,
                ha="center",
                va="center",
                color="black",
                weight="bold",
                zorder=4
            )

    minx, miny, maxx, maxy = base_fazenda.total_bounds
    dx = maxx - minx
    dy = maxy - miny
    margem_x = max(dx * 0.005, 0.35)
    margem_y = max(dy * 0.014, 0.70)

    ax.set_xlim(minx - margem_x, maxx + margem_x)
    ax.set_ylim(miny - margem_y, maxy + margem_y)
    ax.set_aspect("equal")
    ax.axis("off")

    # apenas uma caixa de resumo
    resumo_ax = fig.add_axes([0.765, 0.295, 0.15, 0.31])
    resumo_ax.set_facecolor("none")
    resumo_ax.patch.set_alpha(0)
    resumo_ax.axis("off")

    resumo_box = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        facecolor=(0.94, 0.95, 0.97, 0.96),
        edgecolor="#7B8794",
        linewidth=1.2
    )
    resumo_ax.add_patch(resumo_box)

    resumo_ax.text(
        0.50, 0.90,
        "Resumo da Operação",
        ha="center",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A"
    )

    resumo_ax.text(
        0.50, 0.70,
        f"Área total: {area_total_ha} ha",
        fontsize=9.1,
        color="#0F172A",
        va="center",
        ha="center"
    )
    resumo_ax.text(
        0.50, 0.53,
        f"Trabalhada: {area_trab_ha} ha ({pct_trab}%)",
        fontsize=9.1,
        color="#0F172A",
        va="center",
        ha="center"
    )
    resumo_ax.text(
        0.50, 0.36,
        f"Não trabalhada: {area_nao_ha} ha ({pct_nao}%)",
        fontsize=9.1,
        color="#0F172A",
        va="center",
        ha="center"
    )

    # legenda centralizada sob o mapa
    leg_ax = fig.add_axes([0.11, 0.105, 0.45, 0.060])
    leg_ax.set_facecolor("none")
    leg_ax.patch.set_alpha(0)
    leg_ax.axis("off")

    legenda_box = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        facecolor=(0.93, 0.94, 0.97, 0.94),
        edgecolor="#7B8794",
        linewidth=1.0
    )
    leg_ax.add_patch(legenda_box)

    handles = [
        mpatches.Patch(color=cor_trabalhada, label="Área trabalhada"),
        mpatches.Patch(color=cor_nao_trab, label="Área não trabalhada"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="Limites da fazenda")
    ]

    leg_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=9.8
    )

    adicionar_footer(fig, "#2F3B4A")

    return fig


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


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

with sidebar_container():
    st.markdown("### 🗺️ Tipos de mapa")
    MAPA_AREA = st.checkbox("Área trabalhada", value=True, key="mapa_area_chk")
    MAPA_RPM = st.checkbox("Mapa de RPM", value=False, key="mapa_rpm_chk")
    MAPA_VEL = st.checkbox("Mapa de Velocidade", value=False, key="mapa_vel_chk")

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

COR_TRABALHADA = "#62b27f"
COR_NAO_TRAB = "#f6b1b3"
COR_CAIXA = "#f1f8ff"
COR_RODAPE = "#7a7a7a"

FIG_WIDTH = 25
FIG_HEIGHT = 9

if MAPA_RPM and RPM_MAX <= RPM_MIN:
    st.sidebar.error("⚠️ O RPM máximo deve ser maior que o RPM mínimo.")
if MAPA_VEL and VEL_MAX <= VEL_MIN:
    st.sidebar.error("⚠️ A velocidade máxima deve ser maior que a velocidade mínima.")
if not (MAPA_AREA or MAPA_RPM or MAPA_VEL):
    st.sidebar.warning("Selecione pelo menos um tipo de mapa.")


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
                        df_temp = pd.read_csv(
                            csv_path,
                            sep=";",
                            encoding="latin1",
                            engine="python"
                        )
                        dfs.append(df_temp)
                    except Exception as e:
                        st.error(f"❌ Erro ao ler CSV {os.path.basename(csv_path)}: {e}")

            if not dfs:
                st.error("❌ Nenhum dado válido encontrado nos ZIPs.")
                st.stop()

            df = pd.concat(dfs, ignore_index=True)

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
                    continue

                gdf_linhas = gpd.GeoDataFrame(linhas, geometry="geometry", crs=base_fazenda.crs)

                largura_media = df_faz["vl_largura_implemento"].dropna().mean()
                if pd.isna(largura_media):
                    continue

                largura_final = largura_media * MULTIPLICADOR_BUFFER

                buffer_linhas = gdf_linhas.geometry.buffer(largura_final / 2.0)
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
                            mostrar_talhoes=True,
                            cor_trabalhada=COR_TRABALHADA,
                            cor_nao_trab=COR_NAO_TRAB
                        )

                        st.pyplot(fig_area)
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
                        fig_rpm = criar_figura_tematica(
                            base_fazenda=base_fazenda,
                            gdf_display=gdf_display,
                            coluna_classe="classe_rpm",
                            mapa_cores=rpm_cores,
                            df_legenda=df_leg_rpm,
                            titulo_mapa="Mapa de RPM",
                            titulo_box="Legenda de RPM",
                            faixa_exibida_txt=f"{int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))} até {int(arredondar_para_cima(RPM_MAX, RPM_PASSO))}+",
                            media_txt=f"RPM médio: {formatar_numero(rpm_med_real, 0)}",
                            periodo_ini=periodo_ini,
                            periodo_fim=periodo_fim,
                            fazenda_id=FAZENDA_ID,
                            nome_fazenda=nome_fazenda,
                            casas=0
                        )

                        st.pyplot(fig_rpm)
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
                        fig_vel = criar_figura_tematica(
                            base_fazenda=base_fazenda,
                            gdf_display=gdf_display,
                            coluna_classe="classe_vel",
                            mapa_cores=vel_cores,
                            df_legenda=df_leg_vel,
                            titulo_mapa="Mapa de Velocidade",
                            titulo_box="Legenda de Velocidade",
                            faixa_exibida_txt=f"{formatar_numero(arredondar_para_baixo(VEL_MIN, VEL_PASSO), 1)} até {formatar_numero(arredondar_para_cima(VEL_MAX, VEL_PASSO), 1)}+ km/h",
                            media_txt=f"Vel. média: {formatar_numero(vel_med_real, 1)} km/h",
                            periodo_ini=periodo_ini,
                            periodo_fim=periodo_fim,
                            fazenda_id=FAZENDA_ID,
                            nome_fazenda=nome_fazenda,
                            casas=1
                        )

                        st.pyplot(fig_vel)
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

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
