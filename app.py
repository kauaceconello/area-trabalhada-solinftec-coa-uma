
# APP STREAMLIT – ÁREA TRABALHADA (SOLINFTEC)
# Desenvolvido por Kauã Ceconello
# Atualização: Área por MULTIPOLYGON, mapa por colhedora/operador por turno, RPM/Velocidade mantidos por linhas

import io
import os
import re
import zipfile
import tempfile
from datetime import datetime, time

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pytz

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_hex, hsv_to_rgb
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages

from shapely import wkt
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

if "upload_signature_anterior" not in st.session_state:
    st.session_state["upload_signature_anterior"] = ""


# =========================================================
# ESTILO GLOBAL
# =========================================================
st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"] {
        background: linear-gradient(180deg, #020617 0%, #0B1020 50%, #0F172A 100%) !important;
    }
    [data-testid="stAppViewContainer"], section.main, section.main > div, [data-testid="stHeader"] {
        background: transparent !important;
    }
    .main .block-container, [data-testid="stMainBlockContainer"] {
        background: transparent !important;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    * { font-family: "Inter", "Segoe UI", sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; font-weight: 700 !important; }
    p, span, small, .stMarkdown p { color: #CBD5E1 !important; }
    .stCaption { color: #94A3B8 !important; }

    .hero-card, .upload-hero {
        background: linear-gradient(135deg, #0B1020 0%, #111827 100%);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 14px;
        box-shadow: 0 16px 36px rgba(0,0,0,0.32);
    }
    .hero-title { font-size: 2rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px; line-height: 1.15; }
    .hero-subtitle, .upload-hero-text { font-size: 0.96rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 0; }
    .upload-hero-title { font-size: 1.04rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0B1020 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 8px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.30);
    }
    label { font-size: 0.84rem !important; font-weight: 600 !important; color: #CBD5E1 !important; }
    input, textarea, [data-testid="stNumberInput"] input {
        background-color: #020617 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }
    div.stButton > button {
        width: 100%; height: 3.05em; font-size: 1.0em; font-weight: 700;
        border-radius: 12px; border: 1px solid #2563EB;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF !important; box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
        transition: all 0.18s ease;
    }
    div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(37, 99, 235, 0.45); }
    [data-testid="stFileUploader"] {
        background: linear-gradient(180deg, #0B1020 0%, #111827 100%) !important;
        border: 1px solid rgba(96, 165, 250, 0.22) !important;
        border-radius: 16px !important;
        padding: 10px !important;
        box-shadow: 0 10px 28px rgba(0,0,0,0.28);
    }
    [data-testid="stFileUploader"] *, [data-testid="stFileUploaderDropzone"] * { color: #DCE7F5 !important; fill: #93C5FD !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px dashed rgba(96,165,250,0.45) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.28);
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #F8FAFC !important; }
    [data-testid="stDataFrame"] {
        background: #020617 !important; border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stInfo"], [data-testid="stWarning"], [data-testid="stError"], [data-testid="stSuccess"] {
        background: rgba(255,255,255,0.06) !important; color: #E5E7EB !important;
        border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.12) !important;
    }
    hr { border-color: rgba(255,255,255,0.08) !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📍 Área Trabalhada – Solinftec</div>
        <div class="hero-subtitle">
            Aplicação para cálculo e visualização da área trabalhada com dados da Solinftec e base cartográfica.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UTILITÁRIOS
# =========================================================
def sidebar_container():
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


def formatar_area_ha(valor):
    if pd.isna(valor):
        return "-"
    return f"{float(valor):.2f}".replace(".", ",") + " ha"


def chave_ordenacao_mista(valor):
    texto = str(valor).strip()
    return re.sub(r"\d+", lambda m: f"{int(m.group()):010d}", texto)


def ordenar_tabela_talhoes(df_talhoes):
    if df_talhoes is None or df_talhoes.empty:
        return df_talhoes
    df = df_talhoes.copy()
    if "Gleba" not in df.columns or "Talhão" not in df.columns:
        return df
    df_total = df[df["Gleba"].astype(str).str.upper() == "TOTAL"].copy()
    df_dados = df[df["Gleba"].astype(str).str.upper() != "TOTAL"].copy()
    df_dados["_ord_gleba"] = df_dados["Gleba"].apply(chave_ordenacao_mista)
    df_dados["_ord_talhao"] = df_dados["Talhão"].apply(chave_ordenacao_mista)
    df_dados = df_dados.sort_values(by=["_ord_gleba", "_ord_talhao"]).drop(columns=["_ord_gleba", "_ord_talhao"])
    if not df_total.empty:
        return pd.concat([df_dados, df_total], ignore_index=True)
    return df_dados.reset_index(drop=True)


def validar_colunas(df, colunas_obrigatorias):
    return [c for c in colunas_obrigatorias if c not in df.columns]


def gerar_faixas(vmin, vmax, passo, casas=0):
    inicio = arredondar_para_baixo(vmin, passo)
    fim = arredondar_para_cima(vmax, passo)
    edges = np.arange(inicio, fim + passo, passo)
    faixas = []
    label_under = f"< {int(inicio)}" if casas == 0 else f"< {inicio:.{casas}f}".replace(".", ",")
    faixas.append((-np.inf, inicio, label_under))
    for i in range(len(edges) - 1):
        a = edges[i]
        b = edges[i + 1]
        label = f"{int(a)} a {int(b)}" if casas == 0 else f"{a:.{casas}f} a {b:.{casas}f}".replace(".", ",")
        faixas.append((a, b, label))
    label_over = f"{int(fim)}+" if casas == 0 else f"{fim:.{casas}f}+".replace(".", ",")
    faixas.append((fim, np.inf, label_over))
    return faixas


def criar_cmap_suave(tipo="rpm"):
    if tipo == "rpm":
        cores = ["#0B4F8A", "#1F78B4", "#2D9CDB", "#1FBBA6", "#20B15A", "#8FD14F", "#F2C94C", "#F2994A", "#E05A47"]
    else:
        cores = ["#0A5E8A", "#1479C9", "#16A5C8", "#18B7B2", "#1DBE6B", "#86D44E", "#DCEB46", "#F4C542", "#F59E32", "#E1594F"]
    return LinearSegmentedColormap.from_list(f"cmap_{tipo}", cores, N=256)


def amostrar_cores_classes(cmap, n_classes):
    if n_classes <= 1:
        return [to_hex(cmap(0.55))]
    pontos = np.linspace(0.10, 0.98, n_classes)
    return [to_hex(cmap(x)) for x in pontos]


def gerar_cores_distintas(categorias):
    categorias = list(categorias)
    n = len(categorias)
    if n == 0:
        return {}
    cores = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = 0.70
        v = 0.86
        rgb = hsv_to_rgb([h, s, v])
        cores.append(to_hex(rgb))
    return dict(zip(categorias, cores))


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
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    return buffer.getvalue()


def figuras_para_pdf_multipaginas(figuras):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in figuras:
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
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


def criar_zip_csv_talhoes(df_talhoes_exibicao, nome_csv):
    buffer_zip = io.BytesIO()
    csv_bytes = df_talhoes_exibicao.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    with zipfile.ZipFile(buffer_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(nome_csv, csv_bytes)
    buffer_zip.seek(0)
    return buffer_zip.getvalue()


def preparar_tabela_talhoes_exportacao(df_talhoes, incluir_area_total=True):
    if df_talhoes is None or df_talhoes.empty:
        return pd.DataFrame()
    df_exp = ordenar_tabela_talhoes(df_talhoes)
    colunas = ["Gleba", "Talhão"]
    if incluir_area_total and "Área total (ha)" in df_exp.columns:
        colunas.append("Área total (ha)")
    if "Área trabalhada (ha)" in df_exp.columns:
        colunas.append("Área trabalhada (ha)")
    df_exp = df_exp[colunas].copy()
    for col in ["Área total (ha)", "Área trabalhada (ha)"]:
        if col in df_exp.columns:
            df_exp[col] = pd.to_numeric(df_exp[col], errors="coerce").fillna(0).round(2)
            df_exp[col] = df_exp[col].apply(formatar_area_ha)
    return df_exp


def carregar_geometria_wkt_area(valor):
    try:
        texto = str(valor).strip()
        if not texto.upper().startswith(("POLYGON", "MULTIPOLYGON")):
            return None
        geom = wkt.loads(texto)
        if geom.is_empty:
            return None
        return geom
    except Exception:
        return None


def atribuir_turno(dt):
    if pd.isna(dt):
        return None
    hora = dt.time()
    if time(0, 0) <= hora < time(6, 0):
        return "Turno C"
    if time(6, 0) <= hora < time(15, 0):
        return "Turno A"
    return "Turno B"


# =========================================================
# LINHAS / ÁREAS
# =========================================================
def adicionar_segmento_clipado(
    linhas_saida,
    pontos,
    rpms,
    vels,
    larguras,
    t_inicio,
    t_fim,
    geom_fazenda,
    cd_equipamento=None,
    cd_operador=None,
    desc_operador=None
):
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
    duracao_seg = (t_fim - t_inicio).total_seconds() if t_inicio is not None and t_fim is not None else np.nan

    if linha_clip.geom_type == "LineString":
        geoms = [linha_clip]
    elif linha_clip.geom_type == "MultiLineString":
        geoms = [g for g in linha_clip.geoms if not g.is_empty and g.length > 0]
    else:
        geoms = []

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
            "duracao_seg": duracao_rateada,
            "cd_equipamento": str(cd_equipamento) if cd_equipamento is not None else "",
            "cd_operador": str(cd_operador) if cd_operador is not None and pd.notna(cd_operador) else "",
            "desc_operador": str(desc_operador) if desc_operador is not None and pd.notna(desc_operador) else ""
        })


def construir_linhas_operacionais(gdf_pts, geom_fazenda, tempo_max_seg=60):
    linhas = []
    campos_grupo = ["cd_equipamento"]
    if "cd_operador" in gdf_pts.columns:
        campos_grupo.append("cd_operador")

    for chave, grupo in gdf_pts.groupby(campos_grupo, dropna=False):
        grupo = grupo.sort_values("dt_hr_local_inicial")
        linha_atual = []
        rpm_atual = []
        vel_atual = []
        larguras_atuais = []
        tempo_inicio = None
        ultimo_tempo = None
        cd_equipamento = grupo["cd_equipamento"].iloc[0] if "cd_equipamento" in grupo.columns else ""
        cd_operador = grupo["cd_operador"].iloc[0] if "cd_operador" in grupo.columns else ""
        desc_operador = grupo["desc_operador"].iloc[0] if "desc_operador" in grupo.columns else ""

        for _, row in grupo.iterrows():
            tempo = row["dt_hr_local_inicial"]
            if ultimo_tempo is None:
                linha_atual = [row.geometry]
                rpm_atual = [row.get("vl_rpm", np.nan)]
                vel_atual = [row.get("vl_velocidade", np.nan)]
                larguras_atuais = [row.get("vl_largura_implemento", np.nan)]
                tempo_inicio = tempo
            else:
                delta = (tempo - ultimo_tempo).total_seconds()
                if delta <= tempo_max_seg:
                    linha_atual.append(row.geometry)
                    rpm_atual.append(row.get("vl_rpm", np.nan))
                    vel_atual.append(row.get("vl_velocidade", np.nan))
                    larguras_atuais.append(row.get("vl_largura_implemento", np.nan))
                else:
                    adicionar_segmento_clipado(
                        linhas, linha_atual, rpm_atual, vel_atual, larguras_atuais,
                        tempo_inicio, ultimo_tempo, geom_fazenda,
                        cd_equipamento=cd_equipamento,
                        cd_operador=cd_operador,
                        desc_operador=desc_operador
                    )
                    linha_atual = [row.geometry]
                    rpm_atual = [row.get("vl_rpm", np.nan)]
                    vel_atual = [row.get("vl_velocidade", np.nan)]
                    larguras_atuais = [row.get("vl_largura_implemento", np.nan)]
                    tempo_inicio = tempo
            ultimo_tempo = tempo

        adicionar_segmento_clipado(
            linhas, linha_atual, rpm_atual, vel_atual, larguras_atuais,
            tempo_inicio, ultimo_tempo, geom_fazenda,
            cd_equipamento=cd_equipamento,
            cd_operador=cd_operador,
            desc_operador=desc_operador
        )

    if not linhas:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf_pts.crs)
    return gpd.GeoDataFrame(linhas, geometry="geometry", crs=gdf_pts.crs)


def criar_poligonos_display(gdf_linhas, geom_fazenda):
    registros = []
    for _, row in gdf_linhas.iterrows():
        largura = row.get("largura_media", np.nan)
        if pd.isna(largura) or largura <= 0:
            continue
        try:
            geom_disp = row.geometry.buffer(largura / 2.0, cap_style=2, join_style=2, quad_segs=1).intersection(geom_fazenda)
        except Exception:
            continue
        if geom_disp.is_empty:
            continue
        registro = row.to_dict()
        registro["geometry"] = geom_disp
        registros.append(registro)
    if not registros:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf_linhas.crs)
    return gpd.GeoDataFrame(registros, geometry="geometry", crs=gdf_linhas.crs)


def calcular_legenda_percentual(gdf_display, coluna_classe, faixas, mapa_cores):
    if gdf_display is None or gdf_display.empty or coluna_classe not in gdf_display.columns:
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
            percentual = subset["duracao_seg"].fillna(0).sum() / total_tempo * 100 if total_tempo > 0 else 0
        linhas.append({
            "cor": mapa_cores.get(label, "#cccccc"),
            "inicio": "-" if np.isneginf(a) else a,
            "fim": "+" if np.isinf(b) else b,
            "faixa": label,
            "percentual": percentual
        })
    return pd.DataFrame(linhas)


# =========================================================
# PLOTS
# =========================================================
def desenhar_base_mapa(ax, base_fazenda, facecolor="#FFFFFF", mostrar_talhoes=True, margem_rel_x=0.020, margem_rel_y=0.030):
    ax.set_facecolor("#F8FAFC")
    base_fazenda.plot(ax=ax, facecolor=facecolor, edgecolor="#334155", linewidth=1.0, zorder=1)
    base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=3)
    if mostrar_talhoes and "TALHAO" in base_fazenda.columns:
        for _, row in base_fazenda.iterrows():
            if row.geometry.is_empty:
                continue
            centroid = row.geometry.centroid
            ax.text(
                centroid.x, centroid.y, str(row["TALHAO"]), fontsize=7.8,
                ha="center", va="center", color="#0F172A", weight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.14", facecolor=(1, 1, 1, 0.55), edgecolor="none")
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


def adicionar_header_topo(fig, titulo_mapa, fazenda_id, nome_fazenda, periodo_ini, periodo_fim, subtitulo_extra=None):
    ax_header = fig.add_axes([0.025, 0.905, 0.95, 0.08])
    ax_header.axis("off")
    card = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0)
    ax_header.add_patch(card)
    ax_header.text(0.03, 0.62, titulo_mapa, fontsize=16, weight="bold", color="#0F172A", ha="left", va="center")
    subt = f"Fazenda {fazenda_id} • {nome_fazenda}"
    if subtitulo_extra:
        subt += f" • {subtitulo_extra}"
    ax_header.text(0.03, 0.26, subt, fontsize=10.2, color="#475569", ha="left", va="center")
    ax_header.text(0.97, 0.50, f"Período: {periodo_ini} até {periodo_fim}", fontsize=9.6, color="#64748B", ha="right", va="center")


def adicionar_footer(fig, cor_rodape="#64748B"):
    brasilia = pytz.timezone("America/Sao_Paulo")
    hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")
    fig.text(0.50, 0.030, "Relatório elaborado com base em dados da Solinftec • Resultados dependem da qualidade dos dados operacionais e geoespaciais.", ha="center", fontsize=8.8, color=cor_rodape)
    fig.text(0.50, 0.012, f"Desenvolvido por Kauã Ceconello • Gerado em {hora}", ha="center", fontsize=8.6, color=cor_rodape)


def criar_figura_area(base_fazenda, area_trabalhada, area_total_ha, area_trab_ha, area_nao_ha, pct_trab, pct_nao, periodo_ini, periodo_fim, fazenda_id, nome_fazenda, cor_trabalhada="#22C55E", cor_nao_trab="#E5E7EB", metodo_area="Área operacional fornecida pela Solinftec"):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    fig.add_artist(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.0,rounding_size=0.012", transform=fig.transFigure, facecolor="none", edgecolor="#D8E1EB", linewidth=1.0, zorder=0))
    adicionar_header_topo(fig, "Mapa de Área Trabalhada", fazenda_id, nome_fazenda, periodo_ini, periodo_fim, metodo_area)
    fig.add_artist(FancyBboxPatch((0.03, 0.10), 0.64, 0.78, boxstyle="round,pad=0.004,rounding_size=0.015", transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0, zorder=0))
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    base_fazenda.plot(ax=ax, facecolor=cor_nao_trab, edgecolor="#334155", linewidth=1.0, zorder=1)
    if area_trabalhada is not None and not area_trabalhada.is_empty:
        gpd.GeoSeries([area_trabalhada], crs=base_fazenda.crs).plot(ax=ax, color=cor_trabalhada, alpha=0.88, zorder=2)
    base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=3)
    if "TALHAO" in base_fazenda.columns:
        for _, row in base_fazenda.iterrows():
            if row.geometry.is_empty:
                continue
            centroid = row.geometry.centroid
            ax.text(centroid.x, centroid.y, str(row["TALHAO"]), fontsize=7.8, ha="center", va="center", color="#0F172A", weight="bold", zorder=4, bbox=dict(boxstyle="round,pad=0.14", facecolor=(1, 1, 1, 0.55), edgecolor="none"))
    minx, miny, maxx, maxy = base_fazenda.total_bounds
    dx = maxx - minx
    dy = maxy - miny
    ax.set_xlim(minx - max(dx * 0.020, 0.35), maxx + max(dx * 0.020, 0.35))
    ax.set_ylim(miny - max(dy * 0.030, 0.70), maxy + max(dy * 0.030, 0.70))
    ax.set_aspect("equal")
    ax.axis("off")

    resumo_ax = fig.add_axes([0.71, 0.23, 0.25, 0.48])
    resumo_ax.set_xlim(0, 1)
    resumo_ax.set_ylim(0, 1)
    resumo_ax.axis("off")
    resumo_ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0))
    resumo_ax.text(0.08, 0.92, "Resumo da Operação", ha="left", va="center", fontsize=12, fontweight="bold", color="#0F172A")
    resumo_ax.text(0.08, 0.77, "Área total", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.77, f"{area_total_ha} ha", fontsize=10.2, color="#0F172A", ha="right", weight="bold")
    resumo_ax.text(0.08, 0.62, "Trabalhada", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.62, f"{area_trab_ha} ha ({pct_trab}%)", fontsize=10.0, color="#16A34A", ha="right", weight="bold")
    resumo_ax.text(0.08, 0.47, "Não trabalhada", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.47, f"{area_nao_ha} ha ({pct_nao}%)", fontsize=10.0, color="#475569", ha="right", weight="bold")
    resumo_ax.add_patch(FancyBboxPatch((0.08, 0.28), 0.84, 0.06, boxstyle="round,pad=0.004,rounding_size=0.015", facecolor="#E5E7EB", edgecolor="none"))
    resumo_ax.add_patch(FancyBboxPatch((0.08, 0.28), 0.84 * min(max(pct_trab / 100, 0), 1), 0.06, boxstyle="round,pad=0.004,rounding_size=0.015", facecolor=cor_trabalhada, edgecolor="none"))
    resumo_ax.text(0.50, 0.20, f"Cobertura operacional: {pct_trab}%", fontsize=9.6, color="#0F172A", ha="center", weight="bold")

    leg_ax = fig.add_axes([0.06, 0.09, 0.58, 0.05])
    leg_ax.axis("off")
    handles = [
        mpatches.Patch(color=cor_trabalhada, label="Área trabalhada"),
        mpatches.Patch(color=cor_nao_trab, label="Área não trabalhada"),
        mpatches.Patch(facecolor="white", edgecolor="#0F172A", label="Limites da fazenda")
    ]
    leg_ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=9.8)
    adicionar_footer(fig, "#64748B")
    return fig


def plotar_mapa_classes(ax, base_fazenda, gdf_plot, coluna_classe, mapa_cores, mostrar_talhoes=True):
    desenhar_base_mapa(ax, base_fazenda, facecolor="#FFFFFF", mostrar_talhoes=mostrar_talhoes)
    if gdf_plot is None or gdf_plot.empty:
        return
    gdf_tmp = gdf_plot[[coluna_classe, "geometry"]].dropna(subset=[coluna_classe]).copy()
    for classe, cor in mapa_cores.items():
        sub = gdf_tmp[gdf_tmp[coluna_classe] == classe]
        if not sub.empty:
            sub.plot(ax=ax, color=cor, edgecolor="none", alpha=1.0, zorder=2)


def desenhar_box_legenda_tematica(fig, titulo_box, faixa_exibida_txt, media_txt, df_legenda, reserve_pos=(0.71, 0.16, 0.25, 0.68)):
    rx, ry, rw, rh = reserve_pos
    ax_box = fig.add_axes([rx, ry, rw, rh])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.axis("off")
    ax_box.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0))
    ax_box.text(0.07, 0.945, titulo_box, fontsize=12, weight="bold", color="#0F172A", ha="left", va="center")
    ax_box.text(0.07, 0.895, f"Faixa exibida: {faixa_exibida_txt}", fontsize=8.8, color="#64748B", ha="left", va="center")
    ax_box.add_patch(FancyBboxPatch((0.07, 0.805), 0.86, 0.072, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=0.8))
    ax_box.text(0.50, 0.841, media_txt, fontsize=10.0, color="#1D4ED8", weight="bold", ha="center", va="center")
    ax_box.plot([0.07, 0.93], [0.755, 0.755], color="#E2E8F0", linewidth=1)
    if df_legenda.empty:
        ax_box.text(0.50, 0.55, "Sem dados válidos para exibir.", fontsize=9.2, color="#64748B", ha="center")
        return
    topo = 0.695
    base = 0.08
    n = len(df_legenda)
    row_h = (topo - base) / max(n, 1)
    for i, row in df_legenda.reset_index(drop=True).iterrows():
        y = topo - i * row_h
        pct_txt = f"{row['percentual']:.1f}%".replace(".", ",")
        ax_box.add_patch(FancyBboxPatch((0.07, y - 0.020), 0.025, 0.025, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=row["cor"], edgecolor="none"))
        ax_box.text(0.11, y - 0.007, row["faixa"], fontsize=8.9, color="#0F172A", ha="left", va="center")
        ax_box.add_patch(FancyBboxPatch((0.57, y - 0.020), 0.25, 0.025, boxstyle="round,pad=0.002,rounding_size=0.008", facecolor="#E2E8F0", edgecolor="none"))
        largura_barra = 0.25 * max(0, min(row["percentual"], 100)) / 100
        ax_box.add_patch(FancyBboxPatch((0.57, y - 0.020), largura_barra, 0.025, boxstyle="round,pad=0.002,rounding_size=0.008", facecolor=row["cor"], edgecolor="none"))
        ax_box.text(0.86, y - 0.007, pct_txt, fontsize=8.9, color="#334155", ha="center", va="center", weight="bold")
        if i < n - 1:
            ax_box.plot([0.07, 0.93], [y - 0.045, y - 0.045], color="#F1F5F9", linewidth=0.8)


def criar_figura_tematica(base_fazenda, gdf_display, coluna_classe, mapa_cores, df_legenda, titulo_mapa, titulo_box, faixa_exibida_txt, media_txt, periodo_ini, periodo_fim, fazenda_id, nome_fazenda):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    fig.add_artist(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.0,rounding_size=0.012", transform=fig.transFigure, facecolor="none", edgecolor="#D8E1EB", linewidth=1.0, zorder=0))
    adicionar_header_topo(fig, titulo_mapa, fazenda_id, nome_fazenda, periodo_ini, periodo_fim)
    fig.add_artist(FancyBboxPatch((0.03, 0.10), 0.64, 0.78, boxstyle="round,pad=0.004,rounding_size=0.015", transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0, zorder=0))
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    if gdf_display is not None and not gdf_display.empty:
        plotar_mapa_classes(ax, base_fazenda, gdf_display.dropna(subset=[coluna_classe]).copy(), coluna_classe, mapa_cores, mostrar_talhoes=True)
    else:
        desenhar_base_mapa(ax, base_fazenda, facecolor="#FFFFFF", mostrar_talhoes=True)
    desenhar_box_legenda_tematica(fig, titulo_box, faixa_exibida_txt, media_txt, df_legenda, reserve_pos=(0.71, 0.16, 0.25, 0.68))
    adicionar_footer(fig, "#64748B")
    return fig


def criar_figura_colhedora_operador(base_fazenda, gdf_area_maquina, mapa_cores, legenda_operadores, periodo_ini, periodo_fim, fazenda_id, nome_fazenda, turno_nome):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    fig.add_artist(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.0,rounding_size=0.012", transform=fig.transFigure, facecolor="none", edgecolor="#D8E1EB", linewidth=1.0, zorder=0))
    adicionar_header_topo(fig, "Área Trabalhada por Colhedora / Operador", fazenda_id, nome_fazenda, periodo_ini, periodo_fim, turno_nome)
    fig.add_artist(FancyBboxPatch((0.03, 0.10), 0.64, 0.78, boxstyle="round,pad=0.004,rounding_size=0.015", transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0, zorder=0))
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    desenhar_base_mapa(ax, base_fazenda, facecolor="#FFFFFF", mostrar_talhoes=True)
    if gdf_area_maquina is not None and not gdf_area_maquina.empty:
        for maq, cor in mapa_cores.items():
            sub = gdf_area_maquina[gdf_area_maquina["cd_equipamento"] == maq]
            if not sub.empty:
                sub.plot(ax=ax, color=cor, edgecolor="none", alpha=0.88, zorder=2)
        base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=4)

    ax_box = fig.add_axes([0.71, 0.13, 0.25, 0.72])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.axis("off")
    ax_box.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0))
    ax_box.text(0.07, 0.955, "Colhedoras e operadores", fontsize=12, weight="bold", color="#0F172A", ha="left", va="center")
    ax_box.text(0.07, 0.915, "Cor por colhedora", fontsize=8.8, color="#64748B", ha="left", va="center")

    if not legenda_operadores:
        ax_box.text(0.50, 0.55, "Sem dados válidos para exibir.", fontsize=9.2, color="#64748B", ha="center")
    else:
        y = 0.855
        for maq, operadores in legenda_operadores.items():
            cor = mapa_cores.get(maq, "#cccccc")
            ax_box.add_patch(FancyBboxPatch((0.07, y - 0.016), 0.030, 0.030, boxstyle="round,pad=0.002,rounding_size=0.006", facecolor=cor, edgecolor="none"))
            ax_box.text(0.115, y, f"Colhedora {maq}", fontsize=9.1, color="#0F172A", weight="bold", ha="left", va="center")
            y -= 0.035
            for op in operadores[:4]:
                ax_box.text(0.115, y, f"• {op}", fontsize=7.8, color="#475569", ha="left", va="center")
                y -= 0.030
            if len(operadores) > 4:
                ax_box.text(0.115, y, f"• + {len(operadores)-4} operadores", fontsize=7.8, color="#64748B", ha="left", va="center")
                y -= 0.030
            y -= 0.018
            if y < 0.08:
                ax_box.text(0.50, 0.045, "Legenda continua nos dados da tela.", fontsize=7.6, color="#94A3B8", ha="center")
                break

    adicionar_footer(fig, "#64748B")
    return fig


# =========================================================
# PDF TABELA TALHÕES PROFISSIONAL
# =========================================================
def criar_figura_tabela_talhoes_pdf(df_talhoes, fazenda_id, nome_fazenda, pagina_atual=1, total_paginas=1, area_total_trabalhada=None, area_total_fazenda=None):
    df_base = ordenar_tabela_talhoes(df_talhoes)
    if df_base is None or df_base.empty:
        df_base = pd.DataFrame(columns=["Gleba", "Talhão", "Área total (ha)", "Área trabalhada (ha)"])
    df_dados = df_base[df_base["Gleba"].astype(str).str.upper() != "TOTAL"].copy()
    for col in ["Área total (ha)", "Área trabalhada (ha)"]:
        if col in df_dados.columns:
            df_dados[col] = pd.to_numeric(df_dados[col], errors="coerce").fillna(0).round(2)
    if area_total_trabalhada is None:
        area_total_trabalhada = pd.to_numeric(df_dados.get("Área trabalhada (ha)", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    if area_total_fazenda is None:
        area_total_fazenda = pd.to_numeric(df_dados.get("Área total (ha)", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

    area_nao_trabalhada = max(area_total_fazenda - area_total_trabalhada, 0)
    pct_trabalhado = round((area_total_trabalhada / area_total_fazenda) * 100, 1) if area_total_fazenda and area_total_fazenda > 0 else 0
    pct_nao_trabalhado = round(100 - pct_trabalhado, 1)

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#F4F7FB")
    fig.add_artist(FancyBboxPatch((0.012, 0.012), 0.976, 0.976, boxstyle="round,pad=0.0,rounding_size=0.012", transform=fig.transFigure, facecolor="none", edgecolor="#D8E1EB", linewidth=1.0, zorder=0))

    ax_header = fig.add_axes([0.035, 0.895, 0.93, 0.08])
    ax_header.axis("off")
    ax_header.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=1.0))
    ax_header.text(0.025, 0.62, "Área Trabalhada por Gleba / Talhão", fontsize=15.5, weight="bold", color="#0F172A", ha="left", va="center")
    ax_header.text(0.025, 0.28, f"Fazenda {fazenda_id} • {nome_fazenda}", fontsize=9.8, color="#64748B", ha="left", va="center")
    ax_header.text(0.975, 0.50, f"Página {pagina_atual} de {total_paginas}", fontsize=9.2, color="#64748B", ha="right", va="center")

    ax_cards = fig.add_axes([0.055, 0.765, 0.89, 0.095])
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    ax_cards.axis("off")
    cards = [
        {"x": 0.00, "titulo": "Área total", "valor": formatar_area_ha(area_total_fazenda), "sub": "Base cartográfica", "cor": "#0F172A", "bg": "#FFFFFF"},
        {"x": 0.255, "titulo": "Área trabalhada", "valor": formatar_area_ha(area_total_trabalhada), "sub": f"{str(pct_trabalhado).replace('.', ',')}% da área", "cor": "#16A34A", "bg": "#F0FDF4"},
        {"x": 0.510, "titulo": "Área não trabalhada", "valor": formatar_area_ha(area_nao_trabalhada), "sub": f"{str(pct_nao_trabalhado).replace('.', ',')}% da área", "cor": "#475569", "bg": "#F8FAFC"},
        {"x": 0.765, "titulo": "Cobertura", "valor": f"{str(pct_trabalhado).replace('.', ',')}%", "sub": "Operacional", "cor": "#2563EB", "bg": "#EFF6FF"},
    ]
    for card in cards:
        ax_cards.add_patch(FancyBboxPatch((card["x"], 0.04), 0.225, 0.88, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=card["bg"], edgecolor="#E2E8F0", linewidth=1.0))
        ax_cards.text(card["x"] + 0.025, 0.68, card["titulo"], fontsize=8.5, color="#64748B", ha="left", va="center")
        ax_cards.text(card["x"] + 0.025, 0.42, card["valor"], fontsize=12.2, weight="bold", color=card["cor"], ha="left", va="center")
        ax_cards.text(card["x"] + 0.025, 0.18, card["sub"], fontsize=7.8, color="#94A3B8", ha="left", va="center")

    ax_card = fig.add_axes([0.055, 0.105, 0.89, 0.625])
    ax_card.set_xlim(0, 1)
    ax_card.set_ylim(0, 1)
    ax_card.axis("off")
    ax_card.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0))
    ax_card.text(0.035, 0.955, "Detalhamento por talhão", fontsize=12.8, weight="bold", color="#0F172A", ha="left", va="center")
    ax_card.text(0.035, 0.918, "Áreas calculadas por interseção entre a área trabalhada e a base cartográfica.", fontsize=8.4, color="#64748B", ha="left", va="center")

    if df_dados.empty:
        ax_card.text(0.50, 0.50, "Sem dados de área por talhão para exibir.", fontsize=11, color="#64748B", ha="center", va="center")
        adicionar_footer(fig, "#64748B")
        return fig

    df_tab = df_dados[["Gleba", "Talhão", "Área total (ha)", "Área trabalhada (ha)"]].copy()
    df_tab["% Trabalhado"] = np.where(df_tab["Área total (ha)"] > 0, (df_tab["Área trabalhada (ha)"] / df_tab["Área total (ha)"] * 100).round(1), 0)
    df_tab["Área total formatada"] = df_tab["Área total (ha)"].apply(formatar_area_ha)
    df_tab["Área trabalhada formatada"] = df_tab["Área trabalhada (ha)"].apply(formatar_area_ha)
    df_tab["% Trabalhado formatado"] = df_tab["% Trabalhado"].apply(lambda x: f"{x:.1f}%".replace(".", ","))

    ax_table = fig.add_axes([0.085, 0.155, 0.83, 0.495])
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")
    colunas = [
        {"nome": "GLEBA", "x": 0.03, "w": 0.15, "align": "left"},
        {"nome": "TALHÃO", "x": 0.20, "w": 0.15, "align": "left"},
        {"nome": "ÁREA TOTAL", "x": 0.38, "w": 0.20, "align": "right"},
        {"nome": "ÁREA TRABALHADA", "x": 0.61, "w": 0.22, "align": "right"},
        {"nome": "% TRAB.", "x": 0.86, "w": 0.11, "align": "center"},
    ]
    header_y = 0.945
    header_h = 0.075
    ax_table.add_patch(FancyBboxPatch((0.0, header_y - header_h), 1.0, header_h, boxstyle="round,pad=0.004,rounding_size=0.018", facecolor="#F1F5F9", edgecolor="#E2E8F0", linewidth=0.8))
    for col in colunas:
        if col["align"] == "right":
            x_text = col["x"] + col["w"]
            ha = "right"
        elif col["align"] == "center":
            x_text = col["x"] + col["w"] / 2
            ha = "center"
        else:
            x_text = col["x"]
            ha = "left"
        ax_table.text(x_text, header_y - header_h / 2, col["nome"], fontsize=8.6, weight="bold", color="#0F172A", ha=ha, va="center")

    y_inicio = 0.835
    row_h = 0.060
    gap = 0.010
    for idx, (_, row) in enumerate(df_tab.iterrows()):
        y = y_inicio - idx * (row_h + gap)
        if y < 0.045:
            break
        bg = "#FFFFFF" if idx % 2 == 0 else "#F8FAFC"
        ax_table.add_patch(FancyBboxPatch((0.0, y - row_h / 2), 1.0, row_h, boxstyle="round,pad=0.004,rounding_size=0.012", facecolor=bg, edgecolor="#E2E8F0", linewidth=0.55))
        pct_valor = row["% Trabalhado"]
        if pct_valor >= 80:
            pct_bg, pct_cor = "#DCFCE7", "#166534"
        elif pct_valor >= 50:
            pct_bg, pct_cor = "#FEF9C3", "#854D0E"
        elif pct_valor > 0:
            pct_bg, pct_cor = "#FFEDD5", "#9A3412"
        else:
            pct_bg, pct_cor = "#F1F5F9", "#64748B"
        valores = [str(row["Gleba"]), str(row["Talhão"]), row["Área total formatada"], row["Área trabalhada formatada"], row["% Trabalhado formatado"]]
        for i_col, col in enumerate(colunas):
            valor = valores[i_col]
            if i_col == 4:
                chip_w, chip_h = 0.085, 0.032
                chip_x = col["x"] + (col["w"] - chip_w) / 2
                ax_table.add_patch(FancyBboxPatch((chip_x, y - chip_h / 2), chip_w, chip_h, boxstyle="round,pad=0.004,rounding_size=0.014", facecolor=pct_bg, edgecolor="none"))
                ax_table.text(col["x"] + col["w"] / 2, y, valor, fontsize=8.4, weight="bold", color=pct_cor, ha="center", va="center")
            else:
                if col["align"] == "right":
                    x_text, ha = col["x"] + col["w"], "right"
                elif col["align"] == "center":
                    x_text, ha = col["x"] + col["w"] / 2, "center"
                else:
                    x_text, ha = col["x"], "left"
                cor_texto, peso = ("#166534", "bold") if i_col == 3 else ("#0F172A", "normal")
                ax_table.text(x_text, y, valor, fontsize=8.6, color=cor_texto, weight=peso, ha=ha, va="center")

    ax_card.text(0.035, 0.055, "Observação: os valores podem apresentar pequenas variações por arredondamento e qualidade dos dados geoespaciais.", fontsize=7.8, color="#94A3B8", ha="left", va="center")
    adicionar_footer(fig, "#64748B")
    return fig


def criar_figuras_tabela_talhoes_pdf(df_talhoes, fazenda_id, nome_fazenda, linhas_por_pagina=12):
    if df_talhoes is None or df_talhoes.empty:
        return []
    df_ordenado = ordenar_tabela_talhoes(df_talhoes)
    df_total = df_ordenado[df_ordenado["Gleba"].astype(str).str.upper() == "TOTAL"].copy()
    df_dados = df_ordenado[df_ordenado["Gleba"].astype(str).str.upper() != "TOTAL"].copy()
    if not df_total.empty:
        area_total_trabalhada = pd.to_numeric(df_total["Área trabalhada (ha)"].iloc[0], errors="coerce")
        area_total_fazenda = pd.to_numeric(df_total["Área total (ha)"].iloc[0], errors="coerce")
    else:
        area_total_trabalhada = pd.to_numeric(df_dados["Área trabalhada (ha)"], errors="coerce").fillna(0).sum()
        area_total_fazenda = pd.to_numeric(df_dados["Área total (ha)"], errors="coerce").fillna(0).sum()
    paginas_df = [df_dados.iloc[i:i + linhas_por_pagina].copy() for i in range(0, len(df_dados), linhas_por_pagina)]
    if not paginas_df:
        paginas_df = [df_dados.copy()]
    total_paginas = len(paginas_df)
    figuras = []
    for idx, df_pag in enumerate(paginas_df, start=1):
        figuras.append(criar_figura_tabela_talhoes_pdf(df_pag, fazenda_id, nome_fazenda, idx, total_paginas, area_total_trabalhada, area_total_fazenda))
    return figuras


# =========================================================
# CÁLCULOS ESPECÍFICOS
# =========================================================
def calcular_area_por_talhao(base_fazenda, area_trabalhada):
    if "TALHAO" not in base_fazenda.columns or "GLEBA" not in base_fazenda.columns:
        return None
    base_tmp = base_fazenda.copy()
    base_tmp["Área total (ha)"] = base_tmp.geometry.area / 10000
    total = base_tmp[["GLEBA", "TALHAO", "Área total (ha)"]]
    intersec = gpd.overlay(base_tmp, gpd.GeoDataFrame(geometry=[area_trabalhada], crs=base_tmp.crs), how="intersection")
    if not intersec.empty:
        intersec["Área trabalhada (ha)"] = intersec.geometry.area / 10000
        trab = intersec.groupby(["GLEBA", "TALHAO"])["Área trabalhada (ha)"].sum().reset_index()
    else:
        trab = pd.DataFrame(columns=["GLEBA", "TALHAO", "Área trabalhada (ha)"])
    df_talhoes = total.merge(trab, on=["GLEBA", "TALHAO"], how="left")
    df_talhoes["Área trabalhada (ha)"] = df_talhoes["Área trabalhada (ha)"].fillna(0)
    df_talhoes = df_talhoes.rename(columns={"GLEBA": "Gleba", "TALHAO": "Talhão"})
    df_talhoes = df_talhoes[["Gleba", "Talhão", "Área total (ha)", "Área trabalhada (ha)"]]
    df_talhoes["Área total (ha)"] = pd.to_numeric(df_talhoes["Área total (ha)"], errors="coerce").fillna(0).round(2)
    df_talhoes["Área trabalhada (ha)"] = pd.to_numeric(df_talhoes["Área trabalhada (ha)"], errors="coerce").fillna(0).round(2)
    total_row = pd.DataFrame({
        "Gleba": ["TOTAL"],
        "Talhão": [""],
        "Área total (ha)": [round(df_talhoes["Área total (ha)"].sum(), 2)],
        "Área trabalhada (ha)": [round(df_talhoes["Área trabalhada (ha)"].sum(), 2)]
    })
    return ordenar_tabela_talhoes(pd.concat([df_talhoes, total_row], ignore_index=True))


def preparar_gdf_pontos_linhas(df_faz, base_fazenda):
    gdf_pts = gpd.GeoDataFrame(
        df_faz,
        geometry=gpd.points_from_xy(df_faz["vl_longitude_inicial"], df_faz["vl_latitude_inicial"]),
        crs="EPSG:4326"
    )
    gdf_pts = gdf_pts.to_crs(base_fazenda.crs)
    geom_fazenda = unary_union(base_fazenda.geometry)
    return construir_linhas_operacionais(gdf_pts, geom_fazenda, TEMPO_MAX_SEG)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

with sidebar_container():
    st.markdown("### 🗺️ Tipos de mapa")
    MAPA_AREA = st.checkbox("Área trabalhada (área)", value=True, key="mapa_area_chk")
    MAPA_OPERADOR = st.checkbox("Área por colhedora / operador (linhas)", value=False, key="mapa_operador_chk")
    MAPA_RPM = st.checkbox("Mapa de RPM (linhas)", value=True, key="mapa_rpm_chk")
    MAPA_VEL = st.checkbox("Mapa de Velocidade (linhas)", value=True, key="mapa_vel_chk")

with sidebar_container():
    MULTIPLICADOR_BUFFER = st.number_input("Tamanho do Buffer", min_value=1.0, max_value=10.0, value=2.5, step=0.1, key="buffer_input")
    AREA_MIN_HA = st.number_input("Área mínima trabalhada (ha)", min_value=0.0, value=0.50, step=0.1, key="area_min_input")
    if MAPA_AREA:
        MOSTRAR_TALHOES = st.checkbox("📄 Incluir tabela por Gleba / Talhão no PDF e na tela", value=False, key="mostrar_talhoes_chk")
    else:
        MOSTRAR_TALHOES = False

RPM_MIN, RPM_MAX, RPM_PASSO = 1200, 2000, 100
if MAPA_RPM:
    with sidebar_container():
        st.markdown("### ⚙️ Parâmetros RPM")
        RPM_MIN = st.number_input("RPM mínimo", min_value=0, max_value=10000, value=1200, step=100, key="rpm_min_input")
        RPM_MAX = st.number_input("RPM máximo", min_value=0, max_value=10000, value=2000, step=100, key="rpm_max_input")
        RPM_PASSO = st.number_input("Passo das faixas RPM", min_value=50, max_value=1000, value=100, step=50, key="rpm_passo_input")

VEL_MIN, VEL_MAX, VEL_PASSO = 4.0, 8.0, 1.0
if MAPA_VEL:
    with sidebar_container():
        st.markdown("### ⚙️ Parâmetros Velocidade (km/h)")
        VEL_MIN = st.number_input("Velocidade mínima", min_value=0.0, max_value=100.0, value=4.0, step=0.5, key="vel_min_input")
        VEL_MAX = st.number_input("Velocidade máxima", min_value=0.0, max_value=100.0, value=8.0, step=0.5, key="vel_max_input")
        VEL_PASSO = st.number_input("Passo das faixas Velocidade", min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="vel_passo_input")

TEMPO_MAX_SEG = 60
COR_TRABALHADA = "#22C55E"
COR_NAO_TRAB = "#E5E7EB"

if MAPA_RPM and RPM_MAX <= RPM_MIN:
    st.sidebar.error("⚠️ O RPM máximo deve ser maior que o RPM mínimo.")
if MAPA_VEL and VEL_MAX <= VEL_MIN:
    st.sidebar.error("⚠️ A velocidade máxima deve ser maior que a velocidade mínima.")
if not (MAPA_AREA or MAPA_OPERADOR or MAPA_RPM or MAPA_VEL):
    st.sidebar.warning("Selecione pelo menos um tipo de mapa.")

# assinatura muda o key do upload, limpando o upload anterior ao trocar tipo de mapa
upload_signature = f"area_{MAPA_AREA}_op_{MAPA_OPERADOR}_rpm_{MAPA_RPM}_vel_{MAPA_VEL}"
if upload_signature != st.session_state.get("upload_signature_anterior", ""):
    st.session_state["mapas_gerados"] = False
    st.session_state["upload_signature_anterior"] = upload_signature


# =========================================================
# UPLOAD
# =========================================================
st.markdown(
    """
    <div class="upload-hero">
        <div class="upload-hero-title">📂 Envio dos arquivos</div>
        <div class="upload-hero-text">
            Faça o upload do ZIP correspondente ao tipo de mapa selecionado e da base cartográfica em GPKG.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_zips = st.file_uploader(
    "📦 Upload do ZIP da Solinftec",
    type=["zip"],
    accept_multiple_files=True,
    key=f"uploaded_zips_{upload_signature}"
)

uploaded_gpkg = st.file_uploader(
    "🗺️ Upload da base cartográfica (GPKG)",
    type=["gpkg"],
    key="uploaded_gpkg"
)

GERAR = st.button("▶️ Gerar mapa")

if GERAR:
    st.session_state["mapas_gerados"] = True

if not uploaded_zips or not uploaded_gpkg:
    st.session_state["mapas_gerados"] = False


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
    if not (MAPA_AREA or MAPA_OPERADOR or MAPA_RPM or MAPA_VEL):
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
                        dfs.append(ler_csv_robusto(csv_path))
                    except Exception as e:
                        st.error(f"❌ Erro ao ler CSV {os.path.basename(csv_path)}: {e}")

            if not dfs:
                st.error("❌ Nenhum dado válido encontrado nos ZIPs.")
                st.stop()

            df = pd.concat(dfs, ignore_index=True)

            gpkg_path = os.path.join(tmpdir, "base.gpkg")
            with open(gpkg_path, "wb") as f:
                f.write(uploaded_gpkg.read())
            base = gpd.read_file(gpkg_path)

            faltantes_gpkg = validar_colunas(base, ["FAZENDA", "PROPRIEDADE", "geometry"])
            if faltantes_gpkg:
                st.error("❌ O GPKG não possui as colunas obrigatórias: " + ", ".join(faltantes_gpkg))
                st.stop()

            base["FAZENDA"] = base["FAZENDA"].astype(str)
            if "TALHAO" in base.columns:
                base["TALHAO"] = base["TALHAO"].astype(str)
            if "GLEBA" in base.columns:
                base["GLEBA"] = base["GLEBA"].astype(str)

            # tratamento comum
            if "cd_fazenda" in df.columns:
                df["cd_fazenda"] = df["cd_fazenda"].astype(str)

            # validações por tipo de mapa
            if MAPA_AREA:
                faltantes_area = validar_colunas(df, ["cd_fazenda", "wkt"])
                if faltantes_area:
                    st.error("❌ Para Área trabalhada (área), o CSV precisa das colunas: " + ", ".join(faltantes_area))
                    st.stop()

            precisa_linhas = MAPA_OPERADOR or MAPA_RPM or MAPA_VEL
            if precisa_linhas:
                colunas_linhas = [
                    "dt_hr_local_inicial", "vl_latitude_inicial", "vl_longitude_inicial",
                    "vl_largura_implemento", "cd_estado", "cd_operacao_parada", "cd_fazenda", "cd_equipamento"
                ]
                if MAPA_OPERADOR:
                    colunas_linhas += ["cd_operador", "desc_operador"]
                if MAPA_RPM:
                    colunas_linhas.append("vl_rpm")
                if MAPA_VEL:
                    colunas_linhas.append("vl_velocidade")
                faltantes_linhas = validar_colunas(df, colunas_linhas)
                if faltantes_linhas:
                    st.error("❌ Para mapas com ZIP linhas, o CSV precisa das colunas: " + ", ".join(faltantes_linhas))
                    st.stop()

                if "vl_rpm" not in df.columns:
                    df["vl_rpm"] = np.nan
                if "vl_velocidade" not in df.columns:
                    df["vl_velocidade"] = np.nan
                if "cd_operador" not in df.columns:
                    df["cd_operador"] = ""
                if "desc_operador" not in df.columns:
                    df["desc_operador"] = ""

                df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
                for col in ["vl_latitude_inicial", "vl_longitude_inicial", "vl_largura_implemento", "vl_rpm", "vl_velocidade"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["cd_equipamento"] = df["cd_equipamento"].astype(str)
                df["cd_operador"] = df["cd_operador"].astype(str)
                df["desc_operador"] = df["desc_operador"].astype(str)
                df_linhas = df[(df["cd_estado"] == "E") & (df["cd_operacao_parada"] == -1)].copy()
                df_linhas = df_linhas.dropna(subset=["dt_hr_local_inicial", "vl_latitude_inicial", "vl_longitude_inicial"])
                df_linhas["Turno"] = df_linhas["dt_hr_local_inicial"].apply(atribuir_turno)
            else:
                df_linhas = pd.DataFrame()

            if MAPA_AREA:
                df_area_csv = df.copy()
                df_area_csv["geometry"] = df_area_csv["wkt"].apply(carregar_geometria_wkt_area)
                df_area_csv = df_area_csv.dropna(subset=["geometry"])
                if df_area_csv.empty:
                    st.error("❌ Nenhum POLYGON/MULTIPOLYGON válido encontrado na coluna wkt.")
                    st.stop()
                gdf_area_all = gpd.GeoDataFrame(df_area_csv, geometry="geometry", crs="EPSG:4326")
            else:
                gdf_area_all = None

            rpm_faixas = gerar_faixas(RPM_MIN, RPM_MAX, RPM_PASSO, casas=0) if MAPA_RPM else []
            vel_faixas = gerar_faixas(VEL_MIN, VEL_MAX, VEL_PASSO, casas=1) if MAPA_VEL else []
            rpm_cmap = criar_cmap_suave("rpm")
            vel_cmap = criar_cmap_suave("vel")
            rpm_labels = [f[2] for f in rpm_faixas]
            vel_labels = [f[2] for f in vel_faixas]
            rpm_cores = dict(zip(rpm_labels, amostrar_cores_classes(rpm_cmap, len(rpm_labels)))) if MAPA_RPM else {}
            vel_cores = dict(zip(vel_labels, amostrar_cores_classes(vel_cmap, len(vel_labels)))) if MAPA_VEL else {}

            mapas_gerados_total = 0
            motivos_sem_mapa = []

            # =================================================
            # MAPA ÁREA, RPM E VELOCIDADE: fluxo por fazenda
            # =================================================
            fazendas_ids = set()
            if MAPA_AREA and gdf_area_all is not None:
                fazendas_ids.update(gdf_area_all["cd_fazenda"].dropna().astype(str).unique())
            if precisa_linhas and not df_linhas.empty:
                fazendas_ids.update(df_linhas["cd_fazenda"].dropna().astype(str).unique())

            for FAZENDA_ID in sorted(fazendas_ids, key=chave_ordenacao_mista):
                base_fazenda = base[base["FAZENDA"] == str(FAZENDA_ID)].copy()
                if base_fazenda.empty:
                    motivos_sem_mapa.append(f"Fazenda {FAZENDA_ID}: sem correspondência na base cartográfica.")
                    continue
                nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]
                base_fazenda = base_fazenda.to_crs(epsg=31983)
                geom_fazenda = unary_union(base_fazenda.geometry)

                with st.expander(f"🗺️ Fazenda {FAZENDA_ID} – {nome_fazenda}", expanded=False):

                    # ÁREA PRINCIPAL POR MULTIPOLYGON
                    if MAPA_AREA:
                        gdf_area_faz = gdf_area_all[gdf_area_all["cd_fazenda"].astype(str) == str(FAZENDA_ID)].copy()
                        if gdf_area_faz.empty:
                            st.warning(f"Fazenda {FAZENDA_ID}: sem área WKT no CSV.")
                        else:
                            gdf_area_faz = gdf_area_faz.to_crs(epsg=31983)
                            area_trabalhada = unary_union(gdf_area_faz.geometry).intersection(geom_fazenda)
                            area_nao_trabalhada = geom_fazenda.difference(area_trabalhada)
                            area_total_ha = round(geom_fazenda.area / 10000, 2)
                            area_trab_ha = round(area_trabalhada.area / 10000, 2)
                            area_nao_ha = round(area_nao_trabalhada.area / 10000, 2)
                            if area_trab_ha < AREA_MIN_HA:
                                st.warning(f"Fazenda {FAZENDA_ID}: área trabalhada abaixo do mínimo configurado ({AREA_MIN_HA:.2f} ha).")
                            else:
                                pct_trab = round(area_trab_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0
                                pct_nao = round(area_nao_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0
                                if "dt_hr_local_inicial" in gdf_area_faz.columns:
                                    datas = pd.to_datetime(gdf_area_faz["dt_hr_local_inicial"], errors="coerce").dropna()
                                    if not datas.empty:
                                        periodo_ini = datas.min().strftime("%d/%m/%Y %H:%M")
                                        periodo_fim = datas.max().strftime("%d/%m/%Y %H:%M")
                                    else:
                                        periodo_ini = "-"
                                        periodo_fim = "-"
                                else:
                                    periodo_ini = "-"
                                    periodo_fim = "-"

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
                                    cor_trabalhada=COR_TRABALHADA,
                                    cor_nao_trab=COR_NAO_TRAB,
                                    metodo_area="Área operacional fornecida pela Solinftec"
                                )
                                st.pyplot(fig_area)
                                mapas_gerados_total += 1
                                figuras_pdf_area = [fig_area]
                                df_talhoes = calcular_area_por_talhao(base_fazenda, area_trabalhada) if MOSTRAR_TALHOES else None
                                if MOSTRAR_TALHOES and df_talhoes is not None and not df_talhoes.empty:
                                    figs_tab = criar_figuras_tabela_talhoes_pdf(df_talhoes, FAZENDA_ID, nome_fazenda, linhas_por_pagina=12)
                                    figuras_pdf_area.extend(figs_tab)
                                pdf_area = figuras_para_pdf_multipaginas(figuras_pdf_area)
                                st.download_button("⬇️ Baixar PDF vetorial – Área Trabalhada", data=pdf_area, file_name=f"mapa_area_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_area_{FAZENDA_ID}")
                                plt.close(fig_area)
                                for f in figuras_pdf_area[1:]:
                                    plt.close(f)

                                if df_talhoes is not None:
                                    st.markdown("### 🌾 Área por Gleba / Talhão")
                                    df_talhoes_exibicao = preparar_tabela_talhoes_exportacao(df_talhoes, incluir_area_total=True)
                                    st.dataframe(df_talhoes_exibicao, use_container_width=True, hide_index=True)
                                    zip_csv_talhoes = criar_zip_csv_talhoes(df_talhoes_exibicao, nome_csv=f"area_por_talhao_{FAZENDA_ID}.csv")
                                    st.download_button("⬇️ Baixar ZIP com CSV – Área por Gleba / Talhão", data=zip_csv_talhoes, file_name=f"area_por_talhao_{FAZENDA_ID}.zip", mime="application/zip", key=f"zip_csv_talhoes_{FAZENDA_ID}")

                    # RPM E VELOCIDADE SEM TURNO
                    if (MAPA_RPM or MAPA_VEL) and not df_linhas.empty:
                        df_faz_lin = df_linhas[df_linhas["cd_fazenda"].astype(str) == str(FAZENDA_ID)].copy()
                        if not df_faz_lin.empty:
                            gdf_linhas = preparar_gdf_pontos_linhas(df_faz_lin, base_fazenda)
                            if gdf_linhas.empty:
                                st.warning(f"Fazenda {FAZENDA_ID}: não foi possível formar linhas para RPM/Velocidade.")
                            else:
                                gdf_display = criar_poligonos_display(gdf_linhas, geom_fazenda)
                                if gdf_display is not None and not gdf_display.empty:
                                    if MAPA_RPM:
                                        gdf_display["classe_rpm"] = gdf_display["rpm_medio"].apply(lambda x: classificar_valor(x, rpm_faixas))
                                    if MAPA_VEL:
                                        gdf_display["classe_vel"] = gdf_display["vel_media"].apply(lambda x: classificar_valor(x, vel_faixas))
                                dt_min = df_faz_lin["dt_hr_local_inicial"].min()
                                dt_max = df_faz_lin["dt_hr_local_inicial"].max()
                                periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M")
                                periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M")
                                rpm_validos = df_faz_lin["vl_rpm"].dropna()
                                vel_validos = df_faz_lin["vl_velocidade"].dropna()
                                rpm_med_real = round(rpm_validos.mean(), 0) if not rpm_validos.empty else np.nan
                                vel_med_real = round(vel_validos.mean(), 1) if not vel_validos.empty else np.nan

                                if MAPA_RPM:
                                    df_leg_rpm = calcular_legenda_percentual(gdf_display, "classe_rpm", rpm_faixas, rpm_cores)
                                    faixa_rpm_ini = int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))
                                    faixa_rpm_fim = int(arredondar_para_cima(RPM_MAX, RPM_PASSO))
                                    fig_rpm = criar_figura_tematica(base_fazenda, gdf_display, "classe_rpm", rpm_cores, df_leg_rpm, "Mapa de RPM", "Legenda de RPM", f"< {faixa_rpm_ini} | {faixa_rpm_ini} até {faixa_rpm_fim}+", f"RPM médio: {formatar_numero(rpm_med_real, 0)}", periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda)
                                    st.pyplot(fig_rpm)
                                    mapas_gerados_total += 1
                                    pdf_rpm = figura_para_pdf_bytes(fig_rpm)
                                    st.download_button("⬇️ Baixar PDF vetorial – RPM", data=pdf_rpm, file_name=f"mapa_rpm_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_rpm_{FAZENDA_ID}")
                                    plt.close(fig_rpm)

                                if MAPA_VEL:
                                    df_leg_vel = calcular_legenda_percentual(gdf_display, "classe_vel", vel_faixas, vel_cores)
                                    faixa_vel_ini = arredondar_para_baixo(VEL_MIN, VEL_PASSO)
                                    faixa_vel_fim = arredondar_para_cima(VEL_MAX, VEL_PASSO)
                                    fig_vel = criar_figura_tematica(base_fazenda, gdf_display, "classe_vel", vel_cores, df_leg_vel, "Mapa de Velocidade", "Legenda de Velocidade", f"< {formatar_numero(faixa_vel_ini, 1)} | {formatar_numero(faixa_vel_ini, 1)} até {formatar_numero(faixa_vel_fim, 1)}+ km/h", f"Vel. média: {formatar_numero(vel_med_real, 1)} km/h", periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda)
                                    st.pyplot(fig_vel)
                                    mapas_gerados_total += 1
                                    pdf_vel = figura_para_pdf_bytes(fig_vel)
                                    st.download_button("⬇️ Baixar PDF vetorial – Velocidade", data=pdf_vel, file_name=f"mapa_velocidade_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_vel_{FAZENDA_ID}")
                                    plt.close(fig_vel)

            # =================================================
            # MAPA POR COLHEDORA / OPERADOR: por turno e fazenda
            # =================================================
            if MAPA_OPERADOR:
                if df_linhas.empty:
                    st.warning("⚠️ Não há dados de linhas para gerar o mapa por colhedora / operador.")
                else:
                    st.markdown("## 🚜 Área por colhedora / operador")
                    ordem_turnos = ["Turno C", "Turno A", "Turno B"]
                    for turno_nome in ordem_turnos:
                        df_turno = df_linhas[df_linhas["Turno"] == turno_nome].copy()
                        if df_turno.empty:
                            continue
                        with st.expander(f"🕒 {turno_nome}", expanded=False):
                            for FAZENDA_ID in sorted(df_turno["cd_fazenda"].dropna().astype(str).unique(), key=chave_ordenacao_mista):
                                base_fazenda = base[base["FAZENDA"] == str(FAZENDA_ID)].copy()
                                if base_fazenda.empty:
                                    st.warning(f"Fazenda {FAZENDA_ID}: sem correspondência na base cartográfica.")
                                    continue
                                nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]
                                base_fazenda = base_fazenda.to_crs(epsg=31983)
                                geom_fazenda = unary_union(base_fazenda.geometry)
                                df_faz_turno = df_turno[df_turno["cd_fazenda"].astype(str) == str(FAZENDA_ID)].copy()
                                with st.expander(f"🌾 Fazenda {FAZENDA_ID} – {nome_fazenda}", expanded=False):
                                    gdf_linhas_turno = preparar_gdf_pontos_linhas(df_faz_turno, base_fazenda)
                                    if gdf_linhas_turno.empty:
                                        st.warning("Não foi possível formar linhas operacionais para este turno/fazenda.")
                                        continue
                                    largura_media = df_faz_turno["vl_largura_implemento"].dropna().mean()
                                    if pd.isna(largura_media) or largura_media <= 0:
                                        st.warning("Sem largura válida de implemento.")
                                        continue
                                    largura_final = largura_media * MULTIPLICADOR_BUFFER
                                    registros_area = []
                                    maquinas = sorted(gdf_linhas_turno["cd_equipamento"].dropna().astype(str).unique(), key=chave_ordenacao_mista)
                                    for maq in maquinas:
                                        sub = gdf_linhas_turno[gdf_linhas_turno["cd_equipamento"].astype(str) == maq]
                                        if sub.empty:
                                            continue
                                        try:
                                            area_maq = unary_union(sub.geometry.buffer(largura_final / 2.0)).intersection(geom_fazenda)
                                        except Exception:
                                            continue
                                        if area_maq.is_empty:
                                            continue
                                        registros_area.append({"cd_equipamento": maq, "geometry": area_maq})
                                    if not registros_area:
                                        st.warning("Nenhuma área válida gerada para o mapa por operador.")
                                        continue
                                    gdf_area_maquina = gpd.GeoDataFrame(registros_area, geometry="geometry", crs=base_fazenda.crs)
                                    mapa_cores = gerar_cores_distintas(gdf_area_maquina["cd_equipamento"].tolist())

                                    legenda_operadores = {}
                                    for maq in maquinas:
                                        subop = df_faz_turno[df_faz_turno["cd_equipamento"].astype(str) == maq].copy()
                                        ops = []
                                        for _, op_row in subop[["cd_operador", "desc_operador"]].drop_duplicates().iterrows():
                                            cod = str(op_row["cd_operador"]).strip()
                                            nome = str(op_row["desc_operador"]).strip()
                                            if cod and cod.lower() != "nan" and nome and nome.lower() != "nan":
                                                ops.append(f"{cod} - {nome}")
                                            elif cod and cod.lower() != "nan":
                                                ops.append(cod)
                                        legenda_operadores[maq] = sorted(set(ops)) if ops else ["Operador não informado"]

                                    dt_min = df_faz_turno["dt_hr_local_inicial"].min()
                                    dt_max = df_faz_turno["dt_hr_local_inicial"].max()
                                    periodo_ini = dt_min.strftime("%d/%m/%Y %H:%M")
                                    periodo_fim = dt_max.strftime("%d/%m/%Y %H:%M")
                                    fig_op = criar_figura_colhedora_operador(base_fazenda, gdf_area_maquina, mapa_cores, legenda_operadores, periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda, turno_nome)
                                    st.pyplot(fig_op)
                                    mapas_gerados_total += 1
                                    plt.close(fig_op)

                                    linhas_legenda = []
                                    for maq, ops in legenda_operadores.items():
                                        linhas_legenda.append({"Colhedora": maq, "Operadores": "; ".join(ops)})
                                    st.dataframe(pd.DataFrame(linhas_legenda), use_container_width=True, hide_index=True)

            if mapas_gerados_total == 0:
                st.warning("⚠️ Não foi possível gerar nenhum mapa com os dados enviados.")
                if motivos_sem_mapa:
                    with st.expander("Ver detalhes", expanded=False):
                        for motivo in sorted(set(motivos_sem_mapa)):
                            st.write(f"- {motivo}")

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
