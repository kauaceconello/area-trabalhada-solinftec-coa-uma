
import io
import os
import re
import gc
import zipfile
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_hex
from shapely import wkt
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import pytz

# =========================================================
# CONFIGURAÇÕES
# =========================================================
st.set_page_config(page_title="Área Trabalhada – Solinftec", layout="wide")

BASE_PADRAO_PATH = "base_cartografica/BaseCartografica_10_29_2025_SOLINFTEC.gpkg"
CRS_METRICO = 31983
TEMPO_MAX_SEG = 60
LARGURA_PADRAO_M = 3.0

if "mapas_gerados" not in st.session_state:
    st.session_state["mapas_gerados"] = False

# =========================================================
# ESTILO
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
    }
    [data-testid="stFileUploader"] {
        background: linear-gradient(180deg, #0B1020 0%, #111827 100%) !important;
        border: 1px solid rgba(96, 165, 250, 0.22) !important;
        border-radius: 16px !important; padding: 10px !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #F8FAFC !important; }
    [data-testid="stInfo"], [data-testid="stWarning"], [data-testid="stError"], [data-testid="stSuccess"] {
        background: rgba(255,255,255,0.06) !important;
        color: #E5E7EB !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📍 Área Trabalhada – Solinftec</div>
        <div class="hero-subtitle">
            Mapas de Área Trabalhada, Velocidade e RPM com base nos dados operacionais da Solinftec.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FUNÇÕES GERAIS
# =========================================================
def sidebar_container():
    try:
        return st.sidebar.container(border=True)
    except TypeError:
        return st.sidebar.container()


def chave_ordenacao_mista(valor):
    texto = str(valor).strip()
    return re.sub(r"\d+", lambda m: f"{int(m.group()):010d}", texto)


def validar_colunas(df, colunas):
    return [c for c in colunas if c not in df.columns]


def formatar_numero(valor, casas=0):
    if pd.isna(valor):
        return "-"
    if casas == 0:
        return f"{int(round(valor))}"
    return f"{float(valor):.{casas}f}".replace(".", ",")


def formatar_area_ha(valor):
    if pd.isna(valor):
        return "-"
    return f"{float(valor):.2f}".replace(".", ",") + " ha"


def arredondar_para_baixo(valor, base):
    return np.floor(valor / base) * base


def arredondar_para_cima(valor, base):
    return np.ceil(valor / base) * base


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


def detectar_coluna_geometria(df, tipos):
    tipos_upper = [t.upper() for t in tipos]
    nomes_prioritarios = [
        "wkt", "WKT", "geometry", "GEOMETRY", "geom", "GEOM",
        "the_geom", "THE_GEOM", "linha", "LINHA", "line", "LINE",
        "multipolygon", "MULTIPOLYGON", "polygon", "POLYGON"
    ]

    for col in nomes_prioritarios:
        if col in df.columns:
            serie = df[col].dropna().astype(str).head(50).str.upper()
            if any(serie.str.contains(t, regex=False).any() for t in tipos_upper):
                return col

    for col in df.columns:
        serie = df[col].dropna().astype(str).head(50).str.upper()
        if serie.empty:
            continue
        if any(serie.str.contains(t, regex=False).any() for t in tipos_upper):
            return col

    return None


def carregar_wkt_seguro(valor):
    if pd.isna(valor):
        return None
    texto = str(valor).strip()
    if not texto:
        return None
    try:
        return wkt.loads(texto)
    except Exception:
        return None


def criar_gdf_wkt(df, coluna_wkt, crs="EPSG:4326"):
    if coluna_wkt is None or coluna_wkt not in df.columns:
        return gpd.GeoDataFrame(columns=list(df.columns) + ["geometry"], geometry="geometry", crs=crs)

    df_tmp = df.copy()
    df_tmp["geometry"] = df_tmp[coluna_wkt].apply(carregar_wkt_seguro)
    df_tmp = df_tmp.dropna(subset=["geometry"]).copy()

    if df_tmp.empty:
        return gpd.GeoDataFrame(columns=list(df.columns) + ["geometry"], geometry="geometry", crs=crs)

    gdf = gpd.GeoDataFrame(df_tmp, geometry="geometry", crs=crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if not gdf.empty:
        gdf["geometry"] = gdf.geometry.buffer(0) if not gdf.geometry.geom_type.isin(["LineString", "MultiLineString"]).all() else gdf.geometry
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def preencher_buracos_pequenos(geom, area_max_buraco_m2=5000):
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type == "Polygon":
        interiores_mantidos = []
        for interior in geom.interiors:
            try:
                buraco = Polygon(interior)
                if buraco.area > area_max_buraco_m2:
                    interiores_mantidos.append(interior)
            except Exception:
                interiores_mantidos.append(interior)
        return Polygon(geom.exterior, interiores_mantidos).buffer(0)

    if geom.geom_type == "MultiPolygon":
        partes = [preencher_buracos_pequenos(g, area_max_buraco_m2) for g in geom.geoms if not g.is_empty]
        partes = [g for g in partes if g is not None and not g.is_empty]
        return unary_union(partes).buffer(0) if partes else geom

    return geom


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
    df_dados = df_dados.sort_values(["_ord_gleba", "_ord_talhao"]).drop(columns=["_ord_gleba", "_ord_talhao"])
    if not df_total.empty:
        return pd.concat([df_dados, df_total], ignore_index=True)
    return df_dados.reset_index(drop=True)


def preparar_tabela_talhoes_exportacao(df_talhoes):
    if df_talhoes is None or df_talhoes.empty:
        return pd.DataFrame()
    df_exp = ordenar_tabela_talhoes(df_talhoes)
    colunas = ["Gleba", "Talhão", "Área total (ha)", "Área trabalhada (ha)"]
    colunas = [c for c in colunas if c in df_exp.columns]
    df_exp = df_exp[colunas].copy()
    for col in ["Área total (ha)", "Área trabalhada (ha)"]:
        if col in df_exp.columns:
            df_exp[col] = pd.to_numeric(df_exp[col], errors="coerce").fillna(0).round(2)
            df_exp[col] = df_exp[col].apply(formatar_area_ha)
    return df_exp


def criar_zip_csv_talhoes(df_talhoes_exibicao, nome_csv):
    buffer_zip = io.BytesIO()
    csv_bytes = df_talhoes_exibicao.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    with zipfile.ZipFile(buffer_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(nome_csv, csv_bytes)
    buffer_zip.seek(0)
    return buffer_zip.getvalue()

# =========================================================
# CLASSIFICAÇÃO E MAPAS TEMÁTICOS
# =========================================================
def gerar_faixas(vmin, vmax, passo, casas=0):
    inicio = arredondar_para_baixo(vmin, passo)
    fim = arredondar_para_cima(vmax, passo)
    edges = np.arange(inicio, fim + passo, passo)
    faixas = []
    label_under = f"< {int(inicio)}" if casas == 0 else f"< {inicio:.{casas}f}".replace(".", ",")
    faixas.append((-np.inf, inicio, label_under))
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        label = f"{int(a)} a {int(b)}" if casas == 0 else f"{a:.{casas}f} a {b:.{casas}f}".replace(".", ",")
        faixas.append((a, b, label))
    label_over = f"{int(fim)}+" if casas == 0 else f"{fim:.{casas}f}+".replace(".", ",")
    faixas.append((fim, np.inf, label_over))
    return faixas


def classificar_valor(valor, faixas):
    if pd.isna(valor):
        return None
    for a, b, label in faixas:
        if np.isneginf(a) and valor < b:
            return label
        if np.isinf(b) and valor >= a:
            return label
        if not np.isinf(b) and not np.isneginf(a) and a <= valor < b:
            return label
    return None


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


def calcular_legenda_percentual(gdf_display, coluna_classe, faixas, mapa_cores):
    if gdf_display is None or gdf_display.empty or coluna_classe not in gdf_display.columns:
        return pd.DataFrame(columns=["cor", "faixa", "percentual"])
    dados = gdf_display.dropna(subset=[coluna_classe]).copy()
    if dados.empty:
        return pd.DataFrame(columns=["cor", "faixa", "percentual"])

    total_tempo = dados["duracao_seg"].fillna(0).sum() if "duracao_seg" in dados.columns else 0
    usar_contagem = total_tempo <= 0
    linhas = []
    for _, _, label in faixas:
        subset = dados[dados[coluna_classe] == label]
        if usar_contagem:
            percentual = (len(subset) / len(dados) * 100) if len(dados) else 0
        else:
            percentual = subset["duracao_seg"].fillna(0).sum() / total_tempo * 100 if total_tempo > 0 else 0
        linhas.append({"cor": mapa_cores.get(label, "#cccccc"), "faixa": label, "percentual": percentual})
    return pd.DataFrame(linhas)

# =========================================================
# PROCESSAMENTO DE LINHAS/PONTOS
# =========================================================
def calcular_largura_media_buffer(df_faz_area, df_faz_pontos):
    series = []
    for df_tmp in [df_faz_area, df_faz_pontos]:
        if df_tmp is not None and not df_tmp.empty and "vl_largura_implemento" in df_tmp.columns:
            s = pd.to_numeric(df_tmp["vl_largura_implemento"], errors="coerce").dropna()
            s = s[s > 0]
            if not s.empty:
                series.append(s)
    if not series:
        return np.nan
    return float(pd.concat(series).mean())


def obter_periodo(df_faz_area, df_faz_oper):
    """Retorna período inicial e final considerando os dados disponíveis da fazenda."""
    candidatos = []

    for df_tmp in [df_faz_area, df_faz_oper]:
        if df_tmp is None or df_tmp.empty:
            continue

        for col_data in ["dt_hr_local_inicial", "dt_hr_local_final"]:
            if col_data in df_tmp.columns:
                vals = pd.to_datetime(df_tmp[col_data], errors="coerce").dropna()
                if not vals.empty:
                    candidatos.append(vals)

    if not candidatos:
        return "-", "-"

    datas = pd.concat(candidatos)
    return datas.min().strftime("%d/%m/%Y %H:%M"), datas.max().strftime("%d/%m/%Y %H:%M")


def adicionar_segmento_clipado(linhas_saida, pontos, rpms, vels, larguras, t_inicio, t_fim, geom_fazenda):
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
    largura_media = float(np.nanmean(larguras)) if len(larguras) else LARGURA_PADRAO_M
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
        duracao_rateada = duracao_seg * (geom.length / comprimento_total_clipado) if pd.notna(duracao_seg) and comprimento_total_clipado > 0 else np.nan
        linhas_saida.append({
            "geometry": geom,
            "rpm_medio": rpm_medio,
            "vel_media": vel_media,
            "largura_media": largura_media,
            "duracao_seg": duracao_rateada,
        })


def criar_linhas_por_pontos(df_faz_pontos, geom_fazenda):
    gdf_pts = gpd.GeoDataFrame(
        df_faz_pontos,
        geometry=gpd.points_from_xy(df_faz_pontos["vl_longitude_inicial"], df_faz_pontos["vl_latitude_inicial"]),
        crs="EPSG:4326",
    ).to_crs(epsg=CRS_METRICO)

    linhas = []
    for _, grupo in gdf_pts.groupby("cd_equipamento"):
        grupo = grupo.sort_values("dt_hr_local_inicial")
        linha_atual, rpm_atual, vel_atual, larguras_atuais = [], [], [], []
        tempo_inicio, ultimo_tempo = None, None
        for _, row in grupo.iterrows():
            tempo = row["dt_hr_local_inicial"]
            if ultimo_tempo is None:
                linha_atual = [row.geometry]
                rpm_atual = [row.get("vl_rpm", np.nan)]
                vel_atual = [row.get("vl_velocidade", np.nan)]
                larguras_atuais = [row.get("vl_largura_implemento", LARGURA_PADRAO_M)]
                tempo_inicio = tempo
            else:
                delta = (tempo - ultimo_tempo).total_seconds()
                if delta <= TEMPO_MAX_SEG:
                    linha_atual.append(row.geometry)
                    rpm_atual.append(row.get("vl_rpm", np.nan))
                    vel_atual.append(row.get("vl_velocidade", np.nan))
                    larguras_atuais.append(row.get("vl_largura_implemento", LARGURA_PADRAO_M))
                else:
                    adicionar_segmento_clipado(linhas, linha_atual, rpm_atual, vel_atual, larguras_atuais, tempo_inicio, ultimo_tempo, geom_fazenda)
                    linha_atual = [row.geometry]
                    rpm_atual = [row.get("vl_rpm", np.nan)]
                    vel_atual = [row.get("vl_velocidade", np.nan)]
                    larguras_atuais = [row.get("vl_largura_implemento", LARGURA_PADRAO_M)]
                    tempo_inicio = tempo
            ultimo_tempo = tempo
        adicionar_segmento_clipado(linhas, linha_atual, rpm_atual, vel_atual, larguras_atuais, tempo_inicio, ultimo_tempo, geom_fazenda)

    return gpd.GeoDataFrame(linhas, geometry="geometry", crs=f"EPSG:{CRS_METRICO}") if linhas else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}")


def criar_linhas_por_linestring(df_faz_linhas, coluna_linestring, geom_fazenda):
    gdf = criar_gdf_wkt(df_faz_linhas, coluna_linestring, crs="EPSG:4326")
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}")
    gdf = gdf.to_crs(epsg=CRS_METRICO)

    registros = []
    for _, row in gdf.iterrows():
        geom = row.geometry.intersection(geom_fazenda)
        if geom.is_empty:
            continue
        rpm = pd.to_numeric(row.get("vl_rpm", np.nan), errors="coerce")
        vel = pd.to_numeric(row.get("vl_velocidade", np.nan), errors="coerce")
        largura = pd.to_numeric(row.get("vl_largura_implemento", LARGURA_PADRAO_M), errors="coerce")
        if pd.isna(largura) or largura <= 0:
            largura = LARGURA_PADRAO_M

        duracao = np.nan
        if "dt_hr_local_inicial" in gdf.columns and "dt_hr_local_final" in gdf.columns:
            t1 = pd.to_datetime(row.get("dt_hr_local_inicial"), errors="coerce")
            t2 = pd.to_datetime(row.get("dt_hr_local_final"), errors="coerce")
            if pd.notna(t1) and pd.notna(t2):
                duracao = (t2 - t1).total_seconds()

        if geom.geom_type == "LineString":
            geoms = [geom]
        elif geom.geom_type == "MultiLineString":
            geoms = [g for g in geom.geoms if not g.is_empty and g.length > 0]
        else:
            geoms = []

        for g in geoms:
            registros.append({
                "geometry": g,
                "rpm_medio": rpm,
                "vel_media": vel,
                "largura_media": largura,
                "duracao_seg": duracao,
            })

    return gpd.GeoDataFrame(registros, geometry="geometry", crs=f"EPSG:{CRS_METRICO}") if registros else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}")


def criar_poligonos_display(gdf_linhas, geom_fazenda):
    registros = []
    if gdf_linhas is None or gdf_linhas.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}")
    for _, row in gdf_linhas.iterrows():
        largura = row.get("largura_media", LARGURA_PADRAO_M)
        if pd.isna(largura) or largura <= 0:
            largura = LARGURA_PADRAO_M
        try:
            geom_disp = row.geometry.buffer(largura / 2.0, cap_style=2, join_style=2, quad_segs=1).intersection(geom_fazenda)
        except Exception:
            continue
        if geom_disp.is_empty:
            continue
        registros.append({
            "geometry": geom_disp,
            "rpm_medio": row.get("rpm_medio", np.nan),
            "vel_media": row.get("vel_media", np.nan),
            "largura_media": largura,
            "duracao_seg": row.get("duracao_seg", np.nan),
        })
    return gpd.GeoDataFrame(registros, geometry="geometry", crs=gdf_linhas.crs) if registros else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf_linhas.crs)

# =========================================================
# PDF E PLOTS
# =========================================================
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


def adicionar_footer(fig, cor_rodape="#64748B"):
    brasilia = pytz.timezone("America/Sao_Paulo")
    hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")
    fig.text(0.50, 0.030, "Relatório elaborado com base em dados da Solinftec • Resultados dependem da qualidade dos dados operacionais e geoespaciais.", ha="center", fontsize=8.8, color=cor_rodape)
    fig.text(0.50, 0.012, f"Desenvolvido por Kauã Ceconello • Gerado em {hora}", ha="center", fontsize=8.6, color=cor_rodape)


def adicionar_header_topo(fig, titulo_mapa, fazenda_id, nome_fazenda, periodo_ini, periodo_fim):
    ax_header = fig.add_axes([0.025, 0.905, 0.95, 0.08])
    ax_header.axis("off")
    card = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0)
    ax_header.add_patch(card)
    ax_header.text(0.03, 0.62, titulo_mapa, fontsize=16, weight="bold", color="#0F172A", ha="left", va="center")
    ax_header.text(0.03, 0.26, f"Fazenda {fazenda_id} • {nome_fazenda}", fontsize=10.2, color="#475569", ha="left", va="center")
    ax_header.text(0.97, 0.50, f"Período: {periodo_ini} até {periodo_fim}", fontsize=9.6, color="#64748B", ha="right", va="center")


def ajustar_extensao(ax, base_fazenda):
    minx, miny, maxx, maxy = base_fazenda.total_bounds
    dx, dy = maxx - minx, maxy - miny
    ax.set_xlim(minx - max(dx * 0.020, 0.35), maxx + max(dx * 0.020, 0.35))
    ax.set_ylim(miny - max(dy * 0.030, 0.70), maxy + max(dy * 0.030, 0.70))
    ax.set_aspect("equal")
    ax.axis("off")


def plotar_rotulos_talhao(ax, base_fazenda):
    if "TALHAO" not in base_fazenda.columns:
        return
    for _, row in base_fazenda.iterrows():
        if row.geometry.is_empty:
            continue
        c = row.geometry.centroid
        ax.text(c.x, c.y, str(row["TALHAO"]), fontsize=7.8, ha="center", va="center", color="#0F172A", weight="bold", zorder=4, bbox=dict(boxstyle="round,pad=0.14", facecolor=(1, 1, 1, 0.55), edgecolor="none"))


def criar_figura_area(base_fazenda, area_trabalhada, area_total_ha, area_trab_ha, area_nao_ha, pct_trab, pct_nao, periodo_ini, periodo_fim, fazenda_id, nome_fazenda):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    adicionar_header_topo(fig, "Mapa de Área Trabalhada", fazenda_id, nome_fazenda, periodo_ini, periodo_fim)
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    base_fazenda.plot(ax=ax, facecolor="#E5E7EB", edgecolor="#334155", linewidth=1.0, zorder=1)
    if area_trabalhada is not None and not area_trabalhada.is_empty:
        gpd.GeoSeries([area_trabalhada], crs=base_fazenda.crs).plot(ax=ax, color="#22C55E", alpha=0.88, zorder=2)
    base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=3)
    plotar_rotulos_talhao(ax, base_fazenda)
    ajustar_extensao(ax, base_fazenda)

    resumo_ax = fig.add_axes([0.71, 0.23, 0.25, 0.48])
    resumo_ax.set_xlim(0, 1)
    resumo_ax.set_ylim(0, 1)
    resumo_ax.axis("off")
    resumo_box = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0)
    resumo_ax.add_patch(resumo_box)
    resumo_ax.text(0.08, 0.92, "Resumo da Operação", ha="left", va="center", fontsize=12, fontweight="bold", color="#0F172A")
    resumo_ax.text(0.08, 0.77, "Área total", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.77, f"{area_total_ha} ha", fontsize=10.2, color="#0F172A", ha="right", weight="bold")
    resumo_ax.text(0.08, 0.62, "Trabalhada", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.62, f"{area_trab_ha} ha ({pct_trab}%)", fontsize=10.0, color="#16A34A", ha="right", weight="bold")
    resumo_ax.text(0.08, 0.47, "Não trabalhada", fontsize=8.7, color="#64748B", ha="left")
    resumo_ax.text(0.92, 0.47, f"{area_nao_ha} ha ({pct_nao}%)", fontsize=10.0, color="#475569", ha="right", weight="bold")
    resumo_ax.add_patch(mpatches.FancyBboxPatch((0.08, 0.28), 0.84, 0.06, boxstyle="round,pad=0.004,rounding_size=0.015", facecolor="#E5E7EB", edgecolor="none"))
    resumo_ax.add_patch(mpatches.FancyBboxPatch((0.08, 0.28), 0.84 * min(max(pct_trab / 100, 0), 1), 0.06, boxstyle="round,pad=0.004,rounding_size=0.015", facecolor="#22C55E", edgecolor="none"))
    resumo_ax.text(0.50, 0.20, f"Cobertura operacional: {pct_trab}%", fontsize=9.6, color="#0F172A", ha="center", weight="bold")
    adicionar_footer(fig)
    return fig


def criar_figura_tematica(base_fazenda, gdf_display, coluna_classe, mapa_cores, df_legenda, titulo_mapa, titulo_box, faixa_exibida_txt, media_txt, periodo_ini, periodo_fim, fazenda_id, nome_fazenda):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    adicionar_header_topo(fig, titulo_mapa, fazenda_id, nome_fazenda, periodo_ini, periodo_fim)
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    base_fazenda.plot(ax=ax, facecolor="#FFFFFF", edgecolor="#334155", linewidth=1.0, zorder=1)
    if gdf_display is not None and not gdf_display.empty and coluna_classe in gdf_display.columns:
        for classe, cor in mapa_cores.items():
            sub = gdf_display[gdf_display[coluna_classe] == classe]
            if not sub.empty:
                sub.plot(ax=ax, color=cor, edgecolor="none", alpha=1.0, zorder=2)
    base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=3)
    plotar_rotulos_talhao(ax, base_fazenda)
    ajustar_extensao(ax, base_fazenda)

    ax_box = fig.add_axes([0.71, 0.16, 0.25, 0.68])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.axis("off")
    card = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0)
    ax_box.add_patch(card)
    ax_box.text(0.07, 0.945, titulo_box, fontsize=12, weight="bold", color="#0F172A", ha="left", va="center")
    ax_box.text(0.07, 0.895, f"Faixa exibida: {faixa_exibida_txt}", fontsize=8.8, color="#64748B", ha="left", va="center")
    chip = mpatches.FancyBboxPatch((0.07, 0.805), 0.86, 0.072, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=0.8)
    ax_box.add_patch(chip)
    ax_box.text(0.50, 0.841, media_txt, fontsize=10.0, color="#1D4ED8", weight="bold", ha="center", va="center")
    ax_box.plot([0.07, 0.93], [0.755, 0.755], color="#E2E8F0", linewidth=1)

    if not df_legenda.empty:
        topo, base_y = 0.695, 0.08
        row_h = (topo - base_y) / max(len(df_legenda), 1)
        for i, row in df_legenda.reset_index(drop=True).iterrows():
            y = topo - i * row_h
            pct_txt = f"{row['percentual']:.1f}%".replace(".", ",")
            ax_box.add_patch(mpatches.FancyBboxPatch((0.07, y - 0.020), 0.025, 0.025, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=row["cor"], edgecolor="none"))
            ax_box.text(0.11, y - 0.007, row["faixa"], fontsize=8.9, color="#0F172A", ha="left", va="center")
            ax_box.add_patch(mpatches.FancyBboxPatch((0.57, y - 0.020), 0.25, 0.025, boxstyle="round,pad=0.002,rounding_size=0.008", facecolor="#E2E8F0", edgecolor="none"))
            largura_barra = 0.25 * max(0, min(row["percentual"], 100)) / 100
            ax_box.add_patch(mpatches.FancyBboxPatch((0.57, y - 0.020), largura_barra, 0.025, boxstyle="round,pad=0.002,rounding_size=0.008", facecolor=row["cor"], edgecolor="none"))
            ax_box.text(0.86, y - 0.007, pct_txt, fontsize=8.9, color="#334155", ha="center", va="center", weight="bold")
    else:
        ax_box.text(0.50, 0.55, "Sem dados válidos para exibir.", fontsize=9.2, color="#64748B", ha="center")

    adicionar_footer(fig)
    return fig


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
    pct_trabalhado = round((area_total_trabalhada / area_total_fazenda) * 100, 1) if area_total_fazenda and area_total_fazenda > 0 else 0

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#F4F7FB")
    ax_header = fig.add_axes([0.035, 0.895, 0.93, 0.08])
    ax_header.axis("off")
    header_box = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=1.0)
    ax_header.add_patch(header_box)
    ax_header.text(0.025, 0.62, "Área Trabalhada por Gleba / Talhão", fontsize=15.5, weight="bold", color="#0F172A", ha="left", va="center")
    ax_header.text(0.025, 0.28, f"Fazenda {fazenda_id} • {nome_fazenda}", fontsize=9.8, color="#64748B", ha="left", va="center")
    ax_header.text(0.975, 0.50, f"Página {pagina_atual} de {total_paginas}", fontsize=9.2, color="#64748B", ha="right", va="center")

    ax_cards = fig.add_axes([0.055, 0.765, 0.89, 0.095])
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    ax_cards.axis("off")
    cards = [
        {"x": 0.00, "titulo": "Área total", "valor": formatar_area_ha(area_total_fazenda), "cor": "#0F172A", "bg": "#FFFFFF"},
        {"x": 0.255, "titulo": "Área trabalhada", "valor": formatar_area_ha(area_total_trabalhada), "cor": "#16A34A", "bg": "#F0FDF4"},
        {"x": 0.510, "titulo": "Cobertura", "valor": f"{str(pct_trabalhado).replace('.', ',')}%", "cor": "#2563EB", "bg": "#EFF6FF"},
    ]
    for card in cards:
        box = mpatches.FancyBboxPatch((card["x"], 0.04), 0.225, 0.88, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=card["bg"], edgecolor="#E2E8F0", linewidth=1.0)
        ax_cards.add_patch(box)
        ax_cards.text(card["x"] + 0.025, 0.68, card["titulo"], fontsize=8.5, color="#64748B", ha="left", va="center")
        ax_cards.text(card["x"] + 0.025, 0.42, card["valor"], fontsize=12.2, weight="bold", color=card["cor"], ha="left", va="center")

    ax_table = fig.add_axes([0.065, 0.11, 0.87, 0.61])
    ax_table.axis("off")
    if df_dados.empty:
        ax_table.text(0.5, 0.5, "Sem dados de área por talhão.", ha="center", va="center", fontsize=11, color="#64748B")
        adicionar_footer(fig)
        return fig

    df_tab = df_dados[["Gleba", "Talhão", "Área total (ha)", "Área trabalhada (ha)"]].copy()
    df_tab["% Trabalhado"] = np.where(df_tab["Área total (ha)"] > 0, (df_tab["Área trabalhada (ha)"] / df_tab["Área total (ha)"] * 100).round(1), 0)
    df_tab["Área total (ha)"] = df_tab["Área total (ha)"].apply(formatar_area_ha)
    df_tab["Área trabalhada (ha)"] = df_tab["Área trabalhada (ha)"].apply(formatar_area_ha)
    df_tab["% Trabalhado"] = df_tab["% Trabalhado"].apply(lambda x: f"{x:.1f}%".replace(".", ","))

    tabela = ax_table.table(cellText=df_tab.values, colLabels=df_tab.columns, loc="center", cellLoc="center", colLoc="center")
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8.5)
    tabela.scale(1, 1.35)
    adicionar_footer(fig)
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

    paginas_df = [df_dados.iloc[i:i + linhas_por_pagina].copy() for i in range(0, len(df_dados), linhas_por_pagina)] or [df_dados.copy()]
    figuras = []
    for idx, df_pag in enumerate(paginas_df, start=1):
        figuras.append(criar_figura_tabela_talhoes_pdf(df_pag, fazenda_id, nome_fazenda, idx, len(paginas_df), area_total_trabalhada, area_total_fazenda))
    return figuras

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

with sidebar_container():
    st.markdown("### 🗺️ Tipos de mapa")
    MAPA_AREA = st.checkbox("Área trabalhada", value=True, key="mapa_area_chk")
    MAPA_VEL = st.checkbox("Mapa de Velocidade", value=True, key="mapa_vel_chk")
    MAPA_RPM = st.checkbox("Mapa de RPM", value=True, key="mapa_rpm_chk")

with sidebar_container():
    st.markdown("### 🧭 Base cartográfica")
    st.caption(os.path.basename(BASE_PADRAO_PATH))

# Velocidade e RPM não utilizam parâmetros de buffer.
# Por isso, os parâmetros de Área Trabalhada só aparecem quando Área Trabalhada está selecionada sozinha.
if MAPA_AREA and not (MAPA_VEL or MAPA_RPM):
    with sidebar_container():
        st.markdown("### 📐 Área Trabalhada")
        MULTIPLICADOR_BUFFER_AREA = st.number_input("Tamanho do Buffer", min_value=0.0, max_value=20.0, value=2.5, step=0.1, key="buffer_area_mult_input")
        AREA_MIN_HA = st.number_input("Área mínima trabalhada (ha)", min_value=0.0, value=0.50, step=0.1, key="area_min_input")
        PARAMETROS_AVANCADOS_AREA = st.checkbox("⚙️ Parâmetros avançados", value=False, key="parametros_avancados_area_chk")
        BUFFER_MINIMO_M = 8.0
        FATOR_RECUO_GAPS = 0.30
        AREA_MAX_BURACO_HA = 0.50
        if PARAMETROS_AVANCADOS_AREA:
            BUFFER_MINIMO_M = st.number_input("Buffer mínimo (m)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="buffer_minimo_m_input")
            FATOR_RECUO_GAPS = st.number_input("Fechamento do buffer", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key="fator_recuo_gaps_input")
            AREA_MAX_BURACO_HA = st.number_input("Preencher buracos até (ha)", min_value=0.0, max_value=10.0, value=0.50, step=0.10, key="area_max_buraco_ha_input")
        MOSTRAR_TALHOES = st.checkbox("📄 Incluir tabela por Gleba / Talhão no PDF e CSV", value=False, key="mostrar_talhoes_chk")
else:
    # Padrões internos usados quando os controles ficam ocultos.
    # Caso o arquivo enviado seja MULTIPOLYGON/POLYGON, a Área Trabalhada ainda usa estes valores.
    MULTIPLICADOR_BUFFER_AREA = 2.5
    AREA_MIN_HA = 0.50
    BUFFER_MINIMO_M = 8.0
    FATOR_RECUO_GAPS = 0.30
    AREA_MAX_BURACO_HA = 0.50
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

if MAPA_RPM and RPM_MAX <= RPM_MIN:
    st.sidebar.error("⚠️ O RPM máximo deve ser maior que o RPM mínimo.")
if MAPA_VEL and VEL_MAX <= VEL_MIN:
    st.sidebar.error("⚠️ A velocidade máxima deve ser maior que a velocidade mínima.")

# =========================================================
# UPLOAD
# =========================================================
st.markdown(
    """
    <div class="upload-hero">
        <div class="upload-hero-title">📂 Envio dos arquivos</div>
        <div class="upload-hero-text">
            Envie os ZIPs contendo os CSVs da Solinftec para gerar os mapas.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists(BASE_PADRAO_PATH):
    st.error("❌ Base cartográfica não encontrada. Confira a pasta `base_cartografica` no GitHub.")

uploaded_zips = st.file_uploader("📦 Upload dos ZIPs contendo CSVs da Solinftec", type=["zip"], accept_multiple_files=True)
GERAR = st.button("▶️ Gerar mapa")

if GERAR:
    st.session_state["mapas_gerados"] = True
if not uploaded_zips or not os.path.exists(BASE_PADRAO_PATH):
    st.session_state["mapas_gerados"] = False

mapas_selecionados = []
if MAPA_AREA:
    mapas_selecionados.append("Área trabalhada")
if MAPA_VEL:
    mapas_selecionados.append("Velocidade")
if MAPA_RPM:
    mapas_selecionados.append("RPM")
st.caption("✅ Mapas selecionados: " + ", ".join(mapas_selecionados) if mapas_selecionados else "⚠️ Nenhum mapa selecionado.")

# =========================================================
# PROCESSAMENTO
# =========================================================
if uploaded_zips and os.path.exists(BASE_PADRAO_PATH) and st.session_state.get("mapas_gerados", False):
    if MAPA_RPM and RPM_MAX <= RPM_MIN:
        st.error("❌ Ajuste os parâmetros de RPM.")
        st.stop()
    if MAPA_VEL and VEL_MAX <= VEL_MIN:
        st.error("❌ Ajuste os parâmetros de velocidade.")
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
            del dfs
            gc.collect()

            COLUNA_POLIGONO = detectar_coluna_geometria(df, ["MULTIPOLYGON", "POLYGON"])
            COLUNA_LINHA = detectar_coluna_geometria(df, ["LINESTRING", "MULTILINESTRING"])
            TEM_PONTOS = all(c in df.columns for c in ["vl_latitude_inicial", "vl_longitude_inicial"])

            # Regra solicitada:
            # - Se vier MULTIPOLYGON/POLYGON, gera somente Área Trabalhada.
            # - Se vier LINESTRING ou POINTS, não gera Área Trabalhada.
            MAPA_AREA_EFETIVO = MAPA_AREA and COLUNA_POLIGONO is not None
            MAPA_VEL_EFETIVO = MAPA_VEL and COLUNA_POLIGONO is None and (COLUNA_LINHA is not None or TEM_PONTOS)
            MAPA_RPM_EFETIVO = MAPA_RPM and COLUNA_POLIGONO is None and (COLUNA_LINHA is not None or TEM_PONTOS)
            USAR_LINHAS_SOLINFTEC = (MAPA_VEL_EFETIVO or MAPA_RPM_EFETIVO) and COLUNA_LINHA is not None

            if not (MAPA_AREA_EFETIVO or MAPA_VEL_EFETIVO or MAPA_RPM_EFETIVO):
                st.warning("⚠️ Nenhum mapa compatível com o tipo de arquivo enviado foi selecionado.")
                st.stop()

            colunas_obrigatorias = ["cd_fazenda"]
            if MAPA_AREA_EFETIVO:
                pass
            if (MAPA_VEL_EFETIVO or MAPA_RPM_EFETIVO) and not USAR_LINHAS_SOLINFTEC:
                colunas_obrigatorias.extend(["dt_hr_local_inicial", "vl_latitude_inicial", "vl_longitude_inicial", "cd_estado", "cd_operacao_parada", "cd_equipamento"])
            if MAPA_VEL_EFETIVO:
                colunas_obrigatorias.append("vl_velocidade")
            if MAPA_RPM_EFETIVO:
                colunas_obrigatorias.append("vl_rpm")

            faltantes_csv = validar_colunas(df, colunas_obrigatorias)
            if faltantes_csv:
                st.error("❌ Colunas obrigatórias faltantes: " + ", ".join(faltantes_csv))
                st.stop()

            df["cd_fazenda"] = df["cd_fazenda"].astype(str)
            if "cd_equipamento" in df.columns:
                df["cd_equipamento"] = df["cd_equipamento"].astype(str)
            if "dt_hr_local_inicial" in df.columns:
                df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
            for col in ["vl_latitude_inicial", "vl_longitude_inicial", "vl_largura_implemento", "vl_rpm", "vl_velocidade"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "vl_rpm" not in df.columns:
                df["vl_rpm"] = np.nan
            if "vl_velocidade" not in df.columns:
                df["vl_velocidade"] = np.nan
            if "vl_largura_implemento" not in df.columns:
                df["vl_largura_implemento"] = LARGURA_PADRAO_M

            df_area = df[df[COLUNA_POLIGONO].notna()].copy() if MAPA_AREA_EFETIVO else pd.DataFrame()
            if MAPA_AREA_EFETIVO and df_area.empty:
                st.error("❌ Nenhuma geometria WKT válida encontrada para Área Trabalhada.")
                st.stop()

            df_oper = df.copy()
            if MAPA_VEL_EFETIVO or MAPA_RPM_EFETIVO:
                if not USAR_LINHAS_SOLINFTEC:
                    df_oper = df_oper[(df_oper["cd_estado"] == "E") & (df_oper["cd_operacao_parada"] == -1)].copy()
                    df_oper = df_oper.dropna(subset=["dt_hr_local_inicial", "vl_latitude_inicial", "vl_longitude_inicial"])
                    if df_oper.empty:
                        st.error("❌ Nenhum ponto válido encontrado para Velocidade/RPM.")
                        st.stop()
                else:
                    df_oper = df_oper[df_oper[COLUNA_LINHA].notna()].copy()
                    if df_oper.empty:
                        st.error("❌ Nenhuma linha WKT válida encontrada para Velocidade/RPM.")
                        st.stop()

            base = gpd.read_file(BASE_PADRAO_PATH)
            faltantes_gpkg = validar_colunas(base, ["FAZENDA", "PROPRIEDADE", "geometry"])
            if faltantes_gpkg:
                st.error("❌ O GPKG não possui as colunas obrigatórias: " + ", ".join(faltantes_gpkg))
                st.stop()
            base["FAZENDA"] = base["FAZENDA"].astype(str)
            if "TALHAO" in base.columns:
                base["TALHAO"] = base["TALHAO"].astype(str)
            if "GLEBA" in base.columns:
                base["GLEBA"] = base["GLEBA"].astype(str)

            vel_faixas = gerar_faixas(VEL_MIN, VEL_MAX, VEL_PASSO, casas=1) if MAPA_VEL_EFETIVO else []
            rpm_faixas = gerar_faixas(RPM_MIN, RPM_MAX, RPM_PASSO, casas=0) if MAPA_RPM_EFETIVO else []
            vel_labels = [f[2] for f in vel_faixas]
            rpm_labels = [f[2] for f in rpm_faixas]
            vel_cores = dict(zip(vel_labels, amostrar_cores_classes(criar_cmap_suave("vel"), len(vel_labels)))) if MAPA_VEL_EFETIVO else {}
            rpm_cores = dict(zip(rpm_labels, amostrar_cores_classes(criar_cmap_suave("rpm"), len(rpm_labels)))) if MAPA_RPM_EFETIVO else {}

            fazendas_area = set(df_area["cd_fazenda"].dropna().astype(str).unique()) if MAPA_AREA_EFETIVO else set()
            fazendas_oper = set(df_oper["cd_fazenda"].dropna().astype(str).unique()) if (MAPA_VEL_EFETIVO or MAPA_RPM_EFETIVO) else set()
            fazendas_processar = sorted(fazendas_area.union(fazendas_oper), key=chave_ordenacao_mista)

            mapas_gerados_total = 0

            for FAZENDA_ID in fazendas_processar:
                df_faz_area = df_area[df_area["cd_fazenda"] == FAZENDA_ID].copy() if MAPA_AREA_EFETIVO else pd.DataFrame()
                df_faz_oper = df_oper[df_oper["cd_fazenda"] == FAZENDA_ID].copy() if (MAPA_VEL_EFETIVO or MAPA_RPM_EFETIVO) else pd.DataFrame()
                base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()
                if base_fazenda.empty:
                    continue

                nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]
                base_fazenda = base_fazenda.to_crs(epsg=CRS_METRICO)
                geom_fazenda = unary_union(base_fazenda.geometry)
                periodo_ini, periodo_fim = obter_periodo(df_faz_area, df_faz_oper)

                area_trabalhada = None
                area_total_ha = round(geom_fazenda.area / 10000, 2)
                area_trab_ha = 0
                area_nao_ha = area_total_ha
                pct_trab = 0
                pct_nao = 100 if area_total_ha > 0 else 0
                df_talhoes = None

                if MAPA_AREA_EFETIVO and not df_faz_area.empty:
                    gdf_area = criar_gdf_wkt(df_faz_area, COLUNA_POLIGONO, crs="EPSG:4326")
                    if not gdf_area.empty:
                        gdf_area = gdf_area.to_crs(epsg=CRS_METRICO)
                        area_bruta_wkt = unary_union(gdf_area.geometry)
                        largura_media = calcular_largura_media_buffer(df_faz_area, pd.DataFrame())
                        if pd.notna(largura_media) and largura_media > 0 and MULTIPLICADOR_BUFFER_AREA > 0:
                            distancia_buffer = max(largura_media * MULTIPLICADOR_BUFFER_AREA, BUFFER_MINIMO_M)
                        else:
                            distancia_buffer = BUFFER_MINIMO_M
                        if distancia_buffer > 0:
                            area_trabalhada = area_bruta_wkt.buffer(distancia_buffer, join_style=2).buffer(-distancia_buffer * FATOR_RECUO_GAPS, join_style=2).buffer(0)
                        else:
                            area_trabalhada = area_bruta_wkt.buffer(0)
                        if AREA_MAX_BURACO_HA > 0:
                            area_trabalhada = preencher_buracos_pequenos(area_trabalhada, AREA_MAX_BURACO_HA * 10000)
                        area_trabalhada = area_trabalhada.intersection(geom_fazenda).buffer(0)
                        if not area_trabalhada.is_empty:
                            area_nao = geom_fazenda.difference(area_trabalhada)
                            area_trab_ha = round(area_trabalhada.area / 10000, 2)
                            area_nao_ha = round(area_nao.area / 10000, 2)
                            if area_trab_ha >= AREA_MIN_HA:
                                pct_trab = round(area_trab_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0
                                pct_nao = round(area_nao_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0
                            else:
                                area_trabalhada = None

                            if area_trabalhada is not None and MOSTRAR_TALHOES and "TALHAO" in base_fazenda.columns and "GLEBA" in base_fazenda.columns:
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
                                    "Área trabalhada (ha)": [round(df_talhoes["Área trabalhada (ha)"].sum(), 2)],
                                })
                                df_talhoes = ordenar_tabela_talhoes(pd.concat([df_talhoes, total_row], ignore_index=True))

                gdf_display = None
                if MAPA_VEL_EFETIVO or MAPA_RPM_EFETIVO:
                    if USAR_LINHAS_SOLINFTEC:
                        gdf_linhas = criar_linhas_por_linestring(df_faz_oper, COLUNA_LINHA, geom_fazenda)
                    else:
                        gdf_linhas = criar_linhas_por_pontos(df_faz_oper, geom_fazenda)
                    gdf_display = criar_poligonos_display(gdf_linhas, geom_fazenda)

                vel_validos = df_faz_oper["vl_velocidade"].dropna() if MAPA_VEL_EFETIVO and not df_faz_oper.empty else pd.Series(dtype=float)
                rpm_validos = df_faz_oper["vl_rpm"].dropna() if MAPA_RPM_EFETIVO and not df_faz_oper.empty else pd.Series(dtype=float)
                vel_med_real = round(vel_validos.mean(), 1) if not vel_validos.empty else np.nan
                rpm_med_real = round(rpm_validos.mean(), 0) if not rpm_validos.empty else np.nan

                if gdf_display is not None and not gdf_display.empty:
                    if MAPA_VEL_EFETIVO:
                        gdf_display["classe_vel"] = gdf_display["vel_media"].apply(lambda x: classificar_valor(x, vel_faixas))
                    if MAPA_RPM_EFETIVO:
                        gdf_display["classe_rpm"] = gdf_display["rpm_medio"].apply(lambda x: classificar_valor(x, rpm_faixas))

                df_leg_vel = calcular_legenda_percentual(gdf_display, "classe_vel", vel_faixas, vel_cores) if MAPA_VEL_EFETIVO else pd.DataFrame()
                df_leg_rpm = calcular_legenda_percentual(gdf_display, "classe_rpm", rpm_faixas, rpm_cores) if MAPA_RPM_EFETIVO else pd.DataFrame()

                with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):
                    if MAPA_AREA_EFETIVO and area_trabalhada is not None and not area_trabalhada.is_empty:
                        fig_area = criar_figura_area(base_fazenda, area_trabalhada, area_total_ha, area_trab_ha, area_nao_ha, pct_trab, pct_nao, periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda)
                        st.pyplot(fig_area)
                        figuras_pdf_area = [fig_area]
                        figuras_tabela_pdf = []
                        if MOSTRAR_TALHOES and df_talhoes is not None and not df_talhoes.empty:
                            figuras_tabela_pdf = criar_figuras_tabela_talhoes_pdf(df_talhoes, FAZENDA_ID, nome_fazenda, linhas_por_pagina=12)
                            figuras_pdf_area.extend(figuras_tabela_pdf)
                        pdf_area = figuras_para_pdf_multipaginas(figuras_pdf_area)
                        st.download_button("⬇️ Baixar PDF vetorial – Área Trabalhada", data=pdf_area, file_name=f"mapa_area_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_area_{FAZENDA_ID}")
                        mapas_gerados_total += 1
                        plt.close(fig_area)
                        for fig_tab in figuras_tabela_pdf:
                            plt.close(fig_tab)
                        del pdf_area, figuras_pdf_area, figuras_tabela_pdf

                    # Ordem solicitada: Velocidade antes de RPM
                    if MAPA_VEL_EFETIVO:
                        faixa_vel_ini = arredondar_para_baixo(VEL_MIN, VEL_PASSO)
                        faixa_vel_fim = arredondar_para_cima(VEL_MAX, VEL_PASSO)
                        fig_vel = criar_figura_tematica(
                            base_fazenda, gdf_display, "classe_vel", vel_cores, df_leg_vel,
                            "Mapa de Velocidade", "Legenda de Velocidade",
                            f"< {formatar_numero(faixa_vel_ini, 1)} | {formatar_numero(faixa_vel_ini, 1)} até {formatar_numero(faixa_vel_fim, 1)}+ km/h",
                            f"Vel. média: {formatar_numero(vel_med_real, 1)} km/h",
                            periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda,
                        )
                        st.pyplot(fig_vel)
                        pdf_vel = figura_para_pdf_bytes(fig_vel)
                        st.download_button("⬇️ Baixar PDF vetorial – Velocidade", data=pdf_vel, file_name=f"mapa_velocidade_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_vel_{FAZENDA_ID}")
                        mapas_gerados_total += 1
                        plt.close(fig_vel)
                        del pdf_vel

                    if MAPA_RPM_EFETIVO:
                        faixa_rpm_ini = int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))
                        faixa_rpm_fim = int(arredondar_para_cima(RPM_MAX, RPM_PASSO))
                        fig_rpm = criar_figura_tematica(
                            base_fazenda, gdf_display, "classe_rpm", rpm_cores, df_leg_rpm,
                            "Mapa de RPM", "Legenda de RPM",
                            f"< {faixa_rpm_ini} | {faixa_rpm_ini} até {faixa_rpm_fim}+",
                            f"RPM médio: {formatar_numero(rpm_med_real, 0)}",
                            periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda,
                        )
                        st.pyplot(fig_rpm)
                        pdf_rpm = figura_para_pdf_bytes(fig_rpm)
                        st.download_button("⬇️ Baixar PDF vetorial – RPM", data=pdf_rpm, file_name=f"mapa_rpm_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_rpm_{FAZENDA_ID}")
                        mapas_gerados_total += 1
                        plt.close(fig_rpm)
                        del pdf_rpm

                    if df_talhoes is not None:
                        st.markdown("### 🌾 Área por Gleba / Talhão")
                        df_talhoes_exibicao = preparar_tabela_talhoes_exportacao(df_talhoes)
                        st.dataframe(df_talhoes_exibicao, use_container_width=True, hide_index=True)
                        zip_csv_talhoes = criar_zip_csv_talhoes(df_talhoes_exibicao, nome_csv=f"area_por_talhao_{FAZENDA_ID}.csv")
                        st.download_button("⬇️ Baixar ZIP com CSV – Área por Gleba / Talhão", data=zip_csv_talhoes, file_name=f"area_por_talhao_{FAZENDA_ID}.zip", mime="application/zip", key=f"zip_csv_talhoes_{FAZENDA_ID}")
                        del df_talhoes_exibicao, zip_csv_talhoes

                del df_faz_area, df_faz_oper, base_fazenda, geom_fazenda, gdf_display
                gc.collect()

            if mapas_gerados_total == 0:
                st.warning("⚠️ Não foi possível gerar nenhum mapa com os dados enviados.")

else:
    st.info("⬆️ Envie os ZIPs com CSVs e clique em **Gerar mapa**.")
