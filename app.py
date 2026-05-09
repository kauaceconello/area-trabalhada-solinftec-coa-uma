
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
    [data-testid="stInfo"], [data-testid="stWarning"], [data-testid="stError"] {
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
            Mapas de Área Trabalhada, Área por Colhedora/Operador, Velocidade e RPM com dados da Solinftec.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# UTILITÁRIOS
# =========================================================
def sidebar_container():
    try:
        return st.sidebar.container(border=True)
    except TypeError:
        return st.sidebar.container()


def normalizar_codigo(valor):
    if pd.isna(valor):
        return ""
    texto = str(valor).strip()
    if texto.endswith(".0"):
        texto = texto[:-2]
    return texto


def chave_ordenacao_mista(valor):
    texto = str(valor).strip()
    return re.sub(r"\d+", lambda m: f"{int(m.group()):010d}", texto)


def formatar_numero(valor, casas=0):
    if pd.isna(valor):
        return "-"
    if casas == 0:
        return f"{int(round(float(valor)))}"
    return f"{float(valor):.{casas}f}".replace(".", ",")


def formatar_area_ha(valor):
    if pd.isna(valor):
        return "-"
    return f"{float(valor):.2f}".replace(".", ",") + " ha"


def arredondar_para_baixo(valor, base):
    return np.floor(valor / base) * base


def arredondar_para_cima(valor, base):
    return np.ceil(valor / base) * base


def validar_colunas(df, colunas):
    return [c for c in colunas if c not in df.columns]


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
            df = pd.read_csv(csv_path, engine="python", **cfg)
            if len(df.columns) > 1:
                return df
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


def detectar_coluna_geometria(df, tipos):
    tipos_upper = [t.upper() for t in tipos]
    candidatos = ["wkt", "WKT", "geometry", "GEOMETRY", "geom", "GEOM", "the_geom", "THE_GEOM", "linha", "LINHA", "line", "LINE"]
    for col in candidatos:
        if col in df.columns:
            serie = df[col].dropna().astype(str).head(100).str.upper()
            if any(serie.str.contains(t, regex=False).any() for t in tipos_upper):
                return col
    for col in df.columns:
        serie = df[col].dropna().astype(str).head(100).str.upper()
        if not serie.empty and any(serie.str.contains(t, regex=False).any() for t in tipos_upper):
            return col
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
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


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


def obter_periodo(df1=None, df2=None):
    candidatos = []
    for df_tmp in [df1, df2]:
        if df_tmp is None or df_tmp.empty:
            continue
        for col in ["dt_hr_local_inicial", "dt_hr_local_final"]:
            if col in df_tmp.columns:
                vals = pd.to_datetime(df_tmp[col], errors="coerce").dropna()
                if not vals.empty:
                    candidatos.append(vals)
    if not candidatos:
        return "-", "-"
    datas = pd.concat(candidatos)
    return datas.min().strftime("%d/%m/%Y %H:%M"), datas.max().strftime("%d/%m/%Y %H:%M")


def calcular_largura_media(df):
    if df is None or df.empty or "vl_largura_implemento" not in df.columns:
        return np.nan
    s = pd.to_numeric(df["vl_largura_implemento"], errors="coerce").dropna()
    s = s[s > 0]
    return float(s.mean()) if not s.empty else np.nan


def classificar_turno(dt):
    if pd.isna(dt):
        return None
    hora = pd.to_datetime(dt).hour
    if 0 <= hora < 6:
        return "Turno C"
    if 6 <= hora < 15:
        return "Turno A"
    return "Turno B"


def intervalo_turno(turno):
    return {
        "Turno C": "00:00 às 06:00",
        "Turno A": "06:00 às 15:00",
        "Turno B": "15:00 às 23:59",
    }.get(turno, "")


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
# CLASSIFICAÇÃO DE VELOCIDADE/RPM
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


def criar_cmap_suave(tipo="vel"):
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


def calcular_legenda_percentual(gdf_linhas, coluna_classe, faixas, mapa_cores):
    if gdf_linhas is None or gdf_linhas.empty or coluna_classe not in gdf_linhas.columns:
        return pd.DataFrame(columns=["cor", "faixa", "percentual"])
    dados = gdf_linhas.dropna(subset=[coluna_classe]).copy()
    if dados.empty:
        return pd.DataFrame(columns=["cor", "faixa", "percentual"])
    total_tempo = dados["duracao_seg"].fillna(0).sum() if "duracao_seg" in dados.columns else 0
    usar_contagem = total_tempo <= 0
    linhas = []
    for _, _, label in faixas:
        sub = dados[dados[coluna_classe] == label]
        if usar_contagem:
            pct = len(sub) / len(dados) * 100 if len(dados) else 0
        else:
            pct = sub["duracao_seg"].fillna(0).sum() / total_tempo * 100 if total_tempo else 0
        linhas.append({"cor": mapa_cores.get(label, "#cccccc"), "faixa": label, "percentual": pct})
    return pd.DataFrame(linhas)

# =========================================================
# LINHAS / PONTOS
# =========================================================
def adicionar_segmento_clipado(linhas_saida, pontos, rpms, vels, t_inicio, t_fim, geom_fazenda):
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
    rpm = float(np.nanmean(rpms)) if len(rpms) else np.nan
    vel = float(np.nanmean(vels)) if len(vels) else np.nan
    duracao = (t_fim - t_inicio).total_seconds() if t_inicio is not None and t_fim is not None else np.nan
    if linha_clip.geom_type == "LineString":
        geoms = [linha_clip]
    elif linha_clip.geom_type == "MultiLineString":
        geoms = [g for g in linha_clip.geoms if not g.is_empty and g.length > 0]
    else:
        geoms = []
    for g in geoms:
        linhas_saida.append({"geometry": g, "rpm_medio": rpm, "vel_media": vel, "duracao_seg": duracao, "largura_media": LARGURA_PADRAO_M})


def criar_linhas_por_pontos(df_faz, geom_fazenda):
    gdf_pts = gpd.GeoDataFrame(
        df_faz,
        geometry=gpd.points_from_xy(df_faz["vl_longitude_inicial"], df_faz["vl_latitude_inicial"]),
        crs="EPSG:4326",
    ).to_crs(epsg=CRS_METRICO)
    linhas = []
    for _, grupo in gdf_pts.groupby("cd_equipamento"):
        grupo = grupo.sort_values("dt_hr_local_inicial")
        linha_atual, rpm_atual, vel_atual = [], [], []
        tempo_inicio, ultimo_tempo = None, None
        for _, row in grupo.iterrows():
            tempo = row["dt_hr_local_inicial"]
            if ultimo_tempo is None:
                linha_atual = [row.geometry]
                rpm_atual = [row.get("vl_rpm", np.nan)]
                vel_atual = [row.get("vl_velocidade", np.nan)]
                tempo_inicio = tempo
            else:
                delta = (tempo - ultimo_tempo).total_seconds()
                if delta <= TEMPO_MAX_SEG:
                    linha_atual.append(row.geometry)
                    rpm_atual.append(row.get("vl_rpm", np.nan))
                    vel_atual.append(row.get("vl_velocidade", np.nan))
                else:
                    adicionar_segmento_clipado(linhas, linha_atual, rpm_atual, vel_atual, tempo_inicio, ultimo_tempo, geom_fazenda)
                    linha_atual = [row.geometry]
                    rpm_atual = [row.get("vl_rpm", np.nan)]
                    vel_atual = [row.get("vl_velocidade", np.nan)]
                    tempo_inicio = tempo
            ultimo_tempo = tempo
        adicionar_segmento_clipado(linhas, linha_atual, rpm_atual, vel_atual, tempo_inicio, ultimo_tempo, geom_fazenda)
    return gpd.GeoDataFrame(linhas, geometry="geometry", crs=f"EPSG:{CRS_METRICO}") if linhas else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}")


def criar_linhas_por_wkt(df_faz, coluna_linha, geom_fazenda):
    gdf = criar_gdf_wkt(df_faz, coluna_linha, crs="EPSG:4326")
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
            item = row.to_dict()
            item.update({"geometry": g, "rpm_medio": rpm, "vel_media": vel, "duracao_seg": duracao, "largura_media": largura})
            registros.append(item)
    return gpd.GeoDataFrame(registros, geometry="geometry", crs=f"EPSG:{CRS_METRICO}") if registros else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}")


def criar_area_colhedora_por_linhas(df_faz_turno, coluna_linha, geom_fazenda):
    gdf = criar_gdf_wkt(df_faz_turno, coluna_linha, crs="EPSG:4326")
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["cd_equipamento", "geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}"), pd.DataFrame()
    gdf = gdf.to_crs(epsg=CRS_METRICO)
    registros = []
    linhas_legenda = []
    for equipamento, grupo in gdf.groupby("cd_equipamento"):
        geoms_buffer = []
        for _, row in grupo.iterrows():
            geom = row.geometry.intersection(geom_fazenda)
            if geom.is_empty:
                continue
            largura = pd.to_numeric(row.get("vl_largura_implemento", LARGURA_PADRAO_M), errors="coerce")
            if pd.isna(largura) or largura <= 0:
                largura = LARGURA_PADRAO_M
            try:
                geoms_buffer.append(geom.buffer(largura / 2.0, cap_style=2, join_style=2, quad_segs=1))
            except Exception:
                continue
        if not geoms_buffer:
            continue
        area_geom = unary_union(geoms_buffer).intersection(geom_fazenda).buffer(0)
        if area_geom.is_empty:
            continue
        registros.append({"cd_equipamento": str(equipamento), "geometry": area_geom})

        operadores = []
        if "cd_operador" in grupo.columns and "desc_operador" in grupo.columns:
            ops = grupo[["cd_operador", "desc_operador"]].dropna(how="all").drop_duplicates()
            for _, op in ops.iterrows():
                cd = normalizar_codigo(op.get("cd_operador", ""))
                desc = str(op.get("desc_operador", "")).strip()
                if cd and desc:
                    operadores.append(f"{cd} - {desc}")
                elif cd:
                    operadores.append(cd)
                elif desc:
                    operadores.append(desc)
        operadores_txt = "; ".join(sorted(set(operadores), key=chave_ordenacao_mista)) if operadores else "-"
        linhas_legenda.append({
            "Colhedora": str(equipamento),
            "Operadores": operadores_txt,
            "Área trabalhada (ha)": round(area_geom.area / 10000, 2),
        })
    gdf_saida = gpd.GeoDataFrame(registros, geometry="geometry", crs=f"EPSG:{CRS_METRICO}") if registros else gpd.GeoDataFrame(columns=["cd_equipamento", "geometry"], geometry="geometry", crs=f"EPSG:{CRS_METRICO}")
    df_legenda = pd.DataFrame(linhas_legenda)
    if not df_legenda.empty:
        df_legenda = df_legenda.sort_values("Colhedora", key=lambda s: s.map(chave_ordenacao_mista)).reset_index(drop=True)
    return gdf_saida, df_legenda


# =========================================================
# DISPLAY CARTOGRÁFICO
# =========================================================
def adicionar_moldura_layout(fig):
    """Aplica a moldura e o painel do mapa no mesmo padrão visual dos mapas anteriores."""
    moldura = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="#D8E1EB",
        linewidth=1.0,
        zorder=0,
    )
    fig.add_artist(moldura)
    painel_mapa = mpatches.FancyBboxPatch(
        (0.03, 0.10), 0.64, 0.78,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        transform=fig.transFigure,
        facecolor="#FFFFFF",
        edgecolor="#D8E1EB",
        linewidth=1.0,
        zorder=0,
    )
    fig.add_artist(painel_mapa)


def criar_poligonos_display(gdf_linhas, geom_fazenda):
    """Transforma linhas operacionais em faixas, mantendo o visual antigo de Velocidade/RPM."""
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
            "duracao_seg": row.get("duracao_seg", np.nan),
            "largura_media": largura,
        })
    return gpd.GeoDataFrame(registros, geometry="geometry", crs=gdf_linhas.crs) if registros else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf_linhas.crs)

# =========================================================
# FIGURAS / PDF
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


def adicionar_footer(fig, cor="#64748B"):
    brasilia = pytz.timezone("America/Sao_Paulo")
    hora = datetime.now(brasilia).strftime("%d/%m/%Y %H:%M")
    fig.text(0.50, 0.030, "Relatório elaborado com base em dados da Solinftec • Resultados dependem da qualidade dos dados operacionais e geoespaciais.", ha="center", fontsize=8.8, color=cor)
    fig.text(0.50, 0.012, f"Desenvolvido por Kauã Ceconello • Gerado em {hora}", ha="center", fontsize=8.6, color=cor)


def adicionar_header(fig, titulo, fazenda_id, nome_fazenda, periodo_ini, periodo_fim):
    axh = fig.add_axes([0.025, 0.905, 0.95, 0.08])
    axh.axis("off")
    card = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0)
    axh.add_patch(card)

    # Evita que o período passe por cima de títulos longos.
    fonte_titulo = 14.8 if len(str(titulo)) > 48 else 16
    axh.text(0.03, 0.64, titulo, fontsize=fonte_titulo, weight="bold", color="#0F172A", ha="left", va="center")
    axh.text(0.03, 0.26, f"Fazenda {fazenda_id} • {nome_fazenda}", fontsize=10.2, color="#475569", ha="left", va="center")

    # Período na linha inferior direita, separado do título.
    axh.text(0.97, 0.26, f"Período: {periodo_ini} até {periodo_fim}", fontsize=9.2, color="#64748B", ha="right", va="center")


def ajustar_extensao(ax, base_fazenda):
    minx, miny, maxx, maxy = base_fazenda.total_bounds
    dx, dy = maxx - minx, maxy - miny
    ax.set_xlim(minx - max(dx * 0.02, 0.35), maxx + max(dx * 0.02, 0.35))
    ax.set_ylim(miny - max(dy * 0.03, 0.70), maxy + max(dy * 0.03, 0.70))
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


def criar_cores_distintas(chaves):
    """Cores vivas, alternadas e com alto contraste para diferenciar colhedoras."""
    chaves = list(chaves)
    paleta_viva = [
        "#FF1744",  # vermelho vivo
        "#00E676",  # verde vivo
        "#2979FF",  # azul vivo
        "#FFEA00",  # amarelo vivo
        "#D500F9",  # roxo/magenta vivo
        "#FF6D00",  # laranja vivo
        "#00E5FF",  # ciano vivo
        "#76FF03",  # lima vivo
        "#F50057",  # rosa forte
        "#651FFF",  # violeta forte
        "#00C853",  # verde escuro vivo
        "#FF9100",  # âmbar vivo
        "#0091EA",  # azul forte
        "#C6FF00",  # verde-limão
        "#AA00FF",  # púrpura vivo
        "#FF3D00",  # vermelho alaranjado
        "#1DE9B6",  # turquesa vivo
        "#FFD600",  # dourado vivo
        "#304FFE",  # índigo vivo
        "#64DD17",  # verde claro vivo
    ]
    cores = {}
    for i, chave in enumerate(chaves):
        if i < len(paleta_viva):
            cores[chave] = paleta_viva[i]
        else:
            cores[chave] = to_hex(plt.cm.hsv((i * 0.61803398875) % 1.0))
    return cores


def criar_figura_area(base_fazenda, area_trabalhada, area_total_ha, area_trab_ha, area_nao_ha, pct_trab, pct_nao, periodo_ini, periodo_fim, fazenda_id, nome_fazenda):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    adicionar_moldura_layout(fig)
    adicionar_header(fig, "Mapa de Área Trabalhada", fazenda_id, nome_fazenda, periodo_ini, periodo_fim)
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    base_fazenda.plot(ax=ax, facecolor="#E5E7EB", edgecolor="#334155", linewidth=1.0, zorder=1)
    if area_trabalhada is not None and not area_trabalhada.is_empty:
        gpd.GeoSeries([area_trabalhada], crs=base_fazenda.crs).plot(ax=ax, color="#22C55E", alpha=0.88, zorder=2)
    base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=3)
    plotar_rotulos_talhao(ax, base_fazenda)
    ajustar_extensao(ax, base_fazenda)

    axr = fig.add_axes([0.71, 0.23, 0.25, 0.48])
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.axis("off")
    box = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0)
    axr.add_patch(box)
    axr.text(0.08, 0.92, "Resumo da Operação", fontsize=12, weight="bold", color="#0F172A")
    axr.text(0.08, 0.77, "Área total", fontsize=8.7, color="#64748B")
    axr.text(0.92, 0.77, f"{area_total_ha} ha", fontsize=10.2, color="#0F172A", ha="right", weight="bold")
    axr.text(0.08, 0.62, "Trabalhada", fontsize=8.7, color="#64748B")
    axr.text(0.92, 0.62, f"{area_trab_ha} ha ({pct_trab}%)", fontsize=10.0, color="#16A34A", ha="right", weight="bold")
    axr.text(0.08, 0.47, "Não trabalhada", fontsize=8.7, color="#64748B")
    axr.text(0.92, 0.47, f"{area_nao_ha} ha ({pct_nao}%)", fontsize=10.0, color="#475569", ha="right", weight="bold")
    axr.add_patch(mpatches.FancyBboxPatch((0.08, 0.28), 0.84, 0.06, boxstyle="round,pad=0.004,rounding_size=0.015", facecolor="#E5E7EB", edgecolor="none"))
    axr.add_patch(mpatches.FancyBboxPatch((0.08, 0.28), 0.84 * min(max(pct_trab / 100, 0), 1), 0.06, boxstyle="round,pad=0.004,rounding_size=0.015", facecolor="#22C55E", edgecolor="none"))
    axr.text(0.50, 0.20, f"Cobertura operacional: {pct_trab}%", fontsize=9.6, color="#0F172A", ha="center", weight="bold")
    adicionar_footer(fig)
    return fig


def criar_figura_area_colhedora(base_fazenda, gdf_area_colhedora, df_legenda, cores, turno, periodo_txt, fazenda_id, nome_fazenda):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    adicionar_moldura_layout(fig)
    titulo = f"Mapa de Área por Colhedora/Operador • {turno} ({intervalo_turno(turno)})"
    adicionar_header(fig, titulo, fazenda_id, nome_fazenda, periodo_txt, periodo_txt)

    # Mesmo padrão visual dos demais mapas: mapa à esquerda e resumo/legenda à direita.
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    base_fazenda.plot(ax=ax, facecolor="#FFFFFF", edgecolor="#334155", linewidth=1.0, zorder=1)
    if gdf_area_colhedora is not None and not gdf_area_colhedora.empty:
        for colhedora, cor in cores.items():
            sub = gdf_area_colhedora[gdf_area_colhedora["cd_equipamento"] == colhedora]
            if not sub.empty:
                sub.plot(ax=ax, color=cor, edgecolor="none", alpha=0.92, zorder=2)
    base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=3)
    plotar_rotulos_talhao(ax, base_fazenda)
    ajustar_extensao(ax, base_fazenda)

    axl = fig.add_axes([0.71, 0.16, 0.25, 0.68])
    axl.axis("off")
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    box = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        facecolor="#FFFFFF",
        edgecolor="#D8E1EB",
        linewidth=1.0,
    )
    axl.add_patch(box)
    axl.text(0.07, 0.955, "Resumo por Colhedora", fontsize=12, weight="bold", color="#0F172A", va="center")
    axl.text(0.07, 0.915, f"{turno} • {intervalo_turno(turno)}", fontsize=8.6, color="#64748B", va="center")
    axl.plot([0.07, 0.93], [0.885, 0.885], color="#E2E8F0", linewidth=1)

    if df_legenda is None or df_legenda.empty:
        axl.text(0.50, 0.50, "Sem dados válidos.", fontsize=9, color="#64748B", ha="center")
    else:
        linhas_legenda = []
        for _, row in df_legenda.iterrows():
            colhedora = str(row["Colhedora"])
            area_txt = formatar_area_ha(row["Área trabalhada (ha)"])
            cor = cores.get(colhedora, "#CCCCCC")
            operadores_raw = str(row["Operadores"])
            operadores = [op.strip() for op in operadores_raw.split(";") if op.strip()]
            if not operadores:
                operadores = ["-"]

            linhas_legenda.append({"tipo": "colhedora", "texto": f"{colhedora}: {area_txt}", "cor": cor})
            for operador in operadores:
                linhas_legenda.append({"tipo": "operador", "texto": operador, "cor": cor})

        total_linhas = len(linhas_legenda)
        topo_texto = 0.850
        base_texto = 0.075
        altura_disponivel = topo_texto - base_texto

        # Espaçamento adaptativo:
        # - poucos operadores: não espalha demais;
        # - muitos operadores: reduz o espaçamento para caber tudo;
        # - mantém distância equivalente entre as linhas.
        if total_linhas <= 1:
            espacamento = 0.050
        else:
            espacamento = min(0.052, altura_disponivel / max(total_linhas - 1, 1))

        altura_usada = espacamento * max(total_linhas - 1, 0)
        # Centraliza verticalmente quando há poucas linhas, evitando ficar tudo no topo ou exageradamente espalhado.
        y_inicio = min(topo_texto, base_texto + altura_disponivel / 2 + altura_usada / 2)

        fonte_operador = max(4.4, min(9.2, espacamento * 210))
        fonte_colhedora = max(5.0, min(10.4, fonte_operador + 1.1))
        tamanho_quadrado = max(0.017, min(0.040, espacamento * 1.20))

        if fonte_operador >= 8:
            limite_operador = 44
        elif fonte_operador >= 6.5:
            limite_operador = 39
        elif fonte_operador >= 5.2:
            limite_operador = 34
        else:
            limite_operador = 30

        for i, item in enumerate(linhas_legenda):
            y = y_inicio - i * espacamento
            if item["tipo"] == "colhedora":
                axl.add_patch(mpatches.FancyBboxPatch(
                    (0.07, y - tamanho_quadrado / 2),
                    tamanho_quadrado,
                    tamanho_quadrado,
                    boxstyle="round,pad=0.002,rounding_size=0.004",
                    facecolor=item["cor"],
                    edgecolor="none",
                ))
                txt = item["texto"]
                if len(txt) > 32 and fonte_colhedora < 6:
                    txt = txt[:29] + "..."
                axl.text(
                    0.125,
                    y,
                    txt,
                    fontsize=fonte_colhedora,
                    color="#0F172A",
                    weight="bold",
                    ha="left",
                    va="center",
                )
            else:
                operador_txt = item["texto"] if len(item["texto"]) <= limite_operador else item["texto"][:limite_operador - 3] + "..."
                axl.text(
                    0.125,
                    y,
                    f"• {operador_txt}",
                    fontsize=fonte_operador,
                    color="#475569",
                    ha="left",
                    va="center",
                )

    adicionar_footer(fig)
    return fig

def criar_figura_tematica(base_fazenda, gdf_linhas, coluna_classe, mapa_cores, df_legenda, titulo, titulo_legenda, faixa_txt, media_txt, periodo_ini, periodo_fim, fazenda_id, nome_fazenda):
    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("#F4F7FB")
    adicionar_moldura_layout(fig)
    adicionar_header(fig, titulo, fazenda_id, nome_fazenda, periodo_ini, periodo_fim)

    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66])
    base_fazenda.plot(ax=ax, facecolor="#FFFFFF", edgecolor="#334155", linewidth=1.0, zorder=1)
    if gdf_linhas is not None and not gdf_linhas.empty and coluna_classe in gdf_linhas.columns:
        for classe, cor in mapa_cores.items():
            sub = gdf_linhas[gdf_linhas[coluna_classe] == classe]
            if not sub.empty:
                sub.plot(ax=ax, color=cor, edgecolor="none", alpha=0.95, zorder=2)
    base_fazenda.boundary.plot(ax=ax, color="#0F172A", linewidth=1.1, zorder=3)
    plotar_rotulos_talhao(ax, base_fazenda)
    ajustar_extensao(ax, base_fazenda)

    axb = fig.add_axes([0.71, 0.16, 0.25, 0.68])
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.axis("off")
    box = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor="#FFFFFF", edgecolor="#D8E1EB", linewidth=1.0)
    axb.add_patch(box)
    axb.text(0.07, 0.945, titulo_legenda, fontsize=12, weight="bold", color="#0F172A", ha="left", va="center")
    axb.text(0.07, 0.895, f"Faixa exibida: {faixa_txt}", fontsize=8.8, color="#64748B", ha="left", va="center")
    chip = mpatches.FancyBboxPatch((0.07, 0.805), 0.86, 0.072, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=0.8)
    axb.add_patch(chip)
    axb.text(0.50, 0.841, media_txt, fontsize=10.0, color="#1D4ED8", weight="bold", ha="center", va="center")
    axb.plot([0.07, 0.93], [0.755, 0.755], color="#E2E8F0", linewidth=1)

    if not df_legenda.empty:
        topo, base_y = 0.695, 0.08
        row_h = (topo - base_y) / max(len(df_legenda), 1)
        for i, row in df_legenda.reset_index(drop=True).iterrows():
            y = topo - i * row_h
            pct_txt = f"{row['percentual']:.1f}%".replace(".", ",")
            axb.add_patch(mpatches.FancyBboxPatch((0.07, y - 0.020), 0.025, 0.025, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=row["cor"], edgecolor="none"))
            axb.text(0.11, y - 0.007, row["faixa"], fontsize=8.9, color="#0F172A", ha="left", va="center")
            axb.text(0.88, y - 0.007, pct_txt, fontsize=8.9, color="#334155", ha="center", va="center", weight="bold")
    else:
        axb.text(0.50, 0.55, "Sem dados válidos para exibir.", fontsize=9.2, color="#64748B", ha="center")

    adicionar_footer(fig)
    return fig


def criar_figura_tabela_talhoes_pdf(df_talhoes, fazenda_id, nome_fazenda, pagina_atual=1, total_paginas=1, area_total_trabalhada=None, area_total_fazenda=None):
    df_base = ordenar_tabela_talhoes(df_talhoes)
    df_dados = df_base[df_base["Gleba"].astype(str).str.upper() != "TOTAL"].copy()
    for col in ["Área total (ha)", "Área trabalhada (ha)"]:
        if col in df_dados.columns:
            df_dados[col] = pd.to_numeric(df_dados[col], errors="coerce").fillna(0).round(2)
    if area_total_trabalhada is None:
        area_total_trabalhada = pd.to_numeric(df_dados.get("Área trabalhada (ha)", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    if area_total_fazenda is None:
        area_total_fazenda = pd.to_numeric(df_dados.get("Área total (ha)", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#F4F7FB")
    axh = fig.add_axes([0.035, 0.895, 0.93, 0.08])
    axh.axis("off")
    axh.text(0.025, 0.62, "Área Trabalhada por Gleba / Talhão", fontsize=15.5, weight="bold", color="#0F172A")
    axh.text(0.025, 0.28, f"Fazenda {fazenda_id} • {nome_fazenda}", fontsize=9.8, color="#64748B")
    axh.text(0.975, 0.50, f"Página {pagina_atual} de {total_paginas}", fontsize=9.2, color="#64748B", ha="right")

    ax = fig.add_axes([0.065, 0.12, 0.87, 0.70])
    ax.axis("off")
    if df_dados.empty:
        ax.text(0.5, 0.5, "Sem dados de área por talhão.", ha="center", va="center", fontsize=11, color="#64748B")
        adicionar_footer(fig)
        return fig

    df_tab = df_dados[["Gleba", "Talhão", "Área total (ha)", "Área trabalhada (ha)"]].copy()
    df_tab["% Trabalhado"] = np.where(df_tab["Área total (ha)"] > 0, (df_tab["Área trabalhada (ha)"] / df_tab["Área total (ha)"] * 100).round(1), 0)
    df_tab["Área total (ha)"] = df_tab["Área total (ha)"].apply(formatar_area_ha)
    df_tab["Área trabalhada (ha)"] = df_tab["Área trabalhada (ha)"].apply(formatar_area_ha)
    df_tab["% Trabalhado"] = df_tab["% Trabalhado"].apply(lambda x: f"{x:.1f}%".replace(".", ","))
    tabela = ax.table(cellText=df_tab.values, colLabels=df_tab.columns, loc="center", cellLoc="center", colLoc="center")
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
    return [criar_figura_tabela_talhoes_pdf(df_pag, fazenda_id, nome_fazenda, idx, len(paginas_df), area_total_trabalhada, area_total_fazenda) for idx, df_pag in enumerate(paginas_df, start=1)]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parâmetros")

with sidebar_container():
    st.markdown("### 🗺️ Tipo de processamento")
    MODO_MAPA = st.selectbox(
        "Escolha o tipo de mapa",
        [
            "Área trabalhada (área)",
            "Colhedora/operador (linhas)",
            "Velocidade/RPM (linhas)",
        ],
        index=0,
        key="modo_mapa_selectbox",
    )

with sidebar_container():
    st.markdown("### 🧭 Base cartográfica")
    st.caption("Base padrão SOLINFTEC")

MAPA_AREA = MODO_MAPA == "Área trabalhada (área)"
MAPA_OPERADOR = MODO_MAPA == "Colhedora/operador (linhas)"
MAPA_VEL_RPM = MODO_MAPA == "Velocidade/RPM (linhas)"

if MAPA_AREA:
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
    MULTIPLICADOR_BUFFER_AREA = 2.5
    AREA_MIN_HA = 0.50
    BUFFER_MINIMO_M = 8.0
    FATOR_RECUO_GAPS = 0.30
    AREA_MAX_BURACO_HA = 0.50
    MOSTRAR_TALHOES = False

if MAPA_OPERADOR:
    with sidebar_container():
        st.markdown("### 🌾 Área por colhedora")
        AREA_MIN_OPERADOR_HA = st.number_input(
            "Área mínima para gerar mapa (ha)",
            min_value=0.0,
            value=0.50,
            step=0.10,
            key="area_min_operador_input",
        )
else:
    AREA_MIN_OPERADOR_HA = 0.50

RPM_MIN, RPM_MAX, RPM_PASSO = 1200, 2000, 100
VEL_MIN, VEL_MAX, VEL_PASSO = 4.0, 8.0, 1.0

if MAPA_VEL_RPM:
    with sidebar_container():
        st.markdown("### ⚙️ Parâmetros Velocidade (km/h)")
        VEL_MIN = st.number_input("Velocidade mínima", min_value=0.0, max_value=100.0, value=4.0, step=0.5, key="vel_min_input")
        VEL_MAX = st.number_input("Velocidade máxima", min_value=0.0, max_value=100.0, value=8.0, step=0.5, key="vel_max_input")
        VEL_PASSO = st.number_input("Passo das faixas Velocidade", min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="vel_passo_input")
    with sidebar_container():
        st.markdown("### ⚙️ Parâmetros RPM")
        RPM_MIN = st.number_input("RPM mínimo", min_value=0, max_value=10000, value=1200, step=100, key="rpm_min_input")
        RPM_MAX = st.number_input("RPM máximo", min_value=0, max_value=10000, value=2000, step=100, key="rpm_max_input")
        RPM_PASSO = st.number_input("Passo das faixas RPM", min_value=50, max_value=1000, value=100, step=50, key="rpm_passo_input")

if MAPA_VEL_RPM and RPM_MAX <= RPM_MIN:
    st.sidebar.error("⚠️ O RPM máximo deve ser maior que o RPM mínimo.")
if MAPA_VEL_RPM and VEL_MAX <= VEL_MIN:
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

# key dinâmica limpa o upload quando troca o tipo de mapa.
uploaded_zips = st.file_uploader(
    "📦 Upload dos ZIPs contendo CSVs da Solinftec",
    type=["zip"],
    accept_multiple_files=True,
    key=f"upload_{MODO_MAPA}",
)
GERAR = st.button("▶️ Gerar mapa")

if GERAR:
    st.session_state["mapas_gerados"] = True
if not uploaded_zips or not os.path.exists(BASE_PADRAO_PATH):
    st.session_state["mapas_gerados"] = False

st.caption(f"✅ Tipo selecionado: {MODO_MAPA}")

# =========================================================
# PROCESSAMENTO
# =========================================================
if uploaded_zips and os.path.exists(BASE_PADRAO_PATH) and st.session_state.get("mapas_gerados", False):
    if MAPA_VEL_RPM and RPM_MAX <= RPM_MIN:
        st.error("❌ Ajuste os parâmetros de RPM.")
        st.stop()
    if MAPA_VEL_RPM and VEL_MAX <= VEL_MIN:
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

            if "cd_fazenda" not in df.columns:
                st.error("❌ Coluna obrigatória faltante: cd_fazenda")
                st.stop()
            df["cd_fazenda"] = df["cd_fazenda"].apply(normalizar_codigo)
            if "cd_equipamento" in df.columns:
                df["cd_equipamento"] = df["cd_equipamento"].apply(normalizar_codigo)
            if "cd_operador" in df.columns:
                df["cd_operador"] = df["cd_operador"].apply(normalizar_codigo)
            if "dt_hr_local_inicial" in df.columns:
                df["dt_hr_local_inicial"] = pd.to_datetime(df["dt_hr_local_inicial"], errors="coerce")
            for col in ["vl_latitude_inicial", "vl_longitude_inicial", "vl_largura_implemento", "vl_rpm", "vl_velocidade"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            base = gpd.read_file(BASE_PADRAO_PATH)
            faltantes_gpkg = validar_colunas(base, ["FAZENDA", "PROPRIEDADE", "geometry"])
            if faltantes_gpkg:
                st.error("❌ O GPKG não possui as colunas obrigatórias: " + ", ".join(faltantes_gpkg))
                st.stop()
            base["FAZENDA"] = base["FAZENDA"].apply(normalizar_codigo)
            if "TALHAO" in base.columns:
                base["TALHAO"] = base["TALHAO"].apply(normalizar_codigo)
            if "GLEBA" in base.columns:
                base["GLEBA"] = base["GLEBA"].apply(normalizar_codigo)

            mapas_gerados_total = 0

            # =====================================================
            # MODO 1: ÁREA TRABALHADA PRINCIPAL - CSV ÁREA/WKT
            # =====================================================
            if MAPA_AREA:
                coluna_poligono = "wkt" if "wkt" in df.columns else detectar_coluna_geometria(df, ["MULTIPOLYGON", "POLYGON"])
                if coluna_poligono is None:
                    st.warning("⚠️ O modo Área Trabalhada precisa de um CSV de área da Solinftec.")
                    st.stop()
                mascara_poligono = df[coluna_poligono].notna() & df[coluna_poligono].astype(str).str.upper().str.contains("POLYGON", na=False)
                df_area = df[mascara_poligono].copy()
                if df_area.empty:
                    st.warning("⚠️ Nenhum dado de área válido encontrado no ZIP enviado.")
                    st.stop()
                fazendas_processar = sorted(df_area["cd_fazenda"].dropna().unique(), key=chave_ordenacao_mista)

                for FAZENDA_ID in fazendas_processar:
                    base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()
                    if base_fazenda.empty:
                        continue
                    nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]
                    base_fazenda = base_fazenda.to_crs(epsg=CRS_METRICO)
                    geom_fazenda = unary_union(base_fazenda.geometry)
                    df_faz_area = df_area[df_area["cd_fazenda"] == FAZENDA_ID].copy()
                    periodo_ini, periodo_fim = obter_periodo(df_faz_area, None)
                    gdf_area = criar_gdf_wkt(df_faz_area, coluna_poligono, crs="EPSG:4326")
                    if gdf_area.empty:
                        continue
                    gdf_area = gdf_area.to_crs(epsg=CRS_METRICO)
                    area_bruta = unary_union(gdf_area.geometry)
                    largura_media = calcular_largura_media(df_faz_area)
                    if pd.notna(largura_media) and largura_media > 0 and MULTIPLICADOR_BUFFER_AREA > 0:
                        dist = max(largura_media * MULTIPLICADOR_BUFFER_AREA, BUFFER_MINIMO_M)
                    else:
                        dist = BUFFER_MINIMO_M
                    if dist > 0:
                        area_trabalhada = area_bruta.buffer(dist, join_style=2).buffer(-dist * FATOR_RECUO_GAPS, join_style=2).buffer(0)
                    else:
                        area_trabalhada = area_bruta.buffer(0)
                    if AREA_MAX_BURACO_HA > 0:
                        area_trabalhada = preencher_buracos_pequenos(area_trabalhada, AREA_MAX_BURACO_HA * 10000)
                    area_trabalhada = area_trabalhada.intersection(geom_fazenda).buffer(0)
                    if area_trabalhada.is_empty:
                        continue
                    area_total_ha = round(geom_fazenda.area / 10000, 2)
                    area_trab_ha = round(area_trabalhada.area / 10000, 2)
                    if area_trab_ha < AREA_MIN_HA:
                        continue
                    area_nao_ha = round(max(area_total_ha - area_trab_ha, 0), 2)
                    pct_trab = round(area_trab_ha / area_total_ha * 100, 1) if area_total_ha > 0 else 0
                    pct_nao = round(100 - pct_trab, 1)

                    df_talhoes = None
                    if MOSTRAR_TALHOES and "TALHAO" in base_fazenda.columns and "GLEBA" in base_fazenda.columns:
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

                    with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):
                        fig_area = criar_figura_area(base_fazenda, area_trabalhada, area_total_ha, area_trab_ha, area_nao_ha, pct_trab, pct_nao, periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda)
                        st.pyplot(fig_area)
                        figuras = [fig_area]
                        figs_tab = []
                        if MOSTRAR_TALHOES and df_talhoes is not None and not df_talhoes.empty:
                            figs_tab = criar_figuras_tabela_talhoes_pdf(df_talhoes, FAZENDA_ID, nome_fazenda)
                            figuras.extend(figs_tab)
                        pdf_area = figuras_para_pdf_multipaginas(figuras)
                        st.download_button("⬇️ Baixar PDF vetorial – Área Trabalhada", data=pdf_area, file_name=f"mapa_area_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_area_{FAZENDA_ID}")
                        if df_talhoes is not None:
                            st.markdown("### 🌾 Área por Gleba / Talhão")
                            df_exp = preparar_tabela_talhoes_exportacao(df_talhoes)
                            st.dataframe(df_exp, use_container_width=True, hide_index=True)
                            zip_csv = criar_zip_csv_talhoes(df_exp, f"area_por_talhao_{FAZENDA_ID}.csv")
                            st.download_button("⬇️ Baixar ZIP com CSV – Área por Gleba / Talhão", data=zip_csv, file_name=f"area_por_talhao_{FAZENDA_ID}.zip", mime="application/zip", key=f"zip_csv_talhoes_{FAZENDA_ID}")
                        plt.close(fig_area)
                        for fig in figs_tab:
                            plt.close(fig)
                    mapas_gerados_total += 1

            # =====================================================
            # MODO 2: ÁREA POR COLHEDORA/OPERADOR - CSV LINHAS
            # =====================================================
            elif MAPA_OPERADOR:
                coluna_linha = detectar_coluna_geometria(df, ["LINESTRING", "MULTILINESTRING"])
                faltantes = validar_colunas(df, ["cd_equipamento", "cd_operador", "desc_operador", "dt_hr_local_inicial", "vl_largura_implemento"])
                if coluna_linha is None or faltantes:
                    st.warning("⚠️ O modo Área por Colhedora/Operador precisa de um CSV de linhas da Solinftec com colhedora, operador, horário e largura.")
                    st.stop()
                df_linhas = df[df[coluna_linha].notna()].copy()
                df_linhas = df_linhas[df_linhas[coluna_linha].astype(str).str.upper().str.contains("LINESTRING", na=False)].copy()
                df_linhas["turno"] = df_linhas["dt_hr_local_inicial"].apply(classificar_turno)
                df_linhas = df_linhas.dropna(subset=["turno"])
                if df_linhas.empty:
                    st.warning("⚠️ Nenhuma linha válida encontrada para separar por turno.")
                    st.stop()

                ordem_turnos = ["Turno C", "Turno A", "Turno B"]
                for turno in ordem_turnos:
                    df_turno = df_linhas[df_linhas["turno"] == turno].copy()
                    if df_turno.empty:
                        continue

                    with st.expander(f"🕒 {turno} ({intervalo_turno(turno)})", expanded=False):
                        fazendas_turno = sorted(df_turno["cd_fazenda"].dropna().unique(), key=chave_ordenacao_mista)
                        registros_turno = []

                        # Primeiro gera todos os mapas do turno em memória.
                        # Assim o botão do PDF único aparece no topo do expander do turno.
                        for FAZENDA_ID in fazendas_turno:
                            base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()
                            if base_fazenda.empty:
                                continue
                            nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]
                            base_fazenda = base_fazenda.to_crs(epsg=CRS_METRICO)
                            geom_fazenda = unary_union(base_fazenda.geometry)
                            df_faz_turno = df_turno[df_turno["cd_fazenda"] == FAZENDA_ID].copy()
                            periodo_ini, periodo_fim = obter_periodo(None, df_faz_turno)
                            periodo_txt = f"{periodo_ini} até {periodo_fim}" if periodo_ini != "-" else intervalo_turno(turno)
                            gdf_area_colhedora, df_legenda = criar_area_colhedora_por_linhas(df_faz_turno, coluna_linha, geom_fazenda)
                            if gdf_area_colhedora.empty or df_legenda.empty:
                                continue

                            area_total_mapa_ha = pd.to_numeric(df_legenda["Área trabalhada (ha)"], errors="coerce").fillna(0).sum()
                            if area_total_mapa_ha < AREA_MIN_OPERADOR_HA:
                                continue

                            colhedoras = df_legenda["Colhedora"].astype(str).tolist()
                            cores = criar_cores_distintas(colhedoras)
                            fig_op = criar_figura_area_colhedora(
                                base_fazenda,
                                gdf_area_colhedora,
                                df_legenda,
                                cores,
                                turno,
                                periodo_txt,
                                FAZENDA_ID,
                                nome_fazenda,
                            )
                            registros_turno.append({
                                "fazenda_id": FAZENDA_ID,
                                "nome_fazenda": nome_fazenda,
                                "fig": fig_op,
                            })

                        if not registros_turno:
                            st.info(f"Nenhum mapa acima de {AREA_MIN_OPERADOR_HA:.2f} ha foi gerado para este turno.".replace(".", ","))
                            continue

                        # Botão do PDF único fica antes dos mapas individuais para não ficar escondido no final.
                        figuras_turno = [r["fig"] for r in registros_turno]
                        pdf_turno = figuras_para_pdf_multipaginas(figuras_turno)
                        st.download_button(
                            f"⬇️ Baixar PDF único – Todas as fazendas do {turno}",
                            data=pdf_turno,
                            file_name=f"mapas_area_colhedora_operador_{turno.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            key=f"pdf_operador_turno_{turno}",
                        )

                        st.caption(f"PDF único contém {len(registros_turno)} mapa(s), com uma fazenda por página.")

                        # Depois exibe os mapas individuais e seus downloads separados.
                        for registro in registros_turno:
                            FAZENDA_ID = registro["fazenda_id"]
                            nome_fazenda = registro["nome_fazenda"]
                            fig_op = registro["fig"]
                            with st.expander(f"🗺️ {nome_fazenda}", expanded=False):
                                st.pyplot(fig_op)
                                pdf_op = figura_para_pdf_bytes(fig_op)
                                st.download_button(
                                    "⬇️ Baixar PDF vetorial – Área por Colhedora/Operador",
                                    data=pdf_op,
                                    file_name=f"mapa_area_colhedora_operador_{turno.replace(' ', '_')}_{FAZENDA_ID}.pdf",
                                    mime="application/pdf",
                                    key=f"pdf_operador_{turno}_{FAZENDA_ID}",
                                )
                                mapas_gerados_total += 1

                        for registro in registros_turno:
                            plt.close(registro["fig"])

            # =====================================================
            # MODO 3: VELOCIDADE E RPM - CSV LINHAS OU PONTOS
            # =====================================================
            else:
                coluna_linha = detectar_coluna_geometria(df, ["LINESTRING", "MULTILINESTRING"])
                tem_pontos = all(c in df.columns for c in ["vl_latitude_inicial", "vl_longitude_inicial"])
                if "vl_velocidade" not in df.columns or "vl_rpm" not in df.columns:
                    st.error("❌ O modo Velocidade/RPM precisa das colunas vl_velocidade e vl_rpm.")
                    st.stop()
                if coluna_linha is None and not tem_pontos:
                    st.warning("⚠️ O modo Velocidade/RPM precisa de um CSV de linhas ou de pontos da Solinftec.")
                    st.stop()

                usar_linhas = coluna_linha is not None
                if usar_linhas:
                    df_oper = df[df[coluna_linha].notna()].copy()
                    df_oper = df_oper[df_oper[coluna_linha].astype(str).str.upper().str.contains("LINESTRING", na=False)].copy()
                else:
                    faltantes_pontos = validar_colunas(df, ["dt_hr_local_inicial", "vl_latitude_inicial", "vl_longitude_inicial", "cd_estado", "cd_operacao_parada", "cd_equipamento"])
                    if faltantes_pontos:
                        st.error("❌ Colunas obrigatórias faltantes para pontos: " + ", ".join(faltantes_pontos))
                        st.stop()
                    df_oper = df[(df["cd_estado"] == "E") & (df["cd_operacao_parada"] == -1)].copy()
                    df_oper = df_oper.dropna(subset=["dt_hr_local_inicial", "vl_latitude_inicial", "vl_longitude_inicial"])
                if df_oper.empty:
                    st.warning("⚠️ Nenhum dado operacional válido encontrado para Velocidade/RPM.")
                    st.stop()

                fazendas = sorted(df_oper["cd_fazenda"].dropna().unique(), key=chave_ordenacao_mista)
                for FAZENDA_ID in fazendas:
                    base_fazenda = base[base["FAZENDA"] == FAZENDA_ID].copy()
                    if base_fazenda.empty:
                        continue
                    nome_fazenda = base_fazenda["PROPRIEDADE"].iloc[0]
                    base_fazenda = base_fazenda.to_crs(epsg=CRS_METRICO)
                    geom_fazenda = unary_union(base_fazenda.geometry)
                    df_faz = df_oper[df_oper["cd_fazenda"] == FAZENDA_ID].copy()
                    periodo_ini, periodo_fim = obter_periodo(None, df_faz)
                    if usar_linhas:
                        gdf_linhas = criar_linhas_por_wkt(df_faz, coluna_linha, geom_fazenda)
                    else:
                        gdf_linhas = criar_linhas_por_pontos(df_faz, geom_fazenda)
                    if gdf_linhas.empty:
                        continue
                    gdf_plot = criar_poligonos_display(gdf_linhas, geom_fazenda)
                    if gdf_plot.empty:
                        continue

                    vel_faixas = gerar_faixas(VEL_MIN, VEL_MAX, VEL_PASSO, casas=1)
                    vel_labels = [f[2] for f in vel_faixas]
                    vel_cores = dict(zip(vel_labels, amostrar_cores_classes(criar_cmap_suave("vel"), len(vel_labels))))
                    gdf_plot["classe_vel"] = gdf_plot["vel_media"].apply(lambda x: classificar_valor(x, vel_faixas))
                    df_leg_vel = calcular_legenda_percentual(gdf_plot, "classe_vel", vel_faixas, vel_cores)
                    vel_validos = pd.to_numeric(df_faz["vl_velocidade"], errors="coerce").dropna()
                    vel_med = round(vel_validos.mean(), 1) if not vel_validos.empty else np.nan

                    rpm_faixas = gerar_faixas(RPM_MIN, RPM_MAX, RPM_PASSO, casas=0)
                    rpm_labels = [f[2] for f in rpm_faixas]
                    rpm_cores = dict(zip(rpm_labels, amostrar_cores_classes(criar_cmap_suave("rpm"), len(rpm_labels))))
                    gdf_plot["classe_rpm"] = gdf_plot["rpm_medio"].apply(lambda x: classificar_valor(x, rpm_faixas))
                    df_leg_rpm = calcular_legenda_percentual(gdf_plot, "classe_rpm", rpm_faixas, rpm_cores)
                    rpm_validos = pd.to_numeric(df_faz["vl_rpm"], errors="coerce").dropna()
                    rpm_med = round(rpm_validos.mean(), 0) if not rpm_validos.empty else np.nan

                    with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):
                        faixa_ini = arredondar_para_baixo(VEL_MIN, VEL_PASSO)
                        faixa_fim = arredondar_para_cima(VEL_MAX, VEL_PASSO)
                        fig_vel = criar_figura_tematica(
                            base_fazenda, gdf_plot, "classe_vel", vel_cores, df_leg_vel,
                            "Mapa de Velocidade", "Legenda de Velocidade",
                            f"< {formatar_numero(faixa_ini, 1)} | {formatar_numero(faixa_ini, 1)} até {formatar_numero(faixa_fim, 1)}+ km/h",
                            f"Vel. média: {formatar_numero(vel_med, 1)} km/h",
                            periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda,
                        )
                        st.pyplot(fig_vel)
                        pdf_vel = figura_para_pdf_bytes(fig_vel)
                        st.download_button("⬇️ Baixar PDF vetorial – Velocidade", data=pdf_vel, file_name=f"mapa_velocidade_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_vel_{FAZENDA_ID}")
                        plt.close(fig_vel)
                        mapas_gerados_total += 1

                        faixa_ini_rpm = int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))
                        faixa_fim_rpm = int(arredondar_para_cima(RPM_MAX, RPM_PASSO))
                        fig_rpm = criar_figura_tematica(
                            base_fazenda, gdf_plot, "classe_rpm", rpm_cores, df_leg_rpm,
                            "Mapa de RPM", "Legenda de RPM",
                            f"< {faixa_ini_rpm} | {faixa_ini_rpm} até {faixa_fim_rpm}+",
                            f"RPM médio: {formatar_numero(rpm_med, 0)}",
                            periodo_ini, periodo_fim, FAZENDA_ID, nome_fazenda,
                        )
                        st.pyplot(fig_rpm)
                        pdf_rpm = figura_para_pdf_bytes(fig_rpm)
                        st.download_button("⬇️ Baixar PDF vetorial – RPM", data=pdf_rpm, file_name=f"mapa_rpm_{FAZENDA_ID}.pdf", mime="application/pdf", key=f"pdf_rpm_{FAZENDA_ID}")
                        plt.close(fig_rpm)
                        mapas_gerados_total += 1

            if mapas_gerados_total == 0:
                st.warning("⚠️ Não foi possível gerar nenhum mapa com os dados enviados. Confira se o modo escolhido combina com o arquivo enviado da Solinftec.")

else:
    st.info("⬆️ Envie os ZIPs com CSVs e clique em **Gerar mapa**.")
