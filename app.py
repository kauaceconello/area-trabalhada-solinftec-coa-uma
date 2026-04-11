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
from matplotlib.colors import LinearSegmentedColormap, to_hex

import zipfile
import tempfile
import os
import pytz
from datetime import datetime

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
    Gera faixas arredondadas, por exemplo:
    1700 a 1800, 1800 a 1900, ..., 2000+
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
    Colormap suave para visual mais bonito.
    """
    if tipo == "rpm":
        cores = [
            "#08306b",  # azul escuro
            "#2171b5",  # azul
            "#6baed6",  # azul claro
            "#74c476",  # verde claro
            "#31a354",  # verde
            "#fed976",  # amarelo
            "#fd8d3c",  # laranja
            "#e31a1c",  # vermelho
        ]
    else:
        cores = [
            "#08519c",  # azul escuro
            "#3182bd",  # azul
            "#6baed6",  # azul claro
            "#7bccc4",  # verde água
            "#41ab5d",  # verde
            "#d9ef8b",  # amarelo esverdeado
            "#fdae61",  # laranja
            "#d73027",  # vermelho
        ]

    return LinearSegmentedColormap.from_list(f"cmap_{tipo}", cores, N=256)

def amostrar_cores_classes(cmap, n_classes):
    """
    Retorna cores suaves para cada faixa.
    """
    if n_classes <= 1:
        return [to_hex(cmap(0.5))]
    pontos = np.linspace(0.08, 0.95, n_classes)
    return [to_hex(cmap(x)) for x in pontos]

def classificar_valor(valor, faixas):
    """
    Retorna o label da faixa em que o valor cai.
    """
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

def desenhar_base_mapa(ax, base_fazenda, facecolor="#f7f7f7", mostrar_talhoes=True):
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

def desenhar_legenda_horizontal(fig, pos_ax, df_legenda, titulo):
    """
    Legenda horizontal estilo faixa + % tempo,
    posicionada onde antes ficava a legenda do mapa de área.
    """
    leg_ax = fig.add_axes([pos_ax.x0 + 0.01, pos_ax.y0 - 0.16, pos_ax.width * 0.98, 0.09])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    leg_ax.text(
        0.0, 0.97,
        f"{titulo} — % do tempo por faixa",
        fontsize=10,
        fontweight="bold",
        va="top"
    )

    if df_legenda.empty:
        leg_ax.text(0.0, 0.35, "Sem dados válidos para a legenda.", fontsize=9)
        return

    n = len(df_legenda)
    margem = 0.01
    largura_total = 1 - 2 * margem
    largura_bloco = largura_total / n
    y = 0.32
    h = 0.23

    for i, row in df_legenda.reset_index(drop=True).iterrows():
        x = margem + i * largura_bloco
        w = largura_bloco * 0.94

        rect = mpatches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=row["cor"],
            edgecolor="white",
            linewidth=1.0
        )
        leg_ax.add_patch(rect)

        leg_ax.text(
            x + w / 2,
            y + h + 0.08,
            row["faixa"],
            ha="center",
            va="bottom",
            fontsize=7
        )

        leg_ax.text(
            x + w / 2,
            y - 0.06,
            f"{row['percentual']:.1f}%",
            ha="center",
            va="top",
            fontsize=7
        )

def ler_csvs_de_zip(uploaded_zip, tmpdir, idx_zip):
    """
    Extrai cada ZIP em pasta própria e lê todos os CSVs encontrados.
    """
    dados = []

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
    Cria a linha do segmento, clipa dentro da fazenda e guarda atributos.
    """
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
    Para RPM/Velocidade:
    cria a faixa real da operação usando apenas a largura do implemento (em metros),
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
                cap_style=2,   # flat
                join_style=2   # mitre/bevel mais estável
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

def calcular_legenda_percentual(gdf_display, coluna_valor, faixas, mapa_cores):
    """
    Calcula % do tempo em cada faixa.
    Usa duracao_seg como peso.
    """
    dados = gdf_display.dropna(subset=[coluna_valor]).copy()

    if dados.empty:
        return pd.DataFrame(columns=["faixa", "percentual", "cor"])

    total_tempo = dados["duracao_seg"].fillna(0).sum()
    usar_contagem = total_tempo <= 0

    linhas = []
    for _, _, label in faixas:
        subset = dados[dados["classe"] == label]

        if usar_contagem:
            percentual = len(subset) / len(dados) * 100 if len(dados) > 0 else 0
        else:
            percentual = subset["duracao_seg"].fillna(0).sum() / total_tempo * 100 if total_tempo > 0 else 0

        linhas.append({
            "faixa": label,
            "percentual": percentual,
            "cor": mapa_cores.get(label, "#cccccc")
        })

    return pd.DataFrame(linhas)

def plotar_mapa_classes(ax, base_fazenda, gdf_plot, coluna_classe, mapa_cores, mostrar_talhoes=True):
    """
    Plota o mapa por classe (muito mais leve que plotar elemento por elemento).
    """
    desenhar_base_mapa(ax, base_fazenda, facecolor="#f7f7f7", mostrar_talhoes=mostrar_talhoes)

    if gdf_plot.empty:
        return

    classes_na_ordem = [c for c in mapa_cores.keys() if c in gdf_plot[coluna_classe].dropna().unique()]

    for classe in classes_na_ordem:
        sub = gdf_plot[gdf_plot[coluna_classe] == classe]
        if not sub.empty:
            sub.plot(
                ax=ax,
                color=mapa_cores[classe],
                edgecolor="none",
                alpha=0.95
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
RPM_PASSO = st.sidebar.number_input(
    "Passo das faixas RPM",
    min_value=50,
    max_value=1000,
    value=100,
    step=50
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
VEL_PASSO = st.sidebar.number_input(
    "Passo das faixas Velocidade",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.5
)

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

    with st.spinner("Processando arquivos e gerando mapas..."):
        with tempfile.TemporaryDirectory() as tmpdir:

            # =========================
            # LEITURA DE MÚLTIPLOS ZIPS
            # =========================
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

            df = df.dropna(subset=[
                "dt_hr_local_inicial",
                "vl_latitude_inicial",
                "vl_longitude_inicial"
            ])

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

            # =========================
            # PREPARAÇÃO DE CORES E FAIXAS
            # =========================
            rpm_faixas = gerar_faixas(RPM_MIN, RPM_MAX, RPM_PASSO, casas=0)
            vel_faixas = gerar_faixas(VEL_MIN, VEL_MAX, VEL_PASSO, casas=1)

            rpm_cmap = criar_cmap_suave("rpm")
            vel_cmap = criar_cmap_suave("vel")

            rpm_labels = [f[2] for f in rpm_faixas]
            vel_labels = [f[2] for f in vel_faixas]

            rpm_cores = dict(zip(rpm_labels, amostrar_cores_classes(rpm_cmap, len(rpm_labels))))
            vel_cores = dict(zip(vel_labels, amostrar_cores_classes(vel_cmap, len(vel_labels))))

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
                # CRIAÇÃO DAS LINHAS CLIPADAS
                # =========================
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

                # =========================
                # CÁLCULO DE ÁREA (PRESERVADO)
                # =========================
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
                # GEOMETRIA DE EXIBIÇÃO RPM / VEL
                # usa largura real do implemento (em metros)
                # =========================
                gdf_display = None
                if MAPA_RPM or MAPA_VEL:
                    gdf_display = criar_poligonos_display(gdf_linhas, geom_fazenda)

                    if not gdf_display.empty:
                        gdf_display["classe_rpm"] = gdf_display["rpm_medio"].apply(lambda x: classificar_valor(x, rpm_faixas))
                        gdf_display["classe_vel"] = gdf_display["vel_media"].apply(lambda x: classificar_valor(x, vel_faixas))

                # Legendas (% tempo por faixa)
                df_leg_rpm = pd.DataFrame(columns=["faixa", "percentual", "cor"])
                df_leg_vel = pd.DataFrame(columns=["faixa", "percentual", "cor"])

                if gdf_display is not None and not gdf_display.empty:
                    df_leg_rpm = calcular_legenda_percentual(
                        gdf_display,
                        "rpm_medio",
                        rpm_faixas,
                        rpm_cores
                    )
                    df_leg_vel = calcular_legenda_percentual(
                        gdf_display,
                        "vel_media",
                        vel_faixas,
                        vel_cores
                    )

                # =========================
                # EXPANDER
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
                        plt.subplots_adjust(left=0.15, right=0.84, bottom=0.33, top=0.88)

                        if gdf_display is not None and not gdf_display.empty:
                            plotar_mapa_classes(
                                ax=ax_rpm,
                                base_fazenda=base_fazenda,
                                gdf_plot=gdf_display.dropna(subset=["classe_rpm"]).copy(),
                                coluna_classe="classe_rpm",
                                mapa_cores=rpm_cores,
                                mostrar_talhoes=True
                            )
                        else:
                            desenhar_base_mapa(ax_rpm, base_fazenda, facecolor="#f7f7f7", mostrar_talhoes=True)

                        pos_rpm = ax_rpm.get_position()
                        centro_mapa_rpm = (pos_rpm.x0 + pos_rpm.x1) / 2
                        base_y_rpm = pos_rpm.y0

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
                            f"Faixa exibida:\n"
                            f"{int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))} até {int(arredondar_para_cima(RPM_MAX, RPM_PASSO))}+\n\n"
                            f"Período:\n{periodo_ini} até {periodo_fim}"
                        )

                        adicionar_resumo(fig_rpm, pos_rpm.x1 + 0.02, 0.47, resumo_rpm, COR_CAIXA)
                        adicionar_footer(fig_rpm, centro_mapa_rpm, base_y_rpm, COR_RODAPE)

                        st.pyplot(fig_rpm)

                    # -----------------------------------
                    # 3) MAPA DE VELOCIDADE
                    # -----------------------------------
                    if MAPA_VEL:
                        fig_vel, ax_vel = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                        plt.subplots_adjust(left=0.15, right=0.84, bottom=0.33, top=0.88)

                        if gdf_display is not None and not gdf_display.empty:
                            plotar_mapa_classes(
                                ax=ax_vel,
                                base_fazenda=base_fazenda,
                                gdf_plot=gdf_display.dropna(subset=["classe_vel"]).copy(),
                                coluna_classe="classe_vel",
                                mapa_cores=vel_cores,
                                mostrar_talhoes=True
                            )
                        else:
                            desenhar_base_mapa(ax_vel, base_fazenda, facecolor="#f7f7f7", mostrar_talhoes=True)

                        pos_vel = ax_vel.get_position()
                        centro_mapa_vel = (pos_vel.x0 + pos_vel.x1) / 2
                        base_y_vel = pos_vel.y0

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
                            f"Faixa exibida:\n"
                            f"{formatar_numero(arredondar_para_baixo(VEL_MIN, VEL_PASSO), 1)} até {formatar_numero(arredondar_para_cima(VEL_MAX, VEL_PASSO), 1)}+\n\n"
                            f"Período:\n{periodo_ini} até {periodo_fim}"
                        )

                        adicionar_resumo(fig_vel, pos_vel.x1 + 0.02, 0.47, resumo_vel, COR_CAIXA)
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
