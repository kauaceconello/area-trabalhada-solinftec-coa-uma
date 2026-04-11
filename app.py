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
    .painel-sol {
        border: 1px solid #d0d7de;
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        background: #0b1117;
        color: #f0f6fc;
        margin-bottom: 0.7rem;
    }
    .painel-sol h4 {
        margin: 0 0 0.6rem 0;
        font-size: 1rem;
        color: #f0f6fc;
    }
    .painel-sol .sub {
        color: #8b949e;
        font-size: 0.85rem;
        margin-bottom: 0.55rem;
    }
    .sol-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
    }
    .sol-table th {
        text-align: left;
        border-bottom: 1px solid #30363d;
        padding: 0.45rem 0.3rem;
        color: #c9d1d9;
    }
    .sol-table td {
        border-bottom: 1px solid #21262d;
        padding: 0.45rem 0.3rem;
        color: #f0f6fc;
        vertical-align: middle;
    }
    .sol-dot {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.25);
    }
    .sol-foot {
        font-size: 0.82rem;
        color: #6e7781;
        margin-top: 0.35rem;
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
    Gera faixas arredondadas:
    ex. RPM -> 1700 a 1800, 1800 a 1900, ..., 2000+
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
            "#66c2a4",  # verde água
            "#18c964",  # verde vivo
            "#d9ef8b",  # amarelo esverdeado
            "#f7b733",  # laranja amarelo
            "#f15a24",  # laranja forte
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
    base_fazenda.plot(ax=ax, facecolor=facecolor, edgecolor="black", linewidth=1.1, zorder=1)
    base_fazenda.boundary.plot(ax=ax, color="black", linewidth=1.1, zorder=3)

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
                weight="bold",
                zorder=4
            )

    minx, miny, maxx, maxy = base_fazenda.total_bounds
    dx = maxx - minx
    dy = maxy - miny
    margem_x = dx * 0.05 if dx > 0 else 10
    margem_y = dy * 0.05 if dy > 0 else 10

    ax.set_xlim(minx - margem_x, maxx + margem_x)
    ax.set_ylim(miny - margem_y, maxy + margem_y)
    ax.set_aspect("equal")
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


def ler_csvs_de_zip(uploaded_zip, tmpdir, idx_zip):
    """
    Extrai cada ZIP em pasta própria e retorna todos os CSVs encontrados.
    """
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
    """
    Calcula % do tempo em cada faixa.
    Usa duracao_seg como peso.
    """
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


def plotar_mapa_classes(ax, base_fazenda, gdf_plot, coluna_classe, mapa_cores, mostrar_talhoes=True):
    """
    Plota o mapa por classe (mais leve e mais bonito).
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
                alpha=0.92,
                zorder=2
            )


def renderizar_painel_lateral(
    titulo,
    subtitulo,
    df_legenda,
    metrica_min,
    metrica_media,
    metrica_max,
    unidade_label,
    periodo_ini,
    periodo_fim,
    fazenda_texto,
    casas=0
):
    """
    Renderiza painel lateral estilo Solinftec usando Streamlit + HTML.
    """
    st.markdown(
        f"""
        <div class="painel-sol">
            <h4>{titulo}</h4>
            <div class="sub">{subtitulo}</div>
            <div class="sub">{fazenda_texto}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mín.", f"{formatar_numero(metrica_min, casas)}{unidade_label}")
    c2.metric("Médio", f"{formatar_numero(metrica_media, casas)}{unidade_label}")
    c3.metric("Máx.", f"{formatar_numero(metrica_max, casas)}{unidade_label}")

    html = """
    <div class="painel-sol">
        <table class="sol-table">
            <thead>
                <tr>
                    <th style="width: 18%;">Cor</th>
                    <th style="width: 24%;">Início</th>
                    <th style="width: 22%;">Fim</th>
                    <th style="width: 36%;">% Tempo</th>
                </tr>
            </thead>
            <tbody>
    """

    if df_legenda.empty:
        html += """
            <tr>
                <td colspan="4">Sem dados válidos para exibir.</td>
            </tr>
        """
    else:
        for _, row in df_legenda.iterrows():
            inicio = formatar_numero(row["inicio"], casas)
            fim = row["fim"]
            if fim != "+":
                fim = formatar_numero(fim, casas)

            percentual = f'{row["percentual"]:.1f}%'.replace(".", ",")

            html += f"""
            <tr>
                <td><span class="sol-dot" style="background:{row['cor']};"></span></td>
                <td>{inicio}</td>
                <td>{fim}</td>
                <td>{percentual}</td>
            </tr>
            """

    html += f"""
            </tbody>
        </table>
        <div class="sol-foot">Período: {periodo_ini} até {periodo_fim}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

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

# Só aparece se marcar RPM
RPM_MIN = 1200
RPM_MAX = 2000
RPM_PASSO = 100
if MAPA_RPM:
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

# Só aparece se marcar Velocidade
VEL_MIN = 4.0
VEL_MAX = 8.0
VEL_PASSO = 1.0
if MAPA_VEL:
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

if MAPA_RPM and RPM_MAX <= RPM_MIN:
    st.sidebar.error("⚠️ O RPM máximo deve ser maior que o RPM mínimo.")
if MAPA_VEL and VEL_MAX <= VEL_MIN:
    st.sidebar.error("⚠️ A velocidade máxima deve ser maior que a velocidade mínima.")
if not (MAPA_AREA or MAPA_RPM or MAPA_VEL):
    st.sidebar.warning("Selecione pelo menos um tipo de mapa.")

# =========================
# PROCESSAMENTO
# =========================
if uploaded_zips and uploaded_gpkg and GERAR:

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
            rpm_faixas = gerar_faixas(RPM_MIN, RPM_MAX, RPM_PASSO, casas=0) if MAPA_RPM else []
            vel_faixas = gerar_faixas(VEL_MIN, VEL_MAX, VEL_PASSO, casas=1) if MAPA_VEL else []

            rpm_cmap = criar_cmap_suave("rpm")
            vel_cmap = criar_cmap_suave("vel")

            rpm_labels = [f[2] for f in rpm_faixas] if MAPA_RPM else []
            vel_labels = [f[2] for f in vel_faixas] if MAPA_VEL else []

            rpm_cores = dict(zip(rpm_labels, amostrar_cores_classes(rpm_cmap, len(rpm_labels)))) if MAPA_RPM else {}
            vel_cores = dict(zip(vel_labels, amostrar_cores_classes(vel_cmap, len(vel_labels)))) if MAPA_VEL else {}

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

                if MOSTRAR_TALHOES and "TALHAO" in base.columns and "GLEBA" in base.columns:
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

                    if gdf_display is not None and not gdf_display.empty:
                        if MAPA_RPM:
                            gdf_display["classe_rpm"] = gdf_display["rpm_medio"].apply(
                                lambda x: classificar_valor(x, rpm_faixas)
                            )
                        if MAPA_VEL:
                            gdf_display["classe_vel"] = gdf_display["vel_media"].apply(
                                lambda x: classificar_valor(x, vel_faixas)
                            )

                # Legendas laterais estilo Solinftec
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

                # =========================
                # EXPANDER
                # =========================
                with st.expander(f"🗺️ Mapa – {nome_fazenda}", expanded=False):

                    # -----------------------------------
                    # 1) MAPA DE ÁREA TRABALHADA (MANTIDO)
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
                    # 2) MAPA DE RPM (VISUAL MELHORADO)
                    # -----------------------------------
                    if MAPA_RPM:
                        st.markdown("### ⚙️ Mapa de RPM")

                        col_mapa, col_painel = st.columns([3.2, 1.35], vertical_alignment="top")

                        with col_mapa:
                            fig_rpm, ax_rpm = plt.subplots(figsize=(8.5, 12))
                            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

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

                            st.pyplot(fig_rpm, use_container_width=True)
                            st.caption(
                                "⚠️ Mapa temático de RPM com faixa operacional baseada na largura real do implemento (m)."
                            )

                        with col_painel:
                            renderizar_painel_lateral(
                                titulo="Resumo de RPM",
                                subtitulo=f"Faixa exibida: {int(arredondar_para_baixo(RPM_MIN, RPM_PASSO))} até {int(arredondar_para_cima(RPM_MAX, RPM_PASSO))}+",
                                df_legenda=df_leg_rpm,
                                metrica_min=rpm_min_real,
                                metrica_media=rpm_med_real,
                                metrica_max=rpm_max_real,
                                unidade_label="",
                                periodo_ini=periodo_ini,
                                periodo_fim=periodo_fim,
                                fazenda_texto=f"Fazenda: {FAZENDA_ID} – {nome_fazenda}",
                                casas=0
                            )

                    # -----------------------------------
                    # 3) MAPA DE VELOCIDADE (VISUAL MELHORADO)
                    # -----------------------------------
                    if MAPA_VEL:
                        st.markdown("### 🚜 Mapa de Velocidade")

                        col_mapa, col_painel = st.columns([3.2, 1.35], vertical_alignment="top")

                        with col_mapa:
                            fig_vel, ax_vel = plt.subplots(figsize=(8.5, 12))
                            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

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

                            st.pyplot(fig_vel, use_container_width=True)
                            st.caption(
                                "⚠️ Mapa temático de velocidade com faixa operacional baseada na largura real do implemento (m)."
                            )

                        with col_painel:
                            renderizar_painel_lateral(
                                titulo="Resumo de Velocidade",
                                subtitulo=f"Faixa exibida: {formatar_numero(arredondar_para_baixo(VEL_MIN, VEL_PASSO), 1)} até {formatar_numero(arredondar_para_cima(VEL_MAX, VEL_PASSO), 1)}+ km/h",
                                df_legenda=df_leg_vel,
                                metrica_min=vel_min_real,
                                metrica_media=vel_med_real,
                                metrica_max=vel_max_real,
                                unidade_label=" km/h",
                                periodo_ini=periodo_ini,
                                periodo_fim=periodo_fim,
                                fazenda_texto=f"Fazenda: {FAZENDA_ID} – {nome_fazenda}",
                                casas=1
                            )

                    # -----------------------------------
                    # TABELA DE TALHÕES (PRESERVADA)
                    # -----------------------------------
                    if df_talhoes is not None:
                        st.markdown("### 🌾 Área por Gleba / Talhão")
                        st.dataframe(df_talhoes, use_container_width=True, hide_index=True)

else:
    st.info("⬆️ Envie os arquivos e clique em **Gerar mapa**.")
