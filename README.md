# 📍 Área Trabalhada – Solução de Mapas Operacionais

Aplicação desenvolvida em **Python + Streamlit** para automatizar o processamento de dados operacionais e gerar **mapas temáticos de área trabalhada, RPM e velocidade**, com base em arquivos **CSV compactados em ZIP** e **base cartográfica em GPKG**.

A solução foi criada para reduzir esforço manual, padronizar análises geoespaciais e facilitar a geração de relatórios operacionais em **PDF vetorial**.

---

## 📌 Visão Geral

Este projeto tem como objetivo transformar dados operacionais brutos em mapas analíticos de forma rápida, visual e padronizada.

A aplicação recebe:

- **ZIPs** contendo arquivos CSV com dados operacionais
- **GPKG** com a base cartográfica da fazenda / área de interesse

A partir disso, o sistema:

- identifica e filtra pontos válidos
- organiza trajetórias por equipamento
- calcula a **área trabalhada**
- gera mapas temáticos de **RPM**
- gera mapas temáticos de **velocidade**
- exporta os mapas em **PDF vetorial**
- apresenta tabela de apoio por **gleba / talhão**, quando disponível

---

## 🎯 Problema que o projeto resolve

Antes da aplicação, o processo de análise operacional exigia tratamento manual de dados, organização de arquivos e geração visual de resultados de forma pouco padronizada.

Isso gerava:

- retrabalho
- maior tempo de resposta
- dificuldade de leitura operacional
- dependência de etapas manuais
- desgaste da equipe em períodos de maior demanda

A aplicação foi desenvolvida para transformar esse fluxo em um processo mais simples, visual e replicável.

---

## 🚀 Principais Funcionalidades

- Upload de múltiplos arquivos **ZIP**
- Leitura automática de **CSV** dentro dos ZIPs
- Upload de base cartográfica em **GPKG**
- Conversão e tratamento de dados geográficos
- Segmentação de trajetórias por equipamento
- Interseção geoespacial com a área da fazenda
- Cálculo de **área trabalhada**
- Geração de **mapa de RPM**
- Geração de **mapa de velocidade**
- Exibição opcional de **área por gleba / talhão**
- Exportação em **PDF vetorial**
- Interface web local em **Streamlit**
- Layout visual customizado com foco em legibilidade operacional

---

## 🛠️ Tecnologias Utilizadas

### Linguagem e aplicação
- **Python**
- **Streamlit**

### Tratamento de dados
- **Pandas**
- **NumPy**

### Geoprocessamento
- **GeoPandas**
- **Shapely**

### Visualização
- **Matplotlib**

### Arquivos e exportação
- **ZIP / CSV**
- **GPKG**
- **PDF vetorial**

---

## 🧠 Lógica da Solução

A aplicação segue o fluxo abaixo:

ZIPs com CSVs + GPKG
        ↓
Leitura e tratamento dos dados
        ↓
Filtragem de pontos operacionais válidos
        ↓
Agrupamento por equipamento
        ↓
Geração de linhas / segmentos
        ↓
Interseção com a geometria da fazenda
        ↓
Cálculo de área trabalhada
        ↓
Classificação por RPM e velocidade
        ↓
Geração de mapas temáticos
        ↓
Exportação em PDF vetorial
