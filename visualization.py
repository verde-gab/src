"""
Módulo de Visualização — Mineração de Dados

Fornece funções para gerar gráficos exploratórios a partir de qualquer
DataFrame CSV. Todos os gráficos são salvos automaticamente no diretório
``plots/`` e as mensagens são exibidas em português brasileiro.
"""

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
_DIRETORIO_PLOTS = "plots"
_DPI = 150


def _garantir_diretorio() -> None:
    """Cria o diretório de plots caso ainda não exista."""
    os.makedirs(_DIRETORIO_PLOTS, exist_ok=True)


# ──────────────────────────────────────────────
# 1. Heatmap de Correlação
# ──────────────────────────────────────────────
def heatmap_correlacao(df: pd.DataFrame) -> None:
    """
    Gera um mapa de calor (heatmap) da matriz de correlação
    para todas as colunas numéricas do DataFrame.

    O gráfico é salvo em ``plots/heatmap_correlacao.png``.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados a serem analisados.
    """
    _garantir_diretorio()

    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        print("⚠ Nenhuma coluna numérica encontrada para gerar o heatmap.")
        return

    correlacao = numericas.corr()

    largura = max(10, len(correlacao.columns) * 0.8)
    fig, ax = plt.subplots(figsize=(largura, largura * 0.85))

    sns.heatmap(
        correlacao,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title("Matriz de Correlação", fontsize=16, pad=12)

    plt.tight_layout()
    caminho = os.path.join(_DIRETORIO_PLOTS, "heatmap_correlacao.png")
    plt.savefig(caminho, dpi=_DPI)
    plt.close(fig)
    print(f"✔ Heatmap de correlação salvo em: {caminho}")


# ──────────────────────────────────────────────
# 2. Histogramas de Distribuição
# ──────────────────────────────────────────────
def histogramas_distribuicao(df: pd.DataFrame) -> None:
    """
    Gera histogramas de distribuição para cada coluna numérica,
    organizados em uma grade (grid).

    O gráfico é salvo em ``plots/histogramas.png``.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados a serem analisados.
    """
    _garantir_diretorio()

    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        print("⚠ Nenhuma coluna numérica encontrada para gerar histogramas.")
        return

    colunas = numericas.columns.tolist()
    n = len(colunas)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(colunas):
        ax = axes[i]
        sns.histplot(numericas[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Frequência")

    # Esconde eixos que sobraram
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuição das Variáveis Numéricas", fontsize=16, y=1.01)
    plt.tight_layout()
    caminho = os.path.join(_DIRETORIO_PLOTS, "histogramas.png")
    plt.savefig(caminho, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Histogramas de distribuição salvos em: {caminho}")


# ──────────────────────────────────────────────
# 3. Gráficos Categóricos
# ──────────────────────────────────────────────
def graficos_categoricos(df: pd.DataFrame) -> None:
    """
    Gera gráficos de barras para cada coluna categórica (object /
    category) do DataFrame, antes de qualquer codificação.

    O gráfico é salvo em ``plots/categoricos.png``.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados a serem analisados (pré-encoding).
    """
    _garantir_diretorio()

    categoricas = df.select_dtypes(include=["object", "category", "str"])
    if categoricas.empty:
        print("⚠ Nenhuma coluna categórica encontrada para gerar gráficos.")
        return

    colunas = categoricas.columns.tolist()
    n = len(colunas)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]


    for i, col in enumerate(colunas):
        ax = axes[i]
        contagem = categoricas[col].value_counts()
        sns.barplot(
            x=contagem.index,
            y=contagem.values,
            hue=contagem.index,
            legend=False,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Contagem")
        # Rotaciona labels longas
        if contagem.index.astype(str).str.len().max() > 5:
            ax.tick_params(axis="x", rotation=45)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuição das Variáveis Categóricas", fontsize=16, y=1.01)
    plt.tight_layout()
    caminho = os.path.join(_DIRETORIO_PLOTS, "categoricos.png")
    plt.savefig(caminho, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Gráficos categóricos salvos em: {caminho}")


# ──────────────────────────────────────────────
# 4. Scatter Matrix (Pairplot)
# ──────────────────────────────────────────────
def scatter_matrix(df: pd.DataFrame, max_colunas: int = 6) -> None:
    """
    Gera uma matriz de gráficos de dispersão (scatter plot matrix)
    para as colunas numéricas do DataFrame.

    Para evitar poluição visual, limita-se a no máximo
    ``max_colunas`` variáveis (as primeiras encontradas).

    O gráfico é salvo em ``plots/scatter_matrix.png``.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados a serem analisados.
    max_colunas : int, opcional
        Número máximo de colunas numéricas a incluir (padrão 6).
    """
    _garantir_diretorio()

    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        print("⚠ Nenhuma coluna numérica encontrada para o scatter matrix.")
        return

    colunas_selecionadas = numericas.columns[:max_colunas].tolist()
    if len(numericas.columns) > max_colunas:
        print(
            f"ℹ Scatter matrix limitado a {max_colunas} colunas: "
            f"{colunas_selecionadas}"
        )

    # Amostra para performance em datasets grandes
    amostra = numericas[colunas_selecionadas]
    if len(amostra) > 2000:
        amostra = amostra.sample(n=2000, random_state=42)
        print("ℹ Amostra de 2000 linhas usada para o scatter matrix.")

    g = sns.pairplot(amostra, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
    g.figure.suptitle("Scatter Matrix — Variáveis Numéricas", fontsize=16, y=1.02)

    plt.tight_layout()
    caminho = os.path.join(_DIRETORIO_PLOTS, "scatter_matrix.png")
    plt.savefig(caminho, dpi=_DPI, bbox_inches="tight")
    plt.close(g.figure)
    print(f"✔ Scatter matrix salvo em: {caminho}")


# ──────────────────────────────────────────────
# 5. Boxplots para Detecção de Outliers
# ──────────────────────────────────────────────
def boxplot_outliers(df: pd.DataFrame) -> None:
    """
    Gera boxplots para cada coluna numérica, facilitando a
    identificação visual de outliers.

    O gráfico é salvo em ``plots/boxplots.png``.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados a serem analisados.
    """
    _garantir_diretorio()

    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        print("⚠ Nenhuma coluna numérica encontrada para gerar boxplots.")
        return

    colunas = numericas.columns.tolist()
    n = len(colunas)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(colunas):
        ax = axes[i]
        sns.boxplot(y=numericas[col].dropna(), ax=ax, color="lightcoral")
        ax.set_title(col, fontsize=12)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boxplots — Detecção de Outliers", fontsize=16, y=1.01)
    plt.tight_layout()
    caminho = os.path.join(_DIRETORIO_PLOTS, "boxplots.png")
    plt.savefig(caminho, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Boxplots de outliers salvos em: {caminho}")


# ──────────────────────────────────────────────
# 6. Importância das Features
# ──────────────────────────────────────────────
def grafico_importancia_features(
    importancias: Sequence[float],
    nomes_features: Sequence[str],
) -> None:
    """
    Gera um gráfico de barras horizontais exibindo a importância
    de cada feature, ordenado de forma decrescente.

    O gráfico é salvo em ``plots/importancia_features.png``.

    Parâmetros
    ----------
    importancias : Sequence[float]
        Valores de importância de cada feature (ex.: ``model.feature_importances_``).
    nomes_features : Sequence[str]
        Nomes correspondentes às features.
    """
    _garantir_diretorio()

    indices = np.argsort(importancias)[::-1]
    nomes_ordenados = [nomes_features[i] for i in indices]
    valores_ordenados = [importancias[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(4, len(nomes_ordenados) * 0.4)))

    cores = sns.color_palette("viridis", n_colors=len(nomes_ordenados))
    ax.barh(range(len(nomes_ordenados)), valores_ordenados, color=cores)
    ax.set_yticks(range(len(nomes_ordenados)))
    ax.set_yticklabels(nomes_ordenados)
    ax.invert_yaxis()  # Maior importância no topo
    ax.set_xlabel("Importância")
    ax.set_title("Importância das Features", fontsize=16, pad=12)

    plt.tight_layout()
    caminho = os.path.join(_DIRETORIO_PLOTS, "importancia_features.png")
    plt.savefig(caminho, dpi=_DPI)
    plt.close(fig)
    print(f"✔ Gráfico de importância das features salvo em: {caminho}")
