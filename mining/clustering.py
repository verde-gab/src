"""
Módulo de Clusterização
========================
Implementa algoritmos de agrupamento (KMeans e DBSCAN), visualizações
do método do cotovelo e projeção 2D dos clusters via PCA.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN as _DBSCAN
from sklearn.cluster import KMeans as _KMeans
from sklearn.decomposition import PCA

from mining.evaluation import avaliar_clusterizacao

# Diretório para salvar os gráficos
PLOTS_DIR = "plots"


def _garantir_diretorio_plots() -> None:
    """Cria o diretório de gráficos caso não exista."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
#  K-Means
# ──────────────────────────────────────────────────────────────────
def kmeans(
    df: pd.DataFrame,
    n_clusters: int = 3,
) -> tuple[np.ndarray, _KMeans]:
    """
    Executa o algoritmo K-Means no DataFrame fornecido.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame apenas com features numéricas (já pré-processado).
    n_clusters : int, opcional
        Número de clusters desejados (padrão: 3).

    Retorna
    -------
    tuple[np.ndarray, KMeans]
        (rótulos atribuídos a cada amostra, modelo KMeans treinado).
    """
    print(f"\n[INFO] Executando K-Means com k={n_clusters}...")

    modelo = _KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = modelo.fit_predict(df)

    # Exibir tamanho de cada cluster
    print(f"[OK] K-Means concluído.")
    print(f"  Inércia (SSE): {modelo.inertia_:.2f}")
    print(f"  Distribuição dos clusters:")
    valores, contagens = np.unique(labels, return_counts=True)
    for cluster_id, qtd in zip(valores, contagens):
        print(f"    Cluster {cluster_id}: {qtd} amostras")

    # Avaliação automática via Silhouette Score
    avaliar_clusterizacao(df, labels)

    return labels, modelo


# ──────────────────────────────────────────────────────────────────
#  Método do Cotovelo (Elbow Method)
# ──────────────────────────────────────────────────────────────────
def metodo_cotovelo(df: pd.DataFrame, k_max: int = 10) -> None:
    """
    Gera o gráfico do Método do Cotovelo para auxiliar na escolha
    do número ideal de clusters para o K-Means.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame apenas com features numéricas (já pré-processado).
    k_max : int, opcional
        Valor máximo de k a ser testado (padrão: 10).
    """
    _garantir_diretorio_plots()

    print(f"\n[INFO] Calculando inércia para k de 2 a {k_max}...")

    intervalo_k = range(2, k_max + 1)
    inercias: list[float] = []

    for k in intervalo_k:
        modelo = _KMeans(n_clusters=k, n_init="auto", random_state=42)
        modelo.fit(df)
        inercias.append(modelo.inertia_)
        print(f"  k={k:>2}  →  Inércia = {modelo.inertia_:.2f}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(intervalo_k), inercias, marker="o", linewidth=2, color="#2196F3")
    ax.set_title("Método do Cotovelo (Elbow Method)", fontsize=14)
    ax.set_xlabel("Número de Clusters (k)", fontsize=12)
    ax.set_ylabel("Inércia (SSE)", fontsize=12)
    ax.set_xticks(list(intervalo_k))
    ax.grid(alpha=0.3)

    caminho = os.path.join(PLOTS_DIR, "cotovelo.png")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Gráfico do cotovelo salvo em: {caminho}")


# ──────────────────────────────────────────────────────────────────
#  DBSCAN
# ──────────────────────────────────────────────────────────────────
def dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[np.ndarray, _DBSCAN]:
    """
    Executa o algoritmo DBSCAN no DataFrame fornecido.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame apenas com features numéricas (já pré-processado).
    eps : float, opcional
        Raio máximo de vizinhança (padrão: 0.5).
    min_samples : int, opcional
        Número mínimo de amostras em uma vizinhança para formar um
        core point (padrão: 5).

    Retorna
    -------
    tuple[np.ndarray, DBSCAN]
        (rótulos atribuídos a cada amostra, modelo DBSCAN treinado).
        Pontos de ruído recebem rótulo -1.
    """
    print(f"\n[INFO] Executando DBSCAN (eps={eps}, min_samples={min_samples})...")

    modelo = _DBSCAN(eps=eps, min_samples=min_samples)
    labels = modelo.fit_predict(df)

    # Estatísticas
    n_clusters = len(set(labels) - {-1})
    n_ruido = (labels == -1).sum()

    print(f"[OK] DBSCAN concluído.")
    print(f"  Clusters encontrados : {n_clusters}")
    print(f"  Pontos de ruído      : {n_ruido}")

    valores, contagens = np.unique(labels, return_counts=True)
    for cluster_id, qtd in zip(valores, contagens):
        rotulo = "Ruído" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"    {rotulo}: {qtd} amostras")

    # Avaliação automática (apenas se houver ≥ 2 clusters)
    if n_clusters >= 2:
        # Filtrar pontos de ruído para o cálculo do silhouette
        mascara = labels != -1
        avaliar_clusterizacao(df[mascara], labels[mascara])
    else:
        print("  ⚠️  Silhouette Score não calculado (menos de 2 clusters).")

    return labels, modelo


# ──────────────────────────────────────────────────────────────────
#  Visualização de Clusters (PCA 2D)
# ──────────────────────────────────────────────────────────────────
def visualizar_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    titulo: str = "Clusters",
) -> None:
    """
    Reduz os dados para 2 dimensões via PCA e gera um scatter plot
    colorido pelos rótulos de cluster.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as features originais (numéricas).
    labels : array-like
        Rótulos de cluster para cada amostra.
    titulo : str, opcional
        Título do gráfico (padrão: 'Clusters').
    """
    _garantir_diretorio_plots()

    labels = np.asarray(labels)

    print(f"\n[INFO] Reduzindo dados para 2D com PCA...")
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(df)
    var_explicada = pca.explained_variance_ratio_.sum() * 100

    print(
        f"  Variância explicada pelas 2 componentes: {var_explicada:.2f}%"
    )

    # Preparar DataFrame auxiliar para o plot
    df_plot = pd.DataFrame(
        {
            "PC1": componentes[:, 0],
            "PC2": componentes[:, 1],
            "Cluster": labels.astype(str),
        }
    )

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df_plot,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        s=40,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title(f"{titulo}  (Variância explicada: {var_explicada:.1f}%)", fontsize=14)
    ax.set_xlabel("Componente Principal 1", fontsize=12)
    ax.set_ylabel("Componente Principal 2", fontsize=12)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

    caminho = os.path.join(PLOTS_DIR, "clusters.png")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    print(f"[OK] Gráfico de clusters salvo em: {caminho}")
