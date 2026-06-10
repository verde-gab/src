"""
Módulo de Avaliação de Modelos
==============================
Fornece funções para avaliar modelos de classificação e clusterização,
além de comparar o desempenho entre múltiplos modelos.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)

# Diretório para salvar os gráficos
PLOTS_DIR = "plots"


def _garantir_diretorio_plots() -> None:
    """Cria o diretório de gráficos caso não exista."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def avaliar_classificacao(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    nome_modelo: str = "Modelo",
) -> dict:
    """
    Avalia um modelo de classificação calculando métricas e gerando
    a matriz de confusão como heatmap.

    Parâmetros
    ----------
    y_real : array-like
        Valores reais (rótulos verdadeiros).
    y_pred : array-like
        Valores preditos pelo modelo.
    nome_modelo : str, opcional
        Nome do modelo para identificação nos relatórios e arquivos.

    Retorna
    -------
    dict
        Dicionário com as métricas: accuracy, precision, recall e f1.
    """
    # Cálculo das métricas
    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_real, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_real, y_pred, average="weighted", zero_division=0)

    metricas = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    # Impressão do relatório
    print(f"\n{'='*60}")
    print(f"  Avaliação do modelo: {nome_modelo}")
    print(f"{'='*60}")
    print(f"  Acurácia  : {metricas['accuracy']:.4f}")
    print(f"  Precisão  : {metricas['precision']:.4f}")
    print(f"  Recall    : {metricas['recall']:.4f}")
    print(f"  F1-Score  : {metricas['f1']:.4f}")
    print(f"{'='*60}")
    print("\n  Relatório de Classificação Detalhado:")
    print(classification_report(y_real, y_pred, zero_division=0))

    # Geração da matriz de confusão
    _garantir_diretorio_plots()
    cm = confusion_matrix(y_real, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusão — {nome_modelo}", fontsize=14)
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)

    # Nome do arquivo limpo (sem espaços ou caracteres especiais)
    nome_arquivo = nome_modelo.lower().replace(" ", "_")
    caminho = os.path.join(PLOTS_DIR, f"matriz_confusao_{nome_arquivo}.png")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  📊 Matriz de confusão salva em: {caminho}")

    return metricas


def avaliar_clusterizacao(df: pd.DataFrame, labels: np.ndarray) -> float | None:
    """
    Avalia a qualidade de uma clusterização usando o Silhouette Score.

    Trata o caso especial onde todos os rótulos são iguais (um único cluster),
    situação na qual o silhouette score não pode ser calculado.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as features usadas na clusterização.
    labels : array-like
        Rótulos dos clusters atribuídos a cada amostra.

    Retorna
    -------
    float ou None
        Valor do Silhouette Score, ou None se houver apenas um cluster.
    """
    labels = np.asarray(labels)
    n_clusters = len(set(labels))

    print(f"\n{'='*60}")
    print("  Avaliação de Clusterização")
    print(f"{'='*60}")
    print(f"  Número de clusters encontrados: {n_clusters}")

    # Silhouette score requer pelo menos 2 clusters
    if n_clusters < 2:
        print("  ⚠️  Apenas um cluster detectado. Não é possível calcular o Silhouette Score.")
        print(f"{'='*60}")
        return None

    if n_clusters >= len(labels):
        print("  ⚠️  Número de clusters igual ao número de amostras. Silhouette Score inválido.")
        print(f"{'='*60}")
        return None

    score = silhouette_score(df, labels)
    print(f"  Silhouette Score: {score:.4f}")
    print()
    print("  Interpretação:")
    if score >= 0.7:
        print("    ✅ Estrutura de clusters forte.")
    elif score >= 0.5:
        print("    ✅ Estrutura de clusters razoável.")
    elif score >= 0.25:
        print("    ⚠️  Estrutura de clusters fraca.")
    else:
        print("    ❌ Sem estrutura de clusters significativa.")
    print(f"{'='*60}")

    return round(score, 4)


def comparar_modelos(resultados: dict) -> None:
    """
    Compara o desempenho de múltiplos modelos exibindo uma tabela
    formatada e gerando um gráfico de barras comparativo.

    Parâmetros
    ----------
    resultados : dict
        Dicionário no formato:
        {
            "NomeModelo": {
                "accuracy": float,
                "precision": float,
                "recall": float,
                "f1": float
            },
            ...
        }
    """
    if not resultados:
        print("  ⚠️  Nenhum resultado para comparar.")
        return

    # Tabela formatada
    print(f"\n{'='*76}")
    print("  Comparação de Modelos")
    print(f"{'='*76}")
    print(f"  {'Modelo':<25} {'Acurácia':>10} {'Precisão':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*65}")

    for nome, metricas in resultados.items():
        print(
            f"  {nome:<25} "
            f"{metricas['accuracy']:>10.4f} "
            f"{metricas['precision']:>10.4f} "
            f"{metricas['recall']:>10.4f} "
            f"{metricas['f1']:>10.4f}"
        )

    print(f"{'='*76}")

    # Identificar melhor modelo por F1-Score
    melhor = max(resultados, key=lambda m: resultados[m]["f1"])
    print(f"  🏆 Melhor modelo (F1-Score): {melhor} ({resultados[melhor]['f1']:.4f})")

    # Gráfico de barras comparativo
    _garantir_diretorio_plots()

    nomes = list(resultados.keys())
    metricas_nomes = ["accuracy", "precision", "recall", "f1"]
    metricas_labels = ["Acurácia", "Precisão", "Recall", "F1-Score"]

    x = np.arange(len(nomes))
    largura = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metrica, label) in enumerate(zip(metricas_nomes, metricas_labels)):
        valores = [resultados[nome][metrica] for nome in nomes]
        ax.bar(x + i * largura, valores, largura, label=label)

    ax.set_xlabel("Modelos", fontsize=12)
    ax.set_ylabel("Valor da Métrica", fontsize=12)
    ax.set_title("Comparação de Desempenho dos Modelos", fontsize=14)
    ax.set_xticks(x + largura * 1.5)
    ax.set_xticklabels(nomes, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    caminho = os.path.join(PLOTS_DIR, "comparacao_modelos.png")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  📊 Gráfico de comparação salvo em: {caminho}")
