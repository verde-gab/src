"""
Módulo de Classificação
=======================
Fornece funções para dividir dados, treinar classificadores e
avaliar seus desempenhos de forma padronizada.

Classificadores disponíveis:
- Árvore de Decisão
- KNN (K-Nearest Neighbors)
- Random Forest
- Naive Bayes (Gaussiano)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from mining.evaluation import avaliar_classificacao, comparar_modelos


def separar_dados(
    df: pd.DataFrame,
    coluna_alvo: str,
    test_size: float = 0.3,
) -> tuple:
    """
    Separa o DataFrame em conjuntos de treino e teste.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as features e a coluna alvo.
    coluna_alvo : str
        Nome da coluna alvo (variável dependente).
    test_size : float, opcional
        Proporção dos dados para teste (padrão: 0.3 = 30%).

    Retorna
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if coluna_alvo not in df.columns:
        raise ValueError(f"Coluna alvo '{coluna_alvo}' não encontrada no DataFrame.")

    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    print(f"\n{'='*60}")
    print("  Separação dos Dados")
    print(f"{'='*60}")
    print(f"  Total de amostras    : {len(df)}")
    print(f"  Amostras de treino   : {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
    print(f"  Amostras de teste    : {len(X_test)} ({test_size * 100:.0f}%)")
    print(f"  Número de features   : {X_train.shape[1]}")
    print(f"  Coluna alvo          : {coluna_alvo}")
    print(f"{'='*60}")

    return X_train, X_test, y_train, y_test


def arvore_decisao(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Treina um classificador de Árvore de Decisão e avalia seu desempenho.

    Parâmetros
    ----------
    X_train : array-like
        Features de treino.
    X_test : array-like
        Features de teste.
    y_train : array-like
        Rótulos de treino.
    y_test : array-like
        Rótulos de teste.

    Retorna
    -------
    dict
        Dicionário com as métricas de avaliação.
    """
    print("\n🌳 Treinando Árvore de Decisão...")

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = avaliar_classificacao(y_test, y_pred, nome_modelo="Arvore_de_Decisao")

    return metricas


def knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
) -> dict:
    """
    Treina um classificador KNN (K-Nearest Neighbors) e avalia seu desempenho.

    Parâmetros
    ----------
    X_train : array-like
        Features de treino.
    X_test : array-like
        Features de teste.
    y_train : array-like
        Rótulos de treino.
    y_test : array-like
        Rótulos de teste.
    k : int, opcional
        Número de vizinhos (padrão: 5).

    Retorna
    -------
    dict
        Dicionário com as métricas de avaliação.
    """
    print(f"\n📍 Treinando KNN (k={k})...")

    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = avaliar_classificacao(y_test, y_pred, nome_modelo="KNN")

    return metricas


def random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
) -> dict:
    """
    Treina um classificador Random Forest e avalia seu desempenho.
    Também retorna a importância de cada feature.

    Parâmetros
    ----------
    X_train : array-like
        Features de treino.
    X_test : array-like
        Features de teste.
    y_train : array-like
        Rótulos de treino.
    y_test : array-like
        Rótulos de teste.
    n_estimators : int, opcional
        Número de árvores na floresta (padrão: 100).

    Retorna
    -------
    dict
        Dicionário com as métricas de avaliação e importância das features.
        Inclui a chave 'feature_importances' com um array de importâncias.
    """
    print(f"\n🌲 Treinando Random Forest (n_estimators={n_estimators})...")

    modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = avaliar_classificacao(y_test, y_pred, nome_modelo="Random_Forest")

    # Importância das features
    importancias = modelo.feature_importances_

    if hasattr(X_train, "columns"):
        nomes_features = X_train.columns.tolist()
    else:
        nomes_features = [f"feature_{i}" for i in range(X_train.shape[1])]

    print("\n  Importância das Features (Top 10):")
    indices = np.argsort(importancias)[::-1]
    for pos, idx in enumerate(indices[:10], start=1):
        print(f"    {pos:>2}. {nomes_features[idx]:<30} {importancias[idx]:.4f}")

    metricas["feature_importances"] = dict(zip(nomes_features, importancias))

    return metricas


def naive_bayes(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Treina um classificador Naive Bayes Gaussiano e avalia seu desempenho.

    Parâmetros
    ----------
    X_train : array-like
        Features de treino.
    X_test : array-like
        Features de teste.
    y_train : array-like
        Rótulos de treino.
    y_test : array-like
        Rótulos de teste.

    Retorna
    -------
    dict
        Dicionário com as métricas de avaliação.
    """
    print("\n📐 Treinando Naive Bayes (Gaussiano)...")

    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = avaliar_classificacao(y_test, y_pred, nome_modelo="Naive_Bayes")

    return metricas


def executar_todos(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Executa todos os classificadores disponíveis, coleta os resultados
    e gera uma comparação entre eles.

    Parâmetros
    ----------
    X_train : array-like
        Features de treino.
    X_test : array-like
        Features de teste.
    y_train : array-like
        Rótulos de treino.
    y_test : array-like
        Rótulos de teste.

    Retorna
    -------
    dict
        Dicionário com os resultados de todos os modelos no formato:
        {
            "Árvore de Decisão": {accuracy, precision, recall, f1},
            "KNN": {accuracy, precision, recall, f1},
            "Random Forest": {accuracy, precision, recall, f1, feature_importances},
            "Naive Bayes": {accuracy, precision, recall, f1},
        }
    """
    print(f"\n{'#'*60}")
    print("  🚀 Executando todos os classificadores...")
    print(f"{'#'*60}")

    resultados = {}

    # 1 — Árvore de Decisão
    resultados["Árvore de Decisão"] = arvore_decisao(X_train, X_test, y_train, y_test)

    # 2 — KNN
    resultados["KNN"] = knn(X_train, X_test, y_train, y_test)

    # 3 — Random Forest
    res_rf = random_forest(X_train, X_test, y_train, y_test)
    # Separar feature_importances para não atrapalhar a comparação
    feature_imp = res_rf.pop("feature_importances", None)
    resultados["Random Forest"] = res_rf

    # 4 — Naive Bayes
    resultados["Naive Bayes"] = naive_bayes(X_train, X_test, y_train, y_test)

    # Comparação geral
    comparar_modelos(resultados)

    # Restaurar feature_importances no resultado do Random Forest
    if feature_imp is not None:
        resultados["Random Forest"]["feature_importances"] = feature_imp

    return resultados
