"""
Módulo de Regras de Associação — Mineração de Dados

Implementa o pipeline completo de mineração de regras de associação:
preparação dos dados, execução do algoritmo Apriori e geração /
exibição das regras. Utiliza a biblioteca ``mlxtend``.
"""

from typing import Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# ──────────────────────────────────────────────
# 1. Preparação dos Dados
# ──────────────────────────────────────────────
def preparar_dados_associacao(
    df: pd.DataFrame,
    colunas: Optional[list[str]] = None,
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Converte as colunas selecionadas do DataFrame em formato
    binário (booleano), adequado para o algoritmo Apriori.

    - **Colunas categóricas**: aplica ``pd.get_dummies``.
    - **Colunas numéricas**: discretiza em faixas (ex.: Baixo /
      Médio / Alto) antes de aplicar ``pd.get_dummies``.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame original com os dados.
    colunas : list[str], opcional
        Lista de colunas a incluir. Se ``None``, detecta
        automaticamente as colunas categóricas.
    n_bins : int, opcional
        Número de faixas para discretização de colunas numéricas
        (padrão 3 → Baixo / Médio / Alto).

    Retorna
    -------
    pd.DataFrame
        DataFrame binário (valores ``True`` / ``False``) pronto
        para o Apriori.
    """
    rotulos = {3: ["Baixo", "Medio", "Alto"]}
    rotulos_default = [f"Faixa_{i+1}" for i in range(n_bins)]

    if colunas is None:
        # Auto-detecta categóricas
        colunas = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        if not colunas:
            print(
                "⚠ Nenhuma coluna categórica detectada. "
                "Todas as colunas numéricas serão discretizadas."
            )
            colunas = df.columns.tolist()

    subdf = df[colunas].copy()
    partes: list[pd.DataFrame] = []

    for col in subdf.columns:
        if pd.api.types.is_numeric_dtype(subdf[col]):
            # Discretiza colunas numéricas em faixas
            labels = rotulos.get(n_bins, rotulos_default)
            subdf[col] = pd.cut(
                subdf[col],
                bins=n_bins,
                labels=labels,
                include_lowest=True,
            )
            print(f"ℹ Coluna numérica '{col}' discretizada em {n_bins} faixas.")

        dummies = pd.get_dummies(subdf[[col]], prefix=col, prefix_sep="=")
        partes.append(dummies)

    df_binario = pd.concat(partes, axis=1).astype(bool)

    print(
        f"✔ Dados preparados: {df_binario.shape[0]} registros × "
        f"{df_binario.shape[1]} itens binários."
    )
    return df_binario


# ──────────────────────────────────────────────
# 2. Execução do Apriori
# ──────────────────────────────────────────────
def executar_apriori(
    df_binario: pd.DataFrame,
    suporte_minimo: float = 0.1,
) -> pd.DataFrame:
    """
    Executa o algoritmo Apriori para encontrar os itemsets
    frequentes no DataFrame binário.

    Parâmetros
    ----------
    df_binario : pd.DataFrame
        DataFrame binário gerado por ``preparar_dados_associacao``.
    suporte_minimo : float, opcional
        Suporte mínimo para filtrar itemsets (padrão 0.1 = 10%).

    Retorna
    -------
    pd.DataFrame
        Itemsets frequentes com colunas ``support`` e ``itemsets``.
    """
    print(
        f"⏳ Executando Apriori com suporte mínimo = {suporte_minimo:.2%}..."
    )

    itemsets = apriori(
        df_binario,
        min_support=suporte_minimo,
        use_colnames=True,
    )

    if itemsets.empty:
        print(
            "⚠ Nenhum itemset frequente encontrado. "
            "Tente reduzir o suporte mínimo."
        )
        return itemsets

    print(f"✔ {len(itemsets)} itemsets frequentes encontrados.")
    return itemsets


# ──────────────────────────────────────────────
# 3. Geração de Regras
# ──────────────────────────────────────────────
def gerar_regras(
    itemsets_frequentes: pd.DataFrame,
    confianca_minima: float = 0.5,
) -> pd.DataFrame:
    """
    Gera regras de associação a partir dos itemsets frequentes,
    ordenadas pela métrica *lift* de forma decrescente.

    Parâmetros
    ----------
    itemsets_frequentes : pd.DataFrame
        Resultado de ``executar_apriori``.
    confianca_minima : float, opcional
        Confiança mínima para filtrar regras (padrão 0.5 = 50%).

    Retorna
    -------
    pd.DataFrame
        Regras de associação com métricas (suporte, confiança,
        lift, etc.), ordenadas por lift.
    """
    if itemsets_frequentes.empty:
        print("⚠ Sem itemsets frequentes — impossível gerar regras.")
        return pd.DataFrame()

    print(
        f"⏳ Gerando regras com confiança mínima = {confianca_minima:.2%}..."
    )

    regras = association_rules(
        itemsets_frequentes,
        metric="confidence",
        min_threshold=confianca_minima,
    )

    if regras.empty:
        print(
            "⚠ Nenhuma regra gerada. "
            "Tente reduzir a confiança mínima."
        )
        return regras

    regras = regras.sort_values("lift", ascending=False).reset_index(drop=True)
    print(f"✔ {len(regras)} regras de associação geradas.")
    return regras


# ──────────────────────────────────────────────
# 4. Exibição Formatada
# ──────────────────────────────────────────────
def exibir_regras(regras: pd.DataFrame, top_n: int = 10) -> None:
    """
    Exibe as ``top_n`` regras de associação em formato de tabela
    legível no terminal.

    Parâmetros
    ----------
    regras : pd.DataFrame
        DataFrame de regras retornado por ``gerar_regras``.
    top_n : int, opcional
        Quantidade de regras a exibir (padrão 10).
    """
    if regras.empty:
        print("⚠ Nenhuma regra para exibir.")
        return

    top = regras.head(top_n)

    print()
    print(f"{'='*90}")
    print(f" Top {min(top_n, len(top))} Regras de Associação (ordenadas por Lift)")
    print(f"{'='*90}")
    print(
        f"{'#':<4} {'Antecedente':<28} {'→':^3} {'Consequente':<28} "
        f"{'Suporte':>8} {'Confiança':>10} {'Lift':>7}"
    )
    print(f"{'-'*90}")

    for idx, row in top.iterrows():
        antecedente = ", ".join(sorted(row["antecedents"]))
        consequente = ", ".join(sorted(row["consequents"]))
        print(
            f"{idx+1:<4} {antecedente:<28} {'→':^3} {consequente:<28} "
            f"{row['support']:>8.4f} {row['confidence']:>10.4f} "
            f"{row['lift']:>7.2f}"
        )

    print(f"{'='*90}")
    print()
