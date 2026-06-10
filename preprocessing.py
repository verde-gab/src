"""
Módulo de pré-processamento genérico para dados tabulares.

Funcionalidades:
    - Tratamento de valores nulos (mediana para numéricos, moda para categóricos)
    - Detecção automática de tipos de colunas
    - One-Hot Encoding para variáveis categóricas
    - Escalonamento (StandardScaler ou MinMaxScaler)
    - Pipeline completo de pré-processamento
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def detectar_tipos(df: pd.DataFrame) -> tuple[list, list]:
    """
    Detecta automaticamente colunas numéricas e categóricas do DataFrame.

    Parâmetros:
        df: DataFrame a ser analisado.

    Retorna:
        Tupla (colunas_numericas, colunas_categoricas).
    """
    colunas_numericas = df.select_dtypes(include="number").columns.tolist()
    colunas_categoricas = df.select_dtypes(exclude="number").columns.tolist()

    print(f"[INFO] Colunas numéricas ({len(colunas_numericas)}): {colunas_numericas}")
    print(
        f"[INFO] Colunas categóricas ({len(colunas_categoricas)}): {colunas_categoricas}"
    )

    return colunas_numericas, colunas_categoricas


def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores nulos do DataFrame.

    Estratégia:
        - Colunas numéricas → mediana
        - Colunas categóricas → moda (valor mais frequente)

    Parâmetros:
        df: DataFrame com possíveis valores nulos.

    Retorna:
        DataFrame com nulos preenchidos.
    """
    df = df.copy()
    total_nulos = df.isnull().sum().sum()

    if total_nulos == 0:
        print("[INFO] Nenhum valor nulo encontrado no DataFrame.")
        return df

    print(f"[INFO] Total de valores nulos encontrados: {total_nulos}")

    colunas_numericas, colunas_categoricas = detectar_tipos(df)

    for col in colunas_numericas:
        qtd_nulos = df[col].isnull().sum()
        if qtd_nulos > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(
                f"  → Coluna '{col}': {qtd_nulos} nulo(s) preenchido(s) com mediana = {mediana}"
            )

    for col in colunas_categoricas:
        qtd_nulos = df[col].isnull().sum()
        if qtd_nulos > 0:
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda)
            print(
                f"  → Coluna '{col}': {qtd_nulos} nulo(s) preenchido(s) com moda = '{moda}'"
            )

    print("[OK] Tratamento de nulos concluído.")
    return df


def aplicar_encoding(
    df: pd.DataFrame, colunas_categoricas: list
) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding nas colunas categóricas especificadas.

    Utiliza drop_first=True para evitar a armadilha da variável dummy
    (multicolinearidade perfeita).

    Parâmetros:
        df: DataFrame original.
        colunas_categoricas: lista de nomes das colunas categóricas.

    Retorna:
        DataFrame com as colunas categóricas substituídas pelas dummies.
    """
    df = df.copy()

    if not colunas_categoricas:
        print("[INFO] Nenhuma coluna categórica para codificar.")
        return df

    print(f"[INFO] Aplicando One-Hot Encoding em: {colunas_categoricas}")
    df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

    # Garante que todas as colunas resultantes sejam numéricas (int)
    colunas_dummy = [
        c
        for c in df.columns
        if any(c.startswith(f"{cat}_") for cat in colunas_categoricas)
    ]
    df[colunas_dummy] = df[colunas_dummy].astype(int)

    print(
        f"[OK] Encoding concluído. Novas dimensões do DataFrame: {df.shape}"
    )
    return df


def aplicar_escalonamento(
    df: pd.DataFrame,
    colunas_numericas: list,
    metodo: str = "standard",
) -> pd.DataFrame:
    """
    Aplica escalonamento (normalização/padronização) nas colunas numéricas.

    Parâmetros:
        df: DataFrame com dados numéricos.
        colunas_numericas: lista de colunas a escalonar.
        metodo: 'standard' para StandardScaler (z-score) ou
                'minmax' para MinMaxScaler (0-1).

    Retorna:
        DataFrame com as colunas numéricas escalonadas.
    """
    df = df.copy()

    if not colunas_numericas:
        print("[INFO] Nenhuma coluna numérica para escalonar.")
        return df

    metodo = metodo.lower().strip()

    if metodo == "standard":
        scaler = StandardScaler()
        nome_metodo = "StandardScaler (z-score)"
    elif metodo == "minmax":
        scaler = MinMaxScaler()
        nome_metodo = "MinMaxScaler (0-1)"
    else:
        raise ValueError(
            f"Método '{metodo}' não reconhecido. Use 'standard' ou 'minmax'."
        )

    print(
        f"[INFO] Aplicando {nome_metodo} em {len(colunas_numericas)} coluna(s)..."
    )
    df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
    print("[OK] Escalonamento concluído.")

    return df


def preprocessar_completo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o pipeline completo de pré-processamento:
        1. Tratamento de valores nulos
        2. One-Hot Encoding das colunas categóricas
        3. Escalonamento (StandardScaler) das colunas numéricas

    Parâmetros:
        df: DataFrame bruto.

    Retorna:
        DataFrame totalmente pré-processado.
    """
    print("=" * 60)
    print("  PIPELINE DE PRÉ-PROCESSAMENTO")
    print("=" * 60)
    print(f"[INFO] Dimensões originais: {df.shape}")
    print()

    # --- Etapa 1: Nulos ---
    print("── Etapa 1: Tratamento de nulos ──")
    df = tratar_nulos(df)
    print()

    # --- Detecção de tipos (após tratar nulos) ---
    colunas_numericas, colunas_categoricas = detectar_tipos(df)
    print()

    # --- Etapa 2: Encoding ---
    print("── Etapa 2: One-Hot Encoding ──")
    df = aplicar_encoding(df, colunas_categoricas)
    print()

    # --- Etapa 3: Escalonamento ---
    # Após encoding, todas as colunas são numéricas; escalonar as originais.
    print("── Etapa 3: Escalonamento ──")
    df = aplicar_escalonamento(df, colunas_numericas, metodo="standard")
    print()

    # --- Resumo ---
    print("=" * 60)
    print("  RESUMO DO PRÉ-PROCESSAMENTO")
    print("=" * 60)
    print(f"  Dimensões finais : {df.shape}")
    print(f"  Nulos restantes  : {df.isnull().sum().sum()}")
    print(f"  Tipos de dados   :")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {dtype}: {count} coluna(s)")
    print("=" * 60)

    return df
