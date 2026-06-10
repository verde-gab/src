"""
loader.py — Módulo genérico para carregamento e inspeção de arquivos CSV.

Este módulo fornece funções utilitárias para carregar qualquer arquivo CSV
em um DataFrame do pandas, inspecionar seus dados e listar arquivos CSV
disponíveis em um diretório.
"""

import os
from pathlib import Path

import pandas as pd


def carregar_csv(caminho: str) -> pd.DataFrame:
    """Carrega um arquivo CSV e retorna um DataFrame.

    Parâmetros
    ----------
    caminho : str
        Caminho absoluto ou relativo para o arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame contendo os dados do CSV.

    Levanta
    -------
    FileNotFoundError
        Se o arquivo não for encontrado no caminho informado.
    """
    caminho_arquivo = Path(caminho)

    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho_arquivo)
    print(f"✔ Arquivo carregado com sucesso: {caminho_arquivo.name}")
    print(f"  → {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df


def inspecionar_dados(df: pd.DataFrame) -> None:
    """Exibe um resumo completo do DataFrame no terminal.

    Imprime a forma (shape), tipos de dados, porcentagem de valores nulos,
    as 5 primeiras linhas e estatísticas descritivas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser inspecionado.
    """
    separador = "=" * 60

    # --- Forma do DataFrame ---
    print(f"\n{separador}")
    print("📐 FORMA DO DATASET")
    print(separador)
    print(f"  Linhas : {df.shape[0]}")
    print(f"  Colunas: {df.shape[1]}")

    # --- Tipos de dados ---
    print(f"\n{separador}")
    print("🔤 TIPOS DE DADOS POR COLUNA")
    print(separador)
    for coluna in df.columns:
        print(f"  {coluna:<30} → {df[coluna].dtype}")

    # --- Porcentagem de nulos ---
    print(f"\n{separador}")
    print("❌ PORCENTAGEM DE VALORES NULOS")
    print(separador)
    nulos = df.isnull().mean() * 100
    for coluna, pct in nulos.items():
        indicador = "⚠️" if pct > 0 else "✔"
        print(f"  {indicador} {coluna:<30} → {pct:.2f}%")

    # --- Primeiras linhas ---
    print(f"\n{separador}")
    print("👀 PRIMEIRAS 5 LINHAS")
    print(separador)
    print(df.head().to_string(index=False))

    # --- Estatísticas descritivas ---
    print(f"\n{separador}")
    print("📊 ESTATÍSTICAS DESCRITIVAS")
    print(separador)
    print(df.describe().to_string())

    print(f"\n{separador}\n")


def listar_csvs(diretorio: str) -> list[str]:
    """Lista todos os arquivos CSV presentes em um diretório.

    Parâmetros
    ----------
    diretorio : str
        Caminho absoluto ou relativo para o diretório a ser varrido.

    Retorna
    -------
    list[str]
        Lista com os caminhos completos de cada arquivo CSV encontrado.

    Levanta
    -------
    NotADirectoryError
        Se o caminho informado não for um diretório válido.
    """
    caminho_dir = Path(diretorio)

    if not caminho_dir.is_dir():
        raise NotADirectoryError(f"Diretório não encontrado: {diretorio}")

    arquivos_csv = sorted(
        str(arquivo) for arquivo in caminho_dir.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() == ".csv"
    )

    print(f"📂 {len(arquivos_csv)} arquivo(s) CSV encontrado(s) em '{diretorio}':")
    for csv in arquivos_csv:
        print(f"  → {os.path.basename(csv)}")

    return arquivos_csv
