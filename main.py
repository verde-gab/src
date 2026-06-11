"""
Mineração de Dados — Ferramenta Geral
Menu interativo para carregar, pré-processar, visualizar e minerar dados de qualquer CSV.
"""

import sys
import os

# Adiciona o diretório raiz ao path para imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import carregar_csv, inspecionar_dados, listar_csvs
from preprocessing import (
    preprocessar_completo,
    detectar_tipos,
    tratar_nulos,
    aplicar_encoding,
    aplicar_escalonamento,
)
from visualization import (
    heatmap_correlacao,
    histogramas_distribuicao,
    graficos_categoricos,
    scatter_matrix,
    boxplot_outliers,
    grafico_importancia_features,
)
from mining.classification import separar_dados, executar_todos
from mining.clustering import kmeans, metodo_cotovelo, dbscan, visualizar_clusters
from mining.association import (
    preparar_dados_associacao,
    executar_apriori,
    gerar_regras,
    exibir_regras,
)


BANNER = """
╔══════════════════════════════════════════════════════╗
║       MINERAÇÃO DE DADOS — FERRAMENTA GERAL         ║
║              Grupo 6 — Inteligência Artificial       ║
╚══════════════════════════════════════════════════════╝
"""

MENU_PRINCIPAL = """
┌──────────────────────────────────────────┐
│            MENU PRINCIPAL                │
├──────────────────────────────────────────┤
│  1. Carregar dataset (CSV)               │
│  2. Inspecionar dados                    │
│  3. Pré-processar dados                  │
│  4. Visualizar dados                     │
│  5. Classificação (supervisionada)       │
│  6. Clusterização (não-supervisionada)   │
│  7. Regras de Associação (Apriori)       │
│  8. Exportar dataset processado          │
│  0. Sair                                 │
└──────────────────────────────────────────┘
"""

MENU_VISUALIZACAO = """
  ┌─ Visualizações ─────────────────────┐
  │  1. Heatmap de correlação            │
  │  2. Histogramas de distribuição       │
  │  3. Gráficos categóricos             │
  │  4. Scatter matrix                   │
  │  5. Boxplot (outliers)               │
  │  6. Todas as visualizações           │
  │  0. Voltar                           │
  └──────────────────────────────────────┘
"""

MENU_CLUSTERIZACAO = """
  ┌─ Clusterização ─────────────────────┐
  │  1. Método do cotovelo (sugerir k)   │
  │  2. K-Means                          │
  │  3. DBSCAN                           │
  │  0. Voltar                           │
  └──────────────────────────────────────┘
"""


def limpar_tela():
    """Limpa o terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def pausar():
    """Pausa até o usuário pressionar Enter."""
    input("\n⏎  Pressione Enter para continuar...")


def ler_opcao(prompt: str = "Escolha uma opção: ") -> str:
    """Lê a opção do usuário."""
    return input(f"\n➤  {prompt}").strip()


def menu_carregar(estado: dict):
    """Carrega um dataset CSV."""
    print("\n📂  CARREGAR DATASET")
    print("-" * 40)

    # Listar CSVs disponíveis na pasta data/
    csvs = listar_csvs("data")
    if csvs:
        print("\nCSVs disponíveis na pasta 'data/':")
        for i, csv in enumerate(csvs, 1):
            print(f"  {i}. {os.path.basename(csv)}")
        print(f"  {len(csvs) + 1}. Digitar caminho manualmente")

        opcao = ler_opcao("Número do arquivo ou caminho: ")
        try:
            idx = int(opcao)
            if 1 <= idx <= len(csvs):
                caminho = csvs[idx - 1]
            else:
                caminho = input("  Caminho completo do CSV: ").strip()
        except ValueError:
            caminho = opcao
    else:
        print("\nNenhum CSV encontrado na pasta 'data/'.")
        caminho = input("  Caminho completo do CSV: ").strip()

    try:
        estado["df_original"] = carregar_csv(caminho)
        estado["df_processado"] = None
        estado["caminho"] = caminho
        print(f"\n✅  Dataset carregado com sucesso: {caminho}")
    except Exception as e:
        print(f"\n❌  Erro ao carregar: {e}")


def menu_inspecionar(estado: dict):
    """Inspeciona o dataset carregado."""
    df = estado.get("df_original")
    if df is None:
        print("\n⚠️  Nenhum dataset carregado. Use a opção 1 primeiro.")
        return
    inspecionar_dados(df)


def menu_preprocessar(estado: dict):
    """Pré-processa o dataset."""
    df = estado.get("df_original")
    if df is None:
        print("\n⚠️  Nenhum dataset carregado. Use a opção 1 primeiro.")
        return

    print("\n⚙️  PRÉ-PROCESSAMENTO")
    print("-" * 40)
    print("  1. Pré-processamento completo (automático)")
    print("  2. Apenas tratar nulos")
    print("  3. Apenas aplicar encoding")
    print("  4. Apenas escalonamento")

    opcao = ler_opcao()

    try:
        if opcao == "1":
            estado["df_processado"] = preprocessar_completo(df)
        elif opcao == "2":
            estado["df_processado"] = tratar_nulos(df)
        elif opcao == "3":
            df_temp = estado.get("df_processado", df).copy()
            _, categoricas = detectar_tipos(df_temp)
            estado["df_processado"] = aplicar_encoding(df_temp, categoricas)
        elif opcao == "4":
            df_temp = estado.get("df_processado", df).copy()
            numericas, _ = detectar_tipos(df_temp)
            print("\n  Método de escalonamento:")
            print("    1. StandardScaler (z-score)")
            print("    2. MinMaxScaler (0-1)")
            metodo_esc = ler_opcao()
            metodo = "standard" if metodo_esc == "1" else "minmax"
            estado["df_processado"] = aplicar_escalonamento(df_temp, numericas, metodo)
        else:
            print("Opção inválida.")
            return

        print("\n✅  Pré-processamento concluído!")
    except Exception as e:
        print(f"\n❌  Erro no pré-processamento: {e}")


def menu_visualizar(estado: dict):
    """Menu de visualizações."""
    df = _obter_df(estado)
    if df is None:
        return

    while True:
        print(MENU_VISUALIZACAO)
        opcao = ler_opcao()

        try:
            if opcao == "1":
                heatmap_correlacao(df)
            elif opcao == "2":
                histogramas_distribuicao(df)
            elif opcao == "3":
                graficos_categoricos(estado.get("df_original", df))
            elif opcao == "4":
                scatter_matrix(df)
            elif opcao == "5":
                boxplot_outliers(df)
            elif opcao == "6":
                print("\n📊  Gerando todas as visualizações...")
                heatmap_correlacao(df)
                histogramas_distribuicao(df)
                graficos_categoricos(estado.get("df_original", df))
                scatter_matrix(df)
                boxplot_outliers(df)
                print("\n✅  Todas as visualizações foram salvas na pasta 'plots/'!")
            elif opcao == "0":
                break
            else:
                print("Opção inválida.")
        except Exception as e:
            print(f"\n❌  Erro na visualização: {e}")

        pausar()


def menu_classificacao(estado: dict):
    """Menu de classificação supervisionada."""
    if estado.get("df_processado") is None:
        print("\n⚠️  Você precisa pré-processar os dados antes de executar a classificação.")
        print("   Use a opção 3 (Pré-processar dados → Completo) para converter colunas")
        print("   categóricas em numéricas e escalonar os dados.")
        return

    df = estado["df_processado"]

    # Verificação de segurança: garantir que não há colunas de texto
    colunas_texto = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if colunas_texto:
        print(f"\n⚠️  O dataset ainda contém colunas de texto: {colunas_texto}")
        print("   Os classificadores do sklearn exigem dados 100% numéricos.")
        print("   Execute o pré-processamento completo (opção 3 → 1) antes.")
        return

    print("\n🎯  CLASSIFICAÇÃO SUPERVISIONADA")
    print("-" * 40)

    # Mostrar colunas disponíveis
    print("\nColunas disponíveis:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    opcao = ler_opcao("Número da coluna alvo (target): ")
    try:
        idx = int(opcao) - 1
        coluna_alvo = df.columns[idx]
    except (ValueError, IndexError):
        print("❌  Coluna inválida.")
        return

    print(f"\n  Coluna alvo selecionada: '{coluna_alvo}'")

    try:
        # Se a coluna original não existir mais (ex: virou Dummie no encoding),
        # tenta encontrar a versão com '_Yes' ou similar
        if coluna_alvo not in df.columns:
            possiveis = [c for c in df.columns if c.startswith(f"{coluna_alvo}_")]
            if possiveis:
                print(f"⚠️  Coluna '{coluna_alvo}' não encontrada. Usando '{possiveis[-1]}' gerada no pré-processamento.")
                coluna_alvo = possiveis[-1]

        X_train, X_test, y_train, y_test = separar_dados(df, coluna_alvo)
        resultados = executar_todos(X_train, X_test, y_train, y_test)

        # Tentar gerar gráfico de importância de features (do Random Forest)
        if "Random Forest" in resultados and "feature_importances" in resultados["Random Forest"]:
            fi = resultados["Random Forest"]["feature_importances"]
            grafico_importancia_features(
                list(fi.values()),
                list(fi.keys()),
            )

        print("\n✅  Classificação concluída! Gráficos salvos em 'plots/'.")
    except Exception as e:
        print(f"\n❌  Erro na classificação: {e}")


def menu_clusterizacao(estado: dict):
    """Menu de clusterização."""
    if estado.get("df_processado") is None:
        print("\n⚠️  Você precisa pré-processar os dados antes de executar a clusterização.")
        print("   Use a opção 3 (Pré-processar dados → Completo) para converter colunas")
        print("   categóricas em numéricas e escalonar os dados.")
        return

    df = estado["df_processado"]

    # Usar apenas colunas numéricas para clusterização
    numericas, _ = detectar_tipos(df)
    df_num = df[numericas].dropna()

    if df_num.empty:
        print("\n⚠️  Nenhuma coluna numérica encontrada para clusterização.")
        return

    while True:
        print(MENU_CLUSTERIZACAO)
        opcao = ler_opcao()

        try:
            if opcao == "1":
                k_max_input = ler_opcao("K máximo para teste (padrão 10): ")
                k_max = int(k_max_input) if k_max_input else 10
                metodo_cotovelo(df_num, k_max=k_max)

            elif opcao == "2":
                k_input = ler_opcao("Número de clusters k (padrão 3): ")
                k = int(k_input) if k_input else 3
                labels, modelo = kmeans(df_num, n_clusters=k)
                visualizar_clusters(df_num, labels, titulo=f"K-Means (k={k})")

            elif opcao == "3":
                eps_input = ler_opcao("Valor de eps (padrão 0.5): ")
                eps = float(eps_input) if eps_input else 0.5
                min_s_input = ler_opcao("min_samples (padrão 5): ")
                min_s = int(min_s_input) if min_s_input else 5
                labels, modelo = dbscan(df_num, eps=eps, min_samples=min_s)
                visualizar_clusters(df_num, labels, titulo="DBSCAN")

            elif opcao == "0":
                break
            else:
                print("Opção inválida.")
        except Exception as e:
            print(f"\n❌  Erro na clusterização: {e}")

        pausar()


def menu_associacao(estado: dict):
    """Menu de regras de associação."""
    df = estado.get("df_original")
    if df is None:
        print("\n⚠️  Nenhum dataset carregado. Use a opção 1 primeiro.")
        return

    print("\n🔗  REGRAS DE ASSOCIAÇÃO (Apriori)")
    print("-" * 40)

    suporte_input = ler_opcao("Suporte mínimo (padrão 0.1): ")
    suporte = float(suporte_input) if suporte_input else 0.1

    confianca_input = ler_opcao("Confiança mínima (padrão 0.5): ")
    confianca = float(confianca_input) if confianca_input else 0.5

    try:
        print("\n  Preparando dados para associação...")
        df_binario = preparar_dados_associacao(df)

        print("  Executando Apriori...")
        itemsets = executar_apriori(df_binario, suporte_minimo=suporte)

        if itemsets.empty:
            print("\n⚠️  Nenhum itemset frequente encontrado. Tente reduzir o suporte mínimo.")
            return

        print("  Gerando regras...")
        regras = gerar_regras(itemsets, confianca_minima=confianca)

        if regras.empty:
            print("\n⚠️  Nenhuma regra gerada. Tente reduzir a confiança mínima.")
            return

        top_input = ler_opcao("Quantas regras exibir? (padrão 10): ")
        top_n = int(top_input) if top_input else 10

        exibir_regras(regras, top_n=top_n)
        print("\n✅  Regras de associação geradas com sucesso!")
    except Exception as e:
        print(f"\n❌  Erro nas regras de associação: {e}")


def menu_exportar(estado: dict):
    """Exporta o dataset processado."""
    df = estado.get("df_processado")
    if df is None:
        print("\n⚠️  Nenhum dataset pré-processado. Use a opção 3 primeiro.")
        return

    nome_padrao = "data/dataset_processado.csv"
    nome_input = ler_opcao(f"Nome do arquivo (padrão '{nome_padrao}'): ")
    nome = nome_input if nome_input else nome_padrao

    try:
        df.to_csv(nome, index=False)
        print(f"\n✅  Dataset exportado para: {nome}")
        print(f"   Linhas: {len(df)} | Colunas: {len(df.columns)}")
    except Exception as e:
        print(f"\n❌  Erro ao exportar: {e}")


def _obter_df(estado: dict):
    """Retorna o DataFrame processado (se existir) ou o original."""
    df = estado.get("df_processado")
    if df is None:
        df = estado.get("df_original")
    if df is None:
        print("\n⚠️  Nenhum dataset carregado. Use a opção 1 primeiro.")
        return None
    return df


def main():
    """Loop principal do menu interativo."""
    limpar_tela()
    print(BANNER)

    estado = {
        "df_original": None,
        "df_processado": None,
        "caminho": None,
    }

    acoes = {
        "1": menu_carregar,
        "2": menu_inspecionar,
        "3": menu_preprocessar,
        "4": menu_visualizar,
        "5": menu_classificacao,
        "6": menu_clusterizacao,
        "7": menu_associacao,
        "8": menu_exportar,
    }

    while True:
        # Mostrar status atual
        if estado["caminho"]:
            df = _obter_df(estado)
            status = f"📄 {os.path.basename(estado['caminho'])}"
            if df is not None:
                status += f" ({len(df)} linhas × {len(df.columns)} colunas)"
            if estado["df_processado"] is not None:
                status += " [pré-processado ✓]"
            print(f"\n  Status: {status}")

        print(MENU_PRINCIPAL)
        opcao = ler_opcao()

        if opcao == "0":
            print("\n👋  Encerrando. Até mais!\n")
            break
        elif opcao in acoes:
            acoes[opcao](estado)
            pausar()
        else:
            print("❌  Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
