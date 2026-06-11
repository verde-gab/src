# Mineração de Dados — Ferramenta Geral

Ferramenta interativa de mineração de dados para análise de qualquer dataset CSV.  
**Grupo 2** — Disciplina de Inteligência Computacional.

## 👥 Integrantes

- Gabriel
- Kayky
- Júlia
- Felipe

---

## 📋 O que é Mineração de Dados?

Mineração de dados (*Data Mining*) é o processo de descobrir padrões, correlações e anomalias em grandes volumes de dados usando métodos estatísticos, matemáticos e computacionais. Faz parte do processo de **KDD** (Knowledge Discovery in Databases — Descoberta de Conhecimento em Bases de Dados).

### Etapas do KDD

1. **Seleção** — Escolher os dados relevantes
2. **Pré-processamento** — Limpar e preparar os dados (nulos, encoding, escalonamento)
3. **Transformação** — Converter dados para formatos adequados aos algoritmos
4. **Mineração** — Aplicar algoritmos para encontrar padrões
5. **Avaliação** — Interpretar e validar os resultados

---

## 🛠️ Técnicas Implementadas

### Classificação (Supervisionada)
Técnicas que aprendem com dados rotulados para prever categorias:

| Algoritmo | Descrição |
|---|---|
| **Árvore de Decisão** | Cria regras hierárquicas tipo "se X > 5, então classe A". Interpretável e visual. |
| **KNN** | Classifica com base nos K vizinhos mais próximos. Simples e eficaz. |
| **Random Forest** | Conjunto de várias árvores de decisão (ensemble). Mais robusto e preciso. |
| **Naive Bayes** | Baseado no Teorema de Bayes com premissa de independência entre features. Rápido. |

### Clusterização (Não-Supervisionada)
Técnicas que agrupam dados sem rótulos pré-definidos:

| Algoritmo | Descrição |
|---|---|
| **K-Means** | Agrupa dados em K clusters baseado em distância aos centróides. Usa o método do cotovelo para sugerir K ideal. |
| **DBSCAN** | Agrupa por densidade — encontra clusters de formato irregular e detecta pontos de ruído (outliers). |

### Regras de Associação
Descobre padrões do tipo "quem tem X também tem Y":

| Algoritmo | Descrição |
|---|---|
| **Apriori** | Encontra itemsets frequentes e gera regras com métricas de suporte, confiança e lift. |

---

## 🚀 Como Instalar e Executar

### Pré-requisitos
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/SEU_USUARIO/Mineracao_de_Dados.git
cd Mineracao_de_Dados

# Instalar dependências com uv
uv sync
```

### Execução

```bash
uv run python main.py
```

---

## 📁 Estrutura do Projeto

```
Mineracao_de_Dados/
├── data/                          # Datasets CSV
│   └── heart_disease.csv          # Dataset de exemplo (doenças cardíacas)
├── plots/                         # Gráficos gerados automaticamente
├── src/
│   ├── __init__.py
│   ├── loader.py                  # Carregamento e inspeção de CSV
│   ├── preprocessing.py           # Tratamento de nulos, encoding, escalonamento
│   ├── visualization.py           # Heatmap, histogramas, boxplot, scatter
│   └── mining/
│       ├── __init__.py
│       ├── classification.py      # Árvore de Decisão, KNN, Random Forest, Naive Bayes
│       ├── clustering.py          # K-Means, DBSCAN
│       ├── association.py         # Apriori (regras de associação)
│       └── evaluation.py          # Métricas de avaliação
├── main.py                        # Menu interativo (ponto de entrada)
├── pyproject.toml                 # Configuração do projeto e dependências
└── README.md                      # Este arquivo
```

---

## 💻 Como Usar

Ao executar `uv run python main.py`, você verá o menu interativo:

```
╔══════════════════════════════════════════════════════╗
║       MINERAÇÃO DE DADOS — FERRAMENTA GERAL         ║
╚══════════════════════════════════════════════════════╝

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
```

### Fluxo recomendado

1. **Carregar** qualquer arquivo CSV
2. **Inspecionar** para entender os dados (tipos, nulos, estatísticas)
3. **Pré-processar** (tratar nulos, encoding categórico, escalonamento)
4. **Visualizar** (heatmap, distribuições, outliers)
5. **Minerar** usando classificação, clusterização ou regras de associação

---

## 📊 Exemplo com Dataset de Doenças Cardíacas

O dataset `heart_disease.csv` contém 10.000 registros com 21 variáveis sobre saúde cardiovascular:

- **Variáveis numéricas**: Idade, Pressão Arterial, Colesterol, IMC, etc.
- **Variáveis categóricas**: Gênero, Fumante, Diabetes, Status de Doença Cardíaca, etc.
- **Variável alvo**: `Heart Disease Status` (Sim/Não)

### Exemplos de uso:
- **Classificação**: Prever se um paciente tem doença cardíaca com base nos fatores de risco
- **Clusterização**: Agrupar pacientes por perfil de saúde semelhante
- **Associação**: Descobrir quais fatores de risco aparecem juntos

---

## 📦 Dependências

| Pacote | Versão | Uso |
|---|---|---|
| pandas | ≥ 3.0.3 | Manipulação de dados |
| numpy | ≥ 2.4.6 | Operações numéricas |
| matplotlib | ≥ 3.10.9 | Gráficos e visualizações |
| seaborn | ≥ 0.13.2 | Visualizações estatísticas |
| scikit-learn | ≥ 1.7.0 | Classificação, clusterização, métricas |
| mlxtend | ≥ 0.23.0 | Regras de associação (Apriori) |

---

## 📝 Licença

Projeto acadêmico — Disciplina de Inteligência Computacional.
