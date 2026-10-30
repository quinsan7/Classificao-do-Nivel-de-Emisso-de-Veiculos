# Classificação do Nível de Emissão de Veículos com SVM

Projeto desenvolvido para o **UMC Summit (4ª edição)**, na disciplina de Inteligência Artificial da Universidade de Mogi das Cruzes.

O notebook aplica o algoritmo **Support Vector Machine (SVM)** para classificar o nível de emissão de poluentes de veículos (`Low`, `Medium` ou `High`) e inclui um teste de vazamento de dados que mostra de onde vem o desempenho do modelo.

**Notebook:** `Classificacao_Emissao_Veiculos_SVM_Kaggle.ipynb`
**Link do notebook no Kaggle:** _(adicione aqui)_

## Autores

- Fernando Panichek Quinsan
- Nathan da Silva Carvalho
- Fabiano Bezerra Menegídio
- Higor Barreto Campos

Universidade de Mogi das Cruzes, Mogi das Cruzes, São Paulo, Brasil.

## Dataset

**Vehicle Emission Dataset** (Kaggle, `s3programmer/vehcle-emission-dataset`), arquivo `vehicle_emission_dataset.csv`.

- 10.000 registros e 19 colunas (18 de entrada e 1 alvo).
- Tipos de veículo: ônibus, carros, motos e caminhões.
- Dados operacionais: tamanho do motor, idade, quilometragem, velocidade e aceleração.
- Dados de via, tráfego e clima: tipo de via, condição do tráfego, temperatura, umidade, vento e pressão.
- Emissões: CO2, NOx, PM2.5, VOC e SO2.
- Alvo: `Emission Level` (`High` 44,2%, `Medium` 32,7%, `Low` 23,0%).
- Não há valores ausentes nem linhas duplicadas.

## Como executar

1. No Kaggle, importe o arquivo `.ipynb` (File → Import Notebook).
2. Anexe o dataset Vehicle Emission Dataset ao notebook (Add Input).
3. Execute todas as células (Run All).

A primeira célula lista os arquivos de `/kaggle/input` para confirmar que o dataset está anexado. O caminho usado é `/kaggle/input/datasets/s3programmer/vehcle-emission-dataset/vehicle_emission_dataset.csv`; se o seu dataset estiver em outra pasta, ajuste a linha do `pd.read_csv`.

**Bibliotecas:** `numpy`, `pandas`, `matplotlib`, `seaborn` e `scikit-learn` (todas já disponíveis no ambiente do Kaggle).

## Estrutura do notebook

| Seção | Conteúdo |
|---|---|
| 1 | Importação de bibliotecas |
| 2 | Carregamento dos dados |
| 3 | Avaliação do banco de dados (`info`, `describe`, balanceamento das classes) |
| 4 | Tratamento de dados ausentes e duplicados |
| 5 | Análise exploratória (nível de emissão por tipo de veículo) |
| 6 | Separação das variáveis (`X` e `y`) |
| 7 | Tratamento de encoding das variáveis categóricas (colunas 0/1) |
| 8 | Divisão em treino e teste (80% / 20%, `random_state=42`, `stratify=y`) |
| 9 | Normalização das variáveis numéricas (z-score, calculado só com o treino) |
| 10 | Criação do modelo SVM (kernel RBF) |
| 11 | Ajuste de hiperparâmetros com `GridSearchCV` |
| 12 | Treinamento do modelo final |
| 13 | Previsão |
| 14 | Avaliação (acurácia) |
| 15 | Matriz de confusão |
| 16 | Precisão, recall e F1-Score |
| 17 | Teste de vazamento de dados |
| 18 | Conclusão |

## Método

- **Encoding:** as quatro colunas categóricas (`Vehicle Type`, `Fuel Type`, `Road Type` e `Traffic Conditions`) viram colunas 0/1, uma por categoria, o que resulta em 28 colunas de entrada.
- **Normalização:** as 14 colunas numéricas são padronizadas com média e desvio padrão do conjunto de treino.
- **Modelo:** `SVC(kernel='rbf')`.
- **Ajuste:** `GridSearchCV` com `C` em `[1, 10, 100]`, `gamma` em `['scale', 'auto']`, validação cruzada com 5 divisões e métrica de acurácia.
- **Melhor combinação:** `C=10` e `gamma='auto'`.

## Resultados

| Experimento | Acurácia no teste |
|---|---|
| SVM com todas as variáveis | **96,90%** (validação cruzada: 96,52%) |
| Regra manual de 2 cortes em CO2 (sem treino) | 100,00% |
| SVM só com CO2 | 99,80% |
| SVM sem CO2 | 44,15% |
| Baseline (sempre prever `High`) | 44,20% |

Desempenho por nível (conjunto de teste, 2.000 registros):

| Nível | Precisão | Recall | F1-Score |
|---|---|---|---|
| Low | 0,963 | 0,972 | 0,968 |
| Medium | 0,960 | 0,945 | 0,952 |
| High | 0,979 | 0,985 | 0,982 |

## Principais achados

- O SVM atinge 96,90% de acurácia, mas esse resultado vem de uma única variável. O nível de emissão corresponde exatamente a faixas do `CO2 Emissions`: `Low` abaixo de 150 g, `Medium` de 150 a 300 g e `High` acima de 300 g.
- Uma regra de 2 cortes em CO2 acerta 100% do conjunto de teste, e o SVM treinado só com CO2 acerta 99,80%.
- Sem o CO2, o SVM cai para 44,15%, praticamente igual ao baseline que sempre prevê a classe mais comum (44,20%).
- A correlação das demais variáveis numéricas com o nível é próxima de zero (de -0,010 para a aceleração a 0,022 para o SO2).

## Limitações

- Os 96,90% **não indicam que o modelo descobriu um padrão** nas condições operacionais ou climáticas. Ele reaprendeu a regra que gerou o rótulo (vazamento de dados). Conclusões sobre aceleração, clima ou tráfego como determinantes da emissão não são sustentadas por estes dados.
- Um modelo útil para gestão ambiental exigiria dados com o rótulo definido independentemente das emissões medidas.
- A normalização é calculada antes do `GridSearchCV`, usando todo o conjunto de treino; isso é uma simplificação comum e tem efeito desprezível neste caso.
