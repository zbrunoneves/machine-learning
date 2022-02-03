import os
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# EXPLORAÇÃO E VISUALIZACAO DE DADOS
# =============================================================================

# diretório atual
curr_dir = os.path.dirname(__file__)

# ler arquivo
data = pd.read_csv('../0.dados/iris-data.csv', sep=',', decimal='.')

# n primeiras linhas
data.head(n=10)

# n últimas linhas
data.tail(n=7)

# dimensões do dataset
data.shape

# tipos de dado das colunas
data.dtypes

# selecionar subset de colunas
data[['petal_length', 'petal_width']]

# alterar valores de uma coluna
data['species'] = data['species'].str.replace('Iris-', '')
data['species'] = data['species'].apply(lambda r: r.replace('Iris-', ''))

# contador de recorrências por coluna
data['species'].value_counts()

# informações estatísticas das colunas numéricas
# (média, desvio, quartis e min/max)
data.describe()

# informações estatísticas isoladas
# (média, mediana e desvio padrão)
data.mean()
data.median()
data.std()

# agrupar amostras por valor de uma coluna
data.groupby('species')

# montar tabela com informações personalizadas
data.groupby('species').agg({
    'petal_length': ['median', 'mean', 'std'],
    'petal_width': ['median', 'mean', 'std']
})

# montar tabela com informações personalizadas de maneira genérica
data.groupby('species').agg({
    x: ['median', 'mean', 'std'] for x in data.columns if x != 'species'
})

# trazer toda informação do dataset sem compressão '...'
data.to_string()

# conjunto de dados transposto
data.T

# remove coluna da base de dados
# data = data.drop(['a', 'b'])

# selecionar uma submatriz de dados do dataset
data.iloc[10:99, 0:5]

attributes = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# valores distintos por coluna
classes = data['species'].unique().tolist()

# =============================================================================
# VISUALIZAÇÃO DE PLOTS
# =============================================================================

# histograma de uma variavel
plot = data['petal_length'].plot.hist(bins=20)

# adição de legenda ao plot
plot.set(
    title='Distribuição do comprimento da pétala',
    xlabel='Comprimento (cm)',
    ylabel='Número de amostras'
)

# diagrama de dispersão entre duas colunas do dataset
plot = data.plot.scatter('petal_width', 'petal_length')
plot.set(
    title='Dispersão Largura vs Comprimento da Pétala',
    xlabel='Largura (cm)',
    ylabel='Comprimento (cm)'
)

# associa cores a rotulos
colors = ['red', 'green', 'blue']
attribute_colors = [colors[classes.index(r)] for r in labels]

# matriz de dispersão de todas as colunas
pd.plotting.scatter_matrix(
    attributes,
    c=attribute_colors,
    figsize=(13, 13),
    marker='o',
    s=50,
    alpha=0.5,
    diagonal='hist',
    hist_kwds={'bins': 20}
)

# adiciona titulo superior ao plot
plt.suptitle('Matriz de dispersão dos atributos', y=0.9)

# cria grafico 3d
figure = plt.figure(figsize=(15, 12))
plot = figure.add_subplot(111, projection='3d')
plot.scatter(
    data['sepal_length'],
    data['petal_length'],
    data['petal_width'],
    c=attribute_colors,
    marker='o',
    s=40,
    alpha=0.5
)

# exibir plot
plt.show()
