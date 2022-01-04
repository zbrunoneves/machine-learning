import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

data = pd.read_excel('../0.dados/boston.xlsx')

# descarta coluna com índices
data = data.iloc[:, 1:]

# CRIM    crime per capita
# ZN      fração de propriedades residenciais com mais de 2.500m²
# INDUS   fração de acres destinados à indústria
# CHAS    binário para próximo ao rio Charles
# NOX     concentração de óxidos nítricos em 10 milhões
# RM      número médio de cômodos por moradia
# AGE     fração de moradias com mais de 40 anos
# DIS     distância ponderada para cinco centros comerciais de Boston
# RAD     índice de acessbilidade a rodovias radiais
# TAX     imposto por propriedade em $10.000
# PTRATIO razão aluno/professor por cidade
# B       1000(Bk - 0.63)² onde Bk é a fração de pessoas negras por cidade
# LSTAT   % da parcela mais pobre da população
# MEDV    valor médio de moradias em $1000

print('Pearson correlation coefficient')
for col in data.columns:
    pearson_coef = pearsonr(data[col], data['target'])
    print('%7s = %6.3f' % (col, pearson_coef[0]))

    data.plot.scatter(x=col, y='target')

# explorando correlações entre atributos
att1 = 'LSTAT'
att2 = 'RM'

print('%13s = %6.3f' % (att1+'_'+att2, pearsonr(data[att1], data[att2])[0]))
data.plot.scatter(x=att1, y=att2)

plt.show()
