#==============================================================================
# REGRESSOR LINEAR vs KNN vs POLINOMIAL - CONJUNTO BOSTON
#==============================================================================

import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler, LabelBinarizer, PolynomialFeatures

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Ler as amostras da planilha Excel e gravar como dataframe Pandas
#------------------------------------------------------------------------------

dados = pd.read_csv("C:\\Users\\lucas\Downloads\\regressao\\conjunto_de_treinamento.csv")
resposta = pd.read_csv("C:\\Users\\lucas\Downloads\\regressao\\conjunto_de_teste.csv")

#-------------------------------------------------------------------------------
# Explorar os dados
#-------------------------------------------------------------------------------

print ( '\nImprimir o conjunto de dados:\n')

print(dados)

print ( '\nImprimir o conjunto de dados transposto')
print ('para visualizar os nomes de todas as colunas:\n')

print(dados.T)
print(dados.dtypes)

variaveis_categoricas = [
    x for x in dados.columns if dados[x].dtype == 'object'
    ]

print(variaveis_categoricas)

print ( '\nVerificar a cardinalidade de cada variável categórica:')
print ( 'obs: cardinalidade = qtde de valores distintos que a variável pode assumir\n')

for v in variaveis_categoricas:
    
    print ('\n%15s:'%v , "%4d categorias" % len(dados[v].unique()))
    print (dados[v].unique(),'\n')

#-------------------------------------------------------------------------------
# Executar preprocessamento dos dados
#-------------------------------------------------------------------------------

# tipo              --> nao-ordinal com 4 categorias        --> one-hot enconding
# bairro            --> nao-ordinal com 66 categorias       --> descartar
# tipo_vendedor     --> binaria     --> binarizar mapear para (0/1)
# diferenciais      --> nao-ordinal com 83 categorias       --> descartar

print ( '\nDescartar as variáveis de cardinalidade muito alta:\n')

print (dados.T)
dados = dados.drop(['bairro','diferenciais'],axis=1)
print (dados.T)

print ( '\nAplicar one-hot encoding nas variáveis que tenham')
print ( '3 ou mais categorias:')

dados = pd.get_dummies(dados,columns=['tipo'])
resposta = pd.get_dummies(resposta,columns=['tipo'])
print (dados.head(5).T)

print ( '\nAplicar binarização simples nas variáveis que tenham')
print ( 'apenas 2 categorias:\n')

binarizador = LabelBinarizer()
for v in ['tipo_vendedor']:
    dados[v] = binarizador.fit_transform(dados[v])
    resposta[v] = binarizador.fit_transform(resposta[v])
print (dados.head(5).T)

print ( '\nVerificar a quantidade de amostras de cada classe:\n')

print(dados['preco'].value_counts())

print ( '\nVerificar o valor médio de cada atributo em cada classe:')

print(dados.groupby(['preco']).mean().T)

#-------------------------------------------------------------------------------
# Plotar diagrama de dispersão por classe
#-------------------------------------------------------------------------------
  
atributo1 = 'area_util'
atributo2 = 'area_extra'

cores = [ 'red' if x else 'blue' for x in dados['preco'] ]

grafico = dados.plot.scatter(
    atributo1,
    atributo2,
    c      = cores,
    s      = 10,
    marker = 'o',
    alpha  = 0.5,
    figsize = (14,14)
    )

#plt.show()

#-------------------------------------------------------------------------------
# Selecionar os atributos que serão utilizados pelo classificador
#-------------------------------------------------------------------------------
atributos_selecionados = [
    'Id', 
    'tipo_vendedor', 
    'quartos', 
    'suites', 
    'vagas', 
    'area_util',
    'area_extra', 
    #'churrasqueira', 
    #'estacionamento', 
    #'piscina',
    #'playground', 
    #'quadra', 
    #'s_festas', 
    #'s_jogos', 
    #'s_ginastica', 
    #'sauna',
    'vista_mar',  
    'tipo_Apartamento', 
    'tipo_Casa', 
    'tipo_Loft',
    #'tipo_Quitinete',
    #'preco'
    ]

resposta = resposta[atributos_selecionados]
dados = dados[atributos_selecionados+['preco']]

#-------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
#-------------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1,random_state=12345)

#-------------------------------------------------------------------------------
# Criar os arrays X e Y separando atributos e alvo
#-------------------------------------------------------------------------------

x = dados_embaralhados.loc[:,dados_embaralhados.columns!='preco'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns=='preco'].values

#-------------------------------------------------------------------------------
# Separar X e Y em conjunto de treino e conjunto de teste
#-------------------------------------------------------------------------------   

q = 2342  # qtde de amostras selecionadas para treinamento

# conjunto de treino

x_treino = x[:q,:]
y_treino = y[:q].ravel()
# conjunto de teste

x_teste = x[q:,:]
y_teste = y[q:].ravel()

#------------------------------------------------------------------------------
# Ajustar a escala dos atributos
#------------------------------------------------------------------------------

escala = StandardScaler()

escala.fit(x_treino)

x_treino = escala.transform(x_treino)
x_teste  = escala.transform(x_teste)
x_resposta = escala.transform(resposta)

#------------------------------------------------------------------------------
# Treinar um regressor linear
#------------------------------------------------------------------------------

regressor_linear = LinearRegression()

regressor_linear = regressor_linear.fit(x_treino,y_treino)

#------------------------------------------------------------------------------
# Obter as respostas do regressor linear dentro e fora da amostra
#------------------------------------------------------------------------------

y_resposta_treino = regressor_linear.predict(x_treino)
y_resposta_teste  = regressor_linear.predict(x_teste)

#------------------------------------------------------------------------------
# Calcular as métricas e comparar os resultados
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR LINEAR:')
print(' ')

print(' Métrica  DENTRO da amostra  FORA da amostra')
print(' -------  -----------------  ---------------')

mse_in  = mean_squared_error(y_treino,y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(y_treino,y_resposta_treino)

mse_out  = mean_squared_error(y_teste,y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(y_teste,y_resposta_teste)

print(' %7s  %17.4f  %15.4f' % (  'mse' ,  mse_in ,  mse_out ) )
print(' %7s  %17.4f  %15.4f' % ( 'rmse' , rmse_in , rmse_out ) )
print(' %7s  %17.4f  %15.4f' % (   'r2' ,   r2_in ,   r2_out ) )

#------------------------------------------------------------------------------
# Plotar diagrama de dispersão entre a resposta correta e a resposta do modelo
#------------------------------------------------------------------------------

#plt.scatter(x=y_teste,y=y_resposta_teste)

#------------------------------------------------------------------------------
# Treinar e testar um regressor KNN para vários valores do parâmetros
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR KNN:')
print(' ')

print('  K   DENTRO da amostra  FORA da amostra')
print(' ---  -----------------  ---------------')

for k in range(1,21):
    
    regressor_knn = KNeighborsRegressor(
        n_neighbors = k,
        weights     = 'distance'     # 'uniform' ou 'distance'
        )

    regressor_knn = regressor_knn.fit(x_treino,y_treino)

    y_resposta_treino = regressor_knn.predict(x_treino)
    y_resposta_teste  = regressor_knn.predict(x_teste)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)
    print(' %3d  %17.4f  %15.4f' % ( k , rmse_in , rmse_out ) )

resposta = regressor_knn.predict(x_resposta)
id = []
for j in range(len(resposta)):
    id += [str(j)]
data = {'id':id,'preco':resposta} 
df = pd.DataFrame(data).to_csv('preco_knn.csv',index=False)

#------------------------------------------------------------------------------
# Treinar e testar um regressor POLINOMIAL para graus de 1 a 5
#------------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K:')
print(' ')

print('  K   DENTRO da amostra  FORA da amostra')
print(' ---  -----------------  ---------------')

for k in range(1,6):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    x_resposta_poly = pf.transform(x_resposta)

    regressor_linear = LinearRegression()
    
    regressor_linear = regressor_linear.fit(x_treino_poly,y_treino)
    
    y_resposta_treino = regressor_linear.predict(x_treino_poly)
    y_resposta_teste  = regressor_linear.predict(x_teste_poly)

    mse_in  = mean_squared_error(y_treino,y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in   = r2_score(y_treino,y_resposta_treino)

    mse_out  = mean_squared_error(y_teste,y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out   = r2_score(y_teste,y_resposta_teste)

    print(' %3d  %17.4f  %15.4f' % ( k , rmse_in , rmse_out ) )

    resposta = regressor_linear.predict(x_resposta_poly)