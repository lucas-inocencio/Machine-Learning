#===============================================================================
#
#  EXPERIMENTO 03 - CLASSIFICADOR KNN PARA O CONJUNT ORANGE
#
#    Vamos construir um modelo preditivo de "churn" para uma empresa
#    de telecomunicações.
#
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from matplotlib import pyplot as plt


#-------------------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto IRIS
#-------------------------------------------------------------------------------

dados = pd.read_csv('E:\Estudos\Graduacao\Optativas\EEL891 - Introducao ao Aprendizado de Maquina\Trabalho\Classificacao\\conjunto_de_treinamento.csv')
resposta = pd.read_csv('E:\Estudos\Graduacao\Optativas\EEL891 - Introducao ao Aprendizado de Maquina\Trabalho\Classificacao\\conjunto_de_teste.csv')

#-------------------------------------------------------------------------------
# Explorar os dados
#-------------------------------------------------------------------------------

print ( '\nImprimir o conjunto de dados:\n')

print(dados)

print ( '\nImprimir o conjunto de dados transposto')
print ('para visualizar os nomes de todas as colunas:\n')

print(dados.T)

print ( '\nImprimir os tipos de cada variável:\n')

print(dados.dtypes)

print ( '\nIdentificar as variáveis categóricas:\n')

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
  
# forma_envio_solicitacao           --> nao-ordinal com 3 categorias  --> one-hot encoding
# sexo                              --> nao-ordinal com 4 categorias  --> one-hot encoding
# estado_onde_nasceu                --> nao-ordinal com 28 categorias --> descartar
# estado_onde_reside                --> nao-ordinal com 27 categorias --> descartar
# possui_telefone_residencial       --> binaria --> binarizar (mapear para 0/1)      
# codigo_area_telefone_residencial   --> nao-ordinal com 81 categorias --> descartar
# possui_telefone_celular           --> binaria --> descartar
# vinculo_formal_com_empresa        --> binaria --> binarizar (mapear para 0/1)
# estado_onde_trabalha              --> nao-ordinal com 28 categorias --> descartar
# possui_telefone_trabalho          --> binaria --> binarizar (mapear para 0/1)
# codigo_area_telefone_trabalho     --> nao-ordinal com 77 categorias --> descartar

print ( '\nDescartar as variáveis de cardinalidade muito alta:\n')

print (dados.T)
dados = dados.drop(['estado_onde_nasceu','estado_onde_reside','codigo_area_telefone_residencial','possui_telefone_celular','estado_onde_trabalha',
                    'codigo_area_telefone_trabalho','qtde_contas_bancarias_especiais','grau_instrucao','meses_no_trabalho'],axis=1)
print (dados.T)

print ( '\nAplicar one-hot encoding nas variáveis que tenham')
print ( '3 ou mais categorias:')

dados['sexo'] = dados['sexo'].replace(r'^\s*$', np.NaN, regex=True)
resposta['sexo'] = resposta['sexo'].replace(r'^\s*$', np.NaN, regex=True)
dados['profissao_companheiro'] = dados['profissao_companheiro'].replace(r'^\s*$', np.NaN, regex=True)
resposta['profissao_companheiro'] = resposta['profissao_companheiro'].replace(r'^\s*$', np.NaN, regex=True)
dados['grau_instrucao_companheiro'] = dados['grau_instrucao_companheiro'].replace(r'^\s*$', np.NaN, regex=True)
resposta['grau_instrucao_companheiro'] = resposta['grau_instrucao_companheiro'].replace(r'^\s*$', np.NaN, regex=True)
dados['sexo'] = dados['sexo'].fillna("N")
resposta['sexo'] = resposta['sexo'].fillna("N")
dados['profissao_companheiro'] = dados['profissao_companheiro'].fillna(0)
dados['grau_instrucao_companheiro'] = dados['grau_instrucao_companheiro'].fillna(0)
resposta['profissao_companheiro'] = resposta['profissao_companheiro'].fillna(0)
resposta['grau_instrucao_companheiro'] = resposta['grau_instrucao_companheiro'].fillna(0)

dados = pd.get_dummies(dados,columns=['sexo','forma_envio_solicitacao'])
resposta = pd.get_dummies(resposta,columns=['sexo','forma_envio_solicitacao'])

print (dados.head(5).T)

print ( '\nAplicar binarização simples nas variáveis que tenham')
print ( 'apenas 2 categorias:\n')

binarizador = LabelBinarizer()
for v in ['possui_telefone_residencial','vinculo_formal_com_empresa','possui_telefone_trabalho']:
    dados[v] = binarizador.fit_transform(dados[v])
    resposta[v] = binarizador.fit_transform(resposta[v])
print (dados.head(5).T)

print ( '\nVerificar a quantidade de amostras de cada classe:\n')

print(dados['inadimplente'].value_counts())

print ( '\nVerificar o valor médio de cada atributo em cada classe:')

print(dados.groupby(['inadimplente']).mean().T)

#-------------------------------------------------------------------------------
# Plotar diagrama de dispersão por classe
#-------------------------------------------------------------------------------
  
atributo1 = 'grau_instrucao_companheiro'
atributo2 = 'renda_extra'

cores = [ 'red' if x else 'blue' for x in dados['inadimplente'] ]

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
    #'id_solicitante', 
    #'produto_solicitado', 
    'dia_vencimento',
    #'tipo_endereco', 
    'idade', 
    #'estado_civil', 
    #'qtde_dependentes',
    #'nacionalidade', 
    #'possui_telefone_residencial', 
    #'tipo_residencia',
    #'meses_na_residencia', 
    #'possui_email', 
    #'renda_mensal_regular',
    'renda_extra', 
    #'possui_cartao_visa', 
    'possui_cartao_mastercard',
    'possui_cartao_diners', 
    #'possui_cartao_amex', 
    'possui_outros_cartoes',
    #'qtde_contas_bancarias', 
    #'valor_patrimonio_pessoal', 
    #'possui_carro',
    #'vinculo_formal_com_empresa', 
    #'possui_telefone_trabalho', 
    #'profissao',
    #'ocupacao', 
    'profissao_companheiro', 
    'grau_instrucao_companheiro',
    #'local_onde_reside', 
    #'local_onde_trabalha',
    #'sexo_F', 
    #'sexo_M', 
    #'sexo_N', 
    'forma_envio_solicitacao_correio',
    #'forma_envio_solicitacao_internet',
    #'forma_envio_solicitacao_presencial'
    #'inadimplente'
    ]

resposta = resposta[atributos_selecionados]
dados = dados[atributos_selecionados+['inadimplente']]

#-------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
#-------------------------------------------------------------------------------

#dados_embaralhados = dados.sample(frac=1,random_state=12345)

#-------------------------------------------------------------------------------
# Criar os arrays X e Y separando atributos e alvo
#-------------------------------------------------------------------------------

x = dados.loc[:,dados.columns!='inadimplente'].values
y = dados.loc[:,dados.columns=='inadimplente'].values

#-------------------------------------------------------------------------------
# Separar X e Y em conjunto de treino e conjunto de teste
#-------------------------------------------------------------------------------   

q = 10000  # qtde de amostras selecionadas para treinamento

# conjunto de treino

x_treino = x[:q,:]
y_treino = y[:q].ravel()
# conjunto de teste

x_teste = x[q:,:]
y_teste = y[q:].ravel()

#-------------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
#-------------------------------------------------------------------------------

ajustador_de_escala = MinMaxScaler()
ajustador_de_escala.fit(x_treino)

x_treino = ajustador_de_escala.transform(x_treino)
x_teste  = ajustador_de_escala.transform(x_teste)
x_resposta = ajustador_de_escala.transform(resposta)

#-------------------------------------------------------------------------------
# Treinar um classificador KNN com o conjunto de treino
#-------------------------------------------------------------------------------

classificador = KNeighborsClassifier(n_neighbors=5)

classificador = classificador.fit(x_treino,y_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no mesmo conjunto onde foi treinado
#-------------------------------------------------------------------------------

y_resposta_treino = classificador.predict(x_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no conjunto de teste
#-------------------------------------------------------------------------------

y_resposta_teste = classificador.predict(x_teste)

resposta = classificador.predict(x_resposta)
id = []
for i in range(1,len(resposta)+1):
    id += [str(20000+i)]
data = {'id_solicitante':id,'inadimplente':resposta} 
df = pd.DataFrame(data).to_csv('inadimplente.csv',index=False)

#-------------------------------------------------------------------------------
# Verificar a acurácia do classificador
#-------------------------------------------------------------------------------

print ("\nDESEMPENHO DENTRO DA AMOSTRA DE TREINO\n")

total   = len(y_treino)
acertos = sum(y_resposta_treino==y_treino)
erros   = sum(y_resposta_treino!=y_treino)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

print ("\nDESEMPENHO FORA DA AMOSTRA DE TREINO\n")

total   = len(y_teste)
acertos = sum(y_resposta_teste==y_teste)
erros   = sum(y_resposta_teste!=y_teste)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

#-------------------------------------------------------------------------------
# Verificar a variação da acurácia com o número de vizinhos
#-------------------------------------------------------------------------------

print ( "\n  K TREINO  TESTE")
print ( " -- ------ ------")

for k in range(1,50):

    classificador = KNeighborsClassifier(
        n_neighbors = k,
        weights     = 'uniform',
        p           = 1
        )
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)
    
    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)
    
    print(
        "%3d"%k,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
    )