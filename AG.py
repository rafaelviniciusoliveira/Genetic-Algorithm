import random
import numpy as np
import math

# --------------------ALGORITMO GENÉTICO-----------------------

def gera_individuos(tamanho):
  X = np.array([0]*tamanho,dtype=float)
  Y = np.array([0]*tamanho,dtype=float)
  Z = np.array([0]*tamanho,dtype=float)
  for i in range(0,tamanho):
    X[i] = round(random.uniform(-5.12,5.12),3)
    Y[i] = round(random.uniform(-5.12,5.12),3)
    Z[i] = round(random.uniform(-5.12,5.12),3)
  return np.array([X ,Y, Z]).T

def rastrigin(chromosomes,populacao,dimensao):
    fitness = np.array([0]*populacao,dtype=float)
    fitness[0] = 10*dimensao
    for k in range (populacao):
      for i in range (dimensao):
          fitness[k] += chromosomes[k][i]**2 - (10*math.cos(2*math.pi*chromosomes[k][i]))
    return fitness

def probabilidade(resultado):
  resultado_soma = sum(abs(resultado))
  probabilidade = np.array([0]*len(resultado),dtype=float)
  for i in range(len(resultado)):
    probabilidade[i]=resultado_soma/(abs(resultado[i]))
  return np.array(probabilidade/sum(probabilidade))

def metodo_selecao_roleta(resultado):
  probabilidade1 = probabilidade(resultado)
  fatias = np.cumsum(probabilidade1)
  indices_pais = []
  n_giro_roleta = 0
  while n_giro_roleta < len(resultado):
    idx = (np.abs(fatias - random.uniform(0, 1))).argmin()
    indices_pais.append(idx)
    n_giro_roleta = n_giro_roleta + 1
  return np.array(indices_pais).T

def metodo_selecao_torneio(resultado):
  disputa = np.array([0]*3,dtype=int)
  vencedor = np.array([0]*len(resultado),dtype=int) 
  probabilidade1 = probabilidade(resultado)
  indices_pais = []
  for i in range(len(resultado)):
    for k in range(3):
      disputa[k] = random.uniform(0,len(resultado))

    if (probabilidade1[disputa[0]]>=probabilidade1[disputa[1]]) and (probabilidade1[disputa[0]]>=probabilidade1[disputa[2]]):
      vencedor[i] = disputa[0]
    if (probabilidade1[disputa[1]]>=probabilidade1[disputa[0]]) and (probabilidade1[disputa[1]]>=probabilidade1[disputa[2]]):
      vencedor[i] = disputa[1]
    if (probabilidade1[disputa[2]]>=probabilidade1[disputa[0]]) and (probabilidade1[disputa[2]]>=probabilidade1[disputa[1]]):
      vencedor[i] = disputa[2]

  return vencedor

def cruzamento_um_ponto(chromosomes,selecao):
  pais = np.array([0]*2,dtype=int)
  filhos = []
  for i in range(len(selecao)//2):
    pais[0] = random.uniform(0,len(selecao))
    pais[1] = random.uniform(0,len(selecao))
    filhos.append([chromosomes[pais[0]][0],chromosomes[pais[0]][1],chromosomes[pais[1]][2]])
    filhos.append([chromosomes[pais[1]][0],chromosomes[pais[1]][1],chromosomes[pais[0]][2]])
  return filhos

def cruzamento_multi_ponto(chromosomes,selecao):
  pais = np.array([0]*2,dtype=int)
  filhos = []
  for i in range(len(selecao)//2):
    pais[0] = random.uniform(0,len(selecao))
    pais[1] = random.uniform(0,len(selecao))
    filhos.append([chromosomes[pais[0]][0],chromosomes[pais[1]][1],chromosomes[pais[0]][2]])
    filhos.append([chromosomes[pais[1]][0],chromosomes[pais[0]][1],chromosomes[pais[1]][2]])
  return filhos

def mutacao(descendentes,taxa_Mutacao):
  qnt_mutacao = int(len(descendentes)*taxa_Mutacao)
  mutantes = np.array([0]*len(descendentes),dtype=int)
 
  for i in range(qnt_mutacao):
    mutantes[i] = random.uniform(0,len(descendentes))
    mutantes_caracteristica = random.randint(0,2)
    descendentes[mutantes[i]][mutantes_caracteristica] = round(random.uniform(-5.12,5.12),3)
  return np.array(descendentes)

def Elitismo(chromosomes,resultado,taxa_Elitismo):
  nova_geracao = []
  populacao = []
  probabilidade1 = probabilidade(resultado)
  qnt_elite = int(len(chromosomes)*taxa_Elitismo)
  elite = np.array([0]*qnt_elite,dtype=int)
  
  ordenado_probabilidade = np.argsort(probabilidade1)
  elite = ordenado_probabilidade[len(chromosomes)-qnt_elite:len(chromosomes)]
  aux_polulacao = ordenado_probabilidade[0:len(chromosomes)-qnt_elite]

  nova_geracao = chromosomes[elite]
  populacao = chromosomes[aux_polulacao]

  return nova_geracao,populacao

def GA(tamanho_populacao,taxa_cruzamento,Elitismo_,taxa_mutacao,funcao_selecao,funcao_cruzamento,target,geracoes):
  caracteristicas = 3
  tamanho_populacao = int(tamanho_populacao*taxa_cruzamento)
  chromosomes = gera_individuos(tamanho_populacao)
  cont_geracoes = 0
  resultado_final = 100
  while (cont_geracoes < geracoes and resultado_final > target):
    resultado = rastrigin(chromosomes,len(chromosomes),caracteristicas)

    if Elitismo_: 
      nova_geracao,populacao = Elitismo(chromosomes,resultado,0.01)
      resultado = rastrigin(populacao,len(populacao),caracteristicas)

    if funcao_selecao == 'roleta':
      selecao = metodo_selecao_roleta(resultado)
    elif funcao_selecao == 'torneio': 
      selecao = metodo_selecao_torneio(resultado)

    if funcao_cruzamento == 'um_ponto':
      if Elitismo_:
        descendentes = cruzamento_um_ponto(populacao,selecao)
      else: 
        descendentes = cruzamento_um_ponto(chromosomes,selecao)
    elif funcao_cruzamento == 'multi_ponto':
      if Elitismo_:
        descendentes = cruzamento_multi_ponto(populacao,selecao)
      else: 
        descendentes = cruzamento_multi_ponto(chromosomes,selecao) 

    mutantes = mutacao(descendentes,taxa_mutacao)

    if Elitismo_:
      chromosomes = np.concatenate([nova_geracao,mutantes])
    else: 
      chromosomes = mutantes 
       
    cont_geracoes = cont_geracoes + 1
    resultado_final = round(min(abs(rastrigin(chromosomes,len(chromosomes),caracteristicas))),3)

  return resultado_final, cont_geracoes

tamanho_populacao = 1000
taxa_cruzamento = 0.8
taxa_Elitismo = 0.01
Elitismo_= True
funcao_selecao = 'roleta'
funcao_cruzamento = 'multi_ponto'
taxa_mutacao = 0.01
geracoes = 1000

valor_minino,interacao = GA(tamanho_populacao,taxa_cruzamento,Elitismo_,taxa_mutacao,funcao_selecao,funcao_cruzamento,0.001,geracoes)

print('-------GA-------')
print("Valor minino:",valor_minino,"\n Iterações:",interacao,"\n")
