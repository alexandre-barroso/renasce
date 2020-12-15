### esconder erros
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print('\nInicializando rede_classificatoria...\n')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lematizador = WordNetLemmatizer()
import json
import pickle

palavras = []
classes = []
documentos = []
ignorar_caracteres = ['!', '?', ',', '.']
banco_de_dados = open('dados/banco_de_dados.json').read()
intents = json.loads(banco_de_dados)

print('\n>>> Inicializando corpus...\n')

for intent in intents['matriz_emocoes']:
    for padrao in intent['padroes']:
        
        #tokenizar cada palavra
        palavra = nltk.word_tokenize(padrao)
        palavras.extend(palavra)
        
        #adicionar cada documento ao corpus
        documentos.append((palavra, intent['emocao']))
        
        #e adicionar à lista de classes
        if intent['emocao'] not in classes:
            classes.append(intent['emocao'])

print(documentos)

#lematizar e tirar capitalização de todas as palavras, e aí remover palavras duplicadas
palavras = [lematizador.lemmatize(p.lower()) for p in palavras if p not in ignorar_caracteres]
palavras = sorted(list(set(palavras)))

#ajeita (sort) elemento da lista 'classes'
classes = sorted(list(set(classes)))

#documentos = combinação entre padrões de mensagem e intents
print (len(documentos), "documentos")

#dclasses = intents
print (len(classes), "classes", classes)

#palavras = todas as palavras, vocabulário geral
print (len(palavras), "palavras lematizadas únicas", palavras)

pickle.dump(palavras,open('modelos/palavras.pkl','wb'))
pickle.dump(classes,open('modelos/classes.pkl','wb'))

#criar dados para treinamento
treinamento = []

#criar array vazio para output
output_vazio = [0] * len(classes)

#treinar dados, saco-de-palavras (bag of words, aka. bow) para cada frase
for doc in documentos:
    
    # criar bow
    saco = []
    
    #lista de palavras tokenizadas para os padrões
    palavras_padrao = doc[0]
    
    #lematizar cada palavras - criar palavras-base que vão representar palavras relacionadas
    palavras_padrao = [lematizador.lemmatize(palavra.lower()) for palavra in palavras_padrao]
    
    #criar array bow com 1, se achar match da palavra no padrão atual
    for palavra in palavras:
        saco.append(1) if palavra in palavras_padrao else saco.append(0)
        
    #output é 0 para cada tag e é 1 para tag atual (para cada padrão)
    linha_output = list(output_vazio)
    linha_output[classes.index(doc[1])] = 1
    
    treinamento.append([saco, linha_output])

#embaralhar tudo e transformar em np.array
random.shuffle(treinamento)
treinamento = np.array(treinamento)

#criar, treinar e testar listas. X = padrões, Y = intents
treinar_x = list(treinamento[:,0])
treinar_y = list(treinamento[:,1])
print("\nDados para treino criados\n")

print('\n>>> Inicializando modelo_classificatorio...\n')

# Modelo - 3 camadas
#Camada 1 - 128 neuronios
#Camada 2 - 64 neuronios
#Camada 3 - no. de neuronios = no. de intents para ser capaz de predizer cada output de intent para a funcao softmax

modelo_sequencial = Sequential()
modelo_sequencial.add(Dense(128, input_shape=(len(treinar_x[0]),), activation='relu'))
modelo_sequencial.add(Dropout(0.5))
modelo_sequencial.add(Dense(64, activation='relu'))
modelo_sequencial.add(Dropout(0.5))
modelo_sequencial.add(Dense(len(treinar_y[0]), activation='softmax'))


print('>>> DIGITE: \n\n 1. "c" para continuar treino de onde parou. \n\n 2. "r" para re-treinar do começo. \n\n 3. qualquer outra coisa para encerrar o programa. \n\n OBSERVAÇÃO: digite "c" ou "r" sem aspas!\n')

opcao = input('> ')

epoch = input('\n> Quantas iterações (epochs) a rede neural deve rodar? (mínimo 200)\n>  ')
epoch = int(epoch)

batch_size = input('\n> Qual o batch_size? (recomendado 5)\n>  ')
batch_size = int(batch_size)
print('')

if opcao.lower() == 'r':

    #Compilar modelo 
    #Stochastic gradient descent com Nesterov accelerated gradient é recomendado para esse modelo
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    modelo_sequencial.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    hist = modelo_sequencial.fit(np.array(treinar_x), np.array(treinar_y), epochs=epoch, batch_size=batch_size, verbose=1)
    modelo_sequencial.save('modelos/modelo_classificatorio.h5', hist)

    print('\n>>> Finalizando treino da rede neural...\n')

    
elif opcao.lower() == 'c':

    del modelo_sequencial
    modelo_sequencial = load_model('modelos/modelo_classificatorio.h5')
    hist = modelo_sequencial.fit(np.array(treinar_x), np.array(treinar_y), epochs=epoch, batch_size=batch_size, verbose=1)
    modelo_sequencial.save('modelos/modelo_classificatorio.h5', hist)

    print('\n>>> Finalizando treino da rede neural...\n')
    
else:

   print('\nEncerrando programa...\n') 
