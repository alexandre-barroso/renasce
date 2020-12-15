####### Funções #######

###imports

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
modelo = load_model('modelos/modelo_classificatorio.h5')
import json
import random
intents = json.loads(open('dados/banco_de_dados.json').read())
palavras = pickle.load(open('modelos/palavras.pkl','rb'))
classes = pickle.load(open('modelos/classes.pkl','rb'))

from modulos.interface import *
from redes_neurais.rede_seq2seq import *

### módulo da rede neural lstm gerativa (não funciona ainda):
#from gerador_lstm import *


### Funções

def limpar_frase(frase):

    # tokenizando as palavras (frases -> palavras/tokens)
    palavras_frase = nltk.word_tokenize(frase)

    # achando as raízes das palavras
    palavras_frase = [lemmatizer.lemmatize(word.lower()) for word in palavras_frase]
    return palavras_frase

def conjunto_palavras(frase, palavras, show_details=True):

    # tokenizando padrões
    palavras_frase = limpar_frase(frase)

    # matriz de vocábulos
    bag = [0]*len(palavras)  
    for s in palavras_frase:
        for i,word in enumerate(palavras):
            if word == s: 

                # assinala no. 1 se palavra estiver dentro do vocabulário
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))


def predizer_classe(frase):

    # filtra probabilidades abaixo do mínimo
    p = conjunto_palavras(frase, palavras,show_details=False)
    res = modelo.predict(np.array([p]))[0]
    certeza_minima = 0.25
    resultados = [[i,r] for i,r in enumerate(res) if r>certeza_minima]

    # rankeando força probabílistica
    resultados.sort(key=lambda x: x[1], reverse=True)
    lista_retorno = []
    for r in resultados:
        lista_retorno.append({"intent": classes[r[0]], "probability": str(r[1])})
    return lista_retorno

### modelo ALPHA de resposta
def responder(ints, intents_json):
    emocao = ints[0]['intent']
    lista_intents = intents_json['matriz_emocoes']
    for i in lista_intents:
        if(i['emocao']== emocao):
            resposta = random.choice(i['respostas'])
            break
    return resposta   
    
def emocao_apresentada(ints, intents_json):
    emocao = ints[0]['intent']
    lista_intents = intents_json['matriz_emocoes']
    for i in lista_intents:
        if(i['emocao']== emocao):
            sentimento = (i['emocao'])
            break
    return sentimento

def atualizar_imagem(emocao):
    img = ImageTk.PhotoImage(Image.open(globals()[emocao]))
    foto.configure(image=img)
    foto.image = img

def responder_pergunta(ints, intents_json):
    pergunta = intents_json['matriz_respostas_para_perguntas']
    for i in pergunta:
      resposta = random.choice(i['respostas'])
    return resposta

### processador resposta gerada

def processador(res_g):

    resposta_formatada = res_g.replace('\n','. ').replace(';','.').replace('"','').replace(';','.').replace('—','').replace(':',' ').replace('  ',' ')
    
    resposta_processada = []
    resposta_processada[:] = resposta_formatada
    
    pontuacoes = ['.',',']
    
    if resposta_processada[-1] in pontuacoes:
        del resposta_processada[-1]
        
    if resposta_processada[0] in pontuacoes:
        del resposta_processada[0]
            
    conector = ''
    resposta_pronta = conector.join(resposta_processada)
    
    return str(resposta_pronta)
  

