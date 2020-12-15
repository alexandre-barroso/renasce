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
banco_dados = json.loads(open('dados/banco_de_dados.json').read())
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
                    print ("encontrado(s): %s" % word)
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
        lista_retorno.append({"banco_dados": classes[r[0]], "probabilidade": str(r[1])})
    return lista_retorno 
    
def emocao_apresentada(prev, banco_dados):
    emocao = prev[0]['banco_dados']
    lista_dados = banco_dados['matriz_emocoes']
    for i in lista_dados:
        if(i['emocao'] == emocao):
            sentimento = (i['emocao'])
            break
    return sentimento

def atualizar_imagem(emocao):
    img = ImageTk.PhotoImage(Image.open(globals()[emocao]))
    foto.configure(image=img)
    foto.image = img
