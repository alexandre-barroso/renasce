####### Módulo de Respostas 2 ########
### funções relacionadas à elaboração das respostas do simulador
### voltado para responder falas (não-perguntas)

### esconder erros
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

### imports

from random import choice

import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

from modulos.interface import *
from modulos.funcoes import *
from modulos.agregador_de_falas import *

#mensagens que só contenham esses caracteres são filtradas (não respondidas)
#exemplo: usuário só digita "." e dá enter.
#OBS: EXPANDIR ASSIM QUE POSSÍVEL
msg_vazias = [' ', '', '.', ',', '!', '?', '-', '=', '&']

possibilidade_pontuacoes = [',', '.', '!']

nao_letras = [',', '.', '!', '?','-']

def mensagem_usuario():    
    msg = digitar_conversa.get("1.0",'end-1c').strip()

    if msg not in msg_vazias:
        caixa_chat.config(state=NORMAL)
        caixa_chat.insert(END, '• ' + nome + ": \n" + msg + '\n\n')
        caixa_chat.config(foreground="white", font=("Lakhin", 12 ))
        caixa_chat.config(state=DISABLED)
        caixa_chat.yview(END)

def mensagem_renasce():
    msg = digitar_conversa.get("1.0",'end-1c').strip()
    
    #limpa a caixa de digitação dps que envia
    digitar_conversa.delete("0.0",END)
    
    #final da frase, combinado aleatoriamente
    final_frase_1 = ", " + nome + random.choice([p for p in possibilidade_pontuacoes if p != ',']) + '\n'
    final_frase_2 = '.\n'
    final_frase_3 = [final_frase_1, final_frase_2]
    final_da_frase = random.choice(final_frase_3)

    if msg not in msg_vazias:
        
        #passa o state de DISABLED para NORMAL, permitindo edições
        caixa_chat.config(state=NORMAL)
        
        #define cor da letra (foreground) e fonte da letra
        caixa_chat.config(foreground="white", font=("Lakhin", 12 ))
    
        #prever 'prev', elemento que vai ser usado para prever varios outros
        prev = predizer_classe(msg)
        
        #prever e atualizar imagem 
        emc = emocao_apresentada(prev, banco_dados)
        atualizar_imagem(emc)
        
        #gerar pontuacao do fim aleatoriamente
        pontuacao = random.choice(possibilidade_pontuacoes)
        
        #resposta pré-definida
        res = resposta_renasce(msg)
        
        ##nome do renasce antes
        caixa_chat.insert(END, '• ' + nome_renasce +": \n")

        caixa_chat.insert(END, res)
	
        #pontuando e adicionando nome, se necessário
        caixa_chat.insert(END, final_da_frase) 
                          
   
    #coloca dois paragrafos pra separar msgs
    caixa_chat.insert(END, '\n\n')
    
    #passa o state de NORMAL para DISABLED, impedindo edições
    caixa_chat.config(state=DISABLED)
    
    #rolar chat até o fim (para poder ver a mensagem totalmente)
    caixa_chat.yview(END)
    return
