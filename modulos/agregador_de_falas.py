####### Módulo de Respostas ########
### funções relacionadas à elaboração das respostas do simulador

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
from modulos.modulo_respostas import *

def enviar():
    msg = digitar_conversa.get("1.0",'end-1c').strip()
    
    if msg == '':
    	return mensagem_usuario(), mensagem_renasce()  
    else:
    	return mensagem_usuario(), mensagem_renasce()

def enter(a):
	return enviar() 

###
