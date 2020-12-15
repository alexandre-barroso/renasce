####### Botão 'Enviar' #######
### finalização da interface gráfica do tkinter, levando em conta todas as funções necessárias
### criação do botão de envio, seu lugar na GUI e fechamento "root.mainloop()" do tkinter

### esconder erros
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

### imports

import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

from modulos.interface import *
from modulos.funcoes import *
from modulos.agregador_de_falas import *

### Botão de Enviar

botao_enviar = Button(root, font=("Alice",14,'bold'), text="Enviar", width=4, height=9,
                    bd=0, bg="red", activebackground="red3",fg='#000000',
                    command= enviar)

### Fechar o "mainloop" do tkinter

digitar_conversa.bind("<Return>", enter)
botao_enviar.place(x=660, y=401, height=90, width=90)

from modulos.notas_iniciais import *

root.mainloop()


