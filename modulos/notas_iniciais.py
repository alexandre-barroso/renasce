####### Notas iniciais #######
### módulo de interface visual para notas da versão, de atualização, da edição, do funcionamento, etc

###imports

import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

from modulos.variaveis_fixas import *

notas_iniciais = Tk()
notas_iniciais.title("Introdução e instruções")
notas_iniciais.geometry("500x400")
notas_iniciais.resizable(width=FALSE, height=FALSE)
notas_iniciais.configure(background="gray")	
notas_iniciais.attributes("-topmost", True)

notas = Text(notas_iniciais, font=("Verdana",10),height=500, width=400, bg="white", fg="black")
notas.insert(END,'Notas e observações:\n\nINSTRUÇÕES E COMENTÁRIOS AQUI!')
notas.config(state=DISABLED)
notas.pack()

b2 = Button(notas_iniciais, text = "Fechar", height=1,
            command = notas_iniciais.destroy)
b2.place(x=400,y=363) 

