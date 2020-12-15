####### Interface #######
### módulo de interface visual (tem seu fechamento root.mainloop() em botao_enviar.py, juntamente com o botão de envio)

###imports

import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

from modulos.variaveis_fixas import *

# Base do Chat

root = Tk()
root.title("Redes Neurais Artificiais Simuladoras de Conversa e Emoção (ReNASCE)")
root.geometry("775x500")
root.resizable(width=FALSE, height=FALSE)
root.configure(background="black")

# Fotos

img = ImageTk.PhotoImage(Image.open(neutro))
foto = tk.Label(root, image = img, borderwidth=0, highlightthickness = 0)
foto.configure(bg='black')
foto.pack(side = "bottom", fill = "both", expand = "yes")

# adicionar ".convert('RGB')" assim: "ImageTk.PhotoImage(Image.open(x).convert('RGB'))" e remover "foto.configure(bg="black")
# isso adiciona o canal alpha (transparência) ao fundo das imagens PNG.
# só que não fica tão bom. a imagem diminui bastante de qualidade e as bordas ficam pixeladas

### Janela do chat

caixa_chat = Text(root, bg="black")
caixa_chat.config(state=DISABLED)
caixa_chat.pack()

### Logo

img2 = ImageTk.PhotoImage(Image.open(logo))
logo = Label(root, image = img2)
logo.config(bg='black')
logo.pack()

### barra de rolagem
      
scrollbar = Scrollbar(root, command=caixa_chat.yview, cursor="heart")
caixa_chat['yscrollcommand'] = scrollbar.set

### Caixa para digitar mensagem

digitar_conversa = Text(root, bd=0, bg="white",width="29", height="5", font=("Lakhin",11), fg="black")

### Organização dos componentes

scrollbar.place(x=736,y=6, height=386)
caixa_chat.place(x=301,y=6, height=386, width=440)
digitar_conversa.place(x=301, y=401, height=90, width=335)
foto.place(x=0, y=120)
logo.place(x=0,y=0)
