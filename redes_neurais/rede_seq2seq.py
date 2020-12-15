# créditos pela base/tutorial desse código: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
# (salvo como pdf na pasta de referencias)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

dispositivo = torch.device("cpu")
corpus = os.path.join('dados')
dados = os.path.join(corpus, "amostra_seq2seq.txt")

# padrões de tokens das palavras, usando nomenclatura comum de aprendizagem de maquina:
token_PAD = 0  # usado para frases curtas (Padding (aka, preenchimento): PAD)
token_SOS = 1  # usado para indicar começo de frase (Start-Of-Sentence: SOS)
token_EOS = 2  # usado para indicar fim de frase (End-Of-Sentence: EOS)

class Voc:
    def __init__(self, nome):
        self.nome = nome
        self.aparado = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {token_PAD: "PAD", token_SOS: "SOS", token_EOS: "EOS"}
        
        # contar SOS, EOS, PAD
        self.numero_palavras = 3

    def adicionarFrase(self, frase):
        for palavra in frase.split(' '):
            self.adicionarPalavra(palavra)

    def adicionarPalavra(self, palavra):
        if palavra not in self.word2index:
            self.word2index[palavra] = self.numero_palavras
            self.word2count[palavra] = 1
            self.index2word[self.numero_palavras] = palavra
            self.numero_palavras += 1
        else:
            self.word2count[palavra] += 1

    # remover palavras abaixo da contagem_minima, raras demais para fazerem diferença
    def aparar(self, contagem_minima):
        if self.aparado:
            return
        self.aparado = True

        manter_palavras = []

        for k, v in self.word2count.items():
            if v >= contagem_minima:
                manter_palavras.append(k)

        # reinicializar dicionarios
        self.word2index = {}
        self.word2count = {}
        self.index2word = {token_PAD: "PAD", token_SOS: "SOS", token_EOS: "EOS"}
        
        # contar tokens
        self.numero_palavras = 3

        for palavra in manter_palavras:
            self.adicionarPalavra(palavra)
            
# comprimento máximo da frase a ser considerado           
TAMANHO_MAX = 10  

def unicodeParaAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# descapitalizar (lowercase), aparar (remover palavras raras), e remover caracteres não-letras
def normalizarString(s):
    s = unicodeParaAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# ler pares fala/resposta e retorna um objeto voc
def lerVocs(dados):

    # Ler arquivo e divide em linhas
    linhas = open(dados, encoding='utf-8').\
        read().strip().split('\n')

    # dividir toda linha em pares e ai normalizar
    pares = [[normalizarString(s) for s in l.split('\t')] for l in linhas]
    voc = Voc('')
    return voc, pares

# retorna True iff ambas frases no par ('p') estão abaixo do comprimento TAMANHO_MAX
def filtrarPar(p):
    # a frase de input precisa preservar a ultima palavra para ser um token EOS
    return len(p[0].split(' ')) < TAMANHO_MAX and len(p[1].split(' ')) < TAMANHO_MAX

# filtra pares usando a condição filtrarPares
def filtrarPares(pares):
    return [par for par in pares if filtrarPar(par)]

# com as funcoes definidas acima, retorna um objeto voc e uma lista de pares
def carregarDados(corpus, dados, salvar_dir):
    voc, pares = lerVocs(dados)
    pares = filtrarPares(pares)
    for par in pares:
        voc.adicionarFrase(par[0])
        voc.adicionarFrase(par[1])
    return voc, pares

# carrega/monta voc e pares
salvar_dir = os.path.join("modelos")
voc, pares = carregarDados(corpus, dados, salvar_dir)
    
# palavras abaixo do CONTAGEM_MINIMA são aparadas por serem muito raras
CONTAGEM_MINIMA = 3

def apararPalavrasRaras(voc, pares, CONTAGEM_MINIMA):
    # apara palavras abaixo da CONTAGEM_MINIMA do voc
    voc.aparar(CONTAGEM_MINIMA)
    
    # filtra pares que tem palavras aparadas
    manter_pares = []
    for par in pares:
        frase_input = par[0]
        frase_output = par[1]
        manter_input = True
        manter_output = True
       
        # checar frase de input
        for palavra in frase_input.split(' '):
            if palavra not in voc.word2index:
                manter_input = False
                break
        # checar frase de output
        for palavra in frase_output.split(' '):
            if palavra not in voc.word2index:
                manter_output = False
                break

        # apenas manter pares que não contem palavras aparadas tanto em seu input quanto output
        if manter_input and manter_output:
            manter_pares.append(par)
            
    return manter_pares


# aparar voc e pares
pares = apararPalavrasRaras(voc, pares, CONTAGEM_MINIMA)

def indicesDaFrase(voc, frase):
    return [voc.word2index[palavra] for palavra in frase.split(' ')] + [token_EOS]


def zeroPadding(l, fillvalue=token_PAD):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def matrizBinaria(l, value=token_PAD):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == token_PAD:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# retorna o tensor da frase de input preenchida (preenchimento = padding) e seu comprimento
def inputVar(l, voc):
    indices_batch = [indicesDaFrase(voc, frase) for frase in l]
    comprimento = torch.tensor([len(indices) for indices in indices_batch])
    padLista = zeroPadding(indices_batch)
    padVar = torch.LongTensor(padLista)
    return padVar, comprimento

# retorna o tensor da frase alvo preenchida (padded), a padding mask* e o comprimento máximo da frase alvo (resposta)
# * = ler mais sobre o conceito de Mask em python, que aqui vou usar a traducao 'mascara'
def outputVar(l, voc):
    indices_batch = [indicesDaFrase(voc, frase) for frase in l]
    comprimento_max_alvo = max([len(indices) for indices in indices_batch])
    padLista = zeroPadding(indices_batch)
    mascara = matrizBinaria(padLista)
    mascara = torch.BoolTensor(mascara)
    padVar = torch.LongTensor(padLista)
    return padVar, mascara, comprimento_max_alvo

# retorna todos os itens para um dado lote (batch) de pares
def batchParaTreinarDados(voc, par_batch):
    par_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for par in par_batch:
        input_batch.append(par[0])
        output_batch.append(par[1])
    inp, comprimento = inputVar(input_batch, voc)
    output, mascara, comprimento_max_alvo = outputVar(output_batch, voc)
    return inp, comprimento, output, mascara, comprimento_max_alvo


# exemplo para validação
batch_size_pequeno = 5
batches = batchParaTreinarDados(voc, [random.choice(pares) for _ in range(batch_size_pequeno)])
variavel_input, comprimento, variavel_alvo, mascara, comprimento_max_alvo = batches

class EncoderRNN(nn.Module):
    def __init__(self, tamanho_oculto, embedding, numero_camadas=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.numero_camadas = numero_camadas
        self.tamanho_oculto = tamanho_oculto
        self.embedding = embedding

        # incicializar GRU (Gated Recurrent Unit)
        # parametros definidos como 'tamanho_oculto'
        # porque o tamanho do input é um 'word embedding' com várias características = tamanho_oculto
        self.gru = nn.GRU(tamanho_oculto, tamanho_oculto, numero_camadas,
                          dropout=(0 if numero_camadas == 1 else dropout), bidirectional=True)
    
    # não mexer na nomenclatura de nenhuma função chamada 'forward', algumas delas estão
    # intrinsecamente conectadas a elementos do PyTorch
    def forward(self, sequencia_input, comprimentos_input, oculto=None):
        
        # converter indices de palavras em embeddings
        embedded = self.embedding(sequencia_input)
        
        # empacotar lote (com padding) de sequencias no módulo da rede neural recorrente (RNN)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, comprimentos_input)

        # encaminhar/atravessar GRU
        outputs, oculto = self.gru(packed, oculto)
        
        # desempacotar o padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        # somar os outputs bidirecionais da GRU
        outputs = outputs[:, :, :self.tamanho_oculto] + outputs[:, : ,self.tamanho_oculto:]
        
        # retorna output e o estado oculto final
        return outputs, oculto
        
# ---->>>> COMEÇO: camada de atenção ########################################################################
# mecanismo de atenção de Luong, também conhecido como 'atenção multiplicativa'

class Atencao(nn.Module):
    def __init__(self, metodo, tamanho_oculto):
        super(Atencao, self).__init__()
        self.metodo = metodo
        if self.metodo not in ['dot', 'geral', 'concat']:
            raise ValueError(self.metodo, "não é um método apropriado.")
        self.tamanho_oculto = tamanho_oculto
        if self.metodo == 'geral':
            self.attn = nn.Linear(self.tamanho_oculto, tamanho_oculto)
        elif self.metodo == 'concat':
            self.attn = nn.Linear(self.tamanho_oculto * 2, tamanho_oculto)
            self.v = nn.Parameter(torch.FloatTensor(tamanho_oculto))

    def escore_dot(self, oculto, encoder_output):
        return torch.sum(oculto * encoder_output, dim=2)

    def escore_geral(self, oculto, encoder_output):
        energia = self.attn(encoder_output)
        return torch.sum(oculto * energia, dim=2)

    def escore_concat(self, oculto, encoder_output):
        energia = self.attn(torch.cat((oculto.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energia, dim=2)

    # não mexer no nome dessa função
    def forward(self, oculto, outputs_encoder):
    
        # calcular o peso da atenção (aka, energia/energy) dado o método
        if self.metodo == 'geral':
            energias_atencao = self.escore_geral(oculto, outputs_encoder)
        elif self.metodo == 'concat':
            energias_atencao = self.escore_concat(oculto, outputs_encoder)
        elif self.metodo == 'dot':
            energias_atencao = self.escore_dot(oculto, outputs_encoder)

        # transpôr comprimento máximo e dimensões do tamanho do lote (batch_size)
        energias_atencao = energias_atencao.t()

        # retornar as pontuacoes/escores normalizados da função softmax (com dimensões)
        return F.softmax(energias_atencao, dim=1).unsqueeze(1)
        
class atencaoMultiplicativa(nn.Module):
    def __init__(self, modelo_atencao, embedding, tamanho_oculto, output_size, numero_camadas=1, dropout=0.1):
        super(atencaoMultiplicativa, self).__init__()

        # manter para referência
        self.modelo_atencao = modelo_atencao
        self.tamanho_oculto = tamanho_oculto
        self.output_size = output_size
        self.numero_camadas = numero_camadas
        self.dropout = dropout

        # definir camadas
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(tamanho_oculto, tamanho_oculto, numero_camadas, dropout=(0 if numero_camadas == 1 else dropout))
        self.concat = nn.Linear(tamanho_oculto * 2, tamanho_oculto)
        self.out = nn.Linear(tamanho_oculto, output_size)

        self.attn = Atencao(modelo_atencao, tamanho_oculto)

    def forward(self, input_step, last_hidden, outputs_encoder):
        # uma palavra de cada vez pra gerar o embedding
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # encaminhar (unidirecionalmente) pela GRU
        output_rede, oculto = self.gru(embedded, last_hidden)
        
        # calcular pesos da atenção do output atual da GRU
        pesos_atencao = self.attn(output_rede, outputs_encoder)
        
        # multiplicar pesos da atenção pelos outputs do encoder para alcançar um vetor de contexto
        contexto = pesos_atencao.bmm(outputs_encoder.transpose(0, 1))
        
        # concatenar o vetor de contexto e o output da GRU usando equação de Luong número 5
        # ref: https://isg.beel.org/blog/2019/06/24/memory-augmented-neural-networks-for-machine-translation-pre-print/#2_1_Luong_Attention (salvo como pdf na pasta de referencias)
        # obs: equações estão disponiveis na pasta de referencias
        output_rede = output_rede.squeeze(0)
        contexto = contexto.squeeze(1)
        concat_input = torch.cat((output_rede, contexto), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Predict next word using Luong eq. 6
        # prevê próxima palavra usando equação de Luong número 6
        # ref: mesma da de cima
        # obs: equações estão disponiveis na pasta de referencias
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        
        # Return output and final oculto state
        return output, oculto

# ---->>>> FIM: camada de atenção ########################################################################

# calcular perda, também conhecida como função 'maskNLLLoss'
def calcularPerda(inp, alvo, mascara):
    nTotal = mascara.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, alvo.view(-1, 1)).squeeze(1))
    perda = crossEntropy.masked_select(mascara).mean()
    perda = perda.to(dispositivo)
    return perda, nTotal.item()
    
def treinar(variavel_input, comprimento, variavel_alvo, mascara, comprimento_max_alvo, encoder, decoder, embedding,
          otimizador_encoder, otimizador_decoder, batch_size, clip, tamanho_maximo=TAMANHO_MAX):

    # gradientes zero
    otimizador_encoder.zero_grad()
    otimizador_decoder.zero_grad()

    # definir opções do dispositivo
    variavel_input = variavel_input.to(dispositivo)
    variavel_alvo = variavel_alvo.to(dispositivo)
    mascara = mascara.to(dispositivo)
    comprimento = comprimento.to("cpu")

    # inicializar variaveis
    perda = 0
    imprimir_perdas = []
    n_totals = 0

    # encaminhar/atravessar encoder
    outputs_encoder, encoder_oculto = encoder(variavel_input, comprimento)

    # criar input inicial do decoder (começa com tokens SOS para cada frase)
    input_decoder = torch.LongTensor([[token_SOS for _ in range(batch_size)]])
    input_decoder = input_decoder.to(dispositivo)

    # definir estado inicial do decoder oculto como estado final do encoder oculto
    decoder_oculto = encoder_oculto[:decoder.numero_camadas]

    # determinar se estamos usando 'teacher forcing'
    usar_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # encaminhar/atravessar lote (batch) de frases através do decoder uma de cada vez
    if usar_teacher_forcing:
        for t in range(comprimento_max_alvo):
            output_decoder, decoder_oculto = decoder(
                input_decoder, decoder_oculto, outputs_encoder
            )
            # teacher forcing: proximo input é sempre o presente alvo
            input_decoder = variavel_alvo[t].view(1, -1)
            
            # calcula e acumula perda (loss)
            mask_loss, nTotal = calcularPerda(output_decoder, variavel_alvo[t], mascara[t])
            perda += mask_loss
            imprimir_perdas.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(comprimento_max_alvo):
            output_decoder, decoder_oculto = decoder(
                input_decoder, decoder_oculto, outputs_encoder
            )
            
            # sem teacher forcing: proximo input é o output atual do decoder
            _, topi = output_decoder.topk(1)
            input_decoder = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            input_decoder = input_decoder.to(dispositivo)
            
            # calcula e acumula perda (loss)
            mask_loss, nTotal = calcularPerda(output_decoder, variavel_alvo[t], mascara[t])
            perda += mask_loss
            imprimir_perdas.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # backpropagation
    perda.backward()

    # modificar gradientes
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # ajusta pesos do modelo
    otimizador_encoder.step()
    otimizador_decoder.step()

    return sum(imprimir_perdas) / n_totals

def treinarIters(nome_modelo, voc, pares, encoder, decoder, otimizador_encoder, otimizador_decoder, embedding, n_camadas_encoder, n_camadas_decoder, salvar_dir, n_iteracoes, batch_size, imprimir_cada, salvar_cada, clip, carregarArquivo):

    # carregar lotes (batches) para cada iteração
    batches_treino = [batchParaTreinarDados(voc, [random.choice(pares) for _ in range(batch_size)])
                      for _ in range(n_iteracoes)]

    # inicializações
    iteracao_inicial = 1
    imprimir_perda = 0
    if carregarArquivo:
        iteracao_inicial = checkpoint['iteracao'] + 1

    # loop de treino
    for iteracao in range(iteracao_inicial, n_iteracoes + 1):
        training_batch = batches_treino[iteracao - 1]
        
        # extrair elementos do lote (batch)
        variavel_input, comprimento, variavel_alvo, mascara, comprimento_max_alvo = training_batch

        # pecorre uma iteração de treino com cada lote (batch)
        perda = treinar(variavel_input, comprimento, variavel_alvo, mascara, comprimento_max_alvo, encoder,
                     decoder, embedding, otimizador_encoder, otimizador_decoder, batch_size, clip)
        imprimir_perda += perda

        # salvar checkpoint
        if (iteracao % salvar_cada == 0):
            diretorio = os.path.join(salvar_dir, nome_modelo, '{}-{}_{}'.format(n_camadas_encoder, n_camadas_decoder, tamanho_oculto))
            if not os.path.exists(diretorio):
                os.makedirs(diretorio)
            torch.save({
                'iteracao': iteracao,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'oti_en': otimizador_encoder.state_dict(),
                'oti_de': otimizador_decoder.state_dict(),
                'perda': perda,
                'dic_voc': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(diretorio, '{}_{}.tar'.format(iteracao, 'checkpoint')))

#aqui o algoritmo é estilo 'argmax', 'Greedy Search'
#mais info: https://towardsdatascience.com/word-sequence-decoding-in-seq2seq-architectures-d102000344ad            
class decoderGanancioso(nn.Module):
    def __init__(self, encoder, decoder):
        super(decoderGanancioso, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    #NÃO MEXER NO NOME DA FUNÇÃO 'forward()'!!!
    def forward(self, sequencia_input, comprimento_input, tamanho_maximo):
        
        # encaminhar input atraves do encoder
        outputs_encoder, encoder_oculto = self.encoder(sequencia_input, comprimento_input)
        
        # preparar última camada oculta do encoder para ser a primeira camada oculta do decoder
        decoder_oculto = encoder_oculto[:decoder.numero_camadas]
        
        # inicializar input do decoder com token SOS
        input_decoder = torch.ones(1, 1, device=dispositivo, dtype=torch.long) * token_SOS
        
        # inicializar tensores (para adicionar palavras decodificadas)
        todos_tokens = torch.zeros([0], device=dispositivo, dtype=torch.long)
        todas_pontuacoes = torch.zeros([0], device=dispositivo)
        
        # decodificar um token de palavra por vez
        for _ in range(tamanho_maximo):
            
            # passar pelo decoder
            output_decoder, decoder_oculto = self.decoder(input_decoder, decoder_oculto, outputs_encoder)
            
            # obter o token da palavra mais provavel e sua pontuacao softmax
            pontuacao_decoder, input_decoder = torch.max(output_decoder, dim=1)
            
            # guardar token e a pontuacao
            todos_tokens = torch.cat((todos_tokens, input_decoder), dim=0)
            todas_pontuacoes = torch.cat((todas_pontuacoes, pontuacao_decoder), dim=0)
            
            # preparar token atual para ser o próximo input do decoder (adicionar uma dimensão)
            input_decoder = torch.unsqueeze(input_decoder, 0)
        
        # retornar coletanea de tokens de palavras e pontuacoes softmax
        return todos_tokens, todas_pontuacoes
        
def analisar(encoder, decoder, buscador, voc, frase, tamanho_maximo=TAMANHO_MAX):
    
    ### Formatar frase input como lote (batch)
    
    # palavras para indices (p -> i)
    indices_batch = [indicesDaFrase(voc, frase)]
    
    # criar tensor de comprimento
    comprimento = torch.tensor([len(indices) for indices in indices_batch])
    
    # transpôr dimensões do lote (batch) para as dimensões do modelo
    input_batch = torch.LongTensor(indices_batch).transpose(0, 1)
    
    # indicar dispositivo (no caso, sempre 'cpu')
    input_batch = input_batch.to(dispositivo)
    comprimento = comprimento.to(dispositivo)
    
    # decodificar frase com o buscador
    tokens, pontuacoes = buscador(input_batch, comprimento, tamanho_maximo)
    
    # indices para palavras (i -> p)
    palavras_decodificadas = [voc.index2word[token.item()] for token in tokens]
    return palavras_decodificadas

def resposta_renasce(msg):
    
    try:
        frase_input = msg
        
        # normalizar frase
        frase_input = normalizarString(frase_input)
        
        # analisar frase
        palavras_output = analisar(encoder, decoder, buscador, voc, frase_input)
        
        # formatar resposta
        palavras_output[:] = [x for x in palavras_output if not (x == 'EOS' or x == 'PAD')]
        resposta = ' '.join(palavras_output)
        
        return resposta
    
    except KeyError:
        # esse erro ocorre quando alguma palavra fora do vocabulário é encontrada
        # expandir banco de dados pode solucionar a aparição constante desse erro
        resposta = 'não consegui te entender'
        return resposta
            
def resposta_debug(encoder, decoder, buscador, voc):
    frase_input = ''
    while(1):
        try:
            # frase de input
            frase_input = input('> ')
            
            # checar de é "f" ou "fechar"
            if frase_input.lower() == 'f' or frase_input.lower() == 'fechar': break
            
            # normalizar frase
            frase_input = normalizarString(frase_input)
            
            # analisar frase
            palavras_output = analisar(encoder, decoder, buscador, voc, frase_input)
            
            # formatar e imprimir resposta
            palavras_output[:] = [x for x in palavras_output if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(palavras_output))

        except KeyError:
            # mesma observação da função anterior: erro de vocabulário, expandir banco de dados
            print('não consegui te entender')  
            
                        
# configurar modelo
nome_modelo = 'renasce'
modelo_atencao = 'dot'
#modelo_atencao = 'geral'
#modelo_atencao = 'concat'
tamanho_oculto = 500
n_camadas_encoder = 2
n_camadas_decoder = 2
dropout = 0.1
checkpoint_iter = 10

# >>>>>>>>>>>>>>>>>> COMEÇO: DEFINIR CHECKPOINT ############################################

# se carregarArquivo = None, retreinar rede
# se carregararquivo != None (tipo o que está precedido por '#' embaixo, desativado), continuar de onde treino parou

carregarArquivo = None

#carregarArquivo = os.path.join(salvar_dir, nome_modelo,
#                            '{}-{}_{}'.format(n_camadas_encoder, n_camadas_decoder, tamanho_oculto),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

# >>>>>>>>>>>>>>>>>> FIM: DEFINIR CHECKPOINT ############################################

# carregar modelo se carregarArquivo estiver habilitado
if carregarArquivo:
    # se for o caso de carregar na mesma máquina que foi treinado:
    checkpoint = torch.load(carregarArquivo)
    encoder_salvo = checkpoint['en']
    decoder_salvo = checkpoint['de']
    otimizador_encoder_salvo = checkpoint['oti_en']
    otimizador_decoder_salvo = checkpoint['oti_de']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['dic_voc']

# inicializar word embeddings
embedding = nn.Embedding(voc.numero_palavras, tamanho_oculto)

if carregarArquivo:
    embedding.load_state_dict(embedding_sd)

# inicializar encoders e decoders
encoder = EncoderRNN(tamanho_oculto, embedding, n_camadas_encoder, dropout)
decoder = atencaoMultiplicativa(modelo_atencao, embedding, tamanho_oculto, voc.numero_palavras, n_camadas_decoder, dropout)

if carregarArquivo:
    encoder.load_state_dict(encoder_salvo)
    decoder.load_state_dict(decoder_salvo)

# usar dispositivo indicado (sempre 'cpu' no caso)
encoder = encoder.to(dispositivo)
decoder = decoder.to(dispositivo)

# configurar treino e otimização
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
imprimir_cada = 1
salvar_cada = 1

# certificar-se que as camadas de dropout estão no modo de treino
encoder.train()
decoder.train()

# inicializar otimizadores
otimizador_encoder = optim.Adam(encoder.parameters(), lr=learning_rate)
otimizador_decoder = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if carregarArquivo:
    otimizador_encoder.load_state_dict(otimizador_encoder_salvo)
    otimizador_decoder.load_state_dict(otimizador_decoder_salvo)

# colocar as camadas de dropout no modo de avaliação
encoder.eval()
decoder.eval()

# inicializar módulo buscador
buscador = decoderGanancioso(encoder, decoder)      
           
print(">>> Pronto!\n")      
