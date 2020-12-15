from redes_neurais.rede_seq2seq import *

# treinar

n_iteracoes = input('\n> Quantas iterações (epochs) a rede neural deve rodar?\n>  ')
n_iteracoes = int(n_iteracoes)

batch_size = input('\n> Qual o batch_size? (recomendado 64)\n>  ')
batch_size = int(batch_size)

print("Iniciando treino...")
treinarIters(nome_modelo, voc, pares, encoder, decoder, otimizador_encoder, otimizador_decoder,
           embedding, n_camadas_encoder, n_camadas_decoder, salvar_dir, n_iteracoes, batch_size,
           imprimir_cada, salvar_cada, clip, carregarArquivo)
                
print("Treino completo.")
