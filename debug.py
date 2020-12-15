#treinar = input('Treinar? S/N\n>>> ')
#
#treinar = treinar.lower()

#if treinar == 's':
#    from redes_neurais.treinar_seq2seq import *
#    resposta_debug(encoder, decoder, searcher, voc)
    
#elif treinar == 'n':
#    from redes_neurais.rede_seq2seq import *
#    resposta_debug(encoder, decoder, searcher, voc)

#else:
#    print('Comando não reconhecido. Encerrando programa...')

from redes_neurais.rede_seq2seq import *
resposta_debug(encoder, decoder, buscador, voc) 
