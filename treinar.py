print('\n\n>>> Qual rede neural treinar? Digite a letra correspondente i.e. c para rede_classificatoria, s para rede_seq2seq')
print('\nc. rede_classificatoria\ns. rede_seq2seq\n\nObservação: digitar qualquer outra coisa resultará no encerramento do programa.\n')


rede_neural = input('> ')
print('')

if rede_neural.lower() == 'c':
    exec(open('redes_neurais/rede_classificatoria.py').read())
    
elif rede_neural.lower() == 's':
    exec(open('redes_neurais/treinar_seq2seq.py').read())

else:
    print('Encerrando programa...')
