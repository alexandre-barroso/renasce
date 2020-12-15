# Redes Neurais Artificiais Simuladoras de Conversa e Emoção (ReNASCE)
Como compreendemos a língua sendo gerada artificialmente por uma máquina?

Em poucas palavras, ReNASCE é um chatterbot que utiliza duas redes neurais de arquiteturas distintas, treinadas em banco de dados diferentes, que funcionam paralelamente para dar a ilusão de uma conversa mais humanizada. Esse chatterbot não conta com respostas pré-definidas, elas são respostas geradas por uma rede neural recorrente (estilo seq2seq) que utilizam as mensagens do usuário como input. Ao mesmo tempo, outra rede neural utiliza essa mesma mensagem do usuário também como input, buscando elementos na mensagem que possam ser identificados como pertecendo a 5 emoções: confuso, feliz, irritado, neutro e triste.

Em projeto, seria interessante expôr o chatterbot a alguns usuários e, após uma breve conversa com a máquina, pedir, por meio de formulário, que eles identifiquem detalhes da conversa que humanizaram (ou não) a inteligência artificial. Desta maneira, o objetivo é buscar elementos da linguagem comuns entre as pessoas que podem apontar para o pesquisador o que, de forma mais específica, usuários observam na interação homem-máquina. E, assim, repensar esses elementos linguísticos para oferecer uma experiência mais humanizada e orgânica para as pessoas. Por exemplo, é gramaticalidade um fator importante? E a amplitude do vocabulário, palavras muito repetidas causam incômodo? O fato dele etiquetar emoções torna mais ou menos artificial a interação? A emoção, como elemento talvez apenas tangencialmente linguístico (pragmático), oferece algum benefício na hora de interagir linguisticamente com a máquina? Há uma série de reflexões linguísticas que os próprios usuários fazem quando se deparam esse tipo de situação e são incentivados a descrever em formulário, porque a interação com um chatterbot é de caráter unicamente linguístico.

O código completo está disposto nesse repositório e está completamente anotado. Todas as referências (acadêmicas ou não) relevantes estão ou em anotações ou na pasta "referencias". A rede neural seq2seq está treinada no corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html pois ele oferece diaĺogos formatados "pergunta \t resposta". Criar um corpus satisfatório em português é o primeiro objetivo e talvez a maior dificuldade, apesar de não ser impossível (ferramentas de tradução automática + revisão humana?). O banco de dados da rede classificatória de emoções é um arquivo JSON simples, onde criei uma relação simples entre verbos e emoções. Por exemplo, o verbo "amar" e suas variações é compreendido como a emoção "feliz" e o verbo "odiar", emoção "triste". Há limitações óbvias nesse método: nem sempre o verbo "amar" é algo feliz e nem sempre "odiar" é triste. Vale a pena refletir a funcionalidade e necessidade da rede neural de emoções, cujo papel é auxiliar o usuário a contextualizar, de forma humana, o output da máquina. Considerando que a rede neural de emoções trabalha com rótulos pré-definidos (sou eu, enquanto programador, que defino quais verbos participam de quais emoções), é possível que esse aspecto mais "artesanal" traga algum elemento positivo para o usuário. A próxima dificuldade é a quantidade considerável de espaço de disco rígido que requer treinar uma rede neural. Um bom computador seria ideal. E, finalmente, durante a pandemia do coronavírus, essa é uma pesquisa que pode ser feita completamente online: a interação com o chatterbot e o preenchimento de formulário.

Abaixo, dois links do YouTube. Um é uma demonstração do programa e o outro é um exemplo do treino das redes neurais.
