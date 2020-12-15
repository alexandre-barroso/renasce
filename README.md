# Redes Neurais Artificiais Simuladoras de Conversa e Emoção (ReNASCE)
Como compreendemos a língua sendo gerada artificialmente por uma máquina?

  Em poucas palavras, ReNASCE é um chatterbot de aprendizagem de máquina que utiliza duas redes neurais de arquiteturas distintas, treinadas em bancos de dados diferentes, que funcionam paralelamente para dar a ilusão de uma conversa mais humanizada. Esse chatterbot não conta com respostas pré-definidas, elas são respostas geradas por uma rede neural recorrente (estilo seq2seq) que utiliza as mensagens do usuário como input. Ao mesmo tempo, outra rede neural, de arquitetura sequencial profunda, utiliza essa mesma mensagem do usuário também como seu input, buscando elementos na mensagem que possam ser identificados como pertecendo a 5 emoções: confuso, feliz, irritado, neutro e triste. A criação desse programa se fez necessária para ter uma liberdade maior durante a pesquisa, já que, sendo seu programador, todos os elementos, desde a interface visual até os menores detalhes e equações das redes neurais artificiais, podem ser livremente e facilmente alterados. Tendo compreensão e controle de todos seus mecanismos, é possível interpretar e guiar as relações participante-máquina de forma mais precisa.

  Em projeto, seria interessante expôr o chatterbot a alguns participantes e, após uma breve conversa com a máquina (talvez orientada por um roteiro, restringindo assuntos para facilitar a criação de um banco de dados mais específico?), pedir, por meio de formulário, que eles identifiquem detalhes da conversa que humanizaram (ou não) a inteligência artificial. Dessa maneira, o objetivo é buscar elementos da linguagem citados frequentemente (se é que existem) entre as pessoas que podem apontar para o pesquisador o que, de forma mais específica, usuários observam na interação homem-máquina. E, assim, repensar esses elementos linguísticos para oferecer uma experiência da IA mais humanizada e orgânica para as pessoas, além de usá-los como possíveis exemplos de fatos buscados pelas pessoas na hora de compreender a língua (que elementos buscamos na hora de compreender a língua que nos levam a classificá-la como orgânica e natural, "humana"?). Por exemplo, é gramaticalidade um fator importante? E a amplitude do vocabulário, palavras muito repetidas causam incômodo? E a organização sintática (caso chatterbot construa frases Suj-Ver-Obj de maneira errônea, apesar de inteligível)? O fato dele etiquetar emoções torna mais ou menos artificial a interação? A exposição do participante a um elemento de emoção, apesar de rudimentar e como elemento talvez apenas tangencialmente linguístico (pragmático), oferece algum benefício na hora de interagir linguisticamente com a máquina? E, claro, as próprias limitações de um chatterbot sendo expostas aos participantes. 
  
  O objetivo da pesquisa não é avançar a tecnologia de chatterbot ao máximo de seus limites e extinguir seus erros tecnológicos. O objetivo é, utilizando uma tecnologia de chatterbot relativamente complexa e atual (aprendizagem de máquina em arquitetura seq2seq com respostas não sendo pré-escritas), observar as limitações linguísticas do chatterbot enquanto objeto de estudo linguístico e não de software. Depois de devidamente treinado, usando outros chatterbots online como benchmark, as limitações da programação de um chatterbot atual faz, sim, parte da interação linguística esperada, pois, neste momento atual da tecnologia, é indissociável da experiência homem-máquina no âmbito da linguagem: não espera-se que chatterbots sempre sejam gramaticais, sintáticos ou mesmo que façam sentido durante toda a conversa.

  Há uma série de reflexões linguísticas que os próprios usuários fazem quando se deparam com esse tipo de situação e são incentivados a descrever em formulário, porque a interação com um chatterbot é de caráter unicamente linguístico. A inclusão de um módulo de emoção, na verdade, não é tão relevante assim para a pesquisa, mas é um método de imersão que pode potencialmente deixar o usuário menos consciente do que está sendo estudado e observado (a conversação em si e as reflexões sobre linguagem dos participantes), dando uma falsa impressão de que é tão relevante quanto a escrita. E, assim, pode distrar o participante de uma possível autoconsciência na escrita (dividindo sua atenção, agora, entre 'emoção' e 'escrita' do chatterbot) e dar mais naturalidade aos resultados.

  O código está disponível nesse repositório e está devidamente anotado. O programa já está completo e foi testado, restando apenas elaborar um banco de dados grande o suficiente. 
  
  Todas as referências (acadêmicas ou não) relevantes estão ou em anotações ou na pasta "referencias" (fiz o upload de tudo que usei, então alguns PDFs muito grandes estão lá. Melhor baixar do que abrir aqui no github). A rede neural seq2seq está treinada no corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html pois ele oferece diálogos formatados "pergunta \t resposta" e é comumente utilizado como corpus para testes. Criar um corpus satisfatório em português é o primeiro objetivo e talvez a maior dificuldade, apesar de não ser impossível (ferramentas de tradução automática + revisão humana?). Guiando o usuário através de um roteiro, o tamanho do banco de dados pode ser consideravelmente reduzido. O banco de dados da rede classificatória de emoções é um arquivo JSON simples, onde criei uma relação simples entre verbos e emoções, não precisa ser grande e é formatado de maneira intuitiva (ou seja, é rápido de elaborar). Por exemplo, o verbo "amar" e suas variações é compreendido como a emoção "feliz" e o verbo "odiar", emoção "triste". Há limitações óbvias nesse método: nem sempre o verbo "amar" é algo feliz e nem sempre "odiar" é triste. Vale a pena refletir a funcionalidade e necessidade da rede neural de emoções, cujo papel é auxiliar o usuário a contextualizar, de forma humana, o output da máquina e causar, em algum nível, uma distração, ergo, evitar autoconsciência na escrita. Considerando que a rede neural de emoções trabalha com rótulos pré-definidos (sou eu, enquanto programador, que defino quais verbos participam de quais emoções), é possível que esse aspecto mais artesanal traga algum elemento positivo para o usuário. A próxima dificuldade é a quantidade considerável de espaço de disco rígido que requer treinar uma rede neural. Um bom computador seria ideal. E, finalmente, durante a pandemia do coronavírus, essa é uma pesquisa que pode ser feita completamente online: a interação com o chatterbot e o preenchimento de formulário.

  Abaixo, dois links do YouTube. Um é uma demonstração do programa e o outro é um exemplo do treino das redes neurais. É importante notar que A REDE NEURAL NÃO ESTÁ TREINADA! Ela só treinou 1 iteração para exemplificar como roda o programa. Quando ela diz "não entendi" é porque encontrou uma palavra fora do banco de dados. Como o banco de dados está em inglês (corpus Cornell), palavras em português estão sempre clasificadas como "não entendi". No entanto, o banco de dados de emoções, no vídeo, foi treinado com alguns poucos termos em português, então, mesmo que a resposta seja "não entendi", a outra rede neural ainda é capaz de identificar algumas "emoções". Eu não treinei a inteligência artificial ainda porque estou elaborando um banco de dados em português. Tendo treinado 1 iteração e já conseguir output é suficiente para saber que, num nível básico, seu funcionamento está ok e só resta calibrar coisas como quantidade de camadas, tamanho de lotes, quantidade de iterações, tamanho do bd, etc. Também é pra demonstrar a interface visual. Na descrição dos vídeos tem algumas outras informações.

Demonstração de treino: https://youtu.be/5Fpxy4RUqYc

Demonstração do programa: https://youtu.be/cHKx5qLde8Q

  Você pode rodar o ReNASCE apenas rodando o arquivo, em python3, 'renasce.py'. Rode o 'treinar.py' antes, caso você não tenha treinado a rede seq2seq (a rede classificatória vem com 600 iterações treinadas, num banco de dados mínimo para exemplo). Lembrando que cada iteração de treino gera um arquivo de quase 300 MB no seu computador. Eu estou rodando em um HD externo de 1 TB dedicado exclusivamente para isso. Estou expandindo o banco de dados classificatório (emoções, em JSON) constantemente, pois a parte de programação já está completa (terminada em dezembro/2020). Agora é focar nas amostras, criação de banco de dados e otimizar a arquitetura das redes neurais conforme o necessário. Talvez uma ou outra modificação na interface visual.
  
  Em síntese, ReNASCE é um chatterbot que utiliza aprendizagem de máquina, disposta ao longo de duas redes neurais, para conversar e etiquetar emoções nos inputs (mensagens) dos usuários. O módulo de emoção serve como apoio pragmático, mas, principalmente, como maneira de ajudar na imersão do usuário e desfocá-lo do objeto da pesquisa, a conversação, deixando implícito que a emoção é tão importante quanto. A 'emoção', também, pode ser substituída por outras formas de imersão e distração. Usuários podem ter uma conversa, guiada ou não (dependendo das limitações do banco de dados), onde, após, preencheriam um formulário com algumas perguntas básicas sobre a interação e escreveriam sobre alguns aspectos de linguagem do chatterbot. Entre usuários, podem surgir ou não pontos recorrentes. Esses pontos podem ser estudados e analisados, para melhoria da experiência linguística em IA e como detalhes que tomam a atenção dos usuários não apenas em IA, mas também em conversações humanas em geral (e que ficariam mais evidentes numa conversação homem-máquina). Considerando o que foi proposto, é possível dividir o trabalho em fatias menores de pesquisa, em ordem cronológica: elaboração do banco de dados, calibragem de redes, calibragem da interface visual, experimento prático e análise de resultados, por exemplo. Por fim, tentar compreender porque certos elementos, e não outros, tomam precedência no relacionamento linguístico homem-máquina.
