referência: https://pt.d2l.ai/chapter_preface/index.html

## 🦖 Prefácio
Testar o potencial do deep learning apresenta desafios únicos porque qualquer aplicativo reúne várias disciplinas. Aplicar o deep learning requer compreensão simultânea (i) as motivações para definir um problema de uma maneira particular; (ii) a matemática de uma dada abordagem de modelagem; (iii) os algoritmos de otimização para ajustar os modelos aos dados; e (iv) a engenharia necessária para treinar modelos de forma eficiente, navegando nas armadilhas da computação numérica e obter o máximo do hardware disponível. Ensinar as habilidades de pensamento crítico necessárias para formular problemas, a matemática para resolvê-los e as ferramentas de software para implementar tais soluções em um só lugar apresentam desafios formidáveis.

Propusemo-nos a criar um recurso que pudesse (i) estar disponível gratuitamente para todos; (ii) oferecer profundidade técnica suficiente para fornecer um ponto de partida no caminho para realmente se tornar um cientista de machine learning aplicado; (iii) incluir código executável, mostrando aos leitores como resolver problemas na prática; (iv) permitir atualizações rápidas, tanto por nós e também pela comunidade em geral; e (v) ser complementado por um fórum;

### Conteúdo e Estrutura

![image](https://github.com/user-attachments/assets/e34d70ed-ba2a-4281-9620-b93e22502307)

Às vezes, para evitar repetição desnecessária, encapsulamos as funções, classes, etc. importadas e mencionadas com frequência neste livro no package d2l. Para qualquer bloco, como uma função, uma classe ou vários imports ser salvo no pacote, vamos marcá-lo com # @ save. Oferecemos uma visão geral detalhada dessas funções e classes em: numref: sec_d2l. O package d2l é leve e requer apenas os seguintes packages e módulos como dependências:

```
#@save
import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt

d2l = sys.modules[__name__]
```

A maior parte do código neste livro é baseada no Apache MXNet. MXNet é um framework de código aberto (oper-source) para deep learning e a escolha preferida de AWS (Amazon Web Services), bem como muitas faculdades e empresas. Aqui está como importamos módulos do MXNet.
```
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

A maior parte do código neste livro é baseada no PyTorch. PyTorch é uma estrutura de código aberto para deep learning, que é extremamente popular na comunidade de pesquisa. Aqui está como importamos módulos do PyTorch.
```
#@save
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
```

A maior parte do código deste livro é baseada no TensorFlow. TensorFlow é uma estrutura de código aberto para deep learning, que é extremamente popular na comunidade de pesquisa e na indústria. Aqui está como importamos módulos do TensorFlow.
```
#@save
import numpy as np
import tensorflow as tf
```

## 🦕 Introdução
Até recentemente, quase todos os programas de computador com os quais interagimos diariamente eram codificados por desenvolvedores de software desde os primeiros princípios. Digamos que quiséssemos escrever um aplicativo para gerenciar uma plataforma de e-commerce. Depois de se amontoar em um quadro branco por algumas horas para refletir sobre o problema, iríamos apresentar os traços gerais de uma solução de trabalho que provavelmente se pareceria com isto: (i) os usuários interagem com o aplicativo por meio de uma interface executando em um navegador da web ou aplicativo móvel; (ii) nosso aplicativo interage com um mecanismo de banco de dados de nível comercial para acompanhar o estado de cada usuário e manter registros de histórico de transações; e (iii) no cerne de nossa aplicação, a lógica de negócios (você pode dizer, os cérebros) de nosso aplicativo descreve em detalhes metódicos a ação apropriada que nosso programa deve levar em todas as circunstâncias concebíveis.

Para construir o cérebro de nosso aplicativo, teríamos que percorrer todos os casos esquivos possíveis que antecipamos encontrar, criando regras apropriadas. Cada vez que um cliente clica para adicionar um item ao carrinho de compras, adicionamos uma entrada à tabela de banco de dados do carrinho de compras, associando o ID desse usuário ao ID do produto solicitado. Embora poucos desenvolvedores acertem completamente na primeira vez (podem ser necessários alguns testes para resolver os problemas), na maior parte, poderíamos escrever esse programa a partir dos primeiros princípios e lançá-lo com confiança antes de ver um cliente real. Nossa capacidade de projetar sistemas automatizados a partir dos primeiros princípios que impulsionam o funcionamento de produtos, sistemas e, frequentemente em novas situações, é um feito cognitivo notável. E quando você é capaz de conceber soluções que funcionam do tempo, você não deveria usar o machine learning.

Felizmente para a crescente comunidade de cientistas de machine learning, muitas tarefas que gostaríamos de automatizar não se curvam tão facilmente à habilidade humana. Imagine se amontoar em volta do quadro branco com as mentes mais inteligentes que você conhece, mas desta vez você está lidando com um dos seguintes problemas:

- Escreva um programa que preveja o clima de amanhã com base em informações geográficas, imagens de satélite e uma janela de rastreamento do tempo passado.
- Escreva um programa que aceite uma pergunta, expressa em texto de forma livre, e a responda corretamente.
- Escreva um programa que, dada uma imagem, possa identificar todas as pessoas que ela contém, desenhando contornos em torno de cada uma.
- Escreva um programa que apresente aos usuários produtos que eles provavelmente irão gostar, mas que provavelmente não encontrarão no curso natural da navegação.

Em cada um desses casos, mesmo programadores de elite são incapazes de codificar soluções do zero. As razões para isso podem variar. Às vezes, o programa que procuramos segue um padrão que muda com o tempo, e precisamos que nossos programas se adaptem. Em outros casos, a relação (digamos, entre pixels, e categorias abstratas) podem ser muito complicadas, exigindo milhares ou milhões de cálculos que estão além da nossa compreensão consciente mesmo que nossos olhos administrem a tarefa sem esforço. Machine learning é o estudo de poderosas técnicas que podem aprender com a experiência. À medida que um algoritmo de machine learning acumula mais experiência, normalmente na forma de dados observacionais ou interações com um ambiente, seu desempenho melhora. Compare isso com nossa plataforma de comércio eletrônico determinística, que funciona de acordo com a mesma lógica de negócios, não importa quanta experiência acumule, até que os próprios desenvolvedores aprendam e decidam que é hora de atualizar o software. Neste livro, ensinaremos os fundamentos do machine learning, e foco em particular no deep learning, um poderoso conjunto de técnicas impulsionando inovações em áreas tão diversas como a visão computacional, processamento de linguagem natural, saúde e genômica.

### Um Exemplo motivador

Imagine apenas escrever um programa para responder a uma palavra de alerta como “Alexa”, “OK Google” e “Hey Siri”. Tente codificar em uma sala sozinho com nada além de um computador e um editor de código. Como você escreveria tal programa a partir dos primeiros princípios? Pense nisso … o problema é difícil. A cada segundo, o microfone irá coletar aproximadamente 44.000 amostras. Cada amostra é uma medida da amplitude da onda sonora. Que regra poderia mapear de forma confiável, de um trecho de áudio bruto a previsões confiáveis { yes, no } sobre se o trecho de áudio contém a palavra de ativação? Se você estiver travado, não se preocupe. Também não sabemos escrever tal programa do zero. É por isso que usamos o machine learning.

![image](https://github.com/user-attachments/assets/80b7346d-1fec-425c-8e73-dc617dd25a9d)


Aqui está o truque. Muitas vezes, mesmo quando não sabemos como dizer a um computador explicitamente como mapear de entradas para saídas, ainda assim, somos capazes de realizar a façanha cognitiva por nós mesmos. Em outras palavras, mesmo que você não saiba como programar um computador para reconhecer a palavra “Alexa”, você mesmo é capaz de reconhecê-lo. Armados com essa habilidade, podemos coletar um enorme dataset contendo exemplos de áudio e rotular aqueles que contêm e que não contêm a palavra de ativação. Na abordagem de machine learning, não tentamos projetar um sistema explicitamente para reconhecer palavras de ativação. Em vez disso, definimos um programa flexível cujo comportamento é determinado por vários parâmetros. Em seguida, usamos o conjunto de dados para determinar o melhor conjunto possível de parâmetros, aqueles que melhoram o desempenho do nosso programa com respeito a alguma medida de desempenho na tarefa de interesse.

Você pode pensar nos parâmetros como botões que podemos girar, manipulando o comportamento do programa. Fixando os parâmetros, chamamos o programa de `modelo`. O conjunto de todos os programas distintos (mapeamentos de entrada-saída) que podemos produzir apenas manipulando os parâmetros é chamada de `família` de modelos. E o meta-programa que usa nosso conjunto de dados para escolher os parâmetros é chamado de `algoritmo de aprendizagem`.

Antes de prosseguirmos e envolvermos o algoritmo de aprendizagem, temos que definir o problema com precisão, identificando a natureza exata das entradas e saídas, e escolher uma família modelo apropriada. Nesse caso, nosso modelo recebe um trecho de áudio como `entrada`, e o modelo gera uma seleção entre { yes, no } como `saída`. Se tudo correr de acordo com o plano as suposições da modelo vão normalmente estar corretas quanto a se o áudio contém a palavra de ativação.

Se escolhermos a família certa de modelos, deve haver uma configuração dos botões de forma que o modelo dispara “sim” toda vez que ouve a palavra “Alexa”. Como a escolha exata da palavra de ativação é arbitrária, provavelmente precisaremos de uma família modelo suficientemente rica que, por meio de outra configuração dos botões, ele poderia disparar “sim” somente ao ouvir a palavra “Damasco”. Esperamos que a mesma família de modelo seja adequada para reconhecimento “Alexa” e reconhecimento “Damasco” porque parecem, intuitivamente, tarefas semelhantes. No entanto, podemos precisar de uma família totalmente diferente de modelos se quisermos lidar com entradas ou saídas fundamentalmente diferentes, digamos que se quiséssemos mapear de imagens para legendas, ou de frases em inglês para frases em chinês.

Como você pode imaginar, se apenas definirmos todos os botões aleatoriamente, é improvável que nosso modelo reconheça “Alexa”, “Apricot”, ou qualquer outra palavra em inglês. No machine learning, o aprendizado (learning) é o processo pelo qual descobrimos a configuração certa dos botões coagindo o comportamento desejado de nosso modelo. Em outras palavras, nós treinamos nosso modelo com dados. O processo de treinamento geralmente se parece com o seguinte:
1. Comece com um modelo inicializado aleatoriamente que não pode fazer nada útil.
2. Pegue alguns de seus dados (por exemplo, trechos de áudio e labels { yes, no } correspondentes).
3. Ajuste os botões para que o modelo seja menos ruim em relação a esses exemplos.
4. Repita as etapas 2 e 3 até que o modelo esteja incrível.

![image](https://github.com/user-attachments/assets/29a97296-b1f2-4b1a-b9af-40fcd2c8e16f)

Para resumir, em vez de codificar um reconhecedor de palavra de acionamento, nós codificamos um programa que pode aprender a reconhecê-las se o apresentarmos com um grande dataset rotulado. Você pode pensar neste ato de determinar o comportamento de um programa apresentando-o com um __dataset como programação com dados__. Quer dizer, podemos “programar” um detector de gatos, fornecendo nosso sistema de aprendizado de máquina com muitos exemplos de cães e gatos. Dessa forma, o detector aprenderá a emitir um número positivo muito grande se for um gato, um número negativo muito grande se for um cachorro, e algo mais próximo de zero se não houver certeza, e isso é apenas a ponta do iceberg do que o machine learning pode fazer. Deep learning, que iremos explicar em maiores detalhes posteriormente, é apenas um entre muitos métodos populares para resolver problemas de machine learning.

### Componentes chave

Em nosso exemplo de palavra de ativação, descrevemos um `dataset` consistindo em trechos de áudio e labels binários, e nós demos uma sensação ondulante de como podemos treinar um modelo para aproximar um mapeamento de áudios para classificações. Esse tipo de problema, onde tentamos prever um `label` desconhecido designado com base em entradas conhecidas dado um conjunto de dados que consiste em exemplos para os quais os rótulos são conhecidos, é chamado de `aprendizagem supervisionada`. Esse é apenas um entre muitos tipos de problemas de machine learning. Posteriormente, mergulharemos profundamente em diferentes problemas de machine learning. Primeiro, gostaríamos de lançar mais luz sobre alguns componentes principais que nos acompanharão, independentemente do tipo de problema de machine learning que enfrentarmos:

1. Os `dados` com os quais podemos aprender.
2. Um `modelo` de como transformar os dados.
3. Uma `função objetivo` que quantifica o quão bem (ou mal) o modelo está indo.
4. Um `algoritmo` para ajustar os parâmetros do modelo para otimizar a função obejtivo.

#### Dados
Nem é preciso dizer que você não pode fazer ciência de dados sem dados. Podemos perder centenas de páginas pensando no que exatamente constitui os dados, mas por agora, vamos errar no lado prático e focar nas principais propriedades com as quais se preocupar. Geralmente, estamos preocupados com uma coleção de exemplos. Para trabalhar com dados de maneira útil, nós tipicamente precisamos chegar a uma representação numérica adequada. Cada exemplo (__ou ponto de dados, instância de dados, amostra__) normalmente consiste em um conjunto de atributos chamados `recursos (ou covariáveis)`, a partir do qual o modelo deve fazer suas previsões. Nos problemas de aprendizagem supervisionada acima, a coisa a prever é um atributo especial que é designado como o `rótulo (label) (ou alvo)`.

Se estivéssemos trabalhando com dados de imagem, cada fotografia individual pode constituir um exemplo, cada um representado por uma lista ordenada de valores numéricos correspondendo ao brilho de cada pixel. Uma fotografia colorida de 200 x 200 consistiria em 200 x 200 2 3 = 120000 valores numéricos, correspondentes ao brilho dos canais vermelho, verde e azul para cada pixel. Em outra tarefa tradicional, podemos tentar prever se um paciente vai sobreviver ou não, dado um conjunto padrão de recursos, como idade, sinais vitais e diagnósticos.

Quando cada exemplo é caracterizado pelo mesmo número de valores numéricos, dizemos que os dados consistem em vetores de comprimento fixo e descrevemos o comprimento constante dos vetores como a `dimensionalidade dos dados`. Como você pode imaginar, o comprimento fixo pode ser uma propriedade conveniente. Se quiséssemos treinar um modelo para reconhecer o câncer em imagens microscópicas, entradas de comprimento fixo significam que temos uma coisa a menos com que nos preocupar.

No entanto, nem todos os dados podem ser facilmente representados como vetores de `comprimento fixo`. Embora possamos esperar que as imagens do microscópio venham de equipamentos padrão, não podemos esperar imagens extraídas da Internet aparecerem todas com a mesma resolução ou formato. Para imagens, podemos considerar cortá-los todas em um tamanho padrão, mas essa estratégia só nos leva até certo ponto. Corremos o risco de perder informações nas partes cortadas. Além disso, os dados de texto resistem a representações de comprimento fixo ainda mais obstinadamente. Considere os comentários de clientes deixados em sites de comércio eletrônico como Amazon, IMDB e TripAdvisor. Alguns são curtos: “é uma porcaria!”. Outros vagam por páginas. Uma das principais vantagens do deep learning sobre os métodos tradicionais é a graça comparativa com a qual os modelos modernos podem lidar com dados de `comprimento variável`.

Geralmente, quanto mais dados temos, mais fácil se torna nosso trabalho. Quando temos mais dados, podemos treinar modelos mais poderosos e dependem menos de suposições pré-concebidas. A mudança de regime de (comparativamente) pequeno para big data é um dos principais contribuintes para o sucesso do deep learning moderno. Para esclarecer, muitos dos modelos mais interessantes de deep learning não funcionam sem grandes datasets. Alguns outros trabalham no regime de pequenos dados, mas não são melhores do que as abordagens tradicionais.

Por fim, não basta ter muitos dados e processá-los com inteligência. Precisamos dos dados certos. Se os dados estiverem cheios de erros, ou se os recursos escolhidos não são preditivos da quantidade alvo de interesse, o aprendizado vai falhar. A situação é bem capturada pelo clichê: entra lixo, sai lixo. Além disso, o desempenho preditivo ruim não é a única consequência potencial. Em aplicativos sensíveis de machine learning, como policiamento preditivo, triagem de currículo e modelos de risco usados ​​para empréstimos, devemos estar especialmente alertas para as consequências de dados inúteis. Um modo de falha comum ocorre em conjuntos de dados onde alguns grupos de pessoas não são representados nos dados de treinamento. Imagine aplicar um sistema de reconhecimento de câncer de pele na natureza que nunca tinha visto pele negra antes. A falha também pode ocorrer quando os dados não apenas sub-representem alguns grupos mas refletem preconceitos sociais. Por exemplo, se as decisões de contratação anteriores forem usadas para treinar um modelo preditivo que será usado para selecionar currículos, então os modelos de aprendizado de máquina poderiam inadvertidamente capturar e automatizar injustiças históricas. Observe que tudo isso pode acontecer sem o cientista de dados conspirar ativamente, ou mesmo estar ciente.

#### Modelos
A maior parte do machine learning envolve transformar os dados de alguma forma. Talvez queiramos construir um sistema que ingere fotos e preveja sorrisos. Alternativamente, podemos querer ingerir um conjunto de leituras de sensor e prever quão normais ou anômalas são as leituras. Por modelo, denotamos a maquinaria computacional para ingestão de dados de um tipo, e cuspir previsões de um tipo possivelmente diferente. Em particular, estamos interessados ​​em modelos estatísticos que podem ser estimados a partir de dados. Embora os modelos simples sejam perfeitamente capazes de abordar problemas apropriadamente simples, os problemas nos quais nos concentramos neste livro, ampliam os limites dos métodos clássicos. O deep learning é diferenciado das abordagens clássicas principalmente pelo conjunto de modelos poderosos em que se concentra. Esses modelos consistem em muitas transformações sucessivas dos dados que são encadeados de cima para baixo, daí o nome deep learning. No caminho para discutir modelos profundos, também discutiremos alguns métodos mais tradicionais.


#### Funções Objetivo
Anteriormente, apresentamos o machine learning como aprendizado com a experiência. Por aprender aqui, queremos dizer melhorar em alguma tarefa ao longo do tempo. Mas quem pode dizer o que constitui uma melhoria? Você pode imaginar que poderíamos propor a atualização do nosso modelo, e algumas pessoas podem discordar sobre se a atualização proposta constituiu uma melhoria ou um declínio.

A fim de desenvolver um sistema matemático formal de máquinas de aprendizagem, precisamos ter medidas formais de quão bons (ou ruins) nossos modelos são. No machine learning, e na otimização em geral, chamamos elas de `funções objetivo`. Por convenção, geralmente definimos funções objetivo de modo que quanto menor, melhor. Esta é apenas uma convenção. Você pode assumir qualquer função para a qual mais alto é melhor, e transformá-la em uma nova função que é qualitativamente idêntica, mas para a qual menor é melhor, invertendo o sinal. Porque quanto menor é melhor, essas funções às vezes são chamadas `funções de perda (loss functions)`.

Ao tentar prever valores numéricos, a função de perda mais comum é `erro quadrático`, ou seja, o quadrado da diferença entre a previsão e a verdade fundamental. Para classificação, o objetivo mais comum é minimizar a taxa de erro, ou seja, a fração de exemplos em que nossas previsões discordam da verdade fundamental. Alguns objetivos (por exemplo, erro quadrático) são fáceis de otimizar. Outros (por exemplo, taxa de erro) são difíceis de otimizar diretamente, devido à indiferenciabilidade ou outras complicações. Nesses casos, é comum otimizar um `objetivo substituto`.

Normalmente, a função de perda é definida no que diz respeito aos parâmetros do modelo e depende do conjunto de dados. Nós aprendemos os melhores valores dos parâmetros do nosso modelo minimizando a perda incorrida em um conjunto consistindo em alguns exemplos coletados para treinamento. No entanto, indo bem nos dados de treinamento não garante que teremos um bom desempenho com dados não vistos. Portanto, normalmente queremos dividir os dados disponíveis em duas partições: o `dataset de treinamento` (ou conjunto de treinamento, para ajustar os parâmetros do modelo) e o `dataset de teste` (ou conjunto de teste, que é apresentado para avaliação), relatando o desempenho do modelo em ambos. Você pode pensar no desempenho do treinamento como sendo as pontuações de um aluno em exames práticos usado para se preparar para algum exame final real. Mesmo que os resultados sejam encorajadores, isso não garante sucesso no exame final. Em outras palavras, o desempenho do teste pode divergir significativamente do desempenho do treinamento. Quando um modelo tem um bom desempenho no conjunto de treinamento mas falha em generalizar para dados invisíveis, dizemos que está fazendo `overfitting`. Em termos da vida real, é como ser reprovado no exame real apesar de ir bem nos exames práticos.

#### Algoritmos de Otimização
Assim que tivermos alguma fonte de dados e representação, um modelo e uma função objetivo bem definida, precisamos de um algoritmo capaz de pesquisar para obter os melhores parâmetros possíveis para minimizar a função de perda. Algoritmos de otimização populares para aprendizagem profunda baseiam-se em uma abordagem chamada `gradiente descendente`. Em suma, em cada etapa, este método verifica, para cada parâmetro, para que lado a perda do conjunto de treinamento se moveria se você perturbou esse parâmetro apenas um pouco. Em seguida, atualiza o parâmetro na direção que pode reduzir a perda.

### Tipos de Problemas de Machine Learning
O problema da palavra de ativação em nosso exemplo motivador é apenas um entre muitos problemas que o machine learning pode resolver. Para motivar ainda mais o leitor e nos fornecer uma linguagem comum quando falarmos sobre mais problemas ao longo do livro, a seguir nós listamos uma amostra dos problemas de machine learning. Estaremos constantemente nos referindo a nossos conceitos acima mencionados como dados, modelos e técnicas de treinamento.

#### Aprendizagem Supervisionada
A aprendizagem supervisionada (supervised learning) aborda a tarefa de prever `labels` com recursos de entrada. Cada par recurso-rótulo é chamado de exemplo. Às vezes, quando o contexto é claro, podemos usar o termo exemplos para se referir a uma coleção de entradas, mesmo quando os labels correspondentes são desconhecidos. Nosso objetivo é produzir um modelo que mapeia qualquer entrada para uma previsão de label.

Para fundamentar esta descrição em um exemplo concreto, se estivéssemos trabalhando na área de saúde, então podemos querer prever se um paciente teria um ataque cardíaco ou não. Esta observação, “ataque cardíaco” ou “sem ataque cardíaco”, seria nosso label. Os recursos de entrada podem ser sinais vitais como frequência cardíaca, pressão arterial diastólica, e pressão arterial sistólica.

A supervisão entra em jogo porque para a escolha dos parâmetros, nós (os supervisores) fornecemos ao modelo um conjunto de dados consistindo em exemplos rotulados, onde cada exemplo é correspondido com o label da verdade fundamental. Em termos probabilísticos, normalmente estamos interessados ​​em estimar a probabilidade condicional de determinados recursos de entrada de um label. Embora seja apenas um entre vários paradigmas no machine learning, a aprendizagem supervisionada é responsável pela maioria das bem-sucedidas aplicações de machine learning na indústria. Em parte, isso ocorre porque muitas tarefas importantes podem ser descritas nitidamente como estimar a probabilidade de algo desconhecido dado um determinado dataset disponível:

- Prever câncer versus não câncer, dada uma imagem de tomografia computadorizada.
- Prever a tradução correta em francês, dada uma frase em inglês.
- Prever o preço de uma ação no próximo mês com base nos dados de relatórios financeiros deste mês.

Mesmo com a descrição simples “previsão de labels com recursos de entrada” a aprendizagem supervisionada pode assumir muitas formas e exigem muitas decisões de modelagem, dependendo (entre outras considerações) do tipo, tamanho, e o número de entradas e saídas. Por exemplo, usamos diferentes modelos para processar sequências de comprimentos arbitrários e para processar representações de vetores de comprimento fixo. Visitaremos muitos desses problemas em profundidade ao longo deste livro.

Informalmente, o processo de aprendizagem se parece com o seguinte. Primeiro, pegue uma grande coleção de exemplos para os quais os recursos são conhecidos e selecione deles um subconjunto aleatório, adquirindo os labels da verdade fundamental para cada um. Às vezes, esses labels podem ser dados disponíveis que já foram coletados (por exemplo, um paciente morreu no ano seguinte?) e outras vezes, podemos precisar empregar anotadores humanos para rotular os dados, (por exemplo, atribuição de imagens a categorias). Juntas, essas entradas e os labels correspondentes constituem o conjunto de treinamento. Alimentamos o dataset de treinamento em um algoritmo de aprendizado supervisionado, uma função que recebe como entrada um conjunto de dados e produz outra função: o modelo aprendido. Finalmente, podemos alimentar entradas não vistas anteriormente para o modelo aprendido, usando suas saídas como previsões do rótulo correspondente.

![image](https://github.com/user-attachments/assets/1f9de33c-1212-4b63-8dec-2608f708409d)


<ins>Regressão</ins>

Talvez a tarefa de aprendizagem supervisionada mais simples para entender é regressão. Considere, por exemplo, um conjunto de dados coletados de um banco de dados de vendas de casas. Podemos construir uma mesa, onde cada linha corresponde a uma casa diferente, e cada coluna corresponde a algum atributo relevante, como a metragem quadrada de uma casa, o número de quartos, o número de banheiros e o número de minutos (caminhando) até o centro da cidade. Neste conjunto de dados, cada exemplo seria uma casa específica, e o vetor de recurso correspondente seria uma linha na tabela. Se você mora em Nova York ou São Francisco, e você não é o CEO da Amazon, Google, Microsoft ou Facebook, o vetor de recursos (metragem quadrada, nº de quartos, nº de banheiros, distância a pé) para sua casa pode ser algo como: $ [56, 1, 1, 60] $. No entanto, se você mora em Pittsburgh, pode ser parecido com $ [279, 4, 3, 10] $. Vetores de recursos como este são essenciais para a maioria dos algoritmos clássicos de machine learning.

O que torna um problema uma regressão é, na verdade, o resultado. Digamos que você esteja em busca de uma nova casa. Você pode querer estimar o valor justo de mercado de uma casa, dados alguns recursos como acima. O label, o preço de venda, é um valor numérico. Quando os labels assumem valores numéricos arbitrários, chamamos isso de problema de regressão. Nosso objetivo é produzir um modelo cujas previsões aproximar os valores reais do label.

Mesmo que você nunca tenha trabalhado com machine learning antes, você provavelmente já trabalhou em um problema de regressão informalmente. Imagine, por exemplo, que você mandou consertar seus ralos e que seu contratante gastou 3 horas removendo sujeira de seus canos de esgoto. Então ele lhe enviou uma conta de 350 dólares. Agora imagine que seu amigo contratou o mesmo empreiteiro por 2 horas e que ele recebeu uma nota de 250 dólares. Se alguém lhe perguntasse quanto esperar em sua próxima fatura de remoção de sujeira você pode fazer algumas suposições razoáveis, como mais horas trabalhadas custam mais dólares. Você também pode presumir que há alguma carga básica e que o contratante cobra por hora. Se essas suposições forem verdadeiras, dados esses dois exemplos de dados, você já pode identificar a estrutura de preços do contratante: 100 dólares por hora mais 50 dólares para aparecer em sua casa. Se você acompanhou esse exemplo, então você já entendeu a ideia de alto nível por trás da regressão linear.

Neste caso, poderíamos produzir os parâmetros que correspondem exatamente aos preços do contratante. Às vezes isso não é possível, por exemplo, se alguma variação se deve a algum fator além dos dois citados. Nestes casos, tentaremos aprender modelos que minimizam a distância entre nossas previsões e os valores observados. Na maioria de nossos capítulos, vamos nos concentrar em minimizar a função de perda de erro quadrático. Como veremos mais adiante, essa perda corresponde ao pressuposto que nossos dados foram corrompidos pelo ruído gaussiano.

<ins>Classificação</ins>

Embora os modelos de regressão sejam ótimos para responder às questões quantos?, muitos problemas não se adaptam confortavelmente a este modelo. Por exemplo, um banco deseja adicionar a digitalização de cheques ao seu aplicativo móvel. Isso envolveria o cliente tirando uma foto de um cheque com a câmera do smartphone deles e o aplicativo precisaria ser capaz de entender automaticamente o texto visto na imagem. Especificamente, também precisaria entender o texto manuscrito para ser ainda mais robusto, como mapear um caractere escrito à mão a um dos personagens conhecidos. Este tipo de problema de qual? É chamado de classificação. É tratado com um conjunto diferente de algoritmos do que aqueles usados ​​para regressão, embora muitas técnicas sejam transportadas.

Na classificação, queremos que nosso modelo analise os recursos, por exemplo, os valores de pixel em uma imagem, e, em seguida, prever qual `categoria` (formalmente chamada de classe), entre alguns conjuntos discretos de opções, um exemplo pertence. Para dígitos manuscritos, podemos ter dez classes, correspondendo aos dígitos de 0 a 9. A forma mais simples de classificação é quando existem apenas duas classes, um problema que chamamos de `classificação binária`. Por exemplo, nosso conjunto de dados pode consistir em imagens de animais e nossos rótulos podem ser as classes { cat, dog }. Durante a regressão, buscamos um regressor para produzir um valor numérico, na classificação, buscamos um classificador, cuja saída é a atribuição de classe prevista.

Por razões que abordaremos à medida que o livro se torna mais técnico, pode ser difícil otimizar um modelo que só pode produzir uma tarefa categórica difícil, por exemplo, “gato” ou “cachorro”. Nesses casos, geralmente é muito mais fácil expressar nosso modelo na linguagem das probabilidades. Dados os recursos de um exemplo, nosso modelo atribui uma probabilidade para cada classe possível. Voltando ao nosso exemplo de classificação animal onde as classes são { gato, cachorro }, um classificador pode ver uma imagem e gerar a probabilidade que a imagem é um gato como 0,9. Podemos interpretar esse número dizendo que o classificador tem 90% de certeza de que a imagem representa um gato. A magnitude da probabilidade para a classe prevista transmite uma noção de incerteza. Esta não é a única noção de incerteza e discutiremos outros em capítulos mais avançados.

Quando temos mais de duas classes possíveis, chamamos o problema de `classificação multiclasse`. Exemplos comuns incluem reconhecimento de caracteres escritos à mão { 0,1,2,...,9,a,b,c,... }. Enquanto atacamos problemas de regressão tentando minimizar a função de perda de erro quadrático, a função de perda comum para problemas de classificação é chamada de `entropia cruzada (cross-entropy)`, cujo nome pode ser desmistificado por meio de uma introdução à teoria da informação nos capítulos subsequentes.

A classificação pode ser muito mais complicada do que apenas classificação binária, multi-classe ou mesmo com vários rótulos. Por exemplo, existem algumas variantes de classificação para abordar hierarquias. As hierarquias assumem que existem alguns relacionamentos entre as muitas classes. Portanto, nem todos os erros são iguais — se devemos errar, preferiríamos classificar incorretamente para uma classe parecida em vez de uma classe distante. Normalmente, isso é conhecido como `classificação hierárquica`.

<ins>Tags</ins>

O problema de aprender a prever classes que são não mutuamente exclusivas é chamado de classificação multi-rótulo. Os problemas de tags automáticas são geralmente mais bem descritos como problemas de classificação multi-rótulo. Pense nas tags que as pessoas podem aplicar a postagens em um blog técnico, por exemplo, “machine learning”, “tecnologia”, “gadgets”, “linguagens de programação”, “Linux”, “computação em nuvem”, “AWS”. Um artigo típico pode ter de 5 a 10 tags aplicadas porque esses conceitos estão correlacionados. Postagens sobre “computação em nuvem” provavelmente mencionarão “AWS” e postagens sobre “machine learning” também podem tratar de “linguagens de programação”.

Também temos que lidar com esse tipo de problema ao lidar com a literatura biomédica, onde etiquetar corretamente os artigos é importante porque permite que os pesquisadores façam revisões exaustivas da literatura. Na National Library of Medicine, vários anotadores profissionais revisam cada artigo que é indexado no PubMed para associá-lo aos termos relevantes do MeSH, uma coleção de aproximadamente 28000 tags. Este é um processo demorado e o os anotadores normalmente têm um atraso de um ano entre o arquivamento e a definição das tags. O machine learning pode ser usado aqui para fornecer tags provisórias até que cada artigo possa ter uma revisão manual adequada.

<ins>Busca</ins>

Às vezes, não queremos apenas atribuir cada exemplo a um valor real. No campo da recuperação de informações, queremos impor uma classificação a um conjunto de itens. Tome como exemplo a pesquisa na web. O objetivo é menos determinar se uma página específica é relevante para uma consulta, mas, em vez disso, qual dentre a infinidade de resultados de pesquisa é mais relevante para um determinado usuário. Nós realmente nos preocupamos com a ordem dos resultados de pesquisa relevantes e nosso algoritmo de aprendizagem precisa produzir subconjuntos ordenados de elementos de um conjunto maior. Em outras palavras, se formos solicitados a produzir as primeiras 5 letras do alfabeto, há uma diferença entre retornar “A B C D E” e “C A B E D”. Mesmo que o conjunto de resultados seja o mesmo, a ordenação dentro do conjunto importa.

Uma possível solução para este problema é primeiro atribuir para cada elemento no conjunto uma pontuação de relevância correspondente e, em seguida, para recuperar os elementos com melhor classificação. O molho secreto original por trás do mecanismo de pesquisa do Google foi um dos primeiros exemplos de tal sistema de pontuação, mas foi peculiar por não depender da consulta real. Aqui, eles contaram com um filtro de relevância simples para identificar o conjunto de itens relevantes e, em seguida, no PageRank para ordenar esses resultados que continham o termo de consulta. Hoje em dia, os mecanismos de pesquisa usam machine learning e modelos comportamentais para obter pontuações de relevância dependentes de consulta. Existem conferências acadêmicas inteiras dedicadas a este assunto.

<ins>Sistemas de Recomendação</ins>

Os sistemas de recomendação são outra configuração de problema que está relacionado à pesquisa e classificação. Os problemas são semelhantes na medida em que o objetivo é exibir um conjunto de itens relevantes para o usuário. A principal diferença é a ênfase em personalização para usuários específicos no contexto de sistemas de recomendação. Por exemplo, para recomendações de filmes, a página de resultados para um fã de ficção científica e a página de resultados para um conhecedor das comédias de Peter Sellers podem diferir significativamente. Problemas semelhantes surgem em outras configurações de recomendação, por exemplo, para produtos de varejo, música e recomendação de notícias.

Em alguns casos, os clientes fornecem `feedback explícito` comunicando o quanto eles gostaram de um determinado produto (por exemplo, as avaliações e resenhas de produtos na Amazon, IMDb e GoodReads). Em alguns outros casos, eles fornecem `feedback implícito`, por exemplo, pulando títulos em uma lista de reprodução, o que pode indicar insatisfação, mas pode apenas indicar que a música era inadequada no contexto. Nas formulações mais simples, esses sistemas são treinados para estimar alguma pontuação, como uma avaliação estimada ou a probabilidade de compra, dado um usuário e um item.

Dado esse modelo, para qualquer usuário, poderíamos recuperar o conjunto de objetos com as maiores pontuações, que pode então ser recomendado ao usuário. Os sistemas de produção são consideravelmente mais avançados e levam a atividade detalhada do usuário e características do item em consideração ao computar essas pontuações.

Apesar de seu enorme valor econômico, sistemas de recomendação ingenuamente construídos em cima de modelos preditivos sofrem algumas falhas conceituais graves. Para começar, observamos apenas feedback censurado: os usuários avaliam preferencialmente os filmes que os consideram fortes. Por exemplo, em uma escala de cinco pontos, você pode notar que os itens recebem muitas classificações de cinco e uma estrela mas que existem visivelmente poucas avaliações de três estrelas. Além disso, os hábitos de compra atuais são muitas vezes um resultado do algoritmo de recomendação atualmente em vigor, mas os algoritmos de aprendizagem nem sempre levam esse detalhe em consideração. Assim, é possível que se formem ciclos de feedback onde um sistema de recomendação preferencialmente empurra um item que então é considerado melhor (devido a maiores compras) e, por sua vez, é recomendado com ainda mais frequência. Muitos desses problemas sobre como lidar com a censura, incentivos e ciclos de feedback são importantes questões abertas de pesquisa.

<ins>Aprendizagem Sequencial</ins>

Até agora, vimos problemas em que temos algum número fixo de entradas a partir dos quais produzimos um número fixo de saídas. Por exemplo, consideramos prever os preços das casas a partir de um conjunto fixo de recursos: metragem quadrada, número de quartos, número de banheiros, tempo de caminhada até o centro. Também discutimos o mapeamento de uma imagem (de dimensão fixa) às probabilidades previstas de que pertence a cada de um número fixo de classes, ou pegando um ID de usuário e um ID de produto, e prever uma classificação por estrelas. Nesses casos, uma vez que alimentamos nossa entrada de comprimento fixo no modelo para gerar uma saída, o modelo esquece imediatamente o que acabou de ver.

Isso pode ser bom se todas as nossas entradas realmente tiverem as mesmas dimensões e se as entradas sucessivas realmente não têm nada a ver umas com as outras. Mas como lidaríamos com trechos de vídeo? Nesse caso, cada fragmento pode consistir em um número diferente de quadros. E nosso palpite sobre o que está acontecendo em cada quadro pode ser muito mais forte se levarmos em consideração os quadros anteriores ou posteriores. O mesmo vale para a linguagem. Um problema popular de deep learning é tradução automática: a tarefa de ingerir frases em algum idioma de origem e prevendo sua tradução em outro idioma.

Esses problemas também ocorrem na medicina. Podemos querer um modelo para monitorar pacientes na unidade de terapia intensiva e disparar alertas se seus riscos de morte nas próximas 24 horas excederem algum limite. Definitivamente, não queremos que este modelo jogue fora tudo o que sabe sobre o histórico do paciente a cada hora e apenas fazer suas previsões com base nas medições mais recentes.

Esses problemas estão entre as aplicações mais interessantes de machine learning e são instâncias de aprendizagem sequencial. Eles exigem um modelo para ingerir sequências de entradas ou para emitir sequências de saídas (ou ambos). Especificamente, sequência para aprendizagem de sequencial considera os problemas onde entrada e saída são sequências de comprimento variável, como tradução automática e transcrição de texto da fala falada. Embora seja impossível considerar todos os tipos de transformações de sequência, vale a pena mencionar os seguintes casos especiais.

<b>Marcação e análise.</b> Isso envolve anotar uma sequência de texto com atributos. Em outras palavras, o número de entradas e saídas é essencialmente o mesmo. Por exemplo, podemos querer saber onde estão os verbos e os sujeitos. Como alternativa, podemos querer saber quais palavras são as entidades nomeadas. Em geral, o objetivo é decompor e anotar o texto com base na estrutura e suposições gramaticais para obter algumas anotações. Isso parece mais complexo do que realmente é. Abaixo está um exemplo muito simples de uma frase anotada com marcas que indicam quais palavras se referem a entidades nomeadas (marcadas como “Ent”).

```
Tom has dinner in Washington with Sally
Ent  -    -    -     Ent      -    Ent
```
<b>Reconhecimento automático de fala.</b> Com o reconhecimento de fala, a sequência de entrada é uma gravação de áudio de um alto-falante (mostrado em: numref: fig_speech), e a saída é a transcrição textual do que o locutor disse. O desafio é que existem muito mais quadros de áudio (o som é normalmente amostrado em 8kHz ou 16kHz) do que texto, ou seja, não há correspondência 1: 1 entre áudio e texto, já que milhares de amostras podem correspondem a uma única palavra falada. Estes são problemas de aprendizagem de sequência a sequência em que a saída é muito mais curta do que a entrada.

![image](https://github.com/user-attachments/assets/babf5eeb-7535-4b24-b5ec-910f206dfe4c)

<b>Text to Speech (Texto para fala).</b> Este é o inverso do reconhecimento automático de fala. Em outras palavras, a entrada é um texto e a saída é um arquivo de áudio. Nesse caso, a saída é muito mais longa do que a entrada. Embora seja fácil para os humanos reconhecerem um arquivo de áudio ruim, isso não é tão trivial para computadores.

<b>Tradução por máquina.</b> Ao contrário do caso do reconhecimento de voz, onde correspondente entradas e saídas ocorrem na mesma ordem (após o alinhamento), na tradução automática, a inversão da ordem pode ser vital. Em outras palavras, enquanto ainda estamos convertendo uma sequência em outra, nem o número de entradas e saídas, nem o pedido de exemplos de dados correspondentes são considerados iguais. Considere o seguinte exemplo ilustrativo da tendência peculiar dos alemães para colocar os verbos no final das frases.

```
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Did you already check out this excellent tutorial?
Wrong alignment:  Did you yourself already this excellent tutorial looked-at?
```
Muitos problemas relacionados surgem em outras tarefas de aprendizagem. Por exemplo, determinar a ordem em que um usuário lê uma página da web é um problema de análise de layout bidimensional. Problemas de diálogo apresentam todos os tipos de complicações adicionais, onde determinar o que dizer a seguir requer levar em consideração conhecimento do mundo real e o estado anterior da conversa através de longas distâncias temporais. Estas são áreas ativas de pesquisa.

#### Aprendizagem Não Supervisionada

