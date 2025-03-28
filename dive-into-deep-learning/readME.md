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

#### Funções Objetivo
