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
