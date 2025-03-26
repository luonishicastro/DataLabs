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

## 🐖 Notação
A notação usada ao longo deste livro é resumida a seguir.

* **Números:**
  - **Moeda:** um escalar;
  - **Lastro:** um vetor;
  - **Lançamentos:** uma matrix;


_A ideia do Saldo é de que haja uma divergência positiva e crescente da renda em relação às despesas no tempo._


## 🦕 Introdução

