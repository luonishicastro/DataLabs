referência: https://pt.d2l.ai/chapter_preface/index.html

# 💸 Prefácio
Testar o potencial do deep learning apresenta desafios únicos porque qualquer aplicativo reúne várias disciplinas. Aplicar o deep learning requer compreensão simultânea (i) as motivações para definir um problema de uma maneira particular; (ii) a matemática de uma dada abordagem de modelagem; (iii) os algoritmos de otimização para ajustar os modelos aos dados; e (iv) a engenharia necessária para treinar modelos de forma eficiente, navegando nas armadilhas da computação numérica e obter o máximo do hardware disponível. Ensinar as habilidades de pensamento crítico necessárias para formular problemas, a matemática para resolvê-los e as ferramentas de software para implementar tais soluções em um só lugar apresentam desafios formidáveis.

Propusemo-nos a criar um recurso que pudesse (i) estar disponível gratuitamente para todos; (ii) oferecer profundidade técnica suficiente para fornecer um ponto de partida no caminho para realmente se tornar um cientista de machine learning aplicado; (iii) incluir código executável, mostrando aos leitores como resolver problemas na prática; (iv) permitir atualizações rápidas, tanto por nós e também pela comunidade em geral; e (v) ser complementado por um [fórum]

## 💡 Definições
* **Trabalho:** é a atividade essencial para a geração de `Riqueza`;
  - **Moeda:** é um meio de troca de recursos;
  - **Lastro:** é o valor tangivel que da suporte a um `Ativo`;
  - **Lançamentos:** são movimentações financeiras;
* **Renda:** se refere a receita ou ao dinheiro que ganho ou recebido;
  - **Ativo:** é um recurso que gera receita;
    - **Câmbio:**
    - **Titulo de Divida:** confere ao detentor um direito a receber juros ate o pagamento da dívida;
      - **Conta Bancária:**
      - **Renda Fixa:** Titulos Públicos, Certificado de Depósito Bancário, Debêntures;
    - **Título de Propriedade:** confere ao detentor posse sob um ativo;
      - **Renda Variavel:** Ações;
      - **Fundos de Investimentos:** Cotas de Fundos de Investimentos;
  - **Remuneração:** é a geração de receita em decorrência do `Trabalho`;
  - **Crédito:** é um recurso financeiro que se toma emprestado;
* **Despesa:** se refere ao dinheiro que é gasto;
  - **Gastos:** são os elementos que compõem as `Despesas`. Convém classificar os gastos em Variaveis ou Fixos;
    - **Tipo de Gastos:** Discricionários, Essenciais, Financeiro, Fixas, Outros, Variáveis;
    - **Custo:** representa todos os `Gastos` necessários para sobreviver;
  - **Passivo:** é um recurso que gera despesa;
  - **Dívida:** é o contraponto do `Crédito`, é uma despesa futura;
* **Patrimônio:** se refere ao dinheiro disponível, depois de convertidos os bens e direitos;
  - **Investimento:** utilização do `Patrimônio` para fazê-lo crescer;
    - **Aporte:** é o dinheiro disponível utilizado como `Investimento`;
    - **Retorno:** é o resultado do Investimento, a conclusão do processo;
      - **Lucro:** é o `Retorno` financeiro positivo no periodo;
      - **Prejuízo:** é o `Retorno` financeiro negativo no periodo;
* **Risco:** é tudo aquilo que representa um incerteza para todos os elementos descritos acima;
  - **Risco Calculado:** probabilidade de um resultado ou ganhos reais de um `Investimento` sejam diferentes de um resultado ou `Retorno Esperado`;



## 🧮 Fórmulas
### Medidas de Controle Financeiro
```
Colchão de Segurança = 6 x Custo de Vida Mensal
```
```
Objetivo Financeiro 1 = Renda Média dos Ultimos 12 meses x ((Expectativa Media de Vida em anos - Idade) X 12)
```
```
Graus de Liberdade Financeira

Variáveis:
Q - qualidade de vida
t - tempo livre útil
T - tempo dedicado ao trabalho

Patrimônio Necessário = (Gasto Fixo + Qualidade de Vida) / Taxa de Retorno

Grau de Independência Financeira = Patrimônio Atual / Patrimônio Necessário
```
_Dizemos que possuímos independência financeira quando conseguimos acumular um patrimonio que gere, atraves do rendimento de juros, um valor que possa cobrar toda sas nossas necessidades de gastos._

_Capacidade de fazer seus proprios investimentos, ou seja, ter a capcidade de avaliar e escolher, de forma prática, onde colocar o seu dinheiro sem precisar delegar essa função a ninguém._

```
Limite de Crédito
```
### Medidas de Desempenho Financeiro
```
Saldo Líquido = Rendas - Despesas
```
_A ideia do Saldo é de que haja uma divergência positiva e crescente da renda em relação às despesas no tempo._
```
Rendimento = Total Investido / (Total Investido + Retorno)
```
```
Índice de Sharpe
```
_É uma medida de avaliação da performance de um portfolio, ativo ou fundo de investimento, ajustada a sua volatilidade (ou risco de mercado). Seu calculo consiste na subtração entre o retorno do ativo analisado e a taxa livre de risco, dividida pela volatilidade deste ativo. Quanto maior o Sharpe mais eficiente é a relação entre risco e retorno do portfolio analisado. Caso o indice seja inferior a zero significa que o ativo mensurado corre mais risco e entrega menos retorno do que a taxa livre de risco, em outras palavras, não valeria correr o seu risco para ganhar menos do que o pago por um ativo que não trará oscilações._
### Medidas de Mensuração de Incerteza
```
