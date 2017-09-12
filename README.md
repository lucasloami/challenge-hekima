Este repositório contém o projeto elaborado para a Hekima de análise exploratória de um dataset, criação e avaliação de um algoritmo de predição. Este projeto foi montado seguindo as premissas de [Pesquisas Reprodutíveis](https://pt.coursera.org/learn/reproducible-research), de modo que qualquer pessoa consiga chegar aos mesmos resultados que eu utilizando os passos que segui no Jupyter Notebook.

Foi escolhido o Dataset de [recomendações de Animes](https://www.kaggle.com/CooperUnion/anime-recommendations-database) do Kaggle. Ele contém dois datasets principais:
* `anime.csv`: contém informações descritivas de diversos animes
* `rating.csv`: contém informações sobre as avaliações feitas por usuários a animes

# Dependências do projeto

Todas as dependências podem ser encontradas no arquivo `requirements.txt`, mas abaixo estão listadas:
* Numpy
* Scikit-Learn
* Pandas
* Surprise (biblioteca do Python para sistemas de recomendação)
* Jupyter Notebook


Para instalar as dependências execute na pasta raiz do projeto: `pip install -r requirements.txt`. 

Para acessar o Jupyter Notebook que criei, execute na pasta raiz do projeto `jupyter notebook`. Logo em seguida seu browser será aberto e basta selecionar o arquivo "Sistema de Recomendação.ipynb". 

# Estrutura do projeto

```{sh}
  .
  |-report
  |  |- Report - Sistema de Recomendação.html
  |-data
  |- Sistema de Recomendação.ipynb
  |- requirements.txt
  |- script.py
```

A pasta `data` contém dos datasets utilizados no projeto. Caso você queira reproduzir o projeto na sua máquina, é importante que os dados sejam baixados do link adequado (presente na primeira seção deste README) e adicionados a esta pasta.

A pasta `report` contém um arquivo html com uma versão do relatório gerado a partir do estudo feito nesse projeto. Esse arquivo contém **todos os insights e estudos feitos, bem como uma descrição detalhada de como foi elaborado o sistema de recomendação**

O arquivo **script.py** contém uma versão manual do sistema de recomendação focada apenas na avaliação do desempenho do mesmo diante de um conjunto de dados de teste. Nesta versão não são feitas recomendações efetivamente, mas ela demonstra a compreensão e entendimento do autor do projeto sobre a lógica de construção. 

**Todas as referências utilizadas para a criação desse projeto estão descritas no report**
