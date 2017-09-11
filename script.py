import pandas as pd
import scipy as sp
# from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

animes_df = pd.read_csv('data/anime.csv', na_values=["nan", "Unknown", "NA", ""])
ratings_df = pd.read_csv ('data/rating.csv', na_values=["nan", "Unknown", "NA", "", -1])
ratings_df = ratings_df[ratings_df.user_id <= 15000]

genres_list = []
genres_col_list = animes_df.genre.tolist()

for item in genres_col_list:
  if isinstance(item, basestring):
    l = item.split(",")
    # remove white spaces
    l = map(str.strip, l)
    genres_list.extend(l)


c = Counter(genres_list)
genres_df = pd.DataFrame.from_dict(c, orient='index').rename(columns={'įndex': 'genre', 0:'count'})
genres_df.head(20)

animes_watched = ratings_df.anime_id.unique()
qtd_animes_watched = len(ratings_df.anime_id.unique())

animes_df[~animes_df.anime_id.isin(animes_watched)].head(10)

n_users = ratings_df.user_id.unique().shape[0]
n_items = ratings_df.anime_id.unique().shape[0]
ratings_matrix = np.zeros((n_users, n_items))

for row in ratings_df.itertuples():
    ratings_matrix[row[1]]



ratings_df.rating = ratings_df.rating.fillna(0)
user_u = list(ratings_df.user_id.unique())
anime_u = list(ratings_df.anime_id.unique())

data = ratings_df['rating'].tolist()
row = ratings_df.user_id.astype('category', categories=user_u).cat.codes
col = ratings_df.anime_id.astype('category', categories=anime_u).cat.codes
sparse_matrix = sp.sparse.csr_matrix((data, (row, col)), shape=(len(user_u), len(anime_u)))

sparse_matrix 

ratings_matrix = sparse_matrix.todense()


sparsity  = float(len(ratings_matrix.nonzero()[0]))  
sparsity  /=  (ratings_matrix.shape[0] * ratings_matrix.shape[1]) 
sparsity  *=  100 
print('Sparsity:  {:4.2f}%'.format(sparsity)) 
