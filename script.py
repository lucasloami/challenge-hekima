import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

animes_df = pd.read_csv('data/anime.csv', na_values=["nan", "Unknown", "NA", ""])
ratings_df = pd.read_csv ('data/rating.csv', na_values=["nan", "Unknown", "NA", "", -1])
# animes_df.head(10)

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