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

import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

animes_df = pd.read_csv('data/anime.csv', na_values=["nan", "Unknown", "NA", ""])
ratings_df = pd.read_csv ('data/rating.csv', na_values=["nan", "Unknown", "NA", "", -1])

animes_df2 = animes_df[animes_df.type == 'TV']

dataset = ratings_df.merge(animes_df2, on='anime_id', suffixes=['_user', ''])

dataset = dataset[['user_id', 'name', 'rating_user']]
dataset_sub = dataset[dataset.user_id <= 5000]
dataset_sub = dataset_sub[dataset_sub.rating_user.notnull()]

merged_sub = dataset_sub




user_u = list(dataset_sub.user_id.unique())
anime_u = list(dataset_sub.name.unique())

data = dataset_sub['rating_user'].tolist()
row = dataset_sub.user_id.astype('category', categories=user_u).cat.codes
col = dataset_sub.name.astype('category', categories=anime_u).cat.codes
sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_u), len(anime_u)))

sparse_matrix

# ratings_matrix = sparse_matrix.todense()

# sparsity  = float(len(ratings_matrix.nonzero()[0]))
# sparsity  /=  (ratings_matrix.shape[0]  * ratings_matrix.shape[1])
# sparsity  *=  100
# print('Sparsity:  {:4.2f}%'.format(sparsity))

ratings_train, ratings_test = train_test_split(ratings_matrix, test_size=0.25,  random_state=7)
# ratings_train, ratings_test = train_test_split(sparse_matrix, test_size=0.25,  random_state=7)

user_sim = cosine_similarity(ratings_train)
user_pred = user_sim.dot(ratings_train)/np.array([user_sim.sum(axis=1)]).T

def get_rmse(pred, actual):
  #Ignore nonzero terms.
  pred  = pred[actual.nonzero()].flatten()
  actual  = actual[actual.nonzero()].flatten()
  return sqrt(mean_squared_error(pred,  actual))

get_rmse(user_pred, ratings_test)


def predict(ratings, similarity, type='user'):
  if type == 'user':
    mean_user_rating = ratings.mean(axis=1)
    #You use np.newaxis so that mean_user_rating has same format as ratings
    ratings_diff = (ratings - mean_user_rating)
    pred = mean_user_rating + similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
  return pred

user_prediction = predict(ratings_train, user_sim, type='user')


from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print 'Train User-based CF RMSE: ' + str(rmse(user_prediction, ratings_train))
print 'Test User-based CF RMSE: ' + str(rmse(user_prediction, ratings_test))
