import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqtr

# LOAD DATA
animes_df = pd.read_csv('data/anime.csv', na_values=["nan", "Unknown", "NA", ""])
ratings_df = pd.read_csv ('data/rating.csv', na_values=["nan", "Unknown", "NA", "", -1])


# PREPARE DATASET TO THE STUDIED
dataset = ratings_df.merge(animes_df, on='anime_id', suffixes=['_user', ''])
dataset = dataset[['user_id', 'name', 'rating_user']]
dataset_sub = dataset[dataset.user_id <= 5000]
dataset_sub = dataset_sub[dataset_sub.rating_user.notnull()]


# CREATES A SPARSE MATRIX CONTAINING A USER-ANIME-RATING MATRIX
user_u = list(dataset_sub.user_id.unique())
anime_u = list(dataset_sub.name.unique())

data = dataset_sub['rating_user'].tolist()
row = dataset_sub.user_id.astype('category', categories=user_u).cat.codes
col = dataset_sub.name.astype('category', categories=anime_u).cat.codes
sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_u), len(anime_u)))

# CALCULATE THE SPARSITY OF THE MATRIX
ratings_matrix = sparse_matrix.todense()
sparsity  = float(len(ratings_matrix.nonzero()[0]))
sparsity  /=  (ratings_matrix.shape[0]  * ratings_matrix.shape[1])
sparsity  *=  100
print('Sparsity:  {:4.2f}%'.format(sparsity))

# SPLIT DATA INTO TRAIN AND TEST SET
# ratings_train, ratings_test = train_test_split(ratings_matrix, test_size=0.25,  random_state=7)
ratings_train, ratings_test = train_test_split(sparse_matrix, test_size=0.25,  random_state=7)

# CALCULATE COSINE SIMILARITY
user_sim = cosine_similarity(ratings_train)

def predict(ratings, similarity):
  mean_user_rating = ratings.mean(axis=1)
  ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
  pred = mean_user_rating + similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
  return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

user_prediction = predict(ratings_train, user_sim)
# print 'Train User-based CF RMSE: ' + str(rmse(user_prediction, ratings_train))
print 'Test User-based CF RMSE: ' + str(rmse(user_prediction, ratings_test))
