#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:24:05 2018

@author: jason
"""

import numpy as np
import pandas as pd
import random
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from collections import defaultdict
from surprise import SVD
from surprise import dataset 
from surprise import accuracy
from surprise import prediction_algorithms
from surprise import dump


df = pd.read_csv('beer_reviews1.csv')

# Change all of the user names to integers
df['review_profilename'] = pd.factorize(df['review_profilename'])[0] + 1

# Filter out beer that has not been reviewed 50 times.
df = df.groupby('beer_beerid').filter(lambda x: len(x) >= 50)

reader = dataset.Reader(rating_scale=(1, 5))


data = Dataset.load_from_df(df[['review_profilename', 'beer_beerid', 'review_taste']], reader)
raw_ratings = data.raw_ratings

# Split data 
threshold = int(.90 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = A_raw_ratings  # data is now the set A

def get_top_n(predictions, n=10):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


# retrain on the whole set A
algo = SVD()
trainset = data.build_full_trainset()
algo.train(trainset)

# Compute biased accuracy on A
testset = data.construct_testset(A_raw_ratings)  # testset is now the set A
predictions = algo.test(trainset.build_testset())
print('Biased accuracy on A,', end='   ')
accuracy.rmse(predictions)

# predict r for all pairs (user, item) that are NOT in the training set
# by setting the pairs that were to 0 and the pairs that were not in the
# training set to mean of all ratings.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Compute unbiased accuracy on B
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)
print('Unbiased accuracy on B,', end=' ')
accuracy.rmse(predictions)