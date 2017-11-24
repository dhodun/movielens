import collections
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import csv

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import google.datalab.bigquery as bq
from scipy.sparse import coo_matrix

from tensorflow.contrib.factorization import WALSModel


def load_ratings(table_name):
    # returns all user / movie ratings in the table_name

    query = "SELECT DISTINCT userID AS user, movieID AS item, rating AS rating  FROM `{}`".format(table_name)
    output_ratings = bq.Query(query)
    df = output_ratings.execute(output_options=bq.QueryOutput.dataframe()).result()

    return df


def print_datasets(datasets):
    out = ''
    for dataset in datasets:
        out += 'Users: {:,d}\n'.format(len(dataset["user"].unique()))
        out += 'Items: {:,d}\n'.format(len(dataset["item"].unique()))
        out += 'Ratings: {:,d}\n\n'.format(dataset.shape[0])
    print(out)


def create_mappings(ratings):
    user_encoder = preprocessing.LabelEncoder()
    user_encoder.fit(ratings['user'])
    movie_encoder = preprocessing.LabelEncoder()
    movie_encoder.fit(ratings['item'])

    return user_encoder, movie_encoder


def create_test_and_train(ratings):
    train, test = train_test_split(ratings, train_size=.8, random_state=43)
    print('Data Sets BEFORE pruning:')
    print_datasets([train, test])

    # only items / movies in train will be available for validation
    test_prune = test[test['item'].isin(train['item'])]

    print('Data Sets DURING pruning:')
    print_datasets([train, test_prune])

    # only users in train will be available for validation
    test_prune = test_prune[test_prune['user'].isin(train['user'])]

    print('Data Sets AFTER pruning:')
    print_datasets([train, test_prune])

    return train, test_prune


def create_sparse_sets(train, test, num_users, num_movies):
    train_sparse = coo_matrix(train['rating'], (train['user'], train['item']))
    test_sparse = coo_matrix(test['rating'], (test['user'], test['item']))
    return train_sparse, test_sparse


def train_model(train_sparse, test_sparse, num_users, num_movies):

    num_factors = 5
    regularization = 1e-1

    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # create model
    model = WALSModel(
            num_users,
            num_movies,
            num_factors,
            regularization=regularization,
            unobserved_weight=0)


    with tf.Graph().as_default(), tf.Session() as sess:

    def als_model(self, dataset):
        return WALSModel(
            len(dataset["visitorid"].unique()),
            len(dataset["itemid"].unique()),
            self.num_factors,
            regularization=self.regularization,
            unobserved_weight=0)

    hypertune = args['hypertune']
    dim = args['latent_factors']
    num_iters = args['num_iters']
    reg = args['regularization']
    unobs = args['unobs_weight']
    wt_type = args['wt_type']
    feature_wt_exp = args['feature_wt_exp']
    obs_wt = args['feature_wt_factor']

    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # generate model
    input_tensor, row_factor, col_factor, model = \
        wals.wals_model(tr_sparse, dim, reg, unobs, args['weights'], wt_type, feature_wt_exp, obs_wt)

    # factorize matrix
    session = wals.simple_train(model, input_tensor, num_iters)

    tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # evaluate output factor matrices
    output_row = row_factor.eval(session=session)
    output_col = col_factor.eval(session=session)

    # close the training session now that we've evaluated the output
    session.close()



    def train(self, model, input_matrix, verbose=False):
        rmse_op = self.rmse_op(model, input_matrix) if verbose else None

        row_update_op = model.update_row_factors(sp_input=input_matrix)[1]
        col_update_op = model.update_col_factors(sp_input=input_matrix)[1]

        model.initialize_op.run()
        model.worker_init.run()
        for _ in range(self.num_iters):
            # Update Users
            model.row_update_prep_gramian_op.run()
            model.initialize_row_update_op.run()
            row_update_op.run()
            # Update Items
            model.col_update_prep_gramian_op.run()
            model.initialize_col_update_op.run()
            col_update_op.run()

            if verbose:
                print('RMSE: {:,.3f}'.format(rmse_op.eval()))

    return output_row, output_col


def wals_model():

    pass
