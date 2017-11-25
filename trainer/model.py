import datetime
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.contrib.factorization import WALSModel
from tensorflow.python.lib.io import file_io


def load_ratings(filename):
    # returns all user / movie ratings in the table_name

    df = pd.read_csv(file_io.FileIO(filename, 'r'))
    df.columns = ['user', 'item', 'rating', 'timestamp']
    df.drop('timestamp', axis=1)

    return df


def log_datasets(datasets):
    out = ''
    for dataset in datasets:
        out += 'Users: {:,d}\n'.format(len(dataset["user"].unique()))
        out += 'Items: {:,d}\n'.format(len(dataset["item"].unique()))
        out += 'Ratings: {:,d}\n\n'.format(dataset.shape[0])
    tf.logging.info(out)


def create_mappings(ratings):
    user_encoder = preprocessing.LabelEncoder()
    user_encoder.fit(ratings['user'])
    movie_encoder = preprocessing.LabelEncoder()
    movie_encoder.fit(ratings['item'])

    return user_encoder, movie_encoder


def create_test_and_train(ratings):
    train, test = train_test_split(ratings, train_size=.8, random_state=43)
    tf.logging.info('Data Sets BEFORE pruning:')
    log_datasets([train, test])

    # only items / movies in train will be available for validation
    test_prune = test[test['item'].isin(train['item'])]

    tf.logging.info('Data Sets DURING pruning:')
    log_datasets([train, test_prune])

    # only users in train will be available for validation
    test_prune = test_prune[test_prune['user'].isin(train['user'])]

    tf.logging.info('Data Sets AFTER pruning:')
    log_datasets([train, test_prune])

    return train, test_prune


def create_sparse_sets(train, test, num_users, num_movies):
    train_sparse = coo_matrix((train['rating'].as_matrix(),
                               (train['user'].as_matrix(), train['item'].as_matrix())), shape=(num_users, num_movies))
    test_sparse = coo_matrix((test['rating'].as_matrix(),
                              (test['user'].as_matrix(), test['item'].as_matrix())), shape=(num_users, num_movies))

    return train_sparse, test_sparse


def rmse(model, input_tensor):
    approx_matrix = approx_sparse(model, input_tensor.indices, input_tensor.dense_shape)
    err = tf.sparse_add(input_tensor, approx_matrix * (-1))
    err2 = tf.square(err)
    n = input_tensor.values.shape[0].value
    return tf.sqrt(tf.sparse_reduce_sum(err2) / n)


def approx_sparse(model, indices, shape):
    row_factors = tf.nn.embedding_lookup(
        model.row_factors,
        tf.range(model._input_rows),
        partition_strategy="div")
    col_factors = tf.nn.embedding_lookup(
        model.col_factors,
        tf.range(model._input_cols),
        partition_strategy="div")

    row_indices, col_indices = tf.split(indices,
                                        axis=1,
                                        num_or_size_splits=2)
    gathered_row_factors = tf.gather(row_factors, row_indices)
    gathered_col_factors = tf.gather(col_factors, col_indices)
    approx_vals = tf.squeeze(tf.matmul(gathered_row_factors,
                                       gathered_col_factors,
                                       adjoint_b=True))

    return tf.SparseTensor(indices=indices,
                           values=approx_vals,
                           dense_shape=shape)


def train_model(train_sparse, test_sparse, num_users, num_movies, verbose=False):
    num_factors = 10
    regularization = 1e-1
    epochs = 10

    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    with tf.Graph().as_default(), tf.Session() as sess:

        # create model
        model = WALSModel(
            num_users,
            num_movies,
            num_factors,
            regularization=regularization,
            unobserved_weight=0)

        # create sparse tensor

        input_tensor = tf.SparseTensor(indices=zip(train_sparse.row, train_sparse.col),
                                       values=(train_sparse.data).astype(np.float32),
                                       dense_shape=train_sparse.shape)

        # train model

        rmse_op = rmse(model, input_tensor) if verbose else None

        row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
        col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

        model.initialize_op.run()
        model.worker_init.run()
        for _ in range(epochs):
            # Update Users
            model.row_update_prep_gramian_op.run()
            model.initialize_row_update_op.run()
            row_update_op.run()
            # Update Items
            model.col_update_prep_gramian_op.run()
            model.initialize_col_update_op.run()
            col_update_op.run()

            if verbose:
                tf.logging.info('RMSE: {:,.3f}'.format(rmse_op.eval()))

        row_factor = model.row_factors[0].eval()
        col_factor = model.col_factors[0].eval()

    tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    return row_factor, col_factor


def get_rmse(output_row, output_col, actual):
    mse = 0
    for i in xrange(actual.data.shape[0]):
        row_pred = output_row[actual.row[i]]
        col_pred = output_col[actual.col[i]]
        err = actual.data[i] - np.dot(row_pred, col_pred)
        mse += err * err
    mse = mse / actual.data.shape[0]
    rmse = math.sqrt(mse)
    return rmse


def save_model(args, output_row, output_col):
    model_dir = os.path.join(args.job_dir, 'model')
    np.save(file_io.FileIO(os.path.join(model_dir, 'row_factor'), 'w'), output_row)
    np.save(file_io.FileIO(os.path.join(model_dir, 'col_factor'), 'w'), output_col)
    return None
