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

import util


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


def make_weights(data, weight_factor, axis):
    # first calculate how many of each factor there is to divide weights by
    fraction = np.array(1.0 / (data > 0).sum(axis))

    # fill in NaNs with 0
    fraction[np.ma.masked_invalid(fraction).mask] = 0

    # multiply by factor and flatten
    weights = np.array(fraction * weight_factor).flatten()

    return weights



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


def train_model(train_sparse, test_sparse, num_users, num_movies, args, verbose=False):
    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    with tf.Graph().as_default(), tf.Session() as sess:

        row_weights = np.ones(num_users)
        col_weights = np.ones(num_movies)

        if args.col_weight_bool:
            col_weights = make_weights(train_sparse, args.col_weight_factor, axis=0)

        if args.row_weight_bool:
            row_weights = make_weights(train_sparse, args.row_weight_factor, axis=1)

        # create model
        model = WALSModel(
            num_users,
            num_movies,
            args.num_factors,
            regularization=args.regularization,
            unobserved_weight=args.unobserved_weight,
            row_weights=row_weights,
            col_weights=col_weights)

        # create sparse tensor

        input_tensor = tf.SparseTensor(indices=zip(train_sparse.row, train_sparse.col),
                                       values=(train_sparse.data).astype(np.float32),
                                       dense_shape=train_sparse.shape)

        test_tensor = tf.SparseTensor(indices=zip(test_sparse.row, test_sparse.col),
                                      values=(test_sparse.data).astype(np.float32),
                                      dense_shape=test_sparse.shape)

        # train model

        rmse_op = rmse(model, input_tensor) if verbose else None
        rmse_test_op = rmse(model, test_tensor)

        row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
        col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

        model.initialize_op.run()
        model.worker_init.run()
        for _ in range(args.epochs):
            # Update Users
            model.row_update_prep_gramian_op.run()
            model.initialize_row_update_op.run()
            row_update_op.run()
            # Update Items
            model.col_update_prep_gramian_op.run()
            model.initialize_col_update_op.run()
            col_update_op.run()

            if verbose:
                train_metric = rmse_op.eval()
                test_metric = rmse_test_op.eval()
                tf.logging.info('RMSE Train: {:,.3f}'.format(train_metric))
                tf.logging.info('RMSE Test:  {:,.3f}'.format(test_metric))
                # TODO Collect these in variable for graphing later

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
    model_dir = os.path.join(args.output_dir, 'model')

    # save localy, then try to copy to GCS
    np.save('row_factor', output_row)
    np.save('col_factor', output_col)

    # copy to GCS?
    util.copy_file_to_gcs(model_dir, 'row_factor.npy')
    util.copy_file_to_gcs(model_dir, 'col_factor.npy')

    # these lines did not work in TF 1.2, worked in TF1.3, but CMLE did not support 1.3 w/ GPU at time, even if we
    # overloaded the setup.py

    # np.save(file_io.FileIO(os.path.join(output_dir, 'row_factor'), 'w'), output_row)
    # np.save(file_io.FileIO(os.path.join(output_dir, 'col_factor'), 'w'), output_col)
    return None
