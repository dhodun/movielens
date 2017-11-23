import collections
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import csv

import google.datalab.bigquery as bq


from tensorflow.contrib.factorization import WALSModel

Rating = collections.namedtuple('Rating', ['user_id', 'item_id', 'rating'])


class Dataset(collections.namedtuple('Dataset', ['users', 'items', 'ratings'])):
    # users: set[str]
    # items: set[str]
    # ratings: list[Rating]

    __slots__ = ()

    def __str__(self):
        out = 'Users: {:,d}\n'.format(self.n_users)
        out += 'Items: {:,d}\n'.format(self.n_items)
        out += 'Ratings: {:,d}\n'.format(self.n_ratings)
        return out

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    @property
    def n_ratings(self):
        return len(self.ratings)

    def user_ratings(self, user_id):
        return list(r for r in self.ratings if r.user_id == user_id)

    def item_ratings(self, item_id):
        return list(r for r in self.ratings if r.item_id == item_id)

    def filter_ratings(self, users, items):
        return list(((r.user_id, r.item_id), r.rating)
                    for r in self.ratings
                    if r.user_id in users
                    and r.item_id in items)


def new_dataset(ratings):
    users = set(r.user_id for r in ratings)
    items = set(r.item_id for r in ratings)
    return Dataset(users, items, ratings)


def load_events(table_name, percentage=1):
    # returns all transaction and addtocart events, but distinct on visitorid and itemid (i.e. no difference between 1 event and 5)
    query = "SELECT DISTINCT userID as visitorid, movieID as itemid, rating  FROM `{}`".format(table_name)
    output_ratings = bq.Query(query)
    df = output_ratings.execute(output_options=bq.QueryOutput.dataframe()).result()
    df['rating'] = pd.to_numeric(df['rating']).astype(float)
    df.head(10)

    return df


def load_movielens_ratings(dataset_path):
    ratings_csv = os.path.join(dataset_path, 'ratings.csv')
    if not os.path.isfile(ratings_csv):
        raise Exception('File not found: \'{}\''.format(ratings_csv))
    ratings = list()
    with open(ratings_csv, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for user_id, item_id, rating, timestamp in reader:
            ratings.append(Rating(user_id,
                                  item_id,
                                  float(rating),
                                  int(timestamp)))
    return ratings


def load_movielens(dataset_path):
    if not os.path.isdir(dataset_path):
        raise Exception('Path not found: \'{}\''.format(dataset_path))

    ratings = load_movielens_ratings(dataset_path)
    dataset = new_dataset(ratings)

    return dataset


def print_datasets(datasets):
    out = ''
    for dataset in datasets:
        out += 'Users: {:,d}\n'.format(len(dataset["visitorid"].unique()))
        out += 'Items: {:,d}\n'.format(len(dataset["itemid"].unique()))
        out += 'Ratings: {:,d}\n\n'.format(dataset.shape[0])
    print(out)


dataset = load_events('dhodun1.cs229_movielens.ratings_small')

dataset.head(5)

from sklearn.model_selection import train_test_split

train_valid, test = train_test_split(dataset, train_size=.8, random_state=43)
train, valid = train_test_split(train_valid, train_size=.8, random_state=43)

print("# Train Reviews: {}".format(train.shape[0]))
print("# Valid Reviews: {}".format(valid.shape[0]))
print("# Test Reviews: {}".format(test.shape[0]))

print_datasets([train, valid, test])

# only items in train will be available for validation/test

valid_prune = valid[valid["itemid"].isin(train["itemid"])]
test_prune = test[test["itemid"].isin(train["itemid"])]

# only users in train are available for validation/test

valid_prune = valid_prune[valid_prune["visitorid"].isin(train["visitorid"])]
test_prune = test_prune[test_prune["visitorid"].isin(train["visitorid"])]

print_datasets([train, valid_prune, test_prune])

# Map User <-> index
# Map Item <-> index
IndexMapping = collections.namedtuple('IndexMapping', ['users_to_idx',
                                                       'users_from_idx',
                                                       'items_to_idx',
                                                       'items_from_idx'])


def map_index(values):
    values_from_idx = dict(enumerate(values))
    values_to_idx = dict((value, idx) for idx, value in values_from_idx.items())
    return values_to_idx, values_from_idx


def new_mapping(dataset):
    users_to_idx, users_from_idx = map_index(dataset["visitorid"].unique())
    items_to_idx, items_from_idx = map_index(dataset["itemid"].unique())
    return IndexMapping(users_to_idx, users_from_idx, items_to_idx, items_from_idx)


import tensorflow as tf
import numpy as np

from tensorflow.contrib.factorization import WALSModel


class ALSRecommenderModel:
    def __init__(self, user_factors, item_factors, mapping):
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.mapping = mapping

    def transform(self, x):
        for idx, row in x.iterrows():
            user_id = row["visitorid"]
            item_id = row["itemid"]
            if user_id not in self.mapping.users_to_idx \
                    or item_id not in self.mapping.items_to_idx:
                yield (user_id, item_id), 0.0
                continue
            i = self.mapping.users_to_idx[user_id]
            j = self.mapping.items_to_idx[item_id]
            u = self.user_factors[i]
            v = self.item_factors[j]
            r = np.dot(u, v)
            yield (user_id, item_id), r

    def recommend(self, user_id, num_items=10, items_exclude=set()):
        i = self.mapping.users_to_idx[user_id]
        u = self.user_factors[i]
        V = self.item_factors
        P = np.dot(V, u)
        rank = sorted(enumerate(P), key=lambda p: p[1], reverse=True)

        top = list()
        k = 0
        while k < len(rank) and len(top) < num_items:
            j, r = rank[k]
            k += 1

            item_id = self.mapping.items_from_idx[j]
            if item_id in items_exclude:
                continue

            top.append((item_id, r))

        return top


class ALSRecommender:
    def __init__(self, num_factors=10, num_iters=20, reg=1e-1):
        self.num_factors = num_factors
        self.num_iters = num_iters
        self.regularization = reg

    def fit(self, dataset, verbose=False):
        with tf.Graph().as_default(), tf.Session() as sess:
            input_matrix, mapping = self.sparse_input(dataset)
            model = self.als_model(dataset)
            self.train(model, input_matrix, verbose)
            row_factor = model.row_factors[0].eval()
            col_factor = model.col_factors[0].eval()
            return ALSRecommenderModel(row_factor, col_factor, mapping)

    def sparse_input(self, dataset):
        mapping = new_mapping(dataset)

        indices = [(mapping.users_to_idx[r["visitorid"]],
                    mapping.items_to_idx[r["itemid"]])
                   for idx, r in dataset.iterrows()]
        values = [r["rating"] for idx, r in dataset.iterrows()]
        shape = (len(dataset["visitorid"].unique()), len(dataset["itemid"].unique()))

        return tf.SparseTensor(indices, values, shape), mapping

    def als_model(self, dataset):
        return WALSModel(
            len(dataset["visitorid"].unique()),
            len(dataset["itemid"].unique()),
            self.num_factors,
            regularization=self.regularization,
            unobserved_weight=0)

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

    def approx_sparse(self, model, indices, shape):
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

    def rmse_op(self, model, input_matrix):
        approx_matrix = self.approx_sparse(model, input_matrix.indices, input_matrix.dense_shape)
        err = tf.sparse_add(input_matrix, approx_matrix * (-1))
        err2 = tf.square(err)
        n = input_matrix.values.shape[0].value
        return tf.sqrt(tf.sparse_reduce_sum(err2) / n)


als = ALSRecommender()
als_model = als.fit(train, verbose=True)

_x, y_hat = list(als_model.transform(valid_prune.sample(n=1)))[0]

for k in range(10):
    item = valid_prune.sample(n=1)
    x1 = item["visitorid"].values[0]
    x2 = item["itemid"].values[0]
    y = item["rating"].values[0]
    _, y_hat = list(als_model.transform(item))[0]
    # print(x1, x2, y, y_hat)
    # print(x1)
    print(x1, x2, y, y_hat)


def _rmse(model, data):
    y_hat = list(r_hat for _, r_hat in model.transform(data))
    return np.sqrt(np.mean(np.square(np.subtract(data['rating'].as_matrix(), y_hat))))


def eval_rmse(model):
    rmse = _rmse(model, train)
    print('RMSE (train): {:,.3f}'.format(rmse))

    rmse = _rmse(model, valid_prune)
    print('RMSE (validation): {:,.3f}'.format(rmse))

    rmse = _rmse(model, test_prune)
    print('RMSE (test): {:,.3f}'.format(rmse))


eval_rmse(als_model)