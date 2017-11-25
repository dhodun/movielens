import argparse

import tensorflow as tf

import model


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    # process input data
    ratings = model.load_ratings('gs://dhodun1-ml/movielens/data/ml-latest-small/ratings.csv')

    # create mapping
    user_encoder, movie_encoder = model.create_mappings(ratings)

    ratings['user'] = user_encoder.transform(ratings['user'])
    ratings['item'] = movie_encoder.transform(ratings['item'])

    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)

    train, test = model.create_test_and_train(ratings)
    train_sparse, test_sparse = model.create_sparse_sets(train, test, num_users, num_movies)

    # train model
    output_row, output_col = model.train_model(train_sparse, test_sparse, num_users, num_movies, verbose=True)

    # log results
    train_rmse = model.get_rmse(output_row, output_col, train_sparse)
    test_rmse = model.get_rmse(output_row, output_col, test_sparse)

    tf.logging.info("train RMSE = %.2f" % train_rmse)
    tf.logging.info("test RMSE = %.2f" % test_rmse)

    # save trained model to job directory

    model.save_model(args, output_row, output_col)

    return


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        default='gs://dhodun1-ml/movielens/temp/job_dir/',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--hypertune',
        default=False,
        action="store_true",
        help='Switch to turn on or off hyperparam tuning'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    # TODO replace with tf.app.run() ?
    # TODO compare with west-lake format
