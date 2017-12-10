import argparse
import json
import os

import tensorflow as tf
from tensorflow.python.client import device_lib

import model
import util


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    # double check if there are GPUs or CPUs
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU'], [x.name for x in local_device_protos if
                                                                                 x.device_type == 'CPU']

    gpus, cpus = get_available_gpus()
    tf.logging.info("GPUS {}".format(gpus))
    tf.logging.info("CPUS {}".format(cpus))

    tf.logging.info("Tf Version: {}".format(tf.__version__))

    # process input data
    ratings = model.load_ratings(args.data_file)

    # create mapping
    user_encoder, movie_encoder = model.create_mappings(ratings)

    ratings['user'] = user_encoder.transform(ratings['user'])
    ratings['item'] = movie_encoder.transform(ratings['item'])

    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)

    train, test = model.create_test_and_train(ratings)
    train_sparse, test_sparse = model.create_sparse_sets(train, test, num_users, num_movies)

    # train model
    output_row, output_col = model.train_model(train_sparse, test_sparse, num_users, num_movies, args, verbose=True)

    # log results
    train_rmse = model.get_rmse(output_row, output_col, train_sparse)
    test_rmse = model.get_rmse(output_row, output_col, test_sparse)

    tf.logging.info("train RMSE = %.2f" % train_rmse)
    tf.logging.info("test RMSE = %.2f" % test_rmse)

    # output metric for hyperparameter tuning




    if args.hypertune:
        tf.logging.info("Priting hptuning metric: {}".format(test_rmse))
        util.write_hptuning_metric(args, test_rmse)

    # save trained model to job directory
    model.save_model(args, output_row, output_col)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        default='gs://dhodun1-ml/movielens/temp/job_dir/',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--output-dir',
        default='gs://dhodun1-ml/movielens/temp/job_dir/',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--data-file',
        default='gs://dhodun1-ml/movielens/data/ml-latest-small/ratings.csv',
        help='Where to download Movielens ratings'
    )

    # Model Hyperparameters
    parser.add_argument(
        '--num_factors',
        default=10,
        type=int,
        help='Number of latent factors',
        required=False
    )
    parser.add_argument(
        '--regularization',
        default=1e-1,
        type=float,
        help='Regularization constant',
        required=False
    )
    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='Number of training epochs',
        required=False
    )
    parser.add_argument(
        '--unobserved_weight',
        default=0,
        type=float,
        help='Weight to give unobserved values',
        required=False
    )
    parser.add_argument(
        '--hypertune',
        default=False,
        help='Flag to turn on hyper-param metric summary',
        required=False
    )
    parser.add_argument(
        '--row_weight_bool',
        default=False,
        type=bool,
        help='Are we adding row weights to WALS',
        required=False
    )
    parser.add_argument(
        '--col_weight_bool',
        default=True,
        type=bool,
        help='Are we adding column weights to WALS',
        required=False
    )
    parser.add_argument(
        '--row_weight_factor',
        default=0,
        type=int,
        help='Row Weight Scaling Factor for WALS',
        required=False
    )
    parser.add_argument(
        '--col_weight_factor',
        default=150,
        type=int,
        help='Column Weight Scaling Factor for WALS',
        required=False
    )


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arguments()

    # Add Trial to output-dir if hyper-tuning
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_data = env.get('task') or {'type': 'master', 'index': 0}
    trial = task_data.get('trial')

    if trial is not None:
        args.output_dir = os.path.join(args.output_dir, trial)
        args.hypertune = True

    main(args)
    # TODO replace with tf.app.run() ?
    # TODO compare with west-lake format
