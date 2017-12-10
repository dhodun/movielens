import os

import tensorflow as tf
from google.cloud import storage
from tensorflow.core.framework.summary_pb2 import Summary


def write_hptuning_metric(args, metric):
    summary = Summary(value=[Summary.Value(tag='training/hptuning/metric', simple_value=metric)])

    eval_path = os.path.join(args.output_dir, 'eval')
    summary_writer = tf.summary.FileWriter(eval_path)

    summary_writer.add_summary(summary)
    summary_writer.flush()


def copy_file_to_gcs(job_dir, file_path):
    # remove gs://
    path = job_dir[5:]
    slash = path.find("/")
    bucket = path[:slash]
    dirs = path[slash + 1:]

    destination_name = os.path.join(dirs, file_path)

    upload_blob(bucket, file_path, destination_name)




    # does not work in TF 1.2
    # with file_io.FileIO(file_path, mode='r') as input_f:
    #    with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
    #        output_f.write(input_f.read())


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    tf.logging.INFO('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
