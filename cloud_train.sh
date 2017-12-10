#!/usr/bin/env bash
STAGING_BUCKET="gs://dhodun1-staging"
MODEL_BUCKET="gs://dhodun1-ml"
JOB_ID="movielens_collaborative_filter_$(date +%Y%m%d_%H%M%S)"
DATA_FILE="gs://dhodun1-ml/movielens/data/ml-latest-20M/ratings.csv"
gcloud ml-engine jobs submit training $JOB_ID \
                      --module-name trainer.task \
                      --package-path trainer \
                      --job-dir $MODEL_BUCKET/movielens/$JOB_ID \
                      --staging-bucket $STAGING_BUCKET \
                      --config config_hypertune.yaml \
                      --runtime-version 1.2 \
                      -- \
                      --output-dir $MODEL_BUCKET/movielens/$JOB_ID \
                      #--data-file $DATA_FILE