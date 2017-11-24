#!/usr/bin/env bash
STAGING_BUCKET="gs://dhodun1-staging"
MODEL_BUCKET="gs://dhodun1-ml"
JOB_ID="movielens_collaborative_filter_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training $JOB_ID \
                      --module-name trainer.task \
                      --package-path trainer \
                      --job-dir $MODEL_BUCKET/movielens/$JOB_ID \
                      --staging-bucket $STAGING_BUCKET \
                      -- \
