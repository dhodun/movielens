#!/usr/bin/env bash
STAGING_BUCKET="gs://dhodun1-staging"
MODEL_BUCKET="gs://dhodun1-ml"
JOB_ID="movielens_collaborative_filter_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine local train \
                      --module-name trainer.task \
                      --package-path trainer \
                      --job-dir $MODEL_BUCKET/$JOB_ID \
                      -- \
