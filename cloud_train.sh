#!/usr/bin/env bash
BUCKET="gs://dhodun1-ml"
JOB_ID="movielens_collaborative_filter_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
                      --module-name trainer.task \
                      --package-path trainer \
                      --staging-bucket "$BUCKET" \
                      --region us-east1 \
                      --job-dir "$BUCKET/$JOB_ID" \
                      -- \
                      --raw_metadata_path "${PREPROCESS_OUTPUT}/raw_metadata" \
                      --transform_savedmodel "${PREPROCESS_OUTPUT}/transform_fn" \
                      --eval_data_paths "${PREPROCESS_OUTPUT}/features_eval*.tfrecord.gz" \
                      --train_data_paths "${PREPROCESS_OUTPUT}/features_train*.tfrecord.gz" \
                      --output_path "${GCS_PATH}/model/${JOB_ID}" \
                      --model_type dnn_softmax \
                      --eval_type ranking \
                      --l2_weight_decay 0.01 \
                      --learning_rate 0.05 \
                      --train_steps 500000 \
                      --eval_steps 500 \
                      --top_k_infer 100