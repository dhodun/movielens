#!/usr/bin/env bash
gcloud ml-engine local train \
                      --module-name trainer.task \
                      --package-path trainer \
                      --job-dir "$BUCKET/$JOB_ID" \
                      -- \
