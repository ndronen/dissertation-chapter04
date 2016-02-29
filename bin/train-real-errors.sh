#!/bin/bash

model_dir=models/real_errors/
data_dir=data/

n_embed_dims=10
n_filters=3000
filter_width=6
n_fully_connected=2
n_residual_blocks=2
n_hidden=1000

#corpus="non-word-error-detection-experiment-04-generated-negative-examples.h5"

train.py $model_dir \
    $data_dir/train.h5 \
    $data_dir/validation.h5 \
    marked_chars \
    --target-name binary_target \
    --n-embeddings 255 \
    --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_fully_connected=$n_fully_connected n_residual_blocks=$n_residual_blocks n_hidden=$n_hidden patience=10 \
    --confusion-matrix \
    --classification-report \
    --class-weight-auto \
    --class-weight-exponent 2 \
    --early-stopping-metric val_f1 \
    --checkpoint-metric val_f1 \
    --save-all-checkpoints \
    --verbose \
    --log
