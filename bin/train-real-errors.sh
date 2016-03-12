#!/bin/bash

model_dir=models/real_errors/
if [ $# -eq 0 ]
then
    echo "usage: train.py --mode MODE KEY=VALUE KEY=VALUE ... KEY=VALUE" >&2
    echo "       MODE can be 'transient', 'persistent', or 'persistent-background'" >&2
    echo "       KEY=VALUE pairs are model arguments that may also be set in model.json" >&2
    exit 1
fi

mode=$1
shift

if [ $# -eq 0 ]
then
    bin/train.py $model_dir --mode $mode
else
    bin/train.py $model_dir --mode $mode --model-cfg "$@"
fi
