#!/bin/bash
WS=`pwd`
CUDA_DEVICE_ORDER=PCI_BUS_ID
LOCAL_RUN="xargs -L1 python"

dataset="IPTV"
shared_args="--dataset $dataset --skip_pred_next_event --verbose"

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
    exit 1
fi

# preprocessing/data generation

if [[ $* == *all* ]] || [[ $* == *preprocess* ]]; then
    python preprocessing/process_IPTV.py \
        --name $dataset \
        --max_seq_length 1000 \
        --n_splits 5
fi

# training for each methods

if [[ $* == *all* ]] || [[ $* == *HExp* ]]; then
    printf "%s\n" "$WS/tasks/train.py HExp $shared_args --decay 100 --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *HSG* ]]; then
    printf "%s\n" "$WS/tasks/train.py HSG $shared_args --max_mean 5 --n_gaussians 5 --verbose --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *RPPN* ]]; then
    printf "%s\n" "$WS/tasks/train.py RPPN $shared_args --epochs 150 --batch_size 64 --hidden_size 256 --init_scale 0.01 --bucket_seq --cuda --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *ERPP* ]]; then
    printf "%s\n" "$WS/tasks/train.py ERPP $shared_args --max_mean 10 --n_bases 12 --batch_size 64 --hidden_size 128 --lr 0.001 --epochs 150 --bucket_seq --attr_batch_size 8 --occurred_type_only --cuda --split_id "{0..4} | $LOCAL_RUN
fi

# python postprocessing/summarize_results.py $dataset
