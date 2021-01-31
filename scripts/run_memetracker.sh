#!/bin/bash
WS=`pwd`
CUDA_DEVICE_ORDER=PCI_BUS_ID
LOCAL_RUN="xargs -L1 python"

n_types=100
# dataset="MemeTracker-0.2M-$n_types"
dataset="MemeTracker-0.4M-$n_types"
# dataset="MemeTracker-1.8M-$n_types"
shared_args="--dataset $dataset --skip_pred_next_event --verbose"

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
    exit 1
fi

# preprocessing/data generation

if [[ $* == *all* ]] || [[ $* == *preprocess* ]]; then
    # the preprocessing would take about around 4 hours
    # python preprocessing/export_MemeTracker_to_parquet.py

    # process the parquet files into event sequences
    python preprocessing/process_MemeTracker_spark.py \
        --n_top_sites $n_types \
        --min_seq_length 3 \
        --max_seq_length 500 \
        --time_divisor 3600.0 \
        --end_date 2008-09-30
        # --end_date 2008-08-31
fi

# training for each methods

if [[ $* == *all* ]] || [[ $* == *HExp* ]]; then
    printf "%s\n" "$WS/tasks/train.py HExp $shared_args --decay "0.1" --max_seqs 30000 --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *HSG* ]]; then
    printf "%s\n" "$WS/tasks/train.py HSG $shared_args --max_mean 5 --n_gaussians 5 --max_seqs 100000 --split_id "{0..4} | $LOCAL_RUN
fi

# FIXME: unable to produce results in two days
# if [[ $* == *all* ]] || [[ $* == *NPHC* ]]; then
#     printf "%s\n" "$WS/tasks/train.py NPHC $shared_args --integration_support 20 --max_seqs 20000 --split_id "{0..4} | $LOCAL_RUN
# fi

if [[ $* == *all* ]] || [[ $* == *RPPN* ]]; then
    printf "%s\n" "$WS/tasks/train.py RPPN $shared_args --epochs 150 --batch_size 256 --bucket_seq --hidden_size 128 --cuda --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *ERPP* ]]; then
    printf "%s\n" "$WS/tasks/train.py ERPP $shared_args --max_mean 100 --n_bases 10 --batch_size 256 --bucket_seqs --hidden_size 128 --lr 0.001 --epochs 150 --attr_batch_size 32 --steps 20 --occurred_type_only --cuda --split_id "{0..4} | $LOCAL_RUN
fi

# python postprocessing/summarize_results.py $dataset