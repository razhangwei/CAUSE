#!/bin/bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
TF_CPP_MIN_LOG_LEVEL=2  # disable TF INFO and WARNING messages
GPU_COUNT=`nvidia-smi | grep Default | wc -l`
WS=`pwd`

n_seqs=1000
n_types=10
n_correlations=16
dataset=mscp-$(($n_seqs / 1000))K-$n_types
shared_args="--dataset $dataset"
n_splits=5  # if modified, remember to modify below as well!!!

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
fi

# preprocessing/data generation

if [[ $* == *all* ]] || [[ $* == *preprocess* ]]; then
    python preprocessing/generate_events_by_mscp.py \
        --n_seqs $n_seqs \
        --n_types $n_types \
        --n_correlations $n_correlations \
        --adj_scale 0.5 \
        --n_splits $n_splits
fi

# training for each methods

if [[ $* == *all* ]] || [[ $* == *HExp* ]]; then
    printf "%s\n" "$WS/tasks/train.py HExp $shared_args --decay 0.05 --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *NPHC* ]]; then
    printf "%s\n" "$WS/tasks/train.py NPHC $shared_args --integration_support 5 --verbose --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *HSG* ]]; then
    printf "%s\n" "$WS/tasks/train.py HSG $shared_args --max_mean 1000 --n_gaussians 5 --verbose --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *RPPN* ]]; then
    printf "%s\n" "$WS/tasks/train.py RPPN $shared_args --cuda --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *ERPP* ]]; then
    printf "%s\n" "$WS/tasks/train.py ERPP $shared_args --epoch 200 --cuda --split_id "{0..4} | $LOCAL_RUN
fi

# python postprocessing/summarize_results.py $dataset
