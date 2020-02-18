#!/bin/bash

# file=train_rmdn.py
# # python -m cProfile -o data/output/$file.cprof tasks/$file --epochs 30 --l2_reg 0.1 --cuda
# kernprof -l -o data/output/$file.lprof tasks/$file --epochs 30 --l2_reg 0.1 --cuda
# python -m line_profiler data/output/$file.lprof

# file=attribute_rmdn.py
# grep -n @profile tasks/$file
# kernprof -l -o data/output/$file.lprof tasks/$file --steps 50 --cuda
# python -m line_profiler data/output/$file.lprof

file=train.py
# grep -n @profile tasks/$file
# kernprof -l -o data/output/${file}_RPPN.lprof tasks/$file RPPN --epoch 30 --num_workers 0 --cuda
# python -m line_profiler data/output/${file}_RPPN.lprof

model=ERPP
kernprof -l -o data/output/${file}_$model.lprof tasks/$file $model --n_bases=5 --max_mean=10 --batch_size=32 --lr=0.005 --epoch=30 --cuda
python -m line_profiler data/output/${file}_$model.lprof