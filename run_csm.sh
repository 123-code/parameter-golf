#!/usr/bin/env bash
set -euo pipefail
cd /root/parameter-golf
source .venv/bin/activate

export RUN_ID=csm_ttt_run
export ITERATIONS=200
export WARMUP_STEPS=0
export WARMDOWN_ITERS=0
export TRAIN_LOG_EVERY=20
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_BATCH_TOKENS=131072
export TRAIN_SEQ_LEN=64
export EVAL_SEQ_LEN=64
export VAL_BATCH_SIZE=4096
export NUM_LAYERS=4
export RECURRENCE_DEPTH=3
export TTT_SPARSE_SLOTS=256
export TTT_TOPK=8
export VE_LAYERS=2,3
export XSA_LAST_N=4
export SWA_ENABLED=0
export LATE_QAT_THRESHOLD=0
export EVAL_STRIDE=8
export TTT_ENABLED=1
export TTT_CHUNK_TOKENS=64
export TTT_MAX_VAL_TOKENS=65536
export TTT_BATCH_SEQS=4
export TTT_EPOCHS=1
export TTT_LR=0.01
export TTT_FREEZE_BLOCKS=0
export SKIP_POST_TRAIN_EVAL=1
export DATA_PATH=./data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024

exec torchrun --standalone --nproc_per_node=1 train_csm_ttt.py
