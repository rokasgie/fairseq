export PYTHONPATH=.
ROOT=/home/rokas/fairseq

fairseq-hydra-train \
    --config-dir $ROOT/examples/wav2vec/config/finetuning \
    --config-name base_100h
