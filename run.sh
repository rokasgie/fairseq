fairseq-hydra-train \
    distributed_training.distributed_port=1111 \
    task.data=/home/rokas/ino-voice/data/data_zips/room_zips \
    model.w2v_path=/home/rokas/ino-voice/data/models/xlsr_53_56k.pt \
    --config-dir /home/rokas/ino-voice/fairseq/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h
