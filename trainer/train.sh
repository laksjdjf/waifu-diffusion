#!/bin/bash

# Just an example of how to run the training script.

export HF_API_TOKEN=""
BASE_MODEL="model"
OUTPUT_MODEL="test"
RUN_NAME="test"
DATASET="dataset"
N_GPU=1
N_EPOCHS=1
BATCH_SIZE=8

python3 -m torch.distributed.run --nproc_per_node=$N_GPU trainer/diffusers_trainer.py --model=$BASE_MODEL --output_path=$OUTPUT_MODEL --run_name=$RUN_NAME --dataset=$DATASET --lr_scheduler="cosine" --bucket_side_min=256 --bucket_side_max=1024 --use_8bit_adam=True --ucg=0.1 --gradient_checkpointing=False --batch_size=$BATCH_SIZE --fp16=True --image_log_steps=764 --save_steps 764 --epochs=$N_EPOCHS --resolution=768 --use_ema=False --clip_penultimate=True --use_xformers=True --latent_cache=True --nai_buckets=True --output_bucket_info=False --use_tagger=True

# and to resume... just add the --resume flag and supply it with the path to the checkpoint. 848dfee481af9a356a47b4421eb5f6ed1a27e499
