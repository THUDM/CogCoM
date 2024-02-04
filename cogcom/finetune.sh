#! /bin/bash

NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="path/to/cogcom-base" # download checkpoint and config
VERSION="base"
MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_source_length 1225 \
    --max_target_length 823 \
    --lora_rank 10 \
    --use_lora \
    --local_tokenizer path/to/vicuna-7b-v1.5 \
    --version $VERSION"
# Tips: If training models of resolution 244, you can set --max_length smaller 

OPTIONS_SAT="SAT_HOME=~/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

train_data=("$(realpath .)/data/save/steps_5shot_wds#CoM" "$(realpath .)/data/save/instruct_data##Instruct")

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 800 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --vit_checkpoint_activations \
       --save-interval 200 \
       --eval-interval 200 \
       --save "./checkpoints" \
       --eval-iters 10 \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config test_config_bf16_zero1off.json \
       --skip-init \
       --iterable-dataset \
       --seed 2024
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune.py ${gpt_options}"
# run_cmd="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python finetune.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x