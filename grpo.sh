#!/bin/bash
export VLLM_DISABLE_TQDM=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}_${SLURM_JOB_ID}_${RANK}
mkdir -p $TRITON_CACHE_DIR
export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QWEN_PATH='Put your model path here'
HF_DATASET="VFR-traindata/GRPO_data/GRPO.jsonl"
OUTPUT_DIR="./log/VFR-4B"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="Qwen3-GRPO"
DS_CONFIG="local_scripts/zero3.json"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" /data/user/xtao467/data/llm/bin/python -m torch.distributed.run \
    --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 401760 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model false \
    --len_control true \
    --report_to wandb \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 8 \
    --deepspeed ${DS_CONFIG} \
    --dataloader_num_workers 0 \
    --vllm_gpu_memory_utilization 0.4 \

    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
