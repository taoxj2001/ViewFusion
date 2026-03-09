export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"
ulimit -n 65535
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    sft.py \
    --output_dir "./log/VFR-SFT" \
    --model_name_or_path "Put your model path here" \
    --dataset_name "SFT_data/SFT.jsonl" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --bf16 True \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name VFR-SFT \
    --save_steps 250 \
    --max_grad_norm 5 \
    --save_only_model true \
    --tf32 True \