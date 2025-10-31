DATA_PATH=/data00/chenxiang/workspace/datas/xxx.json

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

OUTPUT_DIR=xxx
LOGGING_DIR=xxx
MODEL_PATH=xxx

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

torchrun -m $DISTRIBUTED_ARGS trainer.sft \
    --data_path $DATA_PATH \
    --data_type "eou" \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --model_name_or_path $MODEL_PATH \
    --num_train_epochs 3 \
    --deepspeed scipts/ds_config/ds_config_zero2.json \
    --prediction_loss_only false \
    --bf16 true \
    --fp16 false \
    --do_train \
    --model_max_length 2048 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_steps 5000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.98 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --use_lora true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "model.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)" \
    --gradient_checkpointing true \
    --attn_implementation "flash_attention_2" 
