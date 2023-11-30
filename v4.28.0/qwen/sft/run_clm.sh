# 常规训练配置
lr=5e-5
per_device_train_batch_size=8
per_device_eval_batch_size=8
gradient_accumulation_steps=4
training_steps=3000

# Qwen 的模型结构如下：
#
# QWenLMHeadModel(
#   (transformer): QWenModel(
#     (wte): Embedding(151936, 4096)
#     (drop): Dropout(p=0.0, inplace=False)
#     (rotary_emb): RotaryEmbedding()
#     (h): ModuleList(
#       (0-31): 32 x QWenBlock(
#         (ln_1): RMSNorm()
#         (attn): QWenAttention(
#           (c_attn): Linear(in_features=4096, out_features=12288, bias=True)
#           (c_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (attn_dropout): Dropout(p=0.0, inplace=False)
#         )
#         (ln_2): RMSNorm()
#         (mlp): QWenMLP(
#           (w1): Linear(in_features=4096, out_features=11008, bias=False)
#           (w2): Linear(in_features=4096, out_features=11008, bias=False)
#           (c_proj): Linear(in_features=11008, out_features=4096, bias=False)
#         )
#       )
#     )
#     (ln_f): RMSNorm()
#   )
#   (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
# )

# LoRA 训练配置
lora_rank=8
lora_alpha=32
target_modules="c_attn,c_proj,w1,w2,c_proj" # 需要根据不同的模型做修改
lora_dropout=0.05

# 数据配置
train_file_path=../../../data/train.json
validation_file=../../../data/dev.json
prompt_column="input"
response_column="target"
max_source_length=1024
max_target_length=512
prompt_type=default

# 预训练模型的路径配置
model_name_or_path=/the/pretrained/model/name/or/path
output_dir=./output

# deepspeed配置
deepspeed_config_file=../../../config/ds_zero2_no_offload.json

# accelerate配置
accelerate_config_file=../../../config/accelerate_config_two_process.yaml

# 正式训练之前设置一下工作路径，一个是当前项目的根目录，一个是当前目录
export PYTHONPATH=.:../../../:$PYTHONPATH

# 直接使用 python 启动
# CUDA_VISIBLE_DEVICES=0 python3 run_clm.py \

# 使用 accelerate 启动
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#     --config_file ${accelerate_config_file} \
#     run_clm.py \

# 使用 deepspeed 启动
torchrun --nnodes 1 --nproc_per_node 1 --master_port=39999 \
    run_clm.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${model_name_or_path} \
    --train_file ${train_file_path} \
    --validation_file ${validation_file} \
    --validation_split_percentage 1 \
    --prompt_column ${prompt_column} \
    --response_column ${response_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --prompt_type ${prompt_type} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_steps ${training_steps} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 99999 \
    --save_steps 500 \
    --preprocessing_num_workers 8 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --use_peft True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --target_modules ${target_modules} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
