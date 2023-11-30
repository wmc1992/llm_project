CUDA_VISIBLE_DEVICES=0 python inference_with_qlora.py \
    --model_name_or_path /data/pretrained_models/llama2/llama2-7b-chat-hf \
    --bits 8 \
    --max_memory_MB 44000 \
    --max_new_tokens 512 \
    --temperature 0.6 \
    --top_p 0.9

# CUDA_VISIBLE_DEVICES=0 python inference_with_qlora.py \
#     --model_name_or_path /data/pretrained_models/llama2/llama2-7b-chat-hf \
#     --bits 8 \
#     --max_memory_MB 44000 \
#     --max_new_tokens 512 \
#     --temperature 0.6 \
#     --top_p 0.9

# model_name_or_path: 模型路径
# bits: 加载模型使用多少位量化
# max_memory_MB: 每张显卡可用显存大小
# max_new_tokens: 推理的最大长度
# temperature: 推理参数
# top_p: 推理参数
