import os
import time
from typing import Optional

import torch
import jsonlines
from tqdm import tqdm
import argparse

import bitsandbytes as bnb
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

from transformers.pipelines.conversational import Conversation


parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", default=None, required=True, type=str, help="指定模型的路径")
parser.add_argument("--cache_dir", default=None, type=Optional[str], help="")
parser.add_argument("--trust_remote_code", default=False, type=bool, help="")
parser.add_argument("--use_auth_token", default=False, type=bool, help="")

parser.add_argument("--bits", default=8, type=int, help="")
parser.add_argument("--max_memory_MB", default=44000, type=int, help="")
parser.add_argument("--double_quant", default=True, type=bool, help="")
parser.add_argument("--quant_type", default="nf4", type=str, help="Quantization data type to use. Should be one of `fp4` or `nf4`.")
parser.add_argument("--fp16", default=True, type=bool, help="")
parser.add_argument("--bf16", default=False, type=bool, help="")

parser.add_argument("--full_finetune", default=True, type=bool, help="")

parser.add_argument("--max_new_tokens", default=512, type=int, help="")
parser.add_argument("--temperature", default=0.6, type=float, help="")
parser.add_argument("--top_p", default=0.9, type=float, help="")

args = parser.parse_args()

# ----------------------------
# 载入模型
# ----------------------------
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    # if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
        })
        tokenizer.pad_id = 0

    # if not args.full_finetune:
    #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # if not args.full_finetune:
    #     if checkpoint_dir is not None:
    #         print("Loading adapters from checkpoint.")
    #         model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, 'adapter_model'), is_trainable=True)
    #     else:
    #         print(f'adding LoRA modules...')
    #         modules = find_all_linear_names(args, model)
    #         config = LoraConfig(
    #             r=args.lora_r,
    #             lora_alpha=args.lora_alpha,
    #             target_modules=modules,
    #             lora_dropout=args.lora_dropout,
    #             bias="none",
    #             task_type="CAUSAL_LM",
    #         )
    #         model = get_peft_model(model, config)

    print(model)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer


model, tokenizer = get_accelerate_model(args, checkpoint_dir=None)

print("预训练模型载入成功")


# ----------------------------
# 准备待预测数据
# ----------------------------
dialog_list = [
    [
        {"role": "user", "content": "what is the recipe of mayonnaise?"}
    ],
    [
        {"role": "system", "content": "Always answer with Haiku"},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ],
    [
        {"role": "system", "content": "Always answer with emojis"},
        {"role": "user", "content": "How to go from Beijing to NY?"},
    ],
    [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "Hello! I'm just an AI, my purpose is to assist and provide helpful information to the best of my abilities. I am programmed to follow ethical guidelines and promote respectful and positive interactions. I am here to help answer any questions you may have, while ensuring that my responses are socially unbiased and do not contain any harmful or inappropriate content. If a question does not make sense or is not factually coherent, I will do my best to explain why and provide clarification. If I am unsure or do not know the answer to a question, I will not provide false information and will instead suggest where you might be able to find the answer or offer alternative solutions. Please feel free to ask me anything, and I will do my best to assist you."},
        {"role": "user", "content": "那你会说中文吗？"},
        {"role": "assistant", "content": "Yes, I can speak Chinese. I'm just an AI, I don't have a physical voice, but I can communicate in Chinese text. Would you like to chat in Chinese?"},
        {"role": "user", "content": "是的，接下来请用中文聊天。"},
        {"role": "assistant", "content": "Sure, I'd be happy to chat with you in Chinese!\n你好！我是一个AI助手，可以帮助你解决各种问题。你可以问我任何问题，我会尽力回答。\n你今天有什么计划吗？"},
        {"role": "user", "content": "帮我制定一个五一出行计划。"},
    ],
]
print("待预测数据集载入成功")


# ----------------------------
# 构造成conversation格式数据
# ----------------------------

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

conversation_list = []
for dialog in dialog_list:
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]

    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    assert (dialog[-1]["role"] == "user"), f"Last message must be from user, got {dialog[-1]['role']}"

    conversation_list.append(
        Conversation(
            text=dialog[-1]["content"],
            past_user_inputs=[item["content"] for item in dialog[:-1] if item["role"] == "user"],
            generated_responses=[item["content"] for item in dialog[:-1] if item["role"] == "assistant"]
        )
    )


# ----------------------------
# 推理
# ----------------------------
start_time = time.time()
total_token_length = 0

with torch.no_grad():
    for dialog, conversation in zip(dialog_list, conversation_list):
        input_ids = tokenizer._build_conversation_input_ids(conversation)
        input_ids = torch.tensor([input_ids])

        out = model.generate(
            input_ids=input_ids.cuda(),
            max_new_tokens=512,
            top_p=0.9,
            temperature=0.6,
        )

        in_text = tokenizer.decode(input_ids[0])
        out_text = tokenizer.decode(out[0])
        print(out_text)
        print("=" * 100)

        total_token_length += (len(out[0]) - len(input_ids[0]))
        break

end_time = time.time()
print("total token length: ", total_token_length)
print("total inference time: ", end_time - start_time)
print("tokens per second: ", total_token_length / (end_time - start_time))
