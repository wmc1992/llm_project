from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)


model_name_or_path = "/data/pretrained_models/Qwen-7B-Chat"
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

print(tokenizer)
print(tokenizer.vocab_size)


# im_start = tokenizer.im_start_id
# im_end = tokenizer.im_end_id
# nl_tokens = tokenizer('\n').input_ids
# _system = tokenizer('system').input_ids + nl_tokens
# _user = tokenizer('user').input_ids + nl_tokens
# _assistant = tokenizer('assistant').input_ids + nl_tokens

print("tokenizer.im_start_id:", tokenizer.im_start_id)
print("tokenizer.im_end_id:", tokenizer.im_end_id)

all_vocabs = tokenizer.get_vocab()
print(type(all_vocabs))
print(len(all_vocabs))
print(model)
