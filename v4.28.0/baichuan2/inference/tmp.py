from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
)


model_name_or_path = "/data/pretrained_models/Baichuan2-13B-Chat"
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

print(tokenizer)
print(tokenizer.vocab_size)

import json

with open("tokenizer.json", "w") as f:
    json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
