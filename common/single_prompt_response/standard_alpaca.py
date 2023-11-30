# 该函数中使用的是出自 standard alpaca 的 prompt，不过由于该 prompt 是英文的，实际测试：该指令会可能导致模型输出部分英文 


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)

def build_source_and_target_from_alpaca_prompt(examples, prompt_column, response_column, history_column):

    sources = []
    targets = []
    for prompt, response in zip(examples[prompt_column], examples[response_column]):
        sources.append(PROMPT_TEMPLATE.format_map({"instruction": prompt}))
        targets.append(response)
    return sources, targets
