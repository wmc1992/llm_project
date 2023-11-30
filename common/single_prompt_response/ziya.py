# 出自封神榜系列的 ziya 模型
# 出处链接：https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/ziya_llama/finetune_ziya_llama.py#L33


ziya_prompt = """<human>:{instruction}\n<bot>:"""


def build_source_and_target_from_ziya_prompt(examples, prompt_column, response_column, history_column):
    sources = []
    targets = []
    for prompt, response in zip(examples[prompt_column], examples[response_column]):
        sources.append(ziya_prompt.format_map({"instruction": prompt}))
        targets.append(response)
    return sources, targets
