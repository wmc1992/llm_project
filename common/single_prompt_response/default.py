def build_source_and_target_default(examples, prompt_column, response_column, history_column):
    """ 默认直接将 source 和 target 拼接到一起 """

    sources = []
    targets = []
    for prompt, response in zip(examples[prompt_column], examples[response_column]):
        sources.append(prompt)
        targets.append(response)
    return sources, targets
