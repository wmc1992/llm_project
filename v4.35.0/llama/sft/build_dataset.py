import torch
from common.single_prompt_response import prompt_type_to_func


IGNORE_INDEX = -100  # 默认的：当 labels 中的 token 为 -100 时就不计算损失，省了再传输一个 mask 了


class DatasetUtil:

    def __init__(self, tokenizer, max_source_length, max_target_length,
                 prompt_column, response_column, history_column, prompt_type="default"):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prompt_type = prompt_type

        # self.print_debug = True
        self.print_debug = False

    def tokenization(self, examples):
        f = prompt_type_to_func[self.prompt_type]
        sources, targets = f(examples, self.prompt_column, self.response_column, self.history_column)

        max_seq_len = self.max_source_length + self.max_target_length

        all_input_ids = []
        all_labels = []
        for source, target in zip(sources, targets):
            token_ids_0 = self.tokenizer.encode(text=source, add_special_tokens=False)
            token_ids_1 = self.tokenizer.encode(text=target, add_special_tokens=False)

            # 如果 source 和 target 的总长度没有超长，就不做截断
            if len(token_ids_0) + len(token_ids_1) > max_seq_len - 2:
                if len(token_ids_0) > self.max_source_length - 1:  # 留一个位置给 bos_token
                    token_ids_0 = token_ids_0[:self.max_source_length - 1]
                if len(token_ids_1) > self.max_target_length - 1:  # 留一个位置给 eos_token
                    token_ids_1 = token_ids_1[:self.max_target_length - 1]

            # 构造的模版：<s> A B </s>
            token_ids_0 = [self.tokenizer.bos_token_id] + token_ids_0
            token_ids_1 = token_ids_1 + [self.tokenizer.eos_token_id]
            token_ids = token_ids_0 + token_ids_1
            labels = [IGNORE_INDEX for _ in range(len(token_ids_0))] + token_ids_1
            assert len(token_ids) == len(labels)

            # PADDING
            if len(token_ids) < max_seq_len:
                token_ids = token_ids + [self.tokenizer.pad_token_id for _ in range(max_seq_len - len(token_ids))]
                labels = labels + [IGNORE_INDEX for _ in range(max_seq_len - len(labels))]

            assert len(token_ids) == max_seq_len
            assert len(labels) == max_seq_len

            all_input_ids.append(torch.LongTensor(token_ids))
            all_labels.append(torch.LongTensor(labels))

        if self.print_debug:
            self.print_debug = False
            print("打印一条数据样本：\n")
            print(sources[0], "\n\n")
            print(targets[0], "\n\n")
            print(all_input_ids[0], "\n\n")
            print(all_labels[0])

        results = {"input_ids": all_input_ids, "labels": all_labels}
        return results
