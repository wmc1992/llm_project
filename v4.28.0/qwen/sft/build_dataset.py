from common.single_prompt_response import prompt_type_to_func


IGNORE_INDEX = -100  # 默认的：当 labels 中的 token 为 -100 时就不计算损失，省了再传输一个 mask 了


class DatasetUtil:
    # 该类的出处参考链接：https://github.com/QwenLM/Qwen/blob/main/finetune.py#L125

    def __init__(self, tokenizer, max_source_length, max_target_length,
                 prompt_column, response_column, history_column, prompt_type="default",
                 system_message: str = "You are a helpful assistant."):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prompt_type = prompt_type
        self.system_message = system_message

        # self.print_debug = True
        self.print_debug = False

    def tokenization(self, examples):
        # 创建几个基础token
        im_start = self.tokenizer.im_start_id
        im_end = self.tokenizer.im_end_id
        nl_tokens = self.tokenizer('\n').input_ids
        _system = self.tokenizer('system').input_ids
        _user = self.tokenizer('user').input_ids
        _assistant = self.tokenizer('assistant').input_ids

        f = prompt_type_to_func[self.prompt_type]
        sources, targets = f(examples, self.prompt_column, self.response_column, self.history_column)

        max_seq_len = self.max_source_length + self.max_target_length

        all_input_ids = []
        all_labels = []
        for source, target in zip(sources, targets):
            input_ids, labels = [], []

            # system 序列，这个放在最前面
            system_tokens = [im_start] + _system + nl_tokens + self.tokenizer(self.system_message).input_ids + [im_end] + nl_tokens
            system_labels = [im_start] + [IGNORE_INDEX] * (len(system_tokens)-3) + [im_end] + nl_tokens
            assert len(system_tokens) == len(system_labels)

            # 构造 source 序列及其 labels
            token_ids_0 = [im_start] + _user + nl_tokens + self.tokenizer(source).input_ids + [im_end] + nl_tokens
            label_ids_0 = [im_start] + [IGNORE_INDEX] * (len(token_ids_0)-3) + [im_end] + nl_tokens
            assert len(token_ids_0) == len(label_ids_0)

            # 构造 response 序列及其 labels
            token_ids_1 = [im_start] + _assistant + nl_tokens + self.tokenizer(target).input_ids + [im_end] + nl_tokens
            label_ids_1 = [im_start] + [IGNORE_INDEX] * len(_assistant + nl_tokens) + \
                self.tokenizer(target).input_ids + [im_end] + nl_tokens
            assert len(token_ids_1) == len(label_ids_1)

            # 如果 system、source、target 三者的总长度没有超长，就不做截断；否则就分别做截断
            if len(system_tokens) + len(token_ids_0) + len(token_ids_1) >= max_seq_len:
                input_ids = (system_tokens + token_ids_0)[:self.max_source_length] + token_ids_1[:self.max_target_length]
                labels = (system_labels + label_ids_0)[:self.max_source_length] + label_ids_1[:self.max_target_length]
            else:
                input_ids = (system_tokens + token_ids_0 + token_ids_1)
                labels = (system_labels + label_ids_0 + label_ids_1)
            assert len(input_ids) == len(labels)

            # PADDING
            input_ids += [self.tokenizer.pad_token_id] * (max_seq_len - len(input_ids))
            labels += [IGNORE_INDEX] * (max_seq_len - len(labels))

            # 为了保证不出问题，再截断一下
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        if self.print_debug:
            self.print_debug = False
            print("打印一条数据样本：\n")
            print(sources[0], "\n\n")
            print(targets[0], "\n\n")
            print(all_input_ids[0], "\n\n")
            print(all_labels[0])

        results = {"input_ids": all_input_ids, "labels": all_labels}
        return results
