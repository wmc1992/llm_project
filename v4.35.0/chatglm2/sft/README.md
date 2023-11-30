版本 v4.35.0 的 transfermers 引入了不少新特性，但也产生了很多兼容性问题，这个 chatglm 模型使用 v4.35.0 版本的 transformers 不能直接训练。

具体的现象是 `ChatGLMTokenizer` 这个类会初始化失败，原因是在该类进行初始化时会调用 `self.sp_tokenizer` 对象，但是此时该对象还没有初始化。报错信息如下所示：

```
Traceback (most recent call last):
  File "/data/llm_project/v4.35.0/chatglm2/sft/run_clm.py", line 881, in <module>
    main()
  File "/data/llm_project/v4.35.0/chatglm2/sft/run_clm.py", line 857, in main
    config, tokenizer, model = load_pretrained_model_and_tokenizer(model_args)
  File "/data/llm_project/v4.35.0/chatglm2/sft/run_clm.py", line 435, in load_pretrained_model_and_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
  File "/home/ubuntu/anaconda3/envs/trans4.35.0/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 755, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "/home/ubuntu/anaconda3/envs/trans4.35.0/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 2024, in from_pretrained
    return cls._from_pretrained(
  File "/home/ubuntu/anaconda3/envs/trans4.35.0/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 2256, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
  File "/home/ubuntu/.cache/huggingface/modules/transformers_modules/chatglm2-6b/tokenization_chatglm.py", line 69, in __init__
    super().__init__(padding_side=padding_side, **kwargs)
  File "/home/ubuntu/anaconda3/envs/trans4.35.0/lib/python3.10/site-packages/transformers/tokenization_utils.py", line 367, in __init__
    self._add_tokens(
  File "/home/ubuntu/anaconda3/envs/trans4.35.0/lib/python3.10/site-packages/transformers/tokenization_utils.py", line 467, in _add_tokens
    current_vocab = self.get_vocab().copy()
  File "/home/ubuntu/.cache/huggingface/modules/transformers_modules/chatglm2-6b/tokenization_chatglm.py", line 108, in get_vocab
    vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
  File "/home/ubuntu/.cache/huggingface/modules/transformers_modules/chatglm2-6b/tokenization_chatglm.py", line 104, in vocab_size
    return self.tokenizer.n_words
AttributeError: 'ChatGLMTokenizer' object has no attribute 'tokenizer'. Did you mean: 'tokenize'?
[2023-11-23 15:00:08,992] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 194383) of binary: /home/ubuntu/anaconda3/envs/trans4.35.0/bin/python
```
