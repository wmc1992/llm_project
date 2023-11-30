版本 v4.35.0 的 transfermers 引入了不少新特性，但也产生了很多兼容性问题，这个 qwen 模型使用 v4.35.0 版本的 transformers 不能直接训练，报错如下所示：

```
Traceback (most recent call last):
  File "/data/llm_project/v4.35.0/qwen/sft/run_clm.py", line 820, in <module>
    main()
  File "/data/llm_project/v4.35.0/qwen/sft/run_clm.py", line 806, in main
    model = init_for_lora(training_args, model)
  File "/data/llm_project/v4.35.0/qwen/sft/run_clm.py", line 645, in init_for_lora
    model.gradient_checkpointing_enable()
  File "/home/ubuntu/anaconda3/envs/trans4.35.0/lib/python3.10/site-packages/transformers/modeling_utils.py", line 1872, in gradient_checkpointing_enable
    self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
TypeError: QWenPreTrainedModel._set_gradient_checkpointing() got an unexpected keyword argument 'enable'
[2023-11-23 14:35:47,101] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 190743) of binary: /home/ubuntu/anaconda3/envs/trans4.35.0/bin/python
```
