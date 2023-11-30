# llm_project

## 训练和推理

#### transformers版本: v4.35.0

> 版本 v4.35.0 的 transformers 引入了很多的新特性，但也引入了很多的兼容性问题，导致了下表中有些模型训练代码无法使用。

|模型|训练代码能<br>否正常使用|推理代码能<br>否正常使用|文档|
|---|:-:|:-:|:-:|
|chatglm|否|否|[报错说明](v4.35.0/chatglm/sft/README.md)|
|chatglm2|否|否|[报错说明](v4.35.0/chatglm2/sft/README.md)|
|baichuan2|否|否|-|
|qwen|否|否|[报错说明](v4.35.0/qwen/sft/README.md)|
|llama|是|否|-|
|llama2|否|否|-|
|mistral|否|否|-|

#### transformers版本: v4.28.0

|模型|训练代码能<br>否正常使用|推理代码能<br>否正常使用|文档|
|---|:-:|:-:|:-:|
|chatglm|是|否|-|
|chatglm2|是|否|-|
|baichuan2|是|否|-|
|qwen|是|否|-|
|llama|是|否|-|

## 其他功能

* 将 lora 部分合并到模型结构中：[run_merge_model_with_lora.sh](utils/run_merge_model_with_lora.sh)
