import os
import torch

import bitsandbytes as bnb
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    BitsAndBytesConfig
)
from peft.tuners.lora import LoraLayer

from transformers.pipelines.conversational import Conversation

import gradio as gr
import mdtex2html


# ----------------------------
# load llama2 model
# ----------------------------
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    # if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
        })
        tokenizer.pad_id = 0

    print(model)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    model.eval()
    return model, tokenizer


class Args:
    model_name_or_path = "/data/pretrained_models/openbuddy-llama2-70b-v10.1-bf16"
    bits = 8
    max_memory_MB = 44000
    fp16 = True
    bf16 = False
    cache_dir = None
    double_quant = True
    quant_type = "nf4"
    trust_remote_code = False
    use_auth_token = False

args = Args()
model, tokenizer = get_accelerate_model(args, checkpoint_dir=None)


# ----------------------------
# inference with llama2
# ----------------------------
def build_dialog(input, history):
    dialog = []
    for one_turn in history:
        dialog.append({"role": "user", "content": one_turn[0]})
        dialog.append({"role": "assistant", "content": one_turn[1]})
    dialog.append({"role": "user", "content": input})
    return dialog


def build_conversation(input, history):
    dialog = build_dialog(input, history)

    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]

    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    assert (dialog[-1]["role"] == "user"), f"Last message must be from user, got {dialog[-1]['role']}"

    print("dialog:")
    print(dialog)

    conversation = Conversation(
        text=dialog[-1]["content"],
        past_user_inputs=[item["content"] for item in dialog[:-1] if item["role"] == "user"],
        generated_responses=[item["content"] for item in dialog[:-1] if item["role"] == "assistant"]
    )
    return conversation


def generate(input, max_length, top_p, temperature, history):
    """ 该函数是需要等待一条数据全部推理完成之后才能返回，页面体验不友好，暂时不使用该函数，改为使用函: stream_generate() """

    conversation = build_conversation(input, history)

    input_ids = tokenizer._build_conversation_input_ids(conversation)
    input_ids = torch.tensor([input_ids])
    print(len(input_ids[0]))

    if len(input_ids[0]) > max_length - 128:
        return f"输入token总长度超过了 max_len-128，输入token长度：{len(input_ids[0])}，max_length：{max_length}。请清空历史数据开始新的对话。"

    out = model.generate(
        input_ids=input_ids.cuda(),
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
    )

    in_text = tokenizer.decode(input_ids[0])
    print(in_text + "\n\n")
    out_text = tokenizer.decode(out[0])
    print(out_text + "\n\n")
    out_text = out_text[len(in_text) + 1: -len("</s>")].strip()
    print(out_text + "\n\n")
    return out_text


def sample_top_p(probs, p):
    # probs: [bs, 1, vocab_size]

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


def stream_generate(input, max_length, top_p, temperature, history):
    """ 流式生成文本，页面交互体验更友好一些 """

    with torch.no_grad():
        conversation = build_conversation(input, history)

        input_ids = tokenizer._build_conversation_input_ids(conversation)
        in_tokens = torch.tensor([input_ids]).cuda()  # [bs, seq_len]
        prompt_len = len(in_tokens[0])

        if len(in_tokens[0]) > max_length - 128:
            return f"输入token总长度超过了 max_len-128，输入token长度：{len(in_tokens[0])}，max_length：{max_length}。请清空历史数据开始新的对话。", ""

        end_token = torch.tensor([tokenizer.eos_token_id]).to(in_tokens.device)
        past_key_values = None
        total_text = ""
        out_tokens = None

        # 生成的文本达到最大长度时停止推理、遇到终止字符时停止推理
        while (out_tokens is None) or ((out_tokens[0][-1] != end_token) and (prompt_len + out_tokens.size()[1] < max_length)):
            forward_result = model(input_ids=in_tokens, past_key_values=past_key_values, use_cache=True)
            logits = forward_result.logits
            past_key_values = forward_result.past_key_values

            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                out_token = sample_top_p(probs, top_p)
            else:
                out_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            in_tokens = out_token

            if out_tokens is None:
                out_tokens = out_token
            else:
                out_tokens = torch.cat([out_tokens, out_token], dim=-1)

            total_text = tokenizer.decode(out_tokens[0])

            yield total_text


"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))

    bool_first = True
    for response in stream_generate(input, max_length, top_p, temperature, history):
        chatbot[-1] = (parse_text(input), parse_text(response))

        if bool_first:
            history.append((input, response))
            bool_first = False
        else:
            history[-1] = (input, response)

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">LLaMA2</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name='0.0.0.0', server_port=7862, show_api=False, share=False, inbrowser=False)
