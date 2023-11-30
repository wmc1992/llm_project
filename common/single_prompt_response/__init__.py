from common.single_prompt_response.default import build_source_and_target_default
from common.single_prompt_response.llama import build_source_and_target_from_llama
from common.single_prompt_response.standard_alpaca import build_source_and_target_from_alpaca_prompt
from common.single_prompt_response.ziya import build_source_and_target_from_ziya_prompt

prompt_type_to_func = {
    "default": build_source_and_target_default,
    "llama": build_source_and_target_from_llama,
    "standard_alpaca": build_source_and_target_from_alpaca_prompt,
    "ziya": build_source_and_target_from_ziya_prompt
}
