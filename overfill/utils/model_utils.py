from einops import rearrange
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from alignment import DataArguments, ModelArguments
from alignment.data import DEFAULT_CHAT_TEMPLATE

import torch
import torch.nn as nn
import os


ABBV2FULL = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "bloom-560m": "bigscience/bloom-560m",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "opt-350m": "facebook/opt-350m",
    "litellama": "ahxt/LiteLlama-460M-1T",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
}
FUUL2ABBV = {v: k for k, v in ABBV2FULL.items()}

TEACHER_INPUT_MODES = ["desc", "desc_in", "desc_in_out", "desc_halfin", "desc_80in", "desc_inst"]
STUDENT_INPUT_MODES = ["desc", "desc_in", "in", "in_out", "dummy", "halfin", "20in"]
STUDENT_OUTPUT_MODES = ["in", "out"]
FEWSHOT_MODES = ["instruction", "fewshot", "both"]
FREEZE_STRATEGIES = ["teacher", "teacher_and_student", "student", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = [
    "last_n",
    "last_and_project",
    "pool_and_project",
    "select_layer_all",
    "select_channel",
    "select_channel_identity",
    "select_channel_linear_softmax",
    "layerwise_pool_and_project",
    "layerwise_pool_and_share_project",
    "layerwise_share_pool_and_share_project",
    "layerwise_last_and_project",
    "layerwise_last_and_share_project",
    "past_kv_copy",
    "past_kv_project_sharelayer_sharepatch",
    "past_kv_project_sharelayer",
    "past_kv_project_sharepatch",
    "past_kv_project_sharepatch_desc",
    "past_kv_mlp_layer",
    "past_kv_tf_layer",
    "past_kv_unet_layer",
    "past_kv_sliding_window_project_sharepatch"
]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    if hasattr(model, "teacher"):
        teacher = count_parameters(model.teacher)
        teacher_trainable = count_trainable_parameters(model.teacher)
        student = count_parameters(model.student)
        student_trainable = count_trainable_parameters(model.student)
        print(
            f"Teacher: {teacher}, trainable {teacher_trainable} ({teacher_trainable/teacher*100:.3f}%), dtype={model.teacher_dtype}"
        )
        print(
            f"Student: {student}, trainable {student_trainable} ({student_trainable/student*100:.3f}%), dtype={model.student_dtype}"
        )
        if hasattr(model, "embedding_proj") and model.embedding_proj is not None:
            embedding_proj = count_trainable_parameters(model.embedding_proj)
            print(
                f"embedding_projection: {embedding_proj}, ({embedding_proj/teacher*100:.3f}% of teacher, {embedding_proj/student*100:.3f}% of student)"
            )
        if hasattr(model, "patch_encoder") and model.patch_encoder is not None:
            patch_encoder = count_trainable_parameters(model.patch_encoder)
            print(
                f"Patch Encoder: {patch_encoder}, ({patch_encoder/teacher*100:.3f}% of teacher, {patch_encoder/student*100:.3f}% of student)"
            )
        print()

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    print(f"Total Parameters: {total}")
    print(f"Total Trainable Parameters: {trainable}")
    print(f"Ratio (%): {trainable/total*100:.3f}")


def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}")


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        # do not freeze pte
        if "pte" in name:
            print("skipping pte")
            continue
        params.requires_grad = False
        total_num_params += params.numel()
    print(f"Froze {total_num_params} params from model type {type(model)}")


def rearrange_kv_tuple(kv, agg_func, attention_mask, patch_len):
    # extract last patch_len tokens from each key and value and stack
    # 2 * [batch_size, num_heads, seq_len, teacher_dim // num_heads]
    # -> [2, batch_size, patch_len, teacher_dim]
    assert len(kv) == 2
    assert len(kv[0].shape) == 4
    assert len(kv[1].shape) == 4

    if agg_func is None:
        # set to identity function
        def agg_func(x, mask, patch_len):
            return x

    key = rearrange(kv[0], 'b h s d -> b s (h d)')
    value = rearrange(kv[1], 'b h s d -> b s (h d)')

    return torch.stack((
        agg_func(key, attention_mask, patch_len),
        agg_func(value, attention_mask, patch_len)
    ), dim=0)


def extract_last_n(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, n: int
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    seq_len = attention_mask.sum(dim=1)
    assert torch.any(seq_len >= n), f"Some sequences are shorter than {n} tokens"
    select_indices = torch.stack([torch.arange(l - n, l) for l in seq_len])
    extracted_outputs = hidden_states[torch.arange(B)[:, None], select_indices]
    assert extracted_outputs.shape == (B, n, D)
    return extracted_outputs


def mean_sliding_window(hidden_states: torch.Tensor, attention_mask: torch.Tensor, n: int) -> torch.Tensor:
    B, S, D = hidden_states.shape
    seq_len = attention_mask.sum(dim=1)
    assert torch.any(seq_len >= n), f"Some sequences are shorter than {n} tokens"
    # split seq_len into n splits
    chunked_hidden = [torch.tensor_split(hidden_states[i][:seq_len[i]], n) for i in range(B)]
    # # check each chunk has the same length
    # assert all(len(chunks) == n for chunks in chunked_hidden):
    # apply mean pooling to each chunk
    pooled_outputs = torch.stack([torch.stack([chunk.mean(dim=0) for chunk in chunks]) for chunks in chunked_hidden])
    assert pooled_outputs.shape == (B, n, D)
    return pooled_outputs


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    assert attention_mask.shape == (B, S)
    attention_mask = attention_mask.unsqueeze(-1).expand(B, S, D)
    hidden_states = hidden_states.masked_fill(~attention_mask, 0)
    pooled_outputs = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)
    return pooled_outputs

    # unmasked_outputs = hidden_states * attention_mask[..., None]
    # pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    # assert pooled_outputs.shape == (B, D)
    # return pooled_outputs


def max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    assert attention_mask.shape == (B, S)
    attention_mask = attention_mask.unsqueeze(-1).expand(B, S, D)
    hidden_states = hidden_states.masked_fill(~attention_mask, -1e9)
    pooled_outputs = hidden_states.max(dim=1).values
    return pooled_outputs

    # unmasked_outputs = hidden_states * attention_mask[..., None]
    # pooled_outputs = unmasked_outputs.max(dim=1).values
    # assert pooled_outputs.shape == (B, D)
    # return pooled_outputs


def stack_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.reshape((B, S * D))  # stack along seq length
    assert pooled_outputs.shape == (B, S * D)
    return pooled_outputs


def model_config_sanity_check(model):
    config = model.config
    is_llama = isinstance(model, LlamaForCausalLM)
    if not hasattr(config, "n_layer"):
        config.n_layer = config.num_hidden_layers

    if is_llama:
        config.head_dim = model.model.layers[0].self_attn.head_dim
        config.num_kv_heads = model.model.layers[0].self_attn.num_key_value_heads
        config.kv_dim = config.num_kv_heads * config.head_dim
    elif hasattr(config, "num_key_value_heads"):
        head_dim = config.hidden_size // config.num_attention_heads
        kv_dim = config.num_key_value_heads * head_dim

        # overwrite the config
        config.head_dim = head_dim
        config.kv_dim = kv_dim
        config.num_kv_heads = config.num_key_value_heads
    else:
        head_dim = config.hidden_size // config.num_attention_heads

        # overwrite the config
        config.head_dim = head_dim
        config.kv_dim = config.hidden_size
        config.num_kv_heads = config.num_attention_heads


def get_model(
    model_name_or_path,
    model_kwargs,
    cache_dir=None
):
    if cache_dir:
        model_name_or_path = os.path.join(cache_dir, model_name_or_path)
        model_kwargs["local_files_only"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model_config_sanity_check(model)
    return model


def get_tokenizer(
    tokenizer_name_or_path: str,
    model_args: ModelArguments,
    data_args: DataArguments,
    auto_set_chat_template: bool = True,
    cache_dir=None,
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model. Modified from aligment/model_utils.py"""
    tokenizer_args = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if cache_dir:
        tokenizer_name_or_path = os.path.join(cache_dir, tokenizer_name_or_path)
        tokenizer_args["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer._pad_token = tokenizer.eos_token
        print(f"Set pad_token_id to {tokenizer.pad_token_id}")
    if not hasattr(tokenizer, "_pad_token"):
        tokenizer._pad_token = tokenizer.eos_token
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer
