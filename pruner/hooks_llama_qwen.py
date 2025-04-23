import torch
import torch.nn as nn

# set up the initial hooks for all the corresponding layers
from transformers import LlamaForCausalLM, Qwen2ForCausalLM


def delete_importance_attr(layer: nn.Module):
    if hasattr(layer, "calculated_importance"):
        del layer.calculated_importance


def remove_all_forward_hooks(model):
    assert isinstance(model, LlamaForCausalLM) or isinstance(model, Qwen2ForCausalLM), "Only LlamaForCausalLM or Qwen2ForCausalLM is supported"
    print("Removing all forward hooks")
    for layer in model.model.layers:
        layer.self_attn.o_proj._forward_hooks.clear()
        layer.input_layernorm._forward_hooks.clear()
        layer.post_attention_layernorm._forward_hooks.clear()
        layer.mlp.down_proj._forward_hooks.clear()
        delete_importance_attr(layer.self_attn.o_proj)
        delete_importance_attr(layer.input_layernorm)
        delete_importance_attr(layer.post_attention_layernorm)
        delete_importance_attr(layer.mlp.down_proj)


def register_all_forward_hooks(model: nn.Module):
    assert isinstance(model, LlamaForCausalLM) or isinstance(model, Qwen2ForCausalLM), "Only LlamaForCausalLM or Qwen2ForCausalLM is supported"
    print("Registering all forward hooks")
    for layer in model.model.layers:
        # self attention
        layer.self_attn.o_proj.register_forward_hook(aggregate_input_hook)
        
        # ln
        layer.input_layernorm.register_forward_hook(aggregate_output_hook)
        layer.post_attention_layernorm.register_forward_hook(aggregate_output_hook)
        
        # ffn
        layer.mlp.down_proj.register_forward_hook(aggregate_input_hook)

def aggregate_input_hook(module, ins, outs):
    # print("Add input hook to", module.__class__.__name__)
    # mean over the sequence length
    scores = ins[0].detach().cpu().mean(1)
    if hasattr(module, "calculated_importance"):
        module.calculated_importance = torch.cat((module.calculated_importance, scores), dim=0)
    else:
        module.calculated_importance = scores


def aggregate_output_hook(module, ins, outs):
    # print("Add output hook to", module.__class__.__name__)
    # mean over the sequence length
    scores = outs.detach().cpu().mean(1)
    if hasattr(module, "calculated_importance"):
        module.calculated_importance = torch.cat((module.calculated_importance, scores), dim=0)
    else:
        module.calculated_importance = scores
