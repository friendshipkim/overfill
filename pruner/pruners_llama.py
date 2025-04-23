import torch.nn as nn
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from utils import get_kept_layers_prune


def prune_linear(module, mask, axis="out") -> None:
    # Linear(in_features, out_features): weight shape (out_features, in_features)
    if isinstance(module, nn.Linear):
        if axis == "out":
            module.out_features = mask.sum()
            # check if already pruned
            if module.weight.data.shape[0] == mask.sum():
                return
            module.weight.data = module.weight.data[mask, :]
            if module.bias is not None:
                module.bias.data = module.bias.data[mask]
        elif axis == "in":
            module.in_features = mask.sum()
            # check if already pruned
            if module.weight.data.shape[1] == mask.sum():
                return
            module.weight.data = module.weight.data[:, mask]


def prune_layernorm(module, mask) -> None:
    if isinstance(module, LlamaRMSNorm) or isinstance(module, nn.LayerNorm) or isinstance(module, Qwen2RMSNorm):
        module.weight.data = module.weight.data[mask]
        module.normalized_shape = mask.sum()
    else:
        raise ValueError(f"Unsupported layer norm type: {type(module)}")


def prune_embedding(module, mask) -> None:
    if isinstance(module, nn.Embedding):
        # apply mask to the embedding
        module.weight.data = module.weight.data[:, mask]
    # change the embedding size
    module.embedding_dim = mask.sum()


def prune_intermediate(model, ratio_or_value=0.2, batch_agg="l2", is_random=False) -> None:
    # goal: trim the MLP layer weights
    intermediate_size = model.config.intermediate_size
    if type(ratio_or_value) is int:
        assert ratio_or_value <= intermediate_size
        intermediate_size_pruned = ratio_or_value
    elif type(ratio_or_value) is float:
        intermediate_size_pruned = int((1 - ratio_or_value) * intermediate_size)
    else:
        raise ValueError(f"Invalid intermediate size: {ratio_or_value}")
    
    idx_list = []
    for layer in model.model.layers:
        # fetch the importances
        importances = layer.mlp.down_proj.calculated_importance
    
        # aggregate batch dimension
        if batch_agg == "l1":
            importances = importances.norm(dim=0, p=1)
        elif batch_agg == "l2":
            importances = importances.norm(dim=0, p=2)
        else:
            raise ValueError(f"Invalid batch aggregation method: {batch_agg}")
        
        idx = importances.argsort(descending=True)[:intermediate_size_pruned]
        
        if is_random:
            idx = torch.randperm(intermediate_size)[:intermediate_size_pruned]
        
        idx_list.append(idx)
        
        # make binary mask from idx
        mask = torch.zeros(intermediate_size).bool()
        mask[idx] = True
        
        # prune weights
        prune_linear(layer.mlp.gate_proj, mask, axis="out")
        prune_linear(layer.mlp.up_proj, mask, axis="out")
        prune_linear(layer.mlp.down_proj, mask, axis="in")
    print("MLP pruned!")
    model.config.intermediate_size = intermediate_size_pruned
    return model, idx_list


def prune_attn_heads(model, ratio_or_value=0.2, batch_agg="l2", is_random=False) -> None:
    if ratio_or_value == 0.0:
        return model, None
    
    for layer in model.model.layers:
        importances = layer.self_attn.o_proj.calculated_importance
        
        # aggregate batch dimension
        if batch_agg == "l1":
            importances = importances.norm(dim=0, p=1)
        elif batch_agg == "l2":
            importances = importances.norm(dim=0, p=2)
        else:
            raise ValueError(f"Invalid batch aggregation method: {batch_agg}")
        
        # TODO prune heads
            
            
def prune_hidden(model, ratio_or_value=0.2, batch_agg="l2", is_random=False) -> None:
    # goal: trim the hidden dimension of the weight matrices in MLP, MHA, and LayerNorm layers.
    hidden_size = model.config.hidden_size
    if type(ratio_or_value) is int:
        assert ratio_or_value <= hidden_size
        hidden_size_pruned = ratio_or_value
    elif type(ratio_or_value) is float:
        hidden_size_pruned = int((1 - ratio_or_value) * hidden_size)
    else:
        raise ValueError(f"Invalid hidden size: {ratio_or_value}")
    
    # fetch the importances across all layers
    importances = [
        abs(layer.input_layernorm.calculated_importance) + abs(layer.post_attention_layernorm.calculated_importance)
        for layer in model.model.layers
    ]
    importances = torch.stack(importances, dim=0)
    # reduce layer dimension
    importances = importances.sum(dim=0)
    
    # reduce batch dimension
    if batch_agg == "l1":
        importances = importances.norm(dim=0, p=1)
    elif batch_agg == "l2":
        importances = importances.norm(dim=0, p=2)
    else:
        raise ValueError(f"Invalid batch aggregation method: {batch_agg}")
    
    idx = importances.argsort(descending=True)[:hidden_size_pruned]
    
    if is_random:
        idx = torch.randperm(hidden_size)[:hidden_size_pruned]
    
    # now let's prune the model
    # make binary mask from idx
    mask = torch.zeros(hidden_size).bool()
    mask[idx] = True
    
    # embedding layer
    prune_embedding(model.model.embed_tokens, mask)
    for layer in model.model.layers:
        prune_layernorm(layer.input_layernorm, mask)
        prune_linear(layer.self_attn.q_proj, mask, axis="in")
        prune_linear(layer.self_attn.k_proj, mask, axis="in")
        prune_linear(layer.self_attn.v_proj, mask, axis="in")
        prune_linear(layer.self_attn.o_proj, mask, axis="out")
        
        prune_linear(layer.mlp.gate_proj, mask, axis="in")
        prune_linear(layer.mlp.up_proj, mask, axis="in")
        prune_linear(layer.mlp.down_proj, mask, axis="out")
        prune_layernorm(layer.post_attention_layernorm, mask)
    
    # ln
    prune_layernorm(model.model.norm, mask)
        
    # lm head is tied to the embedding layer
    prune_linear(model.lm_head, mask, axis="in")
    model.tie_weights()
    # assert torch.allclose(model.lm_head.weight.data, model.model.embed_tokens.weight.data)
    
    model.config.hidden_size = hidden_size_pruned
    print("Embeddings pruned!")
    return model, idx


def prune_depth(model, ratio=0.0, strategy="firstlast") -> None:
    if ratio == 0.0:
        return model, list(range(len(model.model.layers)))

    n_layers = len(model.model.layers)
    n_layers_kept = int((1 - ratio) * n_layers)
    idx = get_kept_layers_prune(strategy, n_layers, ratio)
    
    # prune layers
    new_layers = nn.ModuleList([model.model.layers[i] for i in idx])
    model.model.layers = new_layers
    model.config.num_hidden_layers = n_layers_kept
    print(f"Depth pruned: {n_layers} -> {n_layers_kept}")
    return model, idx


PRUNING_FUNCTIONS = {
    "width_attn": prune_attn_heads,
    "width_hidden": prune_hidden,
    "width_intermediate": prune_intermediate,
    "depth": prune_depth,
}
