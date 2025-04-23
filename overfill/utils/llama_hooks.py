import torch
import torch.nn as nn

# set up the initial hooks for all the corresponding layers
from transformers import LlamaForCausalLM

class LinearOutputPruningHook:
    """Hook to apply channel pruning masks during forward pass"""
    def __init__(self, weight_mask, layer_idx=None):
        self.weight_mask = weight_mask.view(1, 1, -1)
        self.layer_idx = layer_idx

    def __call__(self, module, input_tensor, output_tensor):
        # print(f"Applying output mask to {module.__class__.__name__} layer {self.layer_idx}", output_tensor.shape, self.weight_mask.shape)
        
        # Convert weight mask to same dtype as output tensor
        weight_mask = self.weight_mask.to(device=output_tensor.device, dtype=output_tensor.dtype)
        assert weight_mask.size(-1) == output_tensor.size(-1), f"weight_mask {weight_mask.size(-1)} != output_tensor.size(-1) {output_tensor.size(-1)}"
        masked_output = output_tensor * weight_mask
        return masked_output.to(dtype=output_tensor.dtype)

class OutputProjectionPruningHook:
    """Hook to apply channel pruning masks during forward pass"""
    def __init__(self, weight_mask, layer_idx=None):
        self.weight_mask = weight_mask
        self.layer_idx = layer_idx

    def __call__(self, module, input_tensor, output_tensor):
        # print(f"Applying output mask to {module.__class__.__name__} layer {self.layer_idx}", output_tensor.shape, self.weight_mask.shape)
        
        weight_mask_keep = self.weight_mask.to(device=input_tensor[0].device).bool()
        weight_mask_remove = ~weight_mask_keep
        
        from torch.nn import functional as F
        pruned_weight = module.weight[weight_mask_keep, :]
        # # weight shape: output_dim x input_dim
        # masked_weight = module.weight * weight_mask_keep[:, None]
        pruned_output = F.linear(input_tensor[0], pruned_weight, module.bias)
        
        # Convert weight mask to same dtype as output tensor
        weight_mask = self.weight_mask.to(device=output_tensor.device, dtype=output_tensor.dtype)
        assert weight_mask.size(-1) == output_tensor.size(-1), f"weight_mask {weight_mask.size(-1)} != output_tensor.size(-1) {output_tensor.size(-1)}"
        masked_output = output_tensor * weight_mask
        return masked_output.to(dtype=output_tensor.dtype)
    
class InputProjectionPruningHook:
    """Hook to apply channel pruning masks during forward pass"""
    def __init__(self, weight_mask, layer_idx=None):
        self.weight_mask = weight_mask
        self.layer_idx = layer_idx

    def __call__(self, module, input_tensor, output_tensor):
        # print(f"Applying input mask to {module.__class__.__name__} layer {self.layer_idx}", input_tensor[0].shape, self.weight_mask.shape)
        
        # breakpoint()
        weight_mask_keep = self.weight_mask.to(device=input_tensor[0].device).bool()
        weight_mask_remove = ~weight_mask_keep
        assert torch.all(input_tensor[0][:, :, weight_mask_remove] == 0)
        
        from torch.nn import functional as F
        pruned_input = input_tensor[0][:, :, weight_mask_keep]
        pruned_weight = module.weight[:, weight_mask_keep]
        return F.linear(pruned_input, pruned_weight, module.bias)
        
        module.weight[weight_mask_remove] = 0
        # Convert weight mask to same dtype as output tensor
        weight_mask = self.weight_mask.to(device=input_tensor[0].device, dtype=input_tensor[0].dtype)
        assert weight_mask.size(-1) == input_tensor[0].size(-1), f"weight_mask {weight_mask.size(-1)} != input_tensor[0].size(-1) {input_tensor[0].size(-1)}"
        # here we need to apply the mask to the input tensor
        masked_input = input_tensor[0] * weight_mask
        return module.forward(masked_input)
    
class LayerNormPruningHook:
    """Hook to apply channel pruning masks during forward pass"""
    def __init__(self, weight_mask, layer_idx=None):
        self.weight_mask = weight_mask.view(1, 1, -1)
        # count nonzero elements in weight_mask
        self.active_dims = weight_mask.squeeze(0).squeeze(0).sum().item()
        self.layer_idx = layer_idx
        
    def __call__(self, module, input_tensor, output_tensor):
        # print(f"Applying layer norm mask to {module.__class__.__name__} layer {self.layer_idx}", output_tensor.shape, self.weight_mask.shape)
        
        weight_mask = self.weight_mask.to(device=input_tensor[0].device, dtype=torch.float32)
        assert weight_mask.size(-1) == input_tensor[0].size(-1), f"weight_mask {weight_mask.size(-1)} != input_tensor[0].size(-1) {input_tensor[0].size(-1)}"
        weight = module.weight
        variance_epsilon = module.variance_epsilon
        
        # override the operation of layer norm
        hidden_states = input_tensor[0]
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        masked_states = hidden_states * weight_mask
        
        # Calculate variance only over non-zero elements
        variance = (masked_states.pow(2).sum(-1, keepdim=True) / self.active_dims)
        
        # Normalize the masked states
        normalized_states = masked_states * torch.rsqrt(variance + variance_epsilon)
        
        output = weight * normalized_states.to(input_dtype)
        return output
    
class SanityCheckHook:
    """Hook to check the output of the model"""
    def __init__(self, layer_idx=None):
        self.layer_idx = layer_idx

    def __call__(self, module, input_tensor, output_tensor):
        print(f"Checking in/out of {module.__class__.__name__} layer {self.layer_idx}")
        print("input_tensor[0].shape:", input_tensor[0].shape)
        print("output_tensor.shape:", output_tensor.shape)
        # count the number of non-zero elements in input_tensor[0]
        non_zero_count = (input_tensor[0][0, 0] != 0).sum().item()
        print(f"Number of non-zero elements in input_tensor[0]: {non_zero_count}")
        return output_tensor

class LayerPruningHook:
    """Hook to skip pruned layers during forward pass"""
    def __init__(self, keep_layer: bool, layer_idx: int):
        self.keep_layer = keep_layer
        self.layer_idx = layer_idx

    def __call__(self, module, input_tensor, output_tensor):
        if not self.keep_layer:
            # Skip this layer by returning the input
            return input_tensor  # input_tensor is a tuple, we want the first element
        return output_tensor

def register_pruning_hooks(model: LlamaForCausalLM, weight_masks: dict, depth_only: bool = False):
    """Register forward hooks for pruning on relevant layers"""
    hooks = []
    
    # For layer pruning
    if 'layer' in weight_masks:
        for layer_idx, keep_layer in enumerate(weight_masks['layer']):
            hook = model.model.layers[layer_idx].register_forward_hook(
                LayerPruningHook(keep_layer, layer_idx)
            )
            hooks.append(hook)

    if depth_only:
        return hooks
    
    # For embedding pruning
    emb_mask = weight_masks['hidden']  # [hidden_dim]
    hook = model.model.embed_tokens.register_forward_hook(
        LinearOutputPruningHook(emb_mask)
    )
    hooks.append(hook)
    
    # For MLP and attention projection pruning
    for layer_idx in range(model.config.num_hidden_layers):
        mlp_mask = weight_masks[f'mlp_{layer_idx}']  # [mlp_dim]
        
        # # Attention QKVO ====
        # # NOTE: maybe qkv hooks are not needed
        # hook = model.model.layers[layer_idx].self_attn.q_proj.register_forward_hook(
        #     InputProjectionPruningHook(emb_mask, layer_idx)
        # )
        # hooks.append(hook)

        # hook = model.model.layers[layer_idx].self_attn.k_proj.register_forward_hook(
        #     InputProjectionPruningHook(emb_mask, layer_idx)
        # )
        # hooks.append(hook)

        # hook = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(
        #     InputProjectionPruningHook(emb_mask, layer_idx)
        # )
        # hooks.append(hook)

        hook = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
            OutputProjectionPruningHook(emb_mask, layer_idx)
        )
        hooks.append(hook)
        
        # LayerNorm ====
        # apply to input layernorm = input of self attention
        hook = model.model.layers[layer_idx].input_layernorm.register_forward_hook(
            LayerNormPruningHook(emb_mask, layer_idx)
        )
        hooks.append(hook)
        
        # apply to post attention layernorm = input of mlp
        hook = model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
            LayerNormPruningHook(emb_mask, layer_idx)
        )
        hooks.append(hook)
        
        # MLP ====
        # Apply to up_proj (first MLP layer)
        hook = model.model.layers[layer_idx].mlp.up_proj.register_forward_hook(
            LinearOutputPruningHook(mlp_mask, layer_idx)
        )
        hooks.append(hook)
        
        # Apply to gate_proj (parallel MLP layer)
        hook = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(
            LinearOutputPruningHook(mlp_mask, layer_idx)
        )
        hooks.append(hook)
        
        # Apply to down_proj (final MLP projection)
        hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(
            LinearOutputPruningHook(emb_mask, layer_idx)
        )
        hooks.append(hook)
        
    # final layer norm ====
    hook = model.model.norm.register_forward_hook(
        LayerNormPruningHook(emb_mask, "final")
    )
    hooks.append(hook)
    
    # # check if the final layer output is masked
    # hook = model.lm_head.register_forward_hook(
    #     SanityCheckHook()
    # )
    # hooks.append(hook)
    
    return hooks

def remove_hooks(hooks):
    """Remove registered pruning hooks"""
    for hook in hooks:
        hook.remove()
