import os
from safetensors import safe_open

def extract_prune_config(model_path):
    import re
    # Search for the pattern in the entire path
    pattern = r'h(\d+(?:\.\d+)?)-i(\d+(?:\.\d+)?)-a(\d+(?:\.\d+)?)-d(\d+(?:\.\d+)?)'
    match = re.search(pattern, model_path)
    
    if not match:
        raise ValueError("Could not find valid pruning configuration in the model path")
    
    try:
        prune_config = {
            "width_hidden": float(match.group(1)),
            "width_intermediate": float(match.group(2)),
            "width_attn": float(match.group(3)),
            "depth": float(match.group(4))
        }
        return prune_config
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing pruning configuration: {str(e)}")

def load_safetensors_to_dict(directory):
    safetensors_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    safetensors_dict[key] = f.get_tensor(key)
    return safetensors_dict

def construct_layer_dict(safetensors_dict, num_hidden_layers):
    layer_dict = {}
    is_mamba_layer = [False for _ in range(num_hidden_layers)]
    prefix = "model.layers."
    for full_key, tensor in safetensors_dict.items():
        if full_key.startswith(prefix):
            parts = full_key[len(prefix):].split('.', 1)
            layer_id = int(parts[0])
            param_name = parts[1]
            if layer_id not in layer_dict:
                layer_dict[layer_id] = {}
            if "mamba" in param_name:
                is_mamba_layer[layer_id] = True
            layer_dict[layer_id][param_name] = tensor
    return layer_dict, is_mamba_layer
