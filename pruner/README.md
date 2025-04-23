# Model Pruner

Utilities for static pruning of Transformer language models, currently supporting Llama models. Implements the [Minitron-style](https://arxiv.org/abs/2407.14679) pruning by NVIDIA.

Modified from: https://github.com/alperiox/Compact-Language-Models-via-Pruning-and-Knowledge-Distillation. 


## 🚀 Features

- Supports **channel** and **depth pruning** (Attention head pruning is currently not supported)
- Configurable via YAML
- Prunes using ratio or fixed width


## 🛠️ Pruning Options

The following pruning dimensions are supported:

- `width_hidden`:  
  - Ratio or number of hidden channels to keep
- `width_intermediate`:  
  - Ratio or number of intermediate MLP channels to keep
- `depth`:  
  - Ratio of transformer blocks to keep


## 📁 Configs

Example pruning configs used in our paper are available in the `configs/` directory.

## 🔧 Usage

To prune a Llama 3B model, run:

```bash
python main.py --config configs/llama3b_instruct.yaml
```
Optional:
* Add --push_to_hub to upload to Hugging Face Hub (Make sure to set the HF_TOKEN environment variable)
* The script will also save indices of remaining channels and blocks in a .pkl file