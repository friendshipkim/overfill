"""
one-shot width pruning of LLAMA models
"""
import yaml
import argparse
import os
import copy
from tqdm import tqdm

import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pickle

from hooks_llama_qwen import register_all_forward_hooks, remove_all_forward_hooks
from pruners_llama import PRUNING_FUNCTIONS

device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)
torch.manual_seed(1337)


def create_new_config(config, prune_strategy):
    if type(prune_strategy["width_hidden"]) is int:
        config.hidden_size = prune_strategy["width_hidden"]
    else:
        config.hidden_size = int(config.hidden_size * (1 - prune_strategy["width_hidden"]))
    if type(prune_strategy["width_intermediate"]) is int:
        config.intermediate_size = prune_strategy["width_intermediate"]
    else:
        config.intermediate_size = int(config.intermediate_size * (1 - prune_strategy["width_intermediate"]))
    if type(prune_strategy["depth"]) is int:
        config.num_hidden_layers = prune_strategy["depth"]
    else:
        config.num_hidden_layers = int(config.num_hidden_layers * (1 - prune_strategy["depth"]))
    # attention heads todo
    # base_config.num_attention_heads = int(base_config.num_attention_heads * (1 - prune_strategy["width_attn"]))
    return config


def get_calib_data_iter(tokenizer, data="wikitext", batch_size=64, calib_size=512, max_sequence_length=512):
    if data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    else:
        # Assume a local JSON dataset with a column named "text"
        dataset = load_dataset("json", data_files=data, split="train")
        text_column = "text"
    
    # Add text length column and sort to avoid padding
    def add_length(example):
        example['length'] = len(example[text_column])
        return example
    dataset = dataset.map(add_length)
    dataset = dataset.sort('length', reverse=True)
    
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length)
        # add labels
        tokenized_batch["labels"] = tokenized_batch["input_ids"]
        yield tokenized_batch


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def check_loss(device, model, calibration_loader):
    model.to(device)
    model.eval()
    losses = []
    for batch in tqdm(calibration_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss)
    mean_loss = torch.mean(torch.tensor(losses)).item()
    return mean_loss


def run_calibration(device, model, calibration_loader):
    model.to(device)
    model.eval()

    remove_all_forward_hooks(model)
    register_all_forward_hooks(model)

    print("Running calibration")
    base_loss = check_loss(device, model, calibration_loader)
    
    # sample_batch = calibration_loader[0]
    # sample_batch.to(device)
    # model(**sample_batch)

    return base_loss


def prune(
    model,
    calibration_loader,
    device: str,
    batch_agg_func: str,
    pruning_strategy: dict[str, float | list[int] | int],
    depth_strategy: str,
):

    # Determine model type and use appropriate hooks
    model_type = model.config.model_type
    assert model_type in ["llama", "qwen2"], f"Unsupported model type: {model_type}"
    
    # initialize the base model and run through calibration data
    base_num_params = get_num_params(model)
    base_loss = run_calibration(
        device, model, calibration_loader
    )
    print(f"Base loss before pruning: {base_loss:.4f}")
    print(f"Number of parameters before pruning: {base_num_params}")
    
    # prune
    keep_idx_dict = {}

    # depth pruning first
    depth_ratio, depth_func = pruning_strategy["depth"], PRUNING_FUNCTIONS["depth"]
    print("DEPTH PRUNING | RATIO: ", depth_ratio, " | STRATEGY: ", depth_strategy)
    model, idx = depth_func(model, depth_ratio, depth_strategy)
    keep_idx_dict["depth"] = idx
    del pruning_strategy["depth"]
    
    # width pruning
    funcs = [PRUNING_FUNCTIONS[s] for s, _ in pruning_strategy.items()]
    names = [s for s, _ in pruning_strategy.items()]
    ratios = [constr for _, constr in pruning_strategy.items()]
            
    # # run random pruning
    # print("Running random pruning")
    # random_model = deepcopy(model)
    # for f, r in zip(pruning_funcs, constraints):
    #     f(random_model, r, batch_agg_func, is_random=True)
    # random_model, random_num_params, random_loss = get_model_with_importances(device, random_model, calibration_loader)
    
    # prune
    print("WIDTH PRUNING")
    for f, name, r in zip(funcs, names, ratios):
        print(f"Running {name} with ratio {r}")
        _, idx =f(model, r, batch_agg_func, is_random=False)
        keep_idx_dict[name] = idx
    remove_all_forward_hooks(model)
    
    # pruned model
    print("-" * 100)
    print(model)
    pruned_num_params = get_num_params(model)
    pruned_loss = check_loss(device, model, calibration_loader)

    print(
        f"{'Number of training parameters after pruning:':60} {pruned_num_params}"
    )
    print(
        f"{'Ratio of the pruned size to the base model:':60} {pruned_num_params/base_num_params*100:.2f}%"
    )
    print(f"{'Pruned evaluation loss (before re-training):':60} {pruned_loss:.4f}")
    # print(f"{'Random evaluation loss (before re-training):':60} {random_loss:.4f}")

    return model, keep_idx_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='configs/llama3b_instruct.yaml', help='Path to config YAML file')
    parser.add_argument('--pruned_idx_dir', type= str, default='pruned_idx', help='Path to directory to save pruned idx')
    parser.add_argument('--push_to_hub', action='store_true', help='Push pruned model to HF Hub')
    args = parser.parse_args()

    # Load YAML config and add config to args
    print(f"Loading prune config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)
    
    # load model and tokenizer
    print(f"Loading model from HF: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = AutoConfig.from_pretrained(args.model_name)
    
    # if pad_token is not set, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer._pad_token = tokenizer.eos_token
        print(f"Set pad_token_id to {tokenizer.pad_token_id}")
    
    # prepare calibration data
    data_iter = get_calib_data_iter(
        tokenizer,
        args.dataset_name,
        args.batch_size,
        args.calib_size,
        args.max_seq_len,
    )
    # first_batch = next(data_iter)
    dataloader = [data for data in data_iter]
    
    # prune
    prune_strategy = {
        "width_hidden": args.width_hidden,
        "width_intermediate": args.width_intermediate,
        "width_attn": args.width_attn,
        "depth": args.depth,
    }
    pruned_model, keep_idx_dict = prune(
        model,
        calibration_loader=dataloader,
        device=device,
        batch_agg_func=args.batch_agg_func,
        pruning_strategy=copy.deepcopy(prune_strategy),
        depth_strategy=args.depth_strategy,
    )
    pruned_model.to(torch.bfloat16)
    
    # let's init a new model with the same config as the base model
    pruned_config = create_new_config(model_config, prune_strategy)
    pruned_config.torch_dtype = torch.bfloat16
    try:
        new_pruned_model = AutoModelForCausalLM.from_config(pruned_config).to(device)
        new_pruned_model.load_state_dict(pruned_model.state_dict())
    except Exception as e:
        print(f"Error creating new pruned model: {e}")
        new_pruned_model = pruned_model
    
    # upload to HF hub
    # create repo name from base model and pruning ratios
    repo_name = f"{args.model_name.split('/')[-1]}-pruned-h{args.width_hidden}-i{args.width_intermediate}-a{args.width_attn}-d{args.depth}"
    if args.depth != 0:
        repo_name += f"-{args.depth_strategy}"
    
    # save keep_idx_dict
    os.makedirs(args.pruned_idx_dir, exist_ok=True)
    pruned_idx_path = os.path.join(args.pruned_idx_dir, f"{repo_name}.pkl")
    with open(pruned_idx_path, "wb") as f:
        pickle.dump(keep_idx_dict, f)
    print(f"Saved keep_idx_dict to: {pruned_idx_path}")
    
    # hf login
    if args.push_to_hub:
        print(f"Uploading pruned model to HF Hub as : {repo_name}")
        # check if hf_token is set
        assert os.environ["HF_TOKEN"] is not None, "HF_TOKEN is not set"
        pruned_model.push_to_hub(repo_name, token=os.environ["HF_TOKEN"])
        AutoTokenizer.from_pretrained(args.model_name).push_to_hub(repo_name, token=os.environ["HF_TOKEN"])
        print("Upload complete!")
    
   
