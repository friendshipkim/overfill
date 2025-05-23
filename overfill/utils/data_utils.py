# modified from alignment/data.py
import os
import logging
from typing import List, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from alignment import DataArguments, apply_chat_template

logger = logging.getLogger(__name__)

def get_datasets(
    data_config: DataArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
        cache_dir (`str`, *optional*, defaults to `None`):
            Path to the cache directory to store the datasets.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
        cache_dir=cache_dir,
    )
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
        cache_dir (`str`, *optional*, defaults to `None`):
            Path to the cache directory to store the datasets.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            try:
                if cache_dir:
                    dataset = load_dataset(
                        'json',
                        data_files={
                            split: os.path.join(cache_dir, ds, f'{split}.json'),
                        }
                    )
                    dataset = dataset[split]
                else:
                    # Try first if dataset on a Hub repo
                    dataset = load_dataset(ds, ds_config, split=split)
            except:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


def process_raw_datasets(raw_datasets, tokenizer, preprocessing_num_workers, pre_filter_max_seq_length, auto_insert_empty_system_msg):
    column_names = list(raw_datasets["train"].features)

    #####################
    # remove rows without "assistant" in the label field
    #####################
    logger.info("Removing rows without 'assistant' in the label field")
    def exists_assistant(example):
        for messages in example["messages"]:
            if messages["role"] == "assistant":
                return True
        return False
    raw_datasets = raw_datasets.filter(exists_assistant, num_proc=preprocessing_num_workers)

    #####################
    # Apply chat template
    #####################
    # <|system|>
    # {system_prompt}<|end_of_text|>
    # <|user|>
    # {user_prompt}<|end_of_text|>
    # <|assistant|>
    # {assistant_prompt}<|end_of_text|>
    # everything is in the text field
    logger.info("Applying chat template")
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": auto_insert_empty_system_msg,
        },
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    #####################
    # remove long examples
    #####################
    logger.info("Removing long examples")
    def check_overflows(example, tokenizer, max_seq_length):
        if len(tokenizer.encode(example["text"])) > max_seq_length:
            # warnings.warn(
            #     f"Example {example} has been discarded as the number of tokens exceeds the maximum sequence length."
            # )
            return False
        else:
            return True
    raw_datasets = raw_datasets.filter(
        check_overflows,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": pre_filter_max_seq_length,
        },
        num_proc=preprocessing_num_workers,
    )
    return raw_datasets

def get_response_template(model_name_or_path):
    if "Llama" in model_name_or_path:
        return "<|start_header_id|>assistant<|end_header_id|>"
    elif "Qwen" in model_name_or_path:
        return "<|im_start|>assistant"
    else:
        raise ValueError(f"Model {model_name_or_path} not supported")

def get_chat_template(model_name_or_path):
    if "Llama" in model_name_or_path:
        return "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    else:
        raise ValueError(f"Model {model_name_or_path} not supported")