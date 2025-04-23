#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Two-stage decoding script for decoding language models.
"""

import logging
import random
import sys

import datasets
import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    decontaminate_humaneval,
    get_checkpoint,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl import setup_chat_format
from datetime import timedelta

from overfill.models import TwoStageModel
from overfill.configs.train_configs import SFTDistillConfig
from overfill.utils.data_utils import get_datasets, process_raw_datasets, get_response_template
from overfill.utils.model_utils import get_model, get_tokenizer
from overfill.data import CustomDataCollatorForCompletionOnlyLM
from overfill.custom_trainer import CustomSFTTrainer

logger = logging.getLogger(__name__)


def main():

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=360000))

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTDistillConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
        cache_dir=training_args.data_cache_dir,
    )

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    ################
    # Load tokenizer
    ################
    # here we assume the tokenizer is the same for both the teacher and the student
    teacher_tokenizer = get_tokenizer(model_args.model_name_or_path, model_args, data_args, cache_dir=training_args.model_cache_dir)
    student_tokenizer = get_tokenizer(training_args.student_model_name_or_path, model_args, data_args, cache_dir=training_args.model_cache_dir)
    teacher_tokenizer.model_max_length = training_args.max_seq_length
    student_tokenizer.model_max_length = training_args.max_seq_length

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # model = model_args.model_name_or_path
    # # For ChatML we need to add special tokens and resize the embedding layer
    # if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
    #     model = get_model(model_args.model_name_or_path, model_kwargs, cache_dir=training_args.model_cache_dir)
    #     model, tokenizer = setup_chat_format(model, tokenizer)
    #     model_kwargs = None

    # preprocess the datasets
    if training_args.debug_mode:
        raw_datasets["train"] = raw_datasets["train"].select(range(100))
        raw_datasets["test"] = raw_datasets["test"].select(range(100))
    # use teacher tokenizer to check length, this is NOT the actual tokenization
    raw_datasets = process_raw_datasets(
        raw_datasets,
        teacher_tokenizer, 
        data_args.preprocessing_num_workers,
        training_args.pre_filter_max_seq_length,
        data_args.auto_insert_empty_system_msg
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    assert "text" in train_dataset.features
    assert "text" in eval_dataset.features

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    # load the model
    teacher_model = get_model(
        model_args.model_name_or_path,
        model_kwargs,
        cache_dir=training_args.model_cache_dir
    )
    if training_args.student_model_name_or_path is not None:
        student_model = get_model(
            training_args.student_model_name_or_path,
            model_kwargs,
            cache_dir=training_args.model_cache_dir,
        )

    if training_args.embeddings_from_layer_n is not None:
        training_args.embeddings_from_layer_n = list(map(int, training_args.embeddings_from_layer_n.split(",")))

    model = TwoStageModel(
        teacher=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student=student_model,
        student_tokenizer=student_tokenizer,
        freeze_strategy=training_args.freeze_strategy,
        embedding_transform_strategy=training_args.embedding_transform_strategy,
        depth_prune_ratio=training_args.depth_prune_ratio,
        depth_prune_strategy=training_args.depth_prune_strategy,
    )
    model.to(training_args.device)
    print("model:", model)
    
    if training_args.freeze_strategy != "none":
        for name, param in model.named_parameters():
            if "teacher" in name:
                param.requires_grad = False

    ########################
    # Initialize the Trainer
    ########################
    # # baseline
    # response_template = "<|assistant|>\n"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    # ours
    # response_template = "<|assistant|>\n"
    # for tinyllama
    # response_template = [29966, 29989, 465, 22137, 29989, 29958, 13]
    # for llama3
    # response_template = [27, 91, 78191, 91]
    response_template = get_response_template(model_args.model_name_or_path)

    train_collator = CustomDataCollatorForCompletionOnlyLM(
        teacher_tokenizer,
        response_template,
        teacher_ratio=training_args.teacher_input_ratio,
        completion_only=training_args.completion_only,
        random_data_cutoff=training_args.random_data_cutoff,
    )
    eval_collator = CustomDataCollatorForCompletionOnlyLM(
        teacher_tokenizer,
        response_template,
        teacher_ratio=training_args.teacher_input_ratio,
        completion_only=training_args.completion_only,
        random_data_cutoff=False,
    )

    training_args.dataset_num_proc = data_args.preprocessing_num_workers
    # training_args.eval_on_start = True
    trainer = CustomSFTTrainer(
        model=model,
        # model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_refresh_interval=training_args.eval_refresh_interval,
        dataset_text_field="text",
        tokenizer=teacher_tokenizer,
        peft_config=get_peft_config(model_args),
        data_collator=train_collator,
        eval_data_collator=eval_collator
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
