import logging
import warnings
import os
import pickle
import copy
import inspect
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange

from transformers.cache_utils import Cache, DynamicCache

from overfill.utils.model_utils import (
    EMBEDDING_TRANSFORM_STRATEGIES,
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
)
from pruner.utils import get_kept_layers_prune, get_kept_layers, DEPTH_PRUNE_STRATEGY_LIST

logger = logging.getLogger(__name__)

EMBEDDING_TRANSFORM_STRATEGIES = ["kv_identity", "kv_prune", "kv_proj_layer"]

class TwoStageModel(nn.Module, PyTorchModelHubMixin):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    def __init__(
        self,
        teacher: transformers.AutoModelForCausalLM,
        teacher_tokenizer: transformers.PreTrainedTokenizer,
        student: transformers.AutoModelForCausalLM,
        student_tokenizer: transformers.PreTrainedTokenizer,
        freeze_strategy: str = "teacher",
        teacher_dropout_disabled: bool = False,
        student_dropout_disabled: bool = False,
        teacher_lora: bool = False,
        student_lora: bool = False,
        embedding_transform_strategy: str = "kv_identity",
        refresh_interval: int = -1,
        depth_prune_ratio: float = 0.0,
        depth_prune_strategy: str = "firstlast",
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.config = teacher.config
        self.teacher_dim = teacher.config.hidden_size
        self.student_dim = student.config.hidden_size
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        self.teacher_num_layers = self.teacher.config.n_layer
        self.student_num_layers = self.student.config.n_layer
        self.teacher_dtype = self.teacher.dtype
        self.student_dtype = self.student.dtype

        self.teacher_kv_dim = self.teacher.config.kv_dim
        self.student_kv_dim = self.student.config.kv_dim
        self.teacher_num_kv_heads = self.teacher.config.num_kv_heads
        self.student_num_kv_heads = self.student.config.num_kv_heads
        # LORA ==============================================
        if student_lora or teacher_lora:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_int8_training,
            )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            if teacher_lora:
                print("Initializing LORA model with config:", peft_config)
                self.teacher = prepare_model_for_int8_training(self.teacher)
                self.teacher = get_peft_model(self.teacher, peft_config)
            if student_lora:
                print("Initializing LORA model with config:", peft_config)
                self.student = prepare_model_for_int8_training(self.student)
                self.student = get_peft_model(self.student, peft_config)
        # =================================================== #

        # patch transform strategy
        self.embedding_transform_strategy = embedding_transform_strategy
        assert self.embedding_transform_strategy in EMBEDDING_TRANSFORM_STRATEGIES
        if embedding_transform_strategy == "kv_identity":
            assert self.teacher_num_layers == self.student_num_layers
            assert self.teacher_kv_dim == self.student_kv_dim
            assert self.teacher_num_kv_heads == self.student_num_kv_heads
            self.embedding_proj = None
            self.kept_layers = list(range(self.teacher_num_layers))
            print(f"kept_layers: {self.kept_layers}")
        elif embedding_transform_strategy == "kv_prune":
            assert self.teacher_kv_dim == self.student_kv_dim
            assert self.teacher_num_kv_heads == self.student_num_kv_heads
            self.embedding_proj = None
            
            # depth pruning
            assert depth_prune_strategy in DEPTH_PRUNE_STRATEGY_LIST
            assert self.student_num_layers == int(self.teacher_num_layers * (1 - depth_prune_ratio))
            self.kept_layers = get_kept_layers_prune(depth_prune_strategy, self.teacher_num_layers, depth_prune_ratio)
            assert self.student_num_layers == len(self.kept_layers)
            print(f"kept_layers: {self.kept_layers}")
        elif embedding_transform_strategy == "kv_proj_layer":
            self.embedding_proj = nn.ModuleList(
                [
                    nn.Linear(self.teacher_kv_dim, self.student_kv_dim)
                    for _ in range(self.student_num_layers)
                ]
            )
            self.kept_layers = get_kept_layers(self.teacher_num_layers, self.student_num_layers)
            print(f"kept_layers: {self.kept_layers}")
        else:
            raise ValueError(f"unknown embedding transformation strategy {embedding_transform_strategy}")

        # match data type
        # or use float32 for everything
        if self.embedding_proj is not None:
            self.embedding_proj = self.embedding_proj.to(self.student_dtype)

        # =================================================== #
        # disable dropout
        if student_dropout_disabled:
            print("Dropout disabled for student model")
            disable_dropout(self.student)
        if teacher_dropout_disabled:
            print("Dropout disabled for teacher model")
            disable_dropout(self.teacher)

        # freeze models
        self.freeze(freeze_strategy)
        self.freeze_teacher = True if "teacher" in freeze_strategy else False
        self.freeze_student = True if "student" in freeze_strategy else False

        # for generation
        self.student_prepare_inputs_for_generation = self.student.prepare_inputs_for_generation
        self.refresh_interval = refresh_interval

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.student.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def _freeze_teacher(self):
        print("Freeze teacher model")
        freeze_params(self.teacher)

    def _freeze_student(self):
        print("Freeze student model")
        freeze_params(self.student)

    def freeze(self, freeze_strategy: str):
        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "teacher":
            self._freeze_teacher()
        elif freeze_strategy == "student":
            self._freeze_student()
        elif freeze_strategy == "teacher_and_student":
            self._freeze_teacher()
            self._freeze_student()
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    @property
    def device(self) -> torch.device:
        return next(self.teacher.parameters()).device

    def call_teacher_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        
        # is_checkpointing = getattr(self.model, "is_gradient_checkpointing", False)
        # self.model.gradient_checkpointing_disable()
        if self.freeze_teacher:
            self.teacher.eval()
        output = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )
        # self.model.gradient_checkpointing_enable()
        return output
    
    def project_full_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        full_kv = outputs["past_key_values"]
        
        # output kv shape
        #  - key: [batch_size, self.num_heads, kv_length, head_dim]
        #  - value: same as key
        # tuple of length num_layers, each with tensor of shape [2, <layerwise_shape>]
        # stack inner kv tensors
        def stack(kv):
            return torch.stack(tuple(kv[i] for i in range(2)), dim=0)
        full_kv = tuple(stack(full_kv[l]) for l in self.kept_layers)
        if self.embedding_proj is not None:
            full_kv = tuple(
                rearrange(
                    proj(rearrange(layer_kv, "kv b h l d -> kv b l (h d)")),
                    "kv b l (h d) -> kv b h l d",
                    h=self.student_num_kv_heads
                ) for proj, layer_kv in zip(self.embedding_proj, full_kv)
            )
        return full_kv
    
    def prefill_and_project(
        self,
        teacher_input_ids: Optional[torch.Tensor],
        teacher_attention_mask: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # check if teacher_input_ids is 2D tensor
        assert teacher_input_ids.dim() == 2

        # expand to beam size for generation
        if teacher_input_ids.shape[0] != batch_size:
            # given teacher_input_ids / attention masks of shape [bsz, patch_len]
            # expand them to [bsz * beam, patch_len] by repeating each row 'beam_size' times together
            beam_size = batch_size // teacher_input_ids.shape[0]
            teacher_input_ids = teacher_input_ids[:, None, :].repeat(1, beam_size, 1)
            teacher_input_ids = teacher_input_ids.view(batch_size, -1)

            teacher_attention_mask = teacher_attention_mask[:, None, :].repeat(
                1, beam_size, 1
            )
            teacher_attention_mask = teacher_attention_mask.view(batch_size, -1)

        prefill_output = self.call_teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
        )
        kv_cache = self.project_full_output(prefill_output, teacher_attention_mask)

        return prefill_output, kv_cache
    
    def generate_masks(self, input_ids, tr_attention_mask, st_attention_mask):
        # prefix attention mask, shape: [bsz, patch_len]
        concat_attention_mask = torch.cat([tr_attention_mask, st_attention_mask], dim=1)

        # For a batch where each item may have different prefix lengths
        prefix_lengths = tr_attention_mask.sum(dim=1)  # [batch_size]

        # Create position ids starting from each prefix length
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)  # [1, seq_len]
        position_ids = position_ids + prefix_lengths.unsqueeze(1)  # [batch_size, seq_len]

        return {
            "position_ids": position_ids.to(torch.long),
            "attention_mask": concat_attention_mask,
        }
    
    def generate(
        self,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # NOTE this is called only once per sequence
        # NOTE use regular collator that input_id field is the prefill input

        # Prefill
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        full_output, past_kv_tuple = self.prefill_and_project(
            teacher_input_ids=input_ids,
            teacher_attention_mask=attention_mask,
            batch_size=input_ids.shape[0],
        )
        past_key_values = DynamicCache()
        for (layer_idx, (past_key, past_value)) in enumerate(past_kv_tuple):
            past_key_values.update(past_key, past_value, layer_idx)
        
        kwargs["past_key_values"] = past_key_values
        # full generates 1 token
        next_token_logits = full_output.logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)
        
        # NOTE: always greedy-decode
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_ids = torch.cat([attention_mask, torch.ones_like(next_tokens[:, None])], dim=-1)
        
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_ids

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del full_output

        # override student's prepare_inputs_for_generation
        self.student.prepare_inputs_for_generation = self.student_prepare_inputs_for_generation

        outputs = self.student.generate(**kwargs)

        # restore student's prepare_inputs_for_generation
        self.student.prepare_inputs_for_generation = self.student_prepare_inputs_for_generation
        return outputs

    def student_prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        This is called at every generation step
        transformers/generation/utils.py:348
        """
        # Handle Cache position
        model_inputs = {}
        model_inputs["cache_position"] = cache_position
        # model_inputs["cache_position"] = None

        # Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            # slice input_ids
            if input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
            
        # base model inputs
        assert inputs_embeds is None
        model_inputs["input_ids"] = input_ids.clone(memory_format=torch.contiguous_format)
        model_inputs["inputs_embeds"] = None
        
        # Create missing `position_ids` on the fly
        # 0 to input_ids.shape[1] - 1
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.model.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        
        # Slice position_ids to have the same length as `input_ids`
        # token_type_ids is not used
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)
        model_inputs["position_ids"] = position_ids
        # model_inputs["position_ids"] = None
        
        # Attention mask
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        # Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            # TODO check if this is necessary
            if key == "position_ids":
                continue
            if key not in model_inputs:
                model_inputs[key] = value

        # Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)

        return model_inputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # print(self.tokenizer.batch_decode(teacher_input_ids))
        # print(self.tokenizer.batch_decode(input_ids))
        # print(self.tokenizer.decode(labels[0][labels[0] > 0]))
        teacher_input_ids = kwargs.pop("teacher_input_ids")
        teacher_attention_mask = kwargs.pop("teacher_attention_mask")
        
        if past_key_values is None:
            # Initial prefill
            prefill_output, past_key_values = self.prefill_and_project(
                teacher_input_ids=teacher_input_ids,
                teacher_attention_mask=teacher_attention_mask,
                batch_size=input_ids.shape[0],
            )
        
            past_kv_cache = DynamicCache()
            for (layer_idx, (past_key, past_value)) in enumerate(past_key_values):
                past_kv_cache.update(past_key, past_value, layer_idx)
        else:
            assert past_key_values.get_seq_length() == teacher_input_ids.shape[1], "past_key_values sequence length must match teacher_input_ids"
            past_kv_cache = past_key_values

        # Forward pass with small model
        masks = self.generate_masks(input_ids, teacher_attention_mask, attention_mask)
        outputs = self.student(
            input_ids=input_ids,
            position_ids=masks["position_ids"],
            past_key_values=past_kv_cache,
            attention_mask=masks["attention_mask"],
            labels=labels,
            **kwargs,
        )
        # # get token loss
        # logits = outputs.logits[:, :-1, :]
        # labels = labels[:, 1:]
        # loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # unreduced_loss = loss_fn(logits.transpose(-1, -2), labels)
        # breakpoint()
        # unreduced_loss = unreduced_loss.sum()
        # print(f"unreduced_loss: {unreduced_loss}")
        return outputs
    
    @staticmethod
    def from_pretrained(
        model_path,
        teacher_model_name,
        student_model_name,
        freeze_strategy="teacher",
        depth_prune_ratio=0.0,
        depth_prune_strategy="firstlast",
        embedding_transform_strategy="kv_prune",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):   
        from overfill.utils.model_utils import get_model
        from transformers import AutoTokenizer
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
        }
        # load models
        teacher_model = get_model(
            teacher_model_name, model_kwargs
        )
        student_model = get_model(
            student_model_name, model_kwargs
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        model = TwoStageModel(
            teacher=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            student=student_model,
            student_tokenizer=student_tokenizer,
            freeze_strategy=freeze_strategy,
            embedding_transform_strategy=embedding_transform_strategy,
            depth_prune_ratio=depth_prune_ratio,
            depth_prune_strategy=depth_prune_strategy,
        )
        
        if model_path != "None":
            print(f"Loading checkpoint from {model_path}")
            # load checkpoint
            checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
            print(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            
            # Split state dict into teacher and student parts
            teacher_state_dict = {}
            student_state_dict = {}
            embedding_proj_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith("teacher."):
                    teacher_state_dict[key.replace("teacher.", "")] = value
                elif key.startswith("student."):
                    student_state_dict[key.replace("student.", "")] = value
                elif key.startswith("embedding_proj."):
                    embedding_proj_state_dict[key.replace("embedding_proj.", "")] = value
                else:
                    print(f"unexpected key: {key}")
                    print(f"value: {value.shape}")
                    raise ValueError(f"unexpected key: {key}")
            
            # Load state dicts into respective models
            missing_teacher, unexpected_teacher = model.teacher.load_state_dict(teacher_state_dict, strict=True)
            missing_student, unexpected_student = model.student.load_state_dict(student_state_dict, strict=True)
            if model.embedding_proj is not None:
                missing_embedding_proj, unexpected_embedding_proj = model.embedding_proj.load_state_dict(embedding_proj_state_dict, strict=True)
                print(f"Embedding proj - Missing keys: {missing_embedding_proj}")
                print(f"Embedding proj - Unexpected keys: {unexpected_embedding_proj}")
            
            print(f"Teacher - Missing keys: {missing_teacher}")
            print(f"Teacher - Unexpected keys: {unexpected_teacher}")
            print(f"Student - Missing keys: {missing_student}")
            print(f"Student - Unexpected keys: {unexpected_student}")
            
            # Manually tie the weights after loading
            model.teacher.tie_weights()
            model.student.tie_weights()
        else:
            print("No checkpoint provided, using the weights of given models")
        
        return model
