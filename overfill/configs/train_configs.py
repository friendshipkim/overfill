import os
import sys
import transformers
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, NewType
from transformers import HfArgumentParser
from alignment import SFTConfig
from overfill.utils.model_utils import FREEZE_STRATEGIES

DataClassType = NewType("DataClassType", Any)


@dataclass
class SFTDistillConfig(SFTConfig):
    """
    Arguments related to the distillation process.
    """
    student_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous distilled model in this progressive distillation."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the wandb project."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb entity/user name."},
    )
    embedding_transform_strategy: str = field(
        default="kv_identity",
        metadata={"help": "Embedding transform strategy"},
    )
    embeddings_from_layer_n: List[int] = field(
        default=None,
        metadata={"help": "Embeddings from layer n"},
    )
    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"},
    )
    pre_filter_max_seq_length: Optional[int] = field(
        default=8000,
        metadata={"help": "max sequence length for filtering long sequences before training"},
    )
    teacher_input_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "ratio of teacher input in the training data"},
    )
    channel_mapping_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the channel mapping file for width pruning"},
    )
    freeze_strategy: Optional[str] = field(
        default="teacher",
        metadata={"help": f"Model freeze strategy, choices: {FREEZE_STRATEGIES}"},
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Data local cache directory"},
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Model local cache directory"},
    )
    completion_only: bool = field(
        default=True,
        metadata={"help": "loss is computed only on the completion part of the input"},
    )
    random_data_cutoff: bool = field(
        default=False,
        metadata={"help": "randomly cutoff data at a certain ratio"},
    )
    # pruning arguments
    width_hidden: Optional[float] = field(
        default=None,
        metadata={"help": "width pruning - removed hidden dimension ratio"},
    )
    width_intermediate: Optional[float] = field(
        default=None,
        metadata={"help": "width pruning - removed intermediate dimension ratio"},
    )
    width_attn: Optional[float] = field(
        default=None,
        metadata={"help": "width pruning - removedattention head ratio"},
    )
    depth: Optional[float] = field(
        default=None,
        metadata={"help": "depth pruning - removed layer ratio"},
    )
    eval_refresh_interval: Optional[int] = field(
        default=0,
        metadata={"help": "evaluation refresh interval"},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "teacher model name or path"},
    )
    teacher_model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "teacher model init kwargs"},
    )
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "kl weight"},
    )
    ce_weight: Optional[float] = field(
        default=0.0,
        metadata={"help": "ce weight"},
    )
    depth_prune_ratio: Optional[float] = field(
        default=0.0,
        metadata={"help": "depth prune ratio"},
    )
    depth_prune_strategy: Optional[str] = field(
        default="firstlast",
        metadata={"help": "depth prune strategy"},
    )
