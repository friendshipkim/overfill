import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.api.model import LM
from transformers import AutoTokenizer
from lm_eval.__main__ import cli_evaluate

from overfill.models import TwoStageModel
from overfill.utils.data_utils import get_chat_template

@register_model("custom_hf")
class CustomModel(HFLM):
    def __init__(
        self,
        model_path,
        teacher_model,
        student_model,
        batch_size,
        freeze_strategy="teacher",
        depth_prune_ratio=0.0, 
        depth_prune_strategy="firstlast",
        embedding_transform_strategy="kv_prune",
        device="cuda",
        max_length=2048,
        **kwargs
    ):
        # super().__init__(pretrained=pretrained, tokenizer=tokenizer, **kwargs)
        LM.__init__(self)
        # load your model
        self._model = TwoStageModel.from_pretrained(
            model_path,
            teacher_model_name=teacher_model,
            student_model_name=student_model,
            freeze_strategy=freeze_strategy,
            depth_prune_ratio=depth_prune_ratio,
            depth_prune_strategy=depth_prune_strategy,
            embedding_transform_strategy=embedding_transform_strategy,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model)
        # set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = get_chat_template(teacher_model)
        self.tokenizer.model_max_length = max_length
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size) if batch_size is not None else 64
        self.backend = "causal"
        self.add_bos_token = False
        self.truncation = False
        self._max_length = self.tokenizer.model_max_length
        self.revision = "main"
        self.pretrained = model_path
        self.peft = False
        self.delta = False

if __name__ == "__main__":
    cli_evaluate()
