import torch
import numpy as np
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin


def _torch_collate_batch(examples, tokenizer, pad_value: Optional[int] = None, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # pad with tokenizer.pad_token
    if pad_value is None:
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )
        else:
            pad_value = tokenizer.pad_token_id

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], pad_value)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class DataCollatorForCompletionOnlyLM(DataCollatorMixin):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_template: Union[str, List[int]],
        ignore_index: int = -100,
        return_tensors: str = "pt",
        completion_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.ignore_index = ignore_index
        self.return_tensors = return_tensors
        self.completion_only = completion_only

        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        for i in range(len(examples)):
            processed_inputs = self.mask(examples[i]['input_ids'], examples[i]['attention_mask'])
            examples[i]['input_ids'] = processed_inputs[0]
            examples[i]['attention_mask'] = processed_inputs[1]
            examples[i]['labels'] = processed_inputs[2]

        batch_input_ids = _torch_collate_batch([item['input_ids'] for item in examples], self.tokenizer)
        batch_attention_mask = _torch_collate_batch([item['attention_mask'] for item in examples], self.tokenizer, pad_value=0)
        batch_labels = _torch_collate_batch([item['labels'] for item in examples], self.tokenizer, pad_value=self.ignore_index)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
        # print(self.tokenizer.decode(batch_input_ids[0]))
        # print(self.tokenizer.decode(batch_labels[0][batch_labels[0] > 0]))
        # breakpoint()
        return batch

    def mask(self, input_ids, attention_mask):
        # make it to torch tensors
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)

        # === mask labels
        labels = input_ids.clone()
        response_token_ids_start_idx = None

        for idx in np.where(labels == self.response_token_ids[0])[0]:
            # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
            if (
                self.response_token_ids
                == labels[idx : idx + len(self.response_token_ids)].tolist()
            ):
                response_token_ids_start_idx = idx

        if response_token_ids_start_idx is None:
            warnings.warn(
                f"Could not find response key `{self.response_template}` in the "
                f'following instance: {self.tokenizer.decode(input_ids)} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            labels[:] = self.ignore_index
        else:
            response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response key
            if self.completion_only:
                labels[:response_token_ids_end_idx] = self.ignore_index

        return input_ids, attention_mask, labels


class CustomDataCollatorForCompletionOnlyLM(DataCollatorMixin):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_template: Union[str, List[int]],
        ignore_index: int = -100,
        teacher_ratio: float = 1.0,
        return_tensors: str = "pt",
        double_attn: bool = False,
        partial_cross_attn: bool = False,
        completion_only: bool = True,
        random_data_cutoff: bool = False,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.ignore_index = ignore_index
        self.teacher_ratio = teacher_ratio
        self.return_tensors = return_tensors
        self.double_attn = double_attn
        self.partial_cross_attn = partial_cross_attn
        self.completion_only = completion_only
        self.random_data_cutoff = random_data_cutoff
        self.seed = seed
        np.random.seed(seed)
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template
        assert self.teacher_ratio == 1.0 if self.random_data_cutoff else True, "teacher_ratio must be 1.0 when random_data_cutoff is True"

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # batch = self.parent_torch_call(examples)

        for i in range(len(examples)):
            processed_inputs = self.mask_and_split(examples[i]['input_ids'], examples[i]['attention_mask'])
            examples[i]['teacher_input_ids'] = processed_inputs[0]
            examples[i]['input_ids'] = processed_inputs[1]
            examples[i]['teacher_attention_mask'] = processed_inputs[2]
            examples[i]['attention_mask'] = processed_inputs[3]
            examples[i]['labels'] = processed_inputs[4]
            if self.partial_cross_attn:
                # this is the length of the teacher input
                # make sure it's a tensor
                examples[i]['teacher_input_len'] = torch.LongTensor([processed_inputs[5]])

        batch_teacher_input_ids = _torch_collate_batch([item['teacher_input_ids'] for item in examples], self.tokenizer)
        batch_input_ids = _torch_collate_batch([item['input_ids'] for item in examples], self.tokenizer)
        batch_teacher_attention_mask = _torch_collate_batch([item['teacher_attention_mask'] for item in examples], self.tokenizer, pad_value=0)
        batch_attention_mask = _torch_collate_batch([item['attention_mask'] for item in examples], self.tokenizer, pad_value=0)
        batch_labels = _torch_collate_batch([item['labels'] for item in examples], self.tokenizer, pad_value=self.ignore_index)
        if self.partial_cross_attn:
            batch_teacher_input_len = _torch_collate_batch([item['teacher_input_len'] for item in examples], self.tokenizer, pad_value=0)

        batch = {
            "teacher_input_ids": batch_teacher_input_ids,
            "input_ids": batch_input_ids,
            "teacher_attention_mask": batch_teacher_attention_mask,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
        if self.partial_cross_attn:
            batch['teacher_input_len'] = batch_teacher_input_len
        return batch

    def mask_and_split(self, input_ids, attention_mask):
        # make it to torch tensors
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)

        # === mask labels
        labels = input_ids.clone()
        response_token_ids_start_idx = None

        for idx in np.where(labels == self.response_token_ids[0])[0]:
            # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
            if (
                self.response_token_ids
                == labels[idx : idx + len(self.response_token_ids)].tolist()
            ):
                response_token_ids_start_idx = idx

        if response_token_ids_start_idx is None:
            raise ValueError(
                f"Could not find response key `{self.response_template}` in the "
                f'following instance: {self.tokenizer.decode(input_ids)} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            # labels[:] = self.ignore_index
        else:
            response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response key
            if self.completion_only:
                labels[:response_token_ids_end_idx] = self.ignore_index
        # print(self.tokenizer.decode(input_ids[:response_token_ids_end_idx]))

        # === split input_ids into teacher and student
        # teacher input doesn't include <|assistant|> and after
        # in addition to the user + system prompt, add random portion of the output
        if self.random_data_cutoff:
            output_len = len(input_ids) - response_token_ids_end_idx
            # pick random additional_input_len from [0, output_len - 2]
            # -2 because the last token is always <eos> \n
            add_output_len = np.random.randint(0, output_len - 1)
            teacher_input_len = response_token_ids_end_idx + add_output_len
        else:
            teacher_input_len = int(response_token_ids_end_idx * self.teacher_ratio)
        
        if self.double_attn:
            # pass partial input to teacher, full input to student
            teacher_input_ids, student_input_ids = input_ids[:teacher_input_len], input_ids
            teacher_attention_mask, student_attention_mask = attention_mask[:teacher_input_len], attention_mask
            student_labels = labels
        else:
            teacher_input_ids, student_input_ids = input_ids[:teacher_input_len], input_ids[teacher_input_len:]
            teacher_attention_mask, student_attention_mask = attention_mask[:teacher_input_len], attention_mask[teacher_input_len:]
            student_labels = labels[teacher_input_len:]

        # print(teacher_input_ids.shape, student_input_ids.shape, labels.shape)
        # print(self.tokenizer.decode(teacher_input_ids))
        # print(self.tokenizer.decode(student_input_ids))
        # print(self.tokenizer.decode(student_labels[student_labels > 0]))
        assert len(student_labels) == len(student_input_ids)

        return teacher_input_ids, student_input_ids, teacher_attention_mask, student_attention_mask, student_labels, teacher_input_len

