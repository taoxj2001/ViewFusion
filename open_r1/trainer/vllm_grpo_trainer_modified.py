# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.utils.data
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer
from torch.utils.data import DistributedSampler, SequentialSampler
import copy

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
import torch.nn as nn
from torch.utils.data import Sampler
import gc
import sys
from qwen_vl_utils3_grpo import process_vision_info

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen3VLGRPOVLLMTrainerModified(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            model_init_kwargs.pop("use_cache")
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Qwen3" in model_id:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            else:
                model = Qwen3VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen3-VL" in model_id:
                self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:

            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal = False
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.use_vllm = args.use_vllm

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        if self.use_vllm:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            self.vllm_tensor_parallel_size = getattr(self.args, "vllm_tensor_parallel_size", 1)
            if self.accelerator.num_processes % self.vllm_tensor_parallel_size != 0:
                raise ValueError(
                    f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                    f"({self.accelerator.num_processes}) evenly."
                )

            if self.vllm_tensor_parallel_size > 1:
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(
                            range(
                                i * self.vllm_tensor_parallel_size,
                                (i + 1) * self.vllm_tensor_parallel_size,
                            )
                        )
                        for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                    ]
                )

            os.environ["RANK"] = str(self.accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)

            self.llm = LLM(
                model=model.name_or_path,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                max_num_seqs=(
                        self.args.per_device_train_batch_size
                        * self.vllm_tensor_parallel_size
                        * getattr(self.args, "steps_per_generation", 1)
                ),
                max_model_len=getattr(self.args, "vllm_max_model_length",
                                      args.max_prompt_length + args.max_completion_length),

                distributed_executor_backend="external_launcher",
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,

                max_num_batched_tokens=4096,  # TRL 推荐值（避免 v1 profiler 误判）

                enable_prefix_caching=True,
                limit_mm_per_prompt={"image": 10, "video": 1},
                mm_processor_cache_gb=0,
                mm_processor_kwargs=(
                    {"max_pixels": max_pixels, "min_pixels": min_pixels}
                    if False else None
                ),
                logprobs_mode="processed_logprobs",
            )

            self.sampling_params = SamplingParams(
                n=4,
                temperature=1.0,
                top_p=0.95,
                max_tokens=self.max_completion_length,
            )

            self._last_loaded_step = 0
            self.accelerator.wait_for_everyone()

        else:
            raise ValueError("GRPOVLLMTrainerModified only supports vllm generation, please set --use_vllm True")



        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)



    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
        # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        # import pdb
        # pdb.set_trace()

        logits = model(input_ids, **kwargs).logits


        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data




    def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        print(inputs[0]["id"])
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        for example in inputs:
            example["prompt"] = self.remove_none_from_data(example["prompt"])

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        input_copy = self.remove_none_from_data(input_copy)
        data_type = inputs[0]['data_type']

        image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True,
                                                                       return_video_metadata=True)
        video_inputs_with_metadata = copy.deepcopy(video_inputs)

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **video_kwargs
        )
        prompt_inputs.pop("token_type_ids", None)
        mm_data = [[data_type, image_inputs if image_inputs else video_inputs_with_metadata]]
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        shuffled_mm_data = [None]
        if self.temporal:
            if video_inputs:
                indices = torch.randperm(video_inputs[0].size(0))
                shuffled_video_inputs = [video_inputs[0][indices]]
                shuffled_video_inputs_with_metadata = list(zip(shuffled_video_inputs, video_metadatas))
                shuffled_prompt_inputs = self.processing_class(
                    text=copy.deepcopy(prompts_text),
                    images=image_inputs,
                    videos=shuffled_video_inputs,
                    video_metadata=video_metadatas,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                    **video_kwargs
                )
                shuffled_prompt_inputs.pop("token_type_ids", None)
                shuffled_mm_data = [[self.accelerator.process_index, data_type,
                                     image_inputs if image_inputs else shuffled_video_inputs_with_metadata]]

            else:
                shuffled_mm_data = [None]
        if self.args.use_vllm:
            should_load_weights = False
            if self.vllm_tensor_parallel_size == 1:
                should_load_weights = True  # 所有人都加载
            elif self.accelerator.is_main_process:
                should_load_weights = True  # 兼容 TP>1 的旧逻辑

            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                        self.model,
                        self.accelerator,
                        gather_deepspeed3_params=True,
                ) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        state_dict = unwrapped_model._orig_mod.state_dict()
                    else:
                        state_dict = unwrapped_model.state_dict()

                if should_load_weights:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())

                self.accelerator.wait_for_everyone()
                self._last_loaded_step = self.state.global_step
            if self.vllm_tensor_parallel_size == 1:
                if video_inputs:
                    sample_video_kw = {'do_sample_frames': False}
                else:
                    sample_video_kw = None

                local_multimodal_inputs = []
                for prompt, mm_item in zip(prompts_text, mm_data):
                    local_multimodal_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {mm_item[0]: mm_item[1]},
                        "mm_processor_kwargs": sample_video_kw
                    })
                shuffled_local_multimodal_inputs = []
                if self.temporal and shuffled_mm_data[0] is not None:
                    for mm_item in shuffled_mm_data:
                        shuffled_local_multimodal_inputs.append({
                            "prompt": prompts_text[0],
                            "multi_modal_data": {mm_item[1]: mm_item[2]},
                            "mm_processor_kwargs": sample_video_kw
                        })
                sampling_params = copy.deepcopy(self.sampling_params)
                sampling_params.n = self.num_generations

                outputs = self.llm.generate(
                    local_multimodal_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                completion_ids = [out.token_ids for completion in outputs for out in completion.outputs]

                shuffled_completion_ids = []
                if self.temporal and shuffled_local_multimodal_inputs:
                    shuffled_sampling_params = copy.deepcopy(self.sampling_params)
                    shuffled_sampling_params.n = self.num_generations // 2

                    shuffled_outputs = self.llm.generate(
                        shuffled_local_multimodal_inputs,
                        sampling_params=shuffled_sampling_params,
                        use_tqdm=False,
                    )
                    shuffled_completion_ids = [out.token_ids for completion in shuffled_outputs for out in
                                               completion.outputs]
            else:

                all_prompts_text = gather_object(prompts_text)
                all_mm_data = gather_object(mm_data)
                if self.temporal:
                    shuffled_all_mm_data_none = gather_object(shuffled_mm_data)
                    shuffled_all_mm_data = [x for x in shuffled_all_mm_data_none if x]
                else:
                    shuffled_all_mm_data = []

                all_multimodal_inputs = []
                shuffled_all_multimodal_inputs = []  # init

                if video_inputs:
                    sample_video_kw = {'do_sample_frames': False}
                else:
                    sample_video_kw = None

                for prompt, mm_item in zip(all_prompts_text, all_mm_data):
                    all_multimodal_inputs.append({"prompt": prompt, "multi_modal_data": {mm_item[0]: mm_item[1]},
                                                  "mm_processor_kwargs": sample_video_kw})

                if self.temporal and shuffled_all_mm_data != []:
                    for mm_item in shuffled_all_mm_data:
                        shuffled_all_multimodal_inputs.append(
                            {"prompt": all_prompts_text[mm_item[0]], "multi_modal_data": {mm_item[1]: mm_item[2]},
                             "mm_processor_kwargs": sample_video_kw})

                if self.accelerator.is_main_process:
                    sampling_params = copy.deepcopy(self.sampling_params)
                    sampling_params.n = self.num_generations
                    outputs = self.llm.generate(all_multimodal_inputs, sampling_params=sampling_params, use_tqdm=False)
                    completion_ids = [out.token_ids for completion in outputs for out in completion.outputs]

                    shuffled_completion_ids = []
                    if self.temporal and shuffled_all_mm_data != []:
                        shuffled_sampling_params = copy.deepcopy(self.sampling_params)
                        shuffled_sampling_params.n = self.num_generations // 2
                        shuffled_outputs = self.llm.generate(shuffled_all_multimodal_inputs,
                                                             sampling_params=shuffled_sampling_params, use_tqdm=False)
                        shuffled_completion_ids = [out.token_ids for completion in shuffled_outputs for out in
                                                   completion.outputs]
                else:
                    completion_ids = [None] * len(all_multimodal_inputs) * self.num_generations
                    shuffled_completion_ids = []
                    if self.temporal and shuffled_all_mm_data != []:
                        shuffled_completion_ids = [None] * len(shuffled_all_multimodal_inputs) * (
                                    self.num_generations // 2)

                # Broadcast back
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                # Slice logic
                process_slice = slice(
                    self.accelerator.process_index * len(prompts) * self.num_generations,
                    (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
                )
                completion_ids = completion_ids[process_slice]

                if self.temporal and shuffled_all_mm_data != []:
                    shuffled_completion_ids = broadcast_object_list(shuffled_completion_ids, from_process=0)
                    # Shuffled slice logic (complex, keeping original logic structure)
                    # ... (Keep original messy slicing logic for TP mode if needed) ...
                    # Assuming original logic for shuffled slice was correct for gather mode.
                    # Re-implementing specific slice logic from original code:
                    process_id_list = []
                    for mm_item in shuffled_all_mm_data:
                        process_id_list += [mm_item[0]] * len(prompts) * (self.num_generations // 2)

                    cur_shuffled_completion_ids = []
                    for i in range(len(process_id_list)):
                        if self.accelerator.process_index == process_id_list[i]:
                            cur_shuffled_completion_ids.append(shuffled_completion_ids[i])
                    shuffled_completion_ids = cur_shuffled_completion_ids
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            if self.temporal and shuffled_completion_ids:
                shuffled_completion_ids = [torch.tensor(ids, device=device) for ids in shuffled_completion_ids]
                shuffled_completion_ids = pad(
                    shuffled_completion_ids, padding_value=self.processing_class.pad_token_id
                )
            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            prompt_length = prompt_ids.size(1)

            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        else:
            raise ValueError("Only vLLM generation is supported in this version ")
        is_eos = completion_ids == self.processing_class.eos_token_id

        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")

        if data_type == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)

        if data_type == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(
                len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
        if per_token_logps is None:
            return torch.tensor(0.0, device=self.args.device)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        gc.collect()
        torch.cuda.empty_cache()

        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, **prompt_inputs
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model, prompt_completion_ids, **prompt_inputs
                        )
            except ValueError as e:
                if "Image features and image tokens do not match" in str(e):
                    return torch.tensor(0.0, device=self.args.device)  # <- 直接跳过本 step（上层要识别 None 并 skip）
                raise

        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        gc.collect()
        torch.cuda.empty_cache()

        if self.temporal and video_inputs:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion
                                        in shuffled_completions]
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes)
            ):
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions,
                                                          **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32,
                                                               device=device)

        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Compute the rewards (Normal)
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {
                key: []
                for key in inputs[0].keys()
                if key not in ["prompt", "completion"]
            }
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(
                prompts=prompts, completions=completions, **reward_kwargs
            )
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        # Reward Calculation logic
        if self.temporal and video_inputs:
            temporal_rewards_per_func = rewards_per_func.clone()
            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask, 0] = temporal_rewards_per_func[mask, 0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards = torch.tensor([0.5]).to('cuda')

        if self.temporal and video_inputs:
            rewards = temporal_rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func.sum(dim=1)

        if self.len_control:
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
            if len(selected_indices) > 1:
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 512:
                        rewards[idx] += 0.2

        # print(rewards)
        # print(completion_mask.sum(1))

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        gathered_rewards = self.accelerator.gather_for_metrics(rewards)

        # metrics calculation fix: handle possible mismatch in gather size if drop_last=False
        # Assuming safe gather for now based on original code style
        num_devices = max(1, gathered_rewards.size(0) // self.num_generations)
        rewards_per_device = gathered_rewards.view(num_devices, -1)  # flexible view

        # 简单处理 metric，避免 shape 错误
        self._metrics["reward"].append(gathered_rewards.mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss
    


        
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
