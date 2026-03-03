import os
import json
import random
import requests
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils3_train import process_vision_info
import wandb
from typing import List, Dict, Any

def get_current_device():
    """Get the current device."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    raw_question = example.get('question', "")
    QUESTION_TEMPLATE = f"Question: {raw_question}\n\nPlease reason step by step. First, you need to provide a description of the perspective transformation applied to this set of images. Briefly explain how the perspective changes between the images (e.g., rotated 90 degrees clockwise), and describe the relative positions of the objects based on the background and occlusion relationships. Provide your description process and enclose it within the <spatial_thinking> </spatial_thinking> tags. \n\n When the perspective changes, use multiple images to discover objects that are not visible in a single image, and provide your detailed reasoning for the raw question within <thinking> </thinking>. \n\n Provide only the single option letter (e.g., A, B, C, D) within the <answer> </answer> tags."
    user_content = []
    image_rel_paths = example.get('images', [])
    for path in image_rel_paths:
        if path:
            abs_path = path
            user_content.append({
                "type": "image",
                "image": abs_path
            })

    user_content.append({
        "type": "text",
        "text": QUESTION_TEMPLATE
    })


    full_assistant_response = example.get('response', "")

    messages = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": full_assistant_response}]
        }
    ]

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    texts = []
    batch_images = []
    batch_videos = []
    batch_video_kwargs_list = []

    for i, example in enumerate(examples):
        try:
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                example["messages"],
                return_video_kwargs=True
            )
            batch_images.append(image_inputs)
            batch_videos.append(video_inputs)
            batch_video_kwargs_list.append(video_kwargs)

        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    merged_video_kwargs = {}
    if batch_video_kwargs_list:
        merged_video_kwargs = dict(batch_video_kwargs_list[0])
        for k in merged_video_kwargs.keys():
            v0 = merged_video_kwargs[k]
            for j in range(1, len(batch_video_kwargs_list)):
                if k in batch_video_kwargs_list[j] and batch_video_kwargs_list[j][k] != v0:
                    raise ValueError(f"Inconsistent video_kwargs['{k}'] across batch: idx0={v0}, idx{j}={batch_video_kwargs_list[j][k]}")

    inputs = processor(
        text=texts,
        images=batch_images,
        videos=None,
        return_tensors="pt",
        padding=True,
        **merged_video_kwargs
    )

    labels = inputs["input_ids"].clone()
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100


    qwen3_vision_tokens = [
        151652,  # <|vision_start|>
        151653,  # <|vision_end|>
        151654,  # <|vision_pad|>
        151655,  # <|image_pad|>
        151656,  # <|video_pad|>
    ]

    for visual_token_id in qwen3_vision_tokens:
        labels[labels == visual_token_id] = -100


    im_start_id = 151644  # Qwen 的 <|im_start|> ID
    input_ids = inputs["input_ids"]

    for i in range(len(input_ids)):

        start_indices = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]

        if len(start_indices) > 0:
            last_start_idx = start_indices[-1]
            labels[i, :last_start_idx + 2] = -100

    inputs["labels"] = labels
    return inputs



if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
    )

    if "Qwen3-VL" in model_config.model_name_or_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    if training_args.report_to == "wandb":
        wandb.init(project="llm-training")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()