# -*- coding: utf-8 -*-
import json
import re
import os
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def evaluate(json_path, image_root, checkpoint_path):

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 10, "video": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
        max_tokens=4096,
        repetition_penalty=1.0,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(checkpoint_path)
    dataset = []
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            dataset = json.loads(content)
        else:
            for line in content.split('\n'):
                if line.strip():
                    dataset.append(json.loads(line))

    results_log = []
    correct_count = 0
    for item in tqdm(dataset):
        try:
            content = []
            for img_rel_path in item['images']:
                full_path = os.path.abspath(os.path.join(image_root, img_rel_path))
                content.append({"type": "image", "image": f"file://{full_path}"})


            prompt_text = (
                f"Question: {item['question']}\n\n"
                "Please reason step by step. First, you need to provide a description of the perspective transformation applied to this set of images. "
                "Briefly explain how the perspective changes between the images (e.g., rotated 90 degrees clockwise), and describe the relative positions of the objects based on the background and occlusion relationships. "
                "Provide your description process and enclose it within the <spatial_thinking> </spatial_thinking> tags. \n\n "
                "When the perspective changes, use multiple images to discover objects that are not visible in a single image, and provide your detailed reasoning for the raw question within <thinking> </thinking>. \n\n "
                "Provide only the single option letter (e.g., A, B, C, D) within the <answer> </answer> tags."
            )
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            vllm_input = prepare_inputs_for_vllm(messages, processor)
            outputs = llm.generate([vllm_input], sampling_params=sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text
            pred = extract_answer(generated_text)
            gt = item.get('answer', item.get('gt_answer'))

            is_correct = (pred is not None and pred == gt)

            if is_correct:
                correct_count += 1

            results_log.append({
                "id": item['id'],
                "question_type": item.get('question_type', 'N/A'),
                "pred": pred,
                "gt": gt,
                "is_correct": is_correct,
                "response": generated_text
            })

            if len(results_log) % 10 == 0:
                with open("eval_temp.json", "w", encoding="utf-8") as f:
                    json.dump(results_log, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"{e}")
            continue
    accuracy = correct_count / len(results_log) if results_log else 0
    print("\n" + "=" * 40)
    print(f"Accuracy: {accuracy:.2%}")

    with open("Result_MMSI.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    DATA_JSON = "MMSI/MMSI_Bench.json"
    IMAGE_ROOT = "./MMSI"
    MODEL_PATH = "Put your model path here"
    evaluate(DATA_JSON, IMAGE_ROOT, MODEL_PATH)