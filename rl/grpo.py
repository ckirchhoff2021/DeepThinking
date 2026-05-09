# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

import torch
import datasets
import re

import wandb

checkpoint = "Qwen3.5-0.8B"   # "/mnt/bn/ts-vllm-cpt-0/Qwen3.5-4B"
base_model = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(checkpoint)

lora_config = LoraConfig(
    r=16,  # LoRA 秩
    lora_alpha=32,  # LoRA alpha 参数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,  # Dropout 概率
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 应用 LoRA 到模型
base_model = get_peft_model(base_model, lora_config)
base_model.print_trainable_parameters()  # 打印可训练参数数量


def format_reward_func(completions, **kwargs):
    """
    completions: list of lists, shape: [batch_size, num_generations]
    """
    # pattern = r'<reason>(.*?)</reason>\s*<result>(.*?)</result>'
    pattern = r'<reason>(.*?)</reason>\s*<result>(.*?)</result>\s*<recover>(.*?)</recover>'
    rewards = []
    for completion_group in completions:
        # completion_group 是一个 prompt 生成的多个 completions
        for content in completion_group:
            match = re.search(pattern, content['content'], re.DOTALL)
            rewards.append(0.2 if match else 0.0)
    # print(completions)
    # print('==> format: ', rewards)
    return rewards


def extract_result(completion):
    pattern = r'<result>(.*?)</result>'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()  
    else:
        return ""
    

def extract_recover(completion):
    pattern = r'<recover>(.*?)</recover>'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()  
    else:
        return ""


def curve_accuracy_reward(completions, **kwargs):
    """
    completions: list of lists, shape: [batch_size, num_generations]
    ground_truth: list, shape: [batch_size]
    """
    # print(completions)
    ground_truth = kwargs.get("curve_ground_truth")

    # print(recover_ground_truth)
    rewards = []
    for completion_group, gt in zip(completions, ground_truth):
        # completion_group 是一个 prompt 生成的多个 completions
        for completion in completion_group:
            res = extract_result(completion['content'])
            rewards.append(1.0 if res == gt else 0.0)

    # print(ground_truth)
    # print('==> accuracy: ', rewards)
    return rewards


def recover_accuracy_reward(completions, **kwargs):
    """
    completions: list of lists, shape: [batch_size, num_generations]
    ground_truth: list, shape: [batch_size]
    """
    # print(completions)
    ground_truth = kwargs.get("recover_ground_truth")

    # print(recover_ground_truth)
    rewards = []
    for completion_group, gt in zip(completions, ground_truth):
        # completion_group 是一个 prompt 生成的多个 completions
        for completion in completion_group:
            res = extract_recover(completion['content'])
            rewards.append(0.5 if res == gt else 0.0)

    # print(ground_truth)
    # print('==> accuracy: ', rewards)
    return rewards


dataset = datasets.load_dataset("json", data_files="seed-grpo.json")["train"]
dataset = dataset.rename_column("recover_label", "recover_ground_truth")
dataset = dataset.rename_column("curve_label", "curve_ground_truth")

def fix_file_path(example):
    example["images"][0] = example["images"][0].replace("datas/argos-dp", "")
    return example  

print(dataset[0]["images"])
dataset = dataset.map(fix_file_path, num_proc=4)
print(dataset[0]["images"])


print("dataset num: ", len(dataset))
training_args = GRPOConfig(
    output_dir="train/seed/grpo_outputs",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=5, 
    logging_first_step=True, 
    logging_dir="seed/grpo_outputs/runs", 
    save_steps=100,
    report_to=["wandb"],
    fp16=True,
    learning_rate=1e-5,
    use_vllm=False,
    num_generations=8,              # 每个prompt生成的样本数
    max_completion_length=4096,     # 生成文本的最大长度
)

trainer = GRPOTrainer(
    model=base_model,
    processing_class=processor,
    reward_funcs=[format_reward_func, curve_accuracy_reward, recover_accuracy_reward],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# mlx worker launch --cluster=cloudnative-hl --type=NVIDIA-H20 --resourcetype=arnold --queuename=compute-329-hl-cloudnative-ai-iesqa.llm4se-guarantee --usergroup=LLM4SE --gpu=1 -- bash 2>&1
