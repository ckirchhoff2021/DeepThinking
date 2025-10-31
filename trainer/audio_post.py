import json
import logging
import os

from functools import partial
from typing import Dict

import torch
import transformers
from accelerate.utils import DistributedType

from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers.integrations import deepspeed

from .argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .utils import language_audio_data_collator, safe_save_model_for_hf_trainer, get_parameter_number
from .causal_trainer import CausalLMTrainer
from .sft import make_language_supervised_data_module


local_rank = 0
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def load_from_pretrained(model_path, device_map, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=dtype,
        # attn_implementation="flash_attention_2",
        device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    return model, processor


def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    model, tokenizer = load_from_pretrained(model_args.model_name_or_path, device_map, dtype=compute_dtype)
   
    if not training_args.tune_encoder:
        model.audio_tower.requires_grad_(False)
    if not training_args.tune_projector:
        model.audio_projector.requires_grad_(False)
    if not training_args.tune_llm:
        model.language_model.requires_grad_(False)
   
    if training_args.use_lora: 
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model can only apply LoRA when use_lora is false.")
                  
        rank0_print("==> Currently using LoRA for fine-tuning the model.")
        # for _, param in model.named_parameters():
        #     param.requires_grad = False
            
        modules_to_save = ['embed_tokens', 'audio_projector']  # add training for embed_tokens, especially when adding new tokens
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))
    data_module = make_language_supervised_data_module(
        data_args,
        tokenizer,
        data_collator=language_audio_data_collator,
        max_length=training_args.model_max_length,
    )
    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
     
    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        training_args.lr_scheduler_kwargs = {"min_lr_rate": 0.1}
        
    trainer = CausalLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    train_dataset = trainer.train_dataset
    train_data_num = len(train_dataset)
    rank0_print("==> train dataset num: {}".format(train_data_num))

    trainer.train()
    trainer.save_state()
    
    final_path = os.path.join(training_args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    rank0_print("save final path to {}".format(final_path))
    safe_save_model_for_hf_trainer(trainer, final_path)


if __name__ == "__main__":
    train()
