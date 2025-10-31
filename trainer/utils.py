from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers.integrations import deepspeed
import json
import librosa
from io import BytesIO
from urllib.request import urlopen
import torch
from transformers.feature_extraction_utils import BatchFeature


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    '''
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), bias)
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)  
    '''
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)


def language_data_collator(inputs, padding_value=0, max_length=2048, collate_labels=False):
    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence(
            [s[:max_length] for s in seq],
            batch_first=batch_first,
            padding_value=padding_value,
        )

    input_ids = trim_and_pad(
        [example["input_ids"] for example in inputs],
        batch_first=True,
        padding_value=padding_value,
    )
    
    position_ids = trim_and_pad(
        [example["position_ids"] for example in inputs],
        batch_first=True,
        padding_value=0,
    )

    attention_mask = trim_and_pad(
        [example["attention_mask"] for example in inputs],
        batch_first=True,
        padding_value=0,
    )

    outputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
    
    if collate_labels:
        labels = trim_and_pad(
            [example["labels"] for example in inputs],
            batch_first=True,
            padding_value=-100,
        )
        outputs["labels"] = labels

    return outputs


def language_audio_data_collator(inputs, padding_value=0, max_length=2048, collate_labels=False):
    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence(
            [s[:max_length] for s in seq],
            batch_first=batch_first,
            padding_value=padding_value,
        )

    input_ids = trim_and_pad(
        [example["input_ids"] for example in inputs],
        batch_first=True,
        padding_value=padding_value,
    )

    attention_mask = trim_and_pad(
        [example["attention_mask"] for example in inputs],
        batch_first=True,
        padding_value=0,
    )

    feature_attention_mask = torch.cat(
        [example["feature_attention_mask"] for example in inputs], dim=0
    )
    
    input_features = torch.cat(
        [example["input_features"] for example in inputs], dim=0
    )
    
    outputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_features": input_features,   
        "feature_attention_mask": feature_attention_mask
    }
    
    if collate_labels:
        labels = trim_and_pad(
            [example["labels"] for example in inputs],
            batch_first=True,
            padding_value=-100,
        )
        outputs["labels"] = labels

    return BatchFeature(data={**outputs})


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {"Total": all_param, "Trainable": trainable_params}


def load_json_datas(data_file):
    if data_file.endswith(".json"):
        with open(data_file, "r") as f:
            raw_data_list = json.load(f)
            return raw_data_list
    elif data_file.endswith(".jsonl"):
        with open(data_file, "r", encoding='utf-8') as f:
            raw_data_list = [json.loads(line) for line in f]
            return raw_data_list
    else:
        raise ValueError(f"Unsupported data file format: {data_file}")
    

def role_transfer(conversations):
    '''
    from -> role : human -> user
    value -> content : gpt -> assistant
    '''
    role_dict = {
        "human" : "user",
        "gpt": "assistant"
    }
    for conv in conversations:
        if 'from' in conv:
            conv["role"] = role_dict[conv.pop("from")]
        if 'value' in conv:
            conv["content"] = conv.pop("value")
    
    return conversations


def eou_punc_augmented(system_prompt, conversation):
    if len(system_prompt) == 0:
        return conversation
    
    punctations = ["。", "！", "？", ".", "!", "?" , ",", "，"]
    convs = list()
    conversation = role_transfer(conversation)
    for item in conversation:
        if item['role'] == 'user':
            prob = np.random.rand()
            content = item['content']
            if content[-1] in punctations:
                if prob > 0.5:
                    content = content[:-1]
            else:
                if prob > 0.5:
                    index = np.random.randint(len(punctations))
                    content += punctations[index]
            convs.append({
                'role': item['role'],
                'content': content
            })
        else:
            convs.append({
                'role': item['role'],
                'content': item['content']
            })
    return convs
    
    
def load_audio_file(audio_url, sampling_rate=16000):
    audio_bytes = BytesIO(open(audio_url, 'rb').read()) # BytesIO(urlopen(audio_url).read())
    return librosa.load(audio_bytes, sr=sampling_rate)[0]


def construct_text_audio_conversation(audio_file, system_prompt, task_prompt, text):
    conversation = [
        {
            'role': 'system', 
            'content': f"{system_prompt}"
        }, 
        {
            "role": "user", 
            "content": [
                {
                    "type": "audio", "audio_url": f"{audio_file}"
                },
                {
                    "type": "text", "text": f"{task_prompt}"
                },
            ]
        },
        {
            "role": "assistant", 
            "content": f"{text}"
        }
    ]
    return conversation


def get_labels_by_input_ids(inputs, tokenizer, max_length=2048):
    input_ids = inputs['input_ids']
    input_ids = input_ids.squeeze().numpy()
    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_start|>'))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('assistant'))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_end|>'))[0]

    context = np.ones_like(input_ids, dtype=np.int8)
    for assistant_idx in assistant_idxs:
        if assistant_idx-1 in set(start_idxs):
            st = assistant_idx + 1
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st: end_idx + 1] = 0
                    break

    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))

    if input_ids.shape[-1] > max_length:
        ids = ids[:max_length]
        context = context[:max_length]
     
    if torch.all(context):
        raise Exception("No tokens available to compute loss.")

    # build target
    target = torch.full_like(ids, -100, dtype=torch.long)
    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            target[i - 1] = tokenizer.eos_token_id
    
    inputs['labels'] = target
    inputs['input_ids'] = ids     
    inputs['attention_mask'] = torch.arange(len(ids))
        
    return inputs
