
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict
from transformers.trainer_pt_utils import LabelSmoother
import copy
import os
from .utils import eou_punc_augmented, role_transfer, load_audio_file, construct_text_audio_conversation, get_labels_by_input_ids

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
    

def conversation_to_ids(conversation, tokenizer, max_length=2048, system_prompt='', llm_type='qwen2'):
    if llm_type == 'gemma3':
        input_ids, context = conversation_to_ids_gemma3(conversation, tokenizer, system_prompt)
    else:
        input_ids, context = conversation_to_ids_qwen2(conversation, tokenizer, system_prompt)
        
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
            
    position_ids = torch.arange(ids.size(0)).long()
    attention_mask = torch.ones_like(ids, dtype=torch.bool)
    
    return {
        "input_ids": ids,
        "labels": target,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }


def conversation_to_ids_qwen2(conversation, tokenizer, system_prompt=''):
    chat = []
    if len(system_prompt) > 0:
        chat.append({"role": "system", "content": system_prompt})
        
    context = []
    for _, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant", "system"]
        chat.append({"role": role, "content": message})

    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    input_ids = np.array(input_ids)

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
                  
    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context


def conversation_to_ids_gemma3(conversation, tokenizer, system_prompt=''):
    chat = []
    if len(system_prompt) > 0:
        chat.append({"role": "system", "content": system_prompt})
        
    context = []
    for _, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant", "system"]
        chat.append({"role": role, "content": message})

    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    input_ids = np.array(input_ids)

    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<start_of_turn>'))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('model'))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<end_of_turn>'))[0]

    context = np.ones_like(input_ids, dtype=np.int8)
    for assistant_idx in assistant_idxs:
        if assistant_idx-1 in set(start_idxs):
            st = assistant_idx + 1
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st: end_idx + 1] = 0
                    break
                  
    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context


def preprocess(
    conversations,
    tokenizer,
    system_prompt='',
    max_length=2048,
    llm_type='qwen2'
) -> Dict:   
    conversations = role_transfer(copy.deepcopy(conversations))
    assert len(conversations) > 1, "conversations length must large than 2"
    assert conversations[0]["role"] in ["user", "system"], "the first role must be user"

    sys_prompt = system_prompt
    if conversations[0]["role"] == 'system':
        sys_prompt = conversations[0]
        assert len(conversations) > 2, "conversations length must large than 3"
        assert conversations[1]["role"] == "user", "the second role must be user"
        conversations = conversations[1:]
 
    input_dict = conversation_to_ids(conversations, tokenizer, max_length, sys_prompt, llm_type)
    return input_dict
    

class LanguageSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, 
        raw_data,
        tokenizer,
        max_length=2048,
        llm_type='qwen2'
    ):
        super(LanguageSupervisedDataset, self).__init__()
        
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.llm_type = llm_type

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:  
        conversations = self.raw_data[i]['conversations']    
        system_prompt = self.raw_data[i].get('system', '')
        ret = preprocess(
            conversations,
            self.tokenizer,
            system_prompt=system_prompt,
            max_length=self.max_length,
            llm_type=self.llm_type
        )
        return ret


class EOUAugmentedDataset(LanguageSupervisedDataset):
    def __init__(
        self, 
        raw_data,
        tokenizer,
        max_length=2048,
        llm_type='qwen2'
    ):
        super(EOUAugmentedDataset, self).__init__(raw_data, tokenizer, max_length, llm_type)
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:  
        conversations = self.raw_data[i]['conversations']    
        system_prompt = self.raw_data[i].get('system', '')
        conversations = eou_punc_augmented(system_prompt, conversations)
        ret = preprocess(
            conversations,
            self.tokenizer,
            system_prompt=system_prompt,
            max_length=self.max_length,
            llm_type=self.llm_type
        )
        return ret


class AISHELLAudioDataset(Dataset):
    def __init__(
        self, 
        raw_data, 
        processor, 
        max_length=2048,
        data_root=''
    ):
        super(AISHELLAudioDataset, self).__init__()
        self.raw_data = raw_data
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.data_root = data_root
        self.task_prompt = "Transcribe the speech into text."
        self.system_prompt = "You are a professional speech recognition expert, and you can accurately transcribe various types of speech."

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        item = self.raw_data[index]
        wav = os.path.join(self.data_root, item['wav'])
        text = item['text'].replace(' ', '') # remove white spaces
        conversation = construct_text_audio_conversation(wav, self.system_prompt, self.task_prompt, text)
        audios = [load_audio_file(wav)]
        input_texts = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        inputs = self.processor(text=input_texts, audios=audios, return_tensors="pt", padding=True, sampling_rate=16000)
        inputs = get_labels_by_input_ids(inputs, self.tokenizer)
        return inputs


if __name__ == '__main__':
    path = '/Users/bytedance/Downloads/SIR-models/eou-gpt-0804'
    path = "/home/chenxiang.101/workspace/checkpoints/gemma-3-270m-it"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
 
    # 单图片数据
    eou_datas = [
        {
            "system": "你是一个语义判停专家",
            "conversations": [
                    {
                        'role': 'user', 
                        'content': '今天是什么日子啊，我想知道'
                    }, 
                    {
                        'role': 'assistant', 
                        'content': 'unfinished'
                    }
                ]
            },
        ]
    import json
    ego_datas = json.load(open('data/jsons/identity.json', 'r'))
    ds = EOUAugmentedDataset(ego_datas, tokenizer, llm_type='gemma3')
    print(ds[0])
    
    input_text = tokenizer.apply_chat_template(ego_datas[0]['conversations'], tokenize=False, add_generation_prompt=False)
    print(input_text)
    
    input_text = tokenizer.apply_chat_template(ego_datas[0]['conversations'], tokenize=True, add_generation_prompt=False)
    print(input_text)
    
    # x1 = tokenizer('unfinished')  # 15092
    # print(x1)
    
    # x2 = tokenizer('finished')    # 12129
    # print(x2)
