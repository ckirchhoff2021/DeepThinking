
import torch
from .model import Model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel(Model):
    def __init__(self, path, device, name, dtype=torch.bfloat16):
        self.path = path
        self.device = device
        self.model_name = name
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    
    def from_pretrained(self, dtype):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, torch_dtype=dtype, device_map=self.device, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
       
       
    def update_peft(self, path):
        lora_model = PeftModel.from_pretrained(self.model, path)
        print('Successfully load peft weights ......')
        merged_model = lora_model.merge_and_unload()
        self.model = merged_model
        return merged_model
    
    
    def construct_message(self, text, system_message):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
        return messages
        
        
    def chat(self, prompt, system_message="You are a helpful assistant.", **kwargs):
        messages = self.construct_message(prompt, system_message=system_message)
        response = self.chat_with_messages(messages, **kwargs)
        return response
    
    
    def generate_with_chat_template(self, texts, **kwargs):
        inputs = self.tokenizer(texts, return_tensors="pt").to(self.device)
        inputs = inputs.to(self.model.device)
        outputs = self.generate(**inputs, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return response
    
    
    def generate(self, **kwargs):
        generated_ids = self.model.generate(**kwargs)
        return generated_ids


    def batch_chat(self, prompts, max_new_tokens=128, 
                   system_message="You are a helpful assistant."):
        N = len(prompts)
        texts = list()
        for i in range(N):
            prompt = prompts[i]
            messages = self.construct_message(prompt, system_message)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
    
        response = self.generate_with_chat_template(texts, max_new_tokens)
        return response
    
    
    def chat_with_messages(self, messages, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print(text)
        response = self.generate_with_chat_template([text], **kwargs)
        return response
            
            
    def save_pretrained(self, path):
        self.model.save_pretrained(
            save_directory=path,
            state_dict=self.model.state_dict(),
            max_shard_size="5GB",
            safe_serialization=True,
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(
                save_directory=path,
            )
