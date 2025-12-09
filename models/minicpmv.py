
import os
import torch
from PIL import Image

from .base import BaseModel
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


class MiniCPMV(BaseModel):
    def __init__(self, 
                 path='/home/chenxiang.101/workspace/checkpoints/MiniCPM-V-2_6', 
                 device='cuda:0', 
                 name='minicpm-v-26', 
                 dtype=torch.bfloat16):
        super(MiniCPMV, self).__init__(path, device, name)
        model, tokenizer = self.from_pretrained(dtype)
        self.model = model
        self.tokenizer = tokenizer
     
     
    def from_pretrained(self, dtype):
        model = AutoModel.from_pretrained(
            self.path, trust_remote_code=True, 
            attn_implementation='sdpa', 
            torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval().to(self.device)
        print(model)
        tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True) 
        return model, tokenizer

    def construct_vl_message(self, question, img=None):
        if img is None:
            return [{'role': 'user', 'content': [question]}]
        
        if isinstance(img, Image.Image):
            image = img.convert('RGB')
        else:
            try:
                image = Image.open(img).convert('RGB')
            except:
                image = None
                
        content = [image, question] if image else [question]
        message = [{'role': 'user', 'content': content}]
        return message

    def chat(self, prompt, img):
        message = self.construct_vl_message(prompt, img)
        # x = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # print(x)
        output = self.model.chat(image=None, msgs=message, tokenizer=self.tokenizer)
        return output
   
   
   
if __name__ == '__main__':
    instance = MiniCPMV()
    img_file = 'images/2.jpg'
    prompt = "请描述一下这张图片。"
    response = instance.chat(prompt, img_file)
    print(response)
