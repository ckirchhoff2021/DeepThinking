
import os
import torch
from PIL import Image

from .base import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class MiniCPM4B(BaseModel):
    def __init__(self, path, device, name='MiniCPM3-4B', dtype=torch.bfloat16):
        super(MiniCPM4B, self).__init__(path, device, name)
        model, tokenizer = self.from_pretrained(dtype)
        self.model = model
        self.tokenizer = tokenizer
     
     
    def from_pretrained(self, dtype):
        model = AutoModelForCausalLM.from_pretrained(
            self.path, torch_dtype=dtype, device_map=self.device, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        return model, tokenizer

   
