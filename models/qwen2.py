
import torch
from .base import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen2(BaseModel):
    def __init__(self, path, device, name='Qwen2-7B', dtype=torch.bfloat16):
        super(Qwen2, self).__init__(path, device, name)
        model, tokenizer = self.from_pretrained(dtype)
        self.model = model
        self.tokenizer = tokenizer
     

    def from_pretrained(self, dtype):
        model = AutoModelForCausalLM.from_pretrained(
            self.path, torch_dtype=dtype, device_map=self.device,
            attn_implementation="flash_attention_2"
        )
        # for name, param in model.named_parameters():
        #     print(name)
        tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        return model, tokenizer
            
            

if __name__ == '__main__':
    path = '/home/chenxiang.101/workspace/checkpoints/eou-7B'
    instance = Qwen2(path, 'cuda:0', 'qwen2.5-7b')
    
    messages = [
        {
            'role': 'user',
            'content': '最近有什么好'
        }
    ]
    # print(instance.model.config)
    # prompt = "给我讲个笑话。"
    
    rsp = instance.chat_with_messages(messages)
    print(rsp)
