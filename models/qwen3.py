
import torch
from .base import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3(BaseModel):
    def __init__(self, path, device, name='Qwen3-0.6B', dtype=torch.bfloat16):
        super(Qwen3, self).__init__(path, device, name)
        model, tokenizer = self.from_pretrained(dtype)
        self.model = model
        self.tokenizer = tokenizer
     

    def from_pretrained(self, dtype):
        model = AutoModelForCausalLM.from_pretrained(
            self.path, torch_dtype=dtype, device_map=self.device,
            attn_implementation="flash_attention_2"
        )
        print(model)
        tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        return model, tokenizer
            
    def chat(self, prompt, thinking=False):
        message = [{'role': 'user', 'content': prompt}]
        text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        # print('applt_chat_template: \n', text)
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("==> thinking content: \n", thinking_content)
        # print('---' * 5)
        # print("==> content: \n", content)    
        return content
            

if __name__ == '__main__':
    path = '/home/chenxiang.101/workspace/checkpoints/Qwen3-0.6B'
    instance = Qwen3(path, 'cuda:0', 'qwen3-0.6b')
    prompt = "已知3x+2=10，求x的值。"
    
    rsp = instance.chat(prompt, thinking=True)
    print(rsp)
