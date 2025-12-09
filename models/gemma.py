
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM
from .base import BaseModel

class Gemma(BaseModel):
    def __init__(
        self, 
        path="/home/chenxiang.101/workspace/checkpoints/gemma-3-270m-it", 
        dtype=torch.bfloat16,
        device='cuda:0',
    ):
        
        self.path = path
        self.dtype = dtype
        self.device = device
        
        model, tokenizer = self.from_pretrained()
        self.model = model
        self.tokenizer = tokenizer
     
    def from_pretrained(self):
        model = Gemma3ForCausalLM.from_pretrained(
            self.path, device_map=self.device, torch_dtype=self.dtype
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        return model, tokenizer

    def generate(self, prompt, max_new_tokens=128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
        input_len = inputs["input_ids"].shape[-1]
        response = outputs[0][input_len:]
        outputs = self.tokenizer.decode(response, skip_special_tokens=True)
        return outputs
   
    def chat(self, user_prompt, max_new_tokens=256, system_prompt=""):
        '''
        格式如下也是可以的。
        message = [
            {"role": "system", "content": "你是一个专业的人工智能助手。"},
            {"role": "user", "content":   "你叫什么名字？"}
        ]
        apply_chat_template:
        '<bos><start_of_turn>user\n你是一个专业的人工智能助手。\n\n你叫什么名字？<end_of_turn>\n<start_of_turn>model\n'
        '''
        messages = list()
        if len(system_prompt) > 0:
            system_message = {
                "role": "system",
                "content":  [{"type": "text", "text": system_prompt},]
            }
            messages.append(system_message)
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt},]
        }
        messages.append(user_message)
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # outputs = self.tokenizer.batch_decode(outputs)
        input_len = inputs["input_ids"].shape[-1]
        response = outputs[0][input_len:]
        outputs = self.tokenizer.decode(response, skip_special_tokens=True)
        return outputs
   
   
if __name__ == '__main__':
    model = Gemma()
    # print(model.model)
    prompt = "你是谁？"
    response = model.generate(prompt)
    print(response)
    
    system_prompt = "你是一个语义完整性判断专家，能结合上下文判断语句或者对话内容的完整性。请对用户输入的文本内容或者对话消息进行语义完整性判断，如果内容语义完整请输出finished，不完整请输出unfinished。"
    prompt = "请给我讲个笑话吧。"
    response = model.chat(prompt,  max_new_tokens=256, system_prompt='')
    print(response)
    
    
