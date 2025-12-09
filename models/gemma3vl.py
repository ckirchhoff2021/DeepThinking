
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


class Gemma3VL(object):
    def __init__(
        self, 
        path="/home/chenxiang.101/workspace/checkpoints/gemma-3-270m-it", 
        dtype=torch.bfloat16,
        device='cuda:0',
    ):
        
        self.path = path
        self.dtype = dtype
        self.device = device
        
        model, processor = self.from_pretrained()
        self.model = model
        self.processor = processor
     
    def from_pretrained(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.path, device_map=self.device, torch_dtype=self.dtype
        ).eval()
        print(model)
        processor = AutoProcessor.from_pretrained(self.path)
        return model, processor

    @staticmethod
    def construct_inputs_message(system_prompt, user_prompt, image=None):
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }
        user_message = dict()
        user_message["role"] = "user"
        text_content = {"type": "text", "text": user_prompt}
        
        if image is not None:
            image_content = {"type": "image", "image": image}
            user_message["content"] = [image_content, text_content]
        else:
            user_message["content"] = [text_content]
            
        messages = [system_message, user_message]
        return messages

    @torch.inference_mode()
    def generate(self, prompt, image=None, max_new_tokens=128, system_prompt="You are a helpful assistant."):
        messages = self.construct_inputs_message(system_prompt, prompt, image)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        generation = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generation = generation[0][input_len:]
        outputs = self.processor.decode(generation, skip_special_tokens=True)
        return outputs
   
   
if __name__ == '__main__':
    model = Gemma3VL()
    prompt = "你是谁？"
    response = model.generate(prompt)
    print(response)
    
    
