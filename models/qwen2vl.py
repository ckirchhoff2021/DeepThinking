import os
import torch
from PIL import Image

from .base import BaseModel

from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen2VL(BaseModel):
    def __init__(self, 
                 path="/home/chenxiang.101/workspace/checkpoints/Qwen2-VL-2B-Instruct", 
                 device='cuda:0', 
                 dtype=torch.bfloat16, 
                 name='Qwen2-VL-7B'):
        super(Qwen2VL, self).__init__(path, device, name)
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        # self.min_pixels = 4 * 28 * 28
        # self.max_pixels = 16384 * 28 * 28
        model, processor = self.from_pretrained(dtype)
        self.model = model
        self.processor = processor
        self.processor.tokenizer.padding_side = 'left'
     

    def from_pretrained(self, dtype=torch.bfloat16):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.path, torch_dtype=torch.bfloat16, device_map=self.device,
            attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(
            self.path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

        return model, processor


    def construct_message(self, text, image=None, system_message=None):
        ''' format is as follow.
         messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        '''
        messages = list()
        if system_message:
            messages.append(
                {"role": "system", "content": system_message}
            )
        text_dict = {"type": "text", "text": text}
        conversations = dict()
        conversations['role'] = 'user'
        contents = list()
        if image is not None:
            image_dict = {"type": "image", "image": image}
            contents.append(image_dict)
        contents.append(text_dict) 
        conversations['content'] = contents
        messages.append(conversations)
        return messages


    def chat(self, prompt, image=None, max_new_tokens=128, system_message=None):
        messages = self.construct_message(prompt, image, system_message)
        response = self.chat_with_messages(messages, max_new_tokens=max_new_tokens)
        return response
    
    
    def construct_video_message(self, text, video=None, system_message=None):
        ''' format is as follow.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": [
                                "file:///path/to/frame1.jpg",
                                "file:///path/to/frame2.jpg",
                                "file:///path/to/frame3.jpg",
                                "file:///path/to/frame4.jpg",
                            ],
                            "fps": 1.0,
                        },
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "file:///path/to/video1.mp4",
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ],
        '''
        messages = list()
        if system_message:
            messages.append(
                {"role": "system", "content": system_message}
            )
        text_dict = {"type": "text", "text": text}
        conversations = dict()
        conversations['role'] = 'user'
        contents = list()
        if video is not None:
            video_dict = {"type": "video", "video": video, "fps": 1.0}
            contents.append(video_dict)
        contents.append(text_dict) 
        conversations['content'] = contents
        messages.append(conversations)
        return messages
        
    
    def chat_with_video(self, prompt, video=None, max_new_tokens=1024, system_message=None):
        messages = self.construct_video_message(prompt, video, system_message)
        response = self.chat_with_messages(messages, max_new_tokens=max_new_tokens)
        return response
    
    
    def generate(self, max_new_tokens=128, **kwargs):
        generated_ids = self.model.generate(**kwargs, max_new_tokens=max_new_tokens)
        return generated_ids
      
      
    def generate_with_inputs(self, texts, image_inputs=None, video_inputs=None, max_new_tokens=128):
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        print(inputs.keys())
        inputs = inputs.to(self.model.device)
        outputs = self.generate(max_new_tokens=max_new_tokens, **inputs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response
    
    
    def chat_with_messages(self, messages, max_new_tokens=128):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        response = self.generate_with_inputs(
            [text], 
            image_inputs=image_inputs, 
            video_inputs=video_inputs, 
            max_new_tokens=max_new_tokens)
        return response[0]
        
      
    def batch_chat(self, prompts, images=None, max_new_tokens=1024):
        N = len(prompts)
        text_inputs = list()
        image_inputs = list()
        
        for i in range(N):
            prompt = prompts[i]
            if images is None:
                image = None
            else:
                image = images[i]    
            messages = self.construct_message(prompt, image)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_infos = None
            if image is not None:
                image_infos, _ = process_vision_info(messages)
                
            text_inputs.append(text)
            image_inputs.append(image_infos)
        
        response = self.generate_with_inputs(text_inputs, image_inputs, max_new_tokens=max_new_tokens)
        return response
            

if __name__ == '__main__':
    instance = Qwen2VL()
    
    messages = [
        {
            'role': 'user', 
            'content': [
                {"type": "image", "image": '/home/chenxiang.101/workspace/images/2.jpg'},
                {"type": "image", "image": '/home/chenxiang.101/workspace/images/123.jpg'},
                {"type": "text", "text": '请帮我描述一下第一章图像。'},
            ]
        }
    ]
    
    response = instance.chat_with_messages(messages)
    print(response)
    
        
    
