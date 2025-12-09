
import os
import torch
from PIL import Image

import soundfile as sf

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class QwenOmni(object):
    def __init__(self, 
                 path='/home/chenxiang.101/workspace/checkpoints/Qwen2.5-Omni-7B', 
                 device='cuda:0', 
                 name='QwenOmni', 
                 dtype=torch.bfloat16):
        self.path = path
        self.device = device
        self.name = name
        self.dtype = dtype
        
        model, processor = self.from_pretrained()
        self.model = model
        self.processor = processor
     
     
    def from_pretrained(self):
        # model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        model = Qwen2_5OmniModel.from_pretrained(
            self.path,
            torch_dtype=self.dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        processor = Qwen2_5OmniProcessor.from_pretrained(self.path)

        return model, processor

    @torch.inference_mode()
    def chat(self, text, image=None, video=None, audio=None, max_new_tokens=128):
        '''
        conversation4 = [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/path/to/image.jpg"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                    {"type": "audio", "audio": "/path/to/audio.wav"},
                    {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
                ],
            }
        ]
        '''
        conversations = list()
        conversations.append(
             {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
        )
        content = list()
        if image is not None:
            image_dict = {"type": "image", "image": image}
            content.append(image_dict)
        if video is not None:
            video_dict = {"type": "video", "video": video}
            content.append(video_dict)
        if audio is not None:
            audio_dict = {"type": "audio", "audio": audio}
            content.append(audio_dict)
        text_dict = {"type": "text", "text": text}
        content.append(text_dict)
        conversations.append({'role': 'user', 'content': content})
        
        # Preparation for batch inference
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)

        inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Batch Inference
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=True)
        response = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        sf.write(
            "output.wav",
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        return response
   
   
if __name__ == '__main__':
    instance = QwenOmni()
    image = "/home/chenxiang.101/workspace/images/loss.png"
    prompt = "请描述一下这张图像。"
    response = instance.chat(prompt, image=image, max_new_tokens=256)
    print(response)   
    
   
