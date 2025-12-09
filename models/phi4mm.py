import requests
import torch
import os
import io
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen



class Phi4MM(object):
    def __init__(
        self, 
        path="/home/chenxiang.101/workspace/checkpoints/Phi-4-multimodal-instruct", 
        dtype=torch.bfloat16,
        device='cuda:0',
    ):
        
        self.path = path
        self.dtype = dtype
        self.device = device
        
        model, processor, generation_config = self.from_pretrained()
        self.model = model
        self.processor = processor
        self.generation_config = generation_config
        
        self.user_prompt = '<|user|>'   
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'
     
     
    def from_pretrained(self):
        processor = AutoProcessor.from_pretrained(
            self.path, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.path, 
            device_map=self.device, 
            torch_dtype=self.dtype, 
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation='flash_attention_2',
        )
        # print(model)
        generation_config = GenerationConfig.from_pretrained(self.path)
        return model, processor, generation_config

    @torch.inference_mode()
    def chat(self, prompt, image=None, max_new_tokens=128):
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise ValueError(f"Invalid image type: {type(image)}")
            
        text = f'{self.user_prompt}<|image_1|>{prompt}{self.prompt_suffix}{self.assistant_prompt}'
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors='pt'
        ).to('cuda:0')
        
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
   
   
    @torch.inference_mode()
    def chat_with_audio(self, prompt, audio_file=None, max_new_tokens=128):
        # audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
        # speech_prompt = "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation."
        text = f'{self.user_prompt}<|audio_1|>{prompt}{self.prompt_suffix}{self.assistant_prompt}'

        # Downlowd and open audio file
        # audio, samplerate = sf.read(io.BytesIO(urlopen(audio_file).read()))
        audio, samplerate = sf.read(audio_file)
        # print(audio.shape)
        # print(samplerate)

        # Process with the model
        inputs = self.processor(
            text=text,
            audios=[(audio, samplerate)], 
            return_tensors='pt'
        ).to('cuda:0')

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
   
   
     
    @torch.inference_mode()
    def chat_image_audio(self, image, audio_file, max_new_tokens=128):
        # <|user|><|image_1|><|audio_1|><|end|><|assistant|>
        # <|user|><|image_1|><|image_2|><|image_3|><|audio_1|><|end|><|assistant|>
        # audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
        # speech_prompt = "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation."
        
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise ValueError(f"Invalid image type: {type(image)}")
        
        text = f'{self.user_prompt}<|image_1|><|audio_1|>{self.prompt_suffix}{self.assistant_prompt}'

        # Downlowd and open audio file
        # audio, samplerate = sf.read(io.BytesIO(urlopen(audio_file).read()))
        audio, samplerate = sf.read(audio_file)
        # print(audio.shape)
        # print(samplerate)

        # Process with the model
        inputs = self.processor(
            text=text, 
            images=image,  
            audios=[(audio, samplerate)], 
            return_tensors='pt'
        ).to('cuda:0')

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
   
   
   
if __name__ == '__main__':
    # py38-y, py310-x transformers=4.48.2
    model = Phi4MM()
    image = "/home/chenxiang.101/workspace/images/loss.png"
    prompt = "请用中文描述一下这张图像 "
    response = model.chat(prompt, image=image, max_new_tokens=256)
    print(response)   
    
    audio = '/home/chenxiang.101/workspace/checkpoints/Voice/wavs/en-cn.wav'
    prompt = 'Transcribe the audio to text.'
    response = model.chat_with_audio(prompt, audio)
    print(response)   
    
    
    audio = '/home/chenxiang.101/workspace/checkpoints/Voice/wavs/en-cn2.wav'
    prompt = 'Transcribe the audio to text.'
    response = model.chat_with_audio(prompt, audio)
    print(response)
    
    
    audio = '/home/chenxiang.101/workspace/checkpoints/Voice/wavs/xxx-en.wav'
    response = model.chat_with_audio(prompt, audio)
    print(response)   

