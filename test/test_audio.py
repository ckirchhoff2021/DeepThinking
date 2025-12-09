from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import os
import json
import torch

checkpoints = "/Users/bytedance/Downloads/Git/Gitlab/models/Qwen2-Audio-7B-Instruct"
    
def extract_audios(conversation, processor):
    sampling_rate = processor.feature_extractor.sampling_rate
    def load_audio_file(ele):
        audio_bytes = BytesIO(open(ele['audio_url'], 'rb').read()) # BytesIO(urlopen(ele['audio_url']).read())
        return librosa.load(audio_bytes, sr=sampling_rate)[0]
    
    audios = []
    for message in conversation:
        if not isinstance(message["content"], list):
            continue
        for ele in message['content']:
            if ele['type'] == 'audio':
                audios.append(load_audio_file(ele))
    return audios
    
    
def test_processor():
    processor = AutoProcessor.from_pretrained(checkpoints)
    audio_file = '/Users/bytedance/Downloads/outputs/wavs/English/10055191986589907747.wav'
    conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": f"{audio_file}"},
                {"type": "text", "text": "Transcribe the audio inputs."},
            ]}
        ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    
    '''
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
    Transcribe the audio inputs.<|im_end|>
    <|im_start|>assistant
    '''
    
    audios = extract_audios(conversation, processor)
    print(audios[0].shape)
    print(text)
    
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    print(inputs.keys())
    print(processor.audio_token)

    
def test_multi_turn():
    processor = AutoProcessor.from_pretrained(checkpoints)
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
            {"type": "text", "text": "What's that sound?"},
        ]},
        {"role": "assistant", "content": "It is the sound of glass shattering."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
            {"type": "text", "text": "What can you hear?"},
        ]}
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    '''
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
    What's that sound?<|im_end|>
    <|im_start|>assistant
    It is the sound of glass shattering.<|im_end|>
    <|im_start|>user
    Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>
    What can you hear?<|im_end|>
    <|im_start|>assistant
    '''
    print(text)

    
def test_qwen_audio_encoder():
    from transformers import AutoModel
    configs = json.load(open(os.path.join(checkpoints, 'config.json')))
    print(configs['audio_config'])
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoderConfig
    config = Qwen2AudioEncoderConfig(**configs['audio_config'])
    audio_tower = AutoModel.from_config(config)
    print(audio_tower)
    
    inputs = torch.randn(1, 128, 3000)
    attention_mask = (torch.randn(1, 1, 1500, 1500) > 0.2)
    outputs = audio_tower(inputs, attention_mask=attention_mask)
    print(outputs.last_hidden_state.shape)  # (1, 750, 1280)
    
    
def test_qwen_processor():
    audio_file = "/Users/bytedance/Downloads/outputs/wavs/English/10055191986589907747.wav"
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": f"{audio_file}"},
            {"type": "text", "text": "Transcribe the audio inputs."},
        ]}
    ]
    processor = AutoProcessor.from_pretrained(checkpoints)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = extract_audios(conversation, processor)
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    print(inputs.keys())                                 
    print(inputs['input_ids'].shape)                     # 1, 175
    print(inputs['attention_mask'].shape)                # 1, 175
    print(inputs['input_features'].shape)                # 1, 128, 3000
    print(inputs['feature_attention_mask'].shape)        # 1, 3000
    
    
if __name__ == '__main__':
    # test_processor()
    # test_multi_turn()
    test_qwen_audio_encoder()
    # test_qwen_processor()
