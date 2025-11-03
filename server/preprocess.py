import json
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import List
import numpy as np


def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")
    return base64_str


def pil_to_base64_with_prefix(image: Image.Image, fmt='jpeg') -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    b64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{fmt};base64,' + b64_str


def pil_to_base64_without_prefix(image: Image.Image, fmt='jpeg') -> str:
    b64_str = pil_to_base64_with_prefix(image, fmt)
    return b64_str.split(',')[-1]


def construct_message(prompt, img=None, system_message=None):
    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': system_message})
        
    text = {"type": "text", "text": prompt}
    if img:
        if isinstance(img, Image.Image):
            b64 = pil_to_base64_without_prefix(img)
        else:
            # b64 = img_to_base64(img)
            image = Image.open(img).convert('RGB')
            b64 = pil_to_base64_without_prefix(image)
            
        image = {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        messages.append({"role": "user", "content": [text, image]})
    else:
        messages.append({"role": "user", "content": [text]})
    return messages


def construct_message_with_url(prompt, img_url=None):
    text = {"type": "text", "text": prompt}
    if img_url:
        image = {"type": "image_url",
                 "image_url": {"url": img_url}}
        messages = [{"role": "user", "content": [text, image]}]
    else:
        messages = [{"role": "user", "content": [text]}]
    return messages


def parser_stream_outputs(response):
    outputs = ''
    for res in response:
        outputs += res.decode()
    res_list = outputs.split('\n')
    result = ''
    for res in res_list:
        prefix = 'data:'
        if not res.startswith(prefix):
            continue
        data = json.loads(res[len(prefix):])
        result += data['choices'][0]['delta']['content']
        if data['choices'][0]['finish_reason'] == 'stop':
            break
    return result


def sliced_norm_l2(vec: List[float], dim=2048) -> List[float]:
    norm = float(np.linalg.norm(vec[:dim]))
    return [v / norm for v in vec[:dim]]
