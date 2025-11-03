
from volcenginesdkarkruntime import Ark
# pip install -U 'volcengine-python-sdk[ark]'

class ArkAPI:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = Ark(
            api_key=api_key,
            timeout=1800
        )
    
    def chat(self, prompt, system_message=None, thinking='disabled', max_tokens=512, temperature=0.7, top_p=0.1):
        # thinking: disabled, enabled, auto
        if system_message is not None:
            messages = [
                {"role": "system", "content": f"{system_message}"},
                {"role": "user", "content": f"{prompt}"},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{prompt}"},
            ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            thinking={"type": thinking},
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content
