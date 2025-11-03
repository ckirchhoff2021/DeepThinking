import json
import requests
from openai import OpenAI, NOT_GIVEN
from .preprocess import construct_message


class GeneralAPI(object):
    def __init__(self, api_key, api_base, model_name):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def chat(self, prompt, max_tokens=512, temperature=0.7, top_p=0.1, system_message=None):
        # system_message="You are a helpful assistant."
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
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        # print(response)
        return response.choices[0].message.content

    def chat_with_function_call(self, prompt, tools, system_message='', tool_choice=None):
        messages = []
        if len(system_message) > 0:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        tool_choice = tool_choice if tool_choice else NOT_GIVEN
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        return completion.choices[0]
    
    def generate_with_function_call(self, messages, tools):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools
        )
        return completion.choices[0]
    
    def chat_with_image(self, prompt, img=None, system_message=None):
        messages = construct_message(prompt, img, system_message)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return completion.choices[0]
    
    
    def chat_with_image_and_function_call(self, prompt, img_file, tools, tool_choice=None, system_message=''):
        messages = []
        if len(system_message) > 0:
            messages.append({"role": "system", "content": system_message})
        vllm_message = construct_message(prompt, img_file)
        messages.extend(vllm_message)
        tool_choice = tool_choice if tool_choice else NOT_GIVEN
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        return completion.choices[0]
    
    
    def chat_with_messages(self, messages, max_tokens=512, temperature=0.1, top_p=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content
        

