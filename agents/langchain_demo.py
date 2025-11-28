from pydantic import BaseModel, Field
from langchain.tools import tool
from typing import Type
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

import os
from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain.agents import create_agent

from config import access_config


'''
refer to: https://docs.langchain.com/oss/python/langchain/agents
'''

class WeatherSchema(BaseModel):
    location: str = Field(description="城市或县区，比如北京市、杭州市、余杭区等。")


@tool("get_current_weather", args_schema=WeatherSchema)
def get_current_weather(location: str) -> str:
    """当你想查询指定城市的天气时非常有用。"""
    return f"{location}今天是雨天。"


@tool("get_current_time")
def get_current_time():
    """当你想知道现在的时间时非常有用。"""
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return f"当前时间：{formatted_time}。"


def set_doubao_api_key():
    api_key = access_config["doubao-1.5-pro-32k"]["api_key"]
    api_base = access_config["doubao-1.5-pro-32k"]["api_base"]
    model_name = access_config["doubao-1.5-pro-32k"]["model_name"]
    
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    return model_name


def test_function_call(bind_tools=False):
    model_name = set_doubao_api_key()
    model = ChatOpenAI(model_name=model_name)
    
    functions = [get_current_time, get_current_weather]
    tools = [convert_to_openai_tool(f) for f in functions]
    print(tools)
    
    if bind_tools:
        model_with_tools = model.bind_tools(functions)
    
    step = 1
    messages = [HumanMessage(content="杭州和北京天气怎么样？现在几点了？")]
    if bind_tools:
        assistant_output = model_with_tools.invoke(messages)
    else:
        assistant_output = model.invoke(messages, tools=tools)
    
    print(f'step {step} Assistant output: {assistant_output}')
    functions_to_call = {'get_current_weather': get_current_weather, 'get_current_time': get_current_time}
    
    while assistant_output.tool_calls:
        messages.append(assistant_output)
        for tool in assistant_output.tool_calls:
            args = tool['args'] if 'properties' not in tool['args'] else tool['args']['properties']
            tool_content = functions_to_call[tool['name']].invoke(args)
            messages.append(ToolMessage(
                name=tool['name'], 
                tool_call_id=tool['id'], 
                content=tool_content)
            )
            print(f"Tool output: {tool_content}\n")

        if bind_tools:
            assistant_output = model_with_tools.invoke(messages)
        else:
            assistant_output = model.invoke(messages, tools=tools)
        step += 1
        print(f"step {step} ==> Assistant output：{assistant_output}\n")
        
    print('Final outputs: ', assistant_output.content)



def test_chap_api():
    conversation = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("你叫什名字？")
    ]
    model_name = set_doubao_api_key()
    model = ChatOpenAI(model_name=model_name)
    response = model.invoke(conversation)
    print(response)


def test_agent_call():
    # https://docs.langchain.com/oss/python/langchain/agents
    model_name = set_doubao_api_key()
    model = ChatOpenAI(model_name=model_name)
    agent = create_agent(model, tools=[get_current_weather, get_current_time])
    messages = {
        "messages": [
            {"role": "user", "content": "现在几点了？北京和杭州现在是什么天气？"},
        ]
    }
    response = agent.invoke(messages)
    print(response)


if __name__ == '__main__':
    # print(get_current_weather.name)
    # print(get_current_weather.description)
    # print(get_current_weather.args)
    
    # get_weather_tool = GetCurrentWeatherTool()
    # print(get_weather_tool.name)
    # print(get_weather_tool.description)
    # print(get_weather_tool.args) 
    
    # print(get_current_weather_tool.name)
    # print(get_current_weather_tool.description)
    # print(get_current_weather_tool.args)
    
    # print(convert_to_openai_tool(get_current_weather))
    # test_function_call(False)
    test_agent_call()
