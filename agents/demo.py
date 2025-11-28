
import json
from server import get_chat_api, ChatModelType
from .functions import tools, get_current_weather, get_current_time


def test_function_call():
    server = get_chat_api(ChatModelType.DOUBAO_1_5_PRO_32K)
    messages = [{"role": "user", "content": "广东今天是晴天还是阴天？现在几点了？"}]
    response = server.generate_with_function_call(messages, tools)
    response = response.model_dump()
    outputs = response['message']
    print('query: ', messages[0]['content'])
    
    step = 1
    print(f"\nstep {step}: assistant response = {outputs}\n")
    if  outputs['content'] is None:
        outputs['content'] = ""    
    messages.append(outputs)
    
    if outputs['tool_calls'] == None:  
        print(f"No need to call tools：{outputs['content']}")
        return
    
    while outputs['tool_calls'] != None:
        tool_id = outputs['tool_calls'][0]['id']
        if outputs['tool_calls'][0]['function']['name'] == 'get_current_weather': 
            tool_info = {"name": "get_current_weather", "role":"tool", "tool_call_id": tool_id}
            arguments = outputs['tool_calls'][0]['function']['arguments']
            arguments = json.loads(arguments)
           
            if 'properties' in arguments:
                location = arguments['properties']['location']
            else:
                location = arguments['location']
                
            tool_info['content'] = get_current_weather(location)
            
        elif outputs['tool_calls'][0]['function']['name'] == 'get_current_time':
            tool_info = {"name": "get_current_time", "role":"tool", "tool_call_id": tool_id}
            tool_info['content'] = get_current_time()
        else:
            raise ValueError(f"Unknown function name: {outputs['tool_calls'][0]['function']['name']}")
        
        print(f"tool outputs: {tool_info['content']}\n")
        messages.append(tool_info)
        
        response = server.generate_with_function_call(messages, tools)
        outputs = response.model_dump()['message']
        if  outputs['content'] is None:
            outputs['content'] = ""
            
        messages.append(outputs)
        step += 1
        print(f"step {step}: assistant response = {outputs}\n")
        
    print(f"Final outputs: {outputs['content']}")


if __name__ == '__main__':
    test_function_call()
