import time
from server import get_chat_api, ChatModelType, get_ark_api
from eou.interfaces import gpt_predictor


def test_case_001():
    server = get_chat_api(ChatModelType.DOUBAO_1_5_PRO_32K)
    response = server.chat("你是谁？")
    print(response)
    

def test_case_002():
    server = get_chat_api(ChatModelType.QWEN_MAX)
    response = server.chat("你是谁？")
    print(response)
    
    
def test_case_003():
    server = get_chat_api(ChatModelType.KIMI_K2)
    response = server.chat("你是谁？")
    print(response)


def test_case_004():
    server = get_chat_api(ChatModelType.DEEPSEEKV3)
    response = server.chat("你是谁？")
    print(response)


def test_case_005():
    server = get_chat_api(ChatModelType.EOU_GEN_0_5B_LATEST)
    texts = [
        "你是谁？",
        "你叫什么名字?",
        "你是一个中文模型吗？",
        "你是由谁开发的？",
    ]
    for text in texts:
        response = server.chat(text, system_message='')
        print(response)
   
   
def test_case_006():
    server = get_chat_api(ChatModelType.EOU_GEN_0_5B_LATEST)
    messages = [
        {"role": "user", "content": "你是谁？"},
        {"role": "assistant", "content": "我是一个大模型，我的名字叫火山引擎。"},
        {"role": "user", "content": "你叫"},
    ]
    
    response = server.chat_with_messages(messages)
    print(response)


def test_case_007():
    server = get_chat_api(ChatModelType.QWEN2_5_7B_INSTRUCT)
    response = server.chat("你是谁？")
    print(response)
   
   
def test_case_008():
    server = get_chat_api(ChatModelType.DOUBAO_SEED_1_6)
    response = server.chat("你是谁？")
    print(response)
   

def test_case_009():
    models = [
        ChatModelType.DOUBAO_1_5_PRO_32K,
        ChatModelType.QWEN2_5_7B_INSTRUCT,
        ChatModelType.DEEPSEEKV3,
        ChatModelType.KIMI_K2,
        ChatModelType.QWEN_MAX,
        ChatModelType.DOUBAO_SEED_1_6,
        ChatModelType.MINICPM4_8B
    ]
    for model in models:
        if model == ChatModelType.DOUBAO_SEED_1_6:
            server = gpt_predictor(model, use_ark=True, thinking='disabled')
        else:
            server = gpt_predictor(model)
        start = time.time()
        response = server.predict("你是谁？")
        end = time.time()
        print(response)
        print(f'{model.value} time-cost: ', end-start)
        
    server = get_chat_api(ChatModelType.EOU_GEN_0_5B_LATEST)
    start = time.time()
    response = server.chat("你是谁?", max_tokens=1, temperature=0.1, top_p=0.1)
    end = time.time()
    print(response)
    print(f'{ChatModelType.EOU_GEN_0_5B_LATEST.value} time-cost: ', end-start)


def test_case_010():
    server = get_ark_api(
        ChatModelType.DOUBAO_SEED_1_6
    )
    response = server.chat("你是谁？", thinking='disabled')
    print(response)


def test_case_011():
    server = get_chat_api(ChatModelType.MINICPM4_8B)
    response = server.chat("你是谁？")
    print(response)


def test_case_012():
    server = get_chat_api(ChatModelType.DOUBAO_1_5_PRO_32K_EOU)
    system_message = "你是一个语义完整性判断专家，能结合上下文判断语句或者对话内容的完整性。请对用户输入的文本内容或者对话消息进行语义完整性判断，如果内容语义完整请输出finished，不完整请输出unfinished。"

    response = server.chat("你是谁？", system_message=system_message, max_tokens=1, temperature=0.1, top_p=0.1)
    print(response)
    
    response = server.chat("一帆真是一个", system_message=system_message, max_tokens=2, temperature=0.1, top_p=0.1)
    print(response)
    
    response = server.chat("你是谁？", max_tokens=512, temperature=0.7, top_p=0.1)
    print(response)
    
    response = server.chat("你叫什么名字？", max_tokens=512, temperature=0.7, top_p=0.1)
    print(response)


def test_case_013():
    # server = get_chat_api(ChatModelType.DOUBAO_1_5_VISION_PRO_32K)
    server = get_chat_api(ChatModelType.DOUBAO_SEED_1_6)
    image_file = "paper/datas/images/loss.png"
    response = server.chat_with_image("这张图片的标题是什么？", img=image_file)
    print(response)


def test_case_014():
    server = get_chat_api(ChatModelType.DOUBAO_1_5_PRO_32K)
    response = server.chat("你是谁？", stream=True)
    for chunk in response:
        print(chunk)
        print(chunk.choices[0].delta.content or "", end="")
    print()


def test_case_015():
    server = get_chat_api(ChatModelType.QWEN3_0_6B)
    response = server.chat_with_thinking("请问3x-12=3, 那么x等于多少？", stream=True, max_tokens=8192, thinking=False)
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")
    print()


def test_case_016():
    server = get_chat_api(ChatModelType.QWEN3_0_6B)
   
    response = server.chat_with_thinking("你是谁？", stream=True, max_tokens=8192, thinking=False)
    start = time.time()
    for chunk in response:
        end = time.time()
        print(f'cost time: {end-start}')
        start = end
        # print(chunk.choices[0].delta.content or "", end="")
    print()


if __name__ == '__main__':
    # test_case_001()
    # test_case_002()
    # test_case_003()
    # test_case_004()
    # test_case_005()
    # test_case_006()
    # test_case_007()
    # test_case_008()
    # test_case_009()
    # test_case_010()
    # test_case_011()
    # test_case_012()
    # test_case_013()
    # test_case_014()
    # test_case_015()
    test_case_016()
