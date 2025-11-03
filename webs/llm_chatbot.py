import torch
import gradio as gr
from server import get_chat_api, ChatModelType

models = [
    get_chat_api(ChatModelType.DOUBAO_1_5_PRO_32K),
    get_chat_api(ChatModelType.QWEN2_5_7B_INSTRUCT)
]

def predict(query, chatbot_1, chatbot_2, task_history):
    print(f"User: {query}")    
    chatbots = [chatbot_1, chatbot_2]
    num = len(chatbots)
    for i in range(num):
        chatbots[i].append((query, ""))
    
    for i in range(len(models)):
        api = models[i]
        try:
            response = api.chat(query)
            chatbots[i][-1] = (query, response)
        except:
            chatbots[i][-1] = (query, '[warning]: something is wrong in the server ...')
            print('==> Error happened...', i)
            response = 'error'
       
        yield chatbots
        print(f"{i} Assistant: {response}")
        
    task_history.append((query, response))
    return chatbots
    

def reset_user_input():
    return gr.update(value="")


def garbage_collect():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        

def reset_state(chatbot_1, chatbot_2, task_history):
    task_history.clear()
    chatbots = [chatbot_1, chatbot_2]
    for chatbot in chatbots:
        chatbot.clear()
    garbage_collect()
    return chatbots


def regenerate(chatbot_1, chatbot_2, task_history):
    chatbots = [chatbot_1, chatbot_2]
    if not task_history:
        return chatbots
    item = task_history.pop(-1)
    for chatbot in chatbots:
        chatbot.pop(-1)
    return predict(
        item[0], chatbot_1, chatbot_2, task_history
    )


with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>LLM-Test ChatBot</center>""")
    task_history = gr.State([])

    with gr.Row():
        chatbots = list()
        with gr.Column(scale=1):
            chatbot_1 = gr.Chatbot(label=ChatModelType.DOUBAO_1_5_PRO_32K.value, elem_classes="control-height")
            chatbots.append(chatbot_1)
        with gr.Column(scale=1):
            chatbot_2 = gr.Chatbot(label=ChatModelType.QWEN2_5_7B_INSTRUCT.value, elem_classes="control-height")
            chatbots.append(chatbot_2)
    
    with gr.Row():
        query = gr.Textbox(lines=4, label='Input')
    
    with gr.Row():
        empty_btn = gr.Button("🧹 Clear History (清除历史)")
        submit_btn = gr.Button("🚀 Submit (发送)")
        regen_btn = gr.Button("🤔️ Regenerate (重试)")
    
    submit_btn.click(
        predict, 
        [query, chatbot_1, chatbot_2, task_history], 
        chatbots, 
        show_progress=True
    )
    submit_btn.click(reset_user_input, [], [query])
    empty_btn.click(
        reset_state, 
        [chatbot_1, chatbot_2, task_history], 
        outputs=chatbots, 
        show_progress=True
    )
    regen_btn.click(
        regenerate, 
        [chatbot_1, chatbot_2, task_history], 
        chatbots, 
        show_progress=True
    )

    gr.Markdown("""<font size=2> This is a test for small models (Doubao-1.5-Pro-32K, Qwen2.5-7B-Instruct) which is developed by CX @ByteDance. """)

demo.queue().launch(
    share=False,
    server_port=7788,
    server_name="0.0.0.0",
)
