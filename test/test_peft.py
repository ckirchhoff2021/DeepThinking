from models import Qwen2, Gemma
import peft
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_qwen2(prompts):
    ckpt = '/home/chenxiang.101/workspace/checkpoints/Qwen2.5-0.5B-Instruct'
    model = Qwen2(ckpt, device='cuda:0')
    peft_path = '/home/chenxiang.101/workspace/git/outputs/final'
    model.update_peft(peft_path)
    for prompt in prompts:
        response = model.chat(prompt, max_new_tokens=1024)
        print(response)
    
    
def test_gemma3(prompts):
    ckpt = '/home/chenxiang.101/workspace/checkpoints/gemma-3-270m-it'
    model = Gemma(ckpt,  device='cuda:0')
    peft_path = '/home/chenxiang.101/workspace/git/outputs/final'
    model.update_peft(peft_path)
    for prompt in prompts:
        response = model.chat(prompt, max_new_tokens=1024)
        print(response)
    
    
def load_from_pretrained(model_path, device_map, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    return model, tokenizer


def test_peft():      
    path = '/home/chenxiang.101/workspace/checkpoints/Qwen2.5-0.5B-Instruct'
    model, tokenizer = load_from_pretrained(path, "auto")
    modules_to_save = ['embed_tokens']  # add training for embed_tokens
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=r"model.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)",
        lora_dropout=0.1,
        bias="none",
        layers_to_transform=None,
        modules_to_save=modules_to_save,
    )
    
    peft_model = get_peft_model(model, lora_config)
    print(peft_model)
    
    input_ids = tokenizer('你好', return_tensors='pt').input_ids
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
    output = peft_model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    print(output)
    
    
def test_safetensors():
    from safetensors import safe_open
    with safe_open("final/adapter_model.safetensors", framework="pt", device="cpu") as f:
        tensor_names = f.keys()
        print("tensor names:", list(tensor_names))
        for name in tensor_names:
            print(name, f.get_tensor(name).shape)


def test_save_peft():
    path = "/home/chenxiang.101/workspace/checkpoints/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
    adapter_path = "/home/chenxiang.101/workspace/checkpoints/7B-1028-adapter"
    lora_model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = lora_model.merge_and_unload()
    
    message =  [
        {
            "role": "system",
            "content": "你是一个语义完整性判断专家，能结合上下文判断语句或者对话内容的完整性。请对用户输入的文本内容或者对话消息进行语义完整性判断，如果内容语义完整请输出finished，不完整请输出unfinished。"
        },
        {
            "role": "user",
            "content": "曾经有一个人，他的名字叫"
        }
    ]
    
    inputs = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors='pt').to(model.device)
    outputs = merged_model.generate(**inputs, max_new_tokens=1)
    generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
    response = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True)
    print(response)
    
    save_path = "/home/chenxiang.101/workspace/checkpoints/eou-7B"
    merged_model.save_pretrained(save_path)



def test_save_peft_2():
    path = "/home/chenxiang.101/workspace/checkpoints/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
    adapter_path = "/home/chenxiang.101/workspace/checkpoints/intention/intention2/lora/final"
    lora_model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = lora_model.merge_and_unload()
    
    texts = [
        "你爸爸是谁？",
        "安静",
        "我想打断你",
        "今天去公园看",
        "一会去哪里玩啊",
        "等一等",
        "今天天气不错啊"
    ]
    
    for text in texts:
        message =  [
            {
                "role": "system",
                "content": "你是一个语音对话场景的意图识别专家，能结合上下文分析出用户是否有打断对话的意图。若用户有打断意图，输出interrupt，否则输出continue。"
            },
            {
                "role": "user",
                "content": text
            }
        ]
        
        inputs = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs, return_tensors='pt').to(model.device)
        outputs = merged_model.generate(**inputs, max_new_tokens=1)
        generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
        response = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True)
        print(text, " | ", response)
    
    save_path = "/home/chenxiang.101/workspace/checkpoints/intention/intention-0.5B"
    merged_model.save_pretrained(save_path)


if __name__ == '__main__':
    prompts = [
        '你是谁？',
        '你叫什么名字？',
        '请给我讲个故事',
        '请给我讲个笑话',
        '你爸爸是谁？'
    ]
    # test_qwen2(prompts)
    # test_peft()
    # test_gemma3(prompts)
    # test_save_peft()
    test_save_peft_2()
