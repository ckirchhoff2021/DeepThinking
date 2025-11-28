import os
import yaml

access_config = {
    "doubao-1.5-pro-32k":{
        "api_key": "xxx",
        "model_name": "xxx",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3"
    },
    "doubao-1.5-pro-32k-eou":{
        "api_key": "xxx",
        "model_name": "xxx",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3"
    },  
    "doubao-1.5-vision-pro-32k":{
        "api_key": "xxx",
        "model_name": "xxx",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3"
    },
    "qwen2.5-vl-32b-instruct":{
        "api_key" : "xxx",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name" : "qwen2.5-vl-32b-instruct"
    },
    "qwen3-max":{
        "api_key" : "xxx",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name" : "qwen3-max" # "qwen-max"
    },
    "kimi-k2":{
        "api_key" : "xxx",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name" : "ep-20251015145009-mprsm"
    },
    "deepseekv3":{
        "api_key" : "xxx",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name" : "ep-20250730192841-mlzqr"
    },
    "eou-online":{
        "api_key" : "xxx",
        "api_base" : "https://ai-gateway.vei.volces.com/v1",
        "model_name" : "semantic-integrity-recognition"    
    },
    "turn-detect":{
        "api_key" :"Empty", 
        "api_base": "http://10.37.9.155:8001/v1", 
        "model_name": 'turn-detect'
    },
    "eou-local":{
        "api_key" :"Empty", 
        "api_base": "http://10.37.9.241:8001/v1", 
        "model_name": 'eou-online'
    },
    "eou-gen-200M":{
        "api_key" :"Empty", 
        "api_base": "http://10.37.9.241:8002/v1", 
        "model_name": 'eou-0702'
    },
    "eou-gen-0.5B":{
        "api_key" : "Empty", 
        "api_base": "http://10.37.9.241:8003/v1", 
        "model_name": 'eou-0.5B'
    },
    "eou-gen-0.5B-latest":{
        "api_key" : "xxx",     
        "api_base":  "https://sd3kt4p66uj313gtp5iv0.apigateway-cn-beijing.volceapi.com/mlp/s-20250917185451-zzxxc/v1" ,                 
        "model_name": "eou-detect-v2"                                           
    },
    "doubao-seed-1.6":{
        "api_key" : "xxx",     
        "api_base":  "https://ark.cn-beijing.volces.com/api/v3" ,                 
        "model_name": "ep-20251015144711-wj2wx"                                           
    },
    "qwen2.5-7b":{
        "api_key" : "Empty",     
        "api_base":  "http://10.37.79.92:8000/v1",                
        "model_name": "Qwen2.5-7B-Instruct"                                           
    },
    "minicpm4-8b":{
        "api_key" : "Empty",     
        "api_base":  "http://10.37.79.92:8008/v1",                
        "model_name": "minicpm4"                                           
    },
    "eou-gen-7B":{
        "api_key" : "Empty",     
        "api_base":  "http://10.37.9.155:8001/v1",                
        "model_name": "eou-7B"                                           
    },
    "qwen3-0.6B":{
        "api_key" : "Empty",     
        "api_base":  "http://10.37.9.241:8100/v1",                
        "model_name": "qwen3-0.6B"                                           
    }
}

tos_config = dict(
    ak="xxxxxx", 
    sk="xxxxxx", 
    endpoint="tos-cn-beijing.volces.com", 
    region="cn-beijing",
    bucket_name="chenxiang"
)


def load_configs(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"The config file '{config_file}' does not exist.")

    # Try to load the YAML file
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)  # Use safe_load for safety
            return config
    except yaml.YAMLError as e:
        raise Exception(
            f"Failed to load YAML configuration from '{config_file}': {e}")
        
tos_config = load_configs("api_config.yaml")["api_config"]["tos"]

# print(tos_config)
