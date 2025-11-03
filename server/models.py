from .chat import GeneralAPI
from .volce import ArkAPI
from .embedding import EmbeddingAPI
from config import access_config
from enum import Enum


class ChatModelType(Enum):
    QWEN_MAX = "qwen3-max"
    KIMI_K2 = "kimi-k2"
    DEEPSEEKV3 = "deepseekv3"
    DOUBAO_1_5_PRO_32K = "doubao-1.5-pro-32k"
    DOUBAO_1_5_PRO_32K_EOU = "doubao-1.5-pro-32k-eou"
    DOUBAO_1_5_VISION_PRO_32K = "doubao-1.5-vision-pro-32k"
    QWEN2_5_VL_32B_INSTRUCT = "qwen2.5-vl-32b-instruct"
    TEN_TURN_DETECT = "turn-detect"
    EOU_GEN_200M = "eou-gen-200M"
    EOU_GEN_0_5B = "eou-gen-0.5B"
    EOU_GEN_0_5B_LATEST = "eou-gen-0.5B-latest"
    QWEN2_5_7B_INSTRUCT = "qwen2.5-7b"
    DOUBAO_SEED_1_6 = "doubao-seed-1.6"
    MINICPM4_8B = "minicpm4-8b"
    EOU_GEN_7B = "eou-gen-7B"


class EmbeddingModelType(Enum):
    EOU_ONLINE = "eou-online"
    EOU_LOCAL = "eou-local"


def get_chat_api(model_type: ChatModelType):
    config = access_config[model_type.value]
    api = GeneralAPI(**config)
    return api


def get_embedding_api(model_type: EmbeddingModelType):
    config = access_config[model_type.value]
    api = EmbeddingAPI(**config)
    return api


def get_ark_api(model_type: ChatModelType):
    config = access_config[model_type.value]
    return ArkAPI(config['model_name'], config['api_key'])
