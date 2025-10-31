
from sympy.logic import true
import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/chenxiang.101/workspace/checkpoints/eou")
    llm_type: Optional[str] = field(default='qwen2')
    attn_implementation: Optional[str] = field(default="eager")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    data_root: str = field(
        default='', metadata={"help": "image root."}
    )
    is_hf_dataset: Optional[bool] = field(default=False)
    data_type: Optional[str] = field(default='normal')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: Optional[bool] = field(default=False)
    label_names: List[str] = field(default_factory=lambda: ["labels"])
    tune_llm: Optional[bool] = field(default=False)
    tune_encoder: Optional[bool] = field(default=False)
    tune_projector: Optional[bool] = field(default=True)
  

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear" # r"model.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None
    
