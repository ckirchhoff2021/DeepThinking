import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
   
    
class Qwen3Reranker(object):
    def __init__(self, path="/home/chenxiang.101/workspace/checkpoints/Retrieval/Qwen3-Reranker-0.6B"):
        self.model = AutoModelForCausalLM.from_pretrained(
            path, attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map="auto"
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        self.false_token = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token = self.tokenizer.convert_tokens_to_ids("yes")
        
        
    @staticmethod
    def format_instruction(instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
        return output
    
    def process_inputs(self, input_pairs) -> Tensor:
        max_length = 8192
        inputs = self.tokenizer(
            input_pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs):
        token_false_id = self.false_token
        token_true_id = self.true_token

        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def score(self, queries, documents, instruction=None):
        pairs = [self.format_instruction(instruction, query, doc) for query, doc in zip(queries, documents)]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)
        return scores
        
    
if __name__ == '__main__':
    model = Qwen3Reranker()
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "what is your name?"
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "I like apples."
    ]

    scores = model.score(queries, documents, task)
    print(scores)
