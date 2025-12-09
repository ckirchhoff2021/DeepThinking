import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class Qwen3Embedding(object):
    def __init__(self, path="/home/chenxiang.101/workspace/checkpoints/Retrieval/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(
            path, model_kwargs={
                "attn_implementation": "flash_attention_2", "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            },
            tokenizer_kwargs={"padding_side": "left"}
        )
    
    def encode(self, texts, prompt_name=None):
        if isinstance(texts, str):
            texts = [texts]
        embedding = self.model.encode(texts, prompt_name=prompt_name)
        return embedding
    
    def text_similarity(self, queries, documents):
        embed1 = self.encode(queries, prompt_name='query')
        embed2 = self.encode(documents, prompt_name='document')
        similarity = self.model.similarity(embed1, embed2)
        return similarity
    
    def embedding_similariy(self, embed1, embed2):
        similarity = self.model.similarity(embed1, embed2)
        return similarity
    
    
class Qwen3EmbeddingV2(object):
    def __init__(self, path="/home/chenxiang.101/workspace/checkpoints/Retrieval/Qwen3-Embedding-0.6B"):
        self.model = AutoModel.from_pretrained(
            path, attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def encode(self, input_texts):
        max_length = 8192
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def similarity(self, queries, documents):
        query_embeddings = self.encode(queries)
        document_embeddings = self.encode(documents)
        scores = query_embeddings @ document_embeddings.T
        return scores
    
    
if __name__ == '__main__':
    model = Qwen3Embedding()
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    
    similarity = model.text_similarity(queries, documents)
    print(similarity)
    
    model = Qwen3EmbeddingV2()
    similarity = model.similarity(queries, documents)
    print(similarity)
    
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [ model.get_detailed_instruct(task, query) for query in queries ]
    similarity = model.similarity(queries, documents)
    print(similarity)
