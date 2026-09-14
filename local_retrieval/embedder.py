import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


QWEN_NAME = 'qwen3-0.6b'
EMBEDDER_NAMES = (QWEN_NAME,)
QWEN_QUERY_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'


def last_token_pool(last_hidden_state, attention_mask):
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


class Qwen3Embedder:
    name = QWEN_NAME
    model_name = 'Qwen/Qwen3-Embedding-0.6B'
    pooling = 'last_token'
    query_instruction = QWEN_QUERY_INSTRUCTION
    max_length = 512

    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or self.model_name
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
        model_kwargs = {'dtype': torch.bfloat16} if self.device.type == 'cuda' else {}
        self.model = AutoModel.from_pretrained(self.model_path, **model_kwargs).to(self.device)
        self.model.eval()
        self.dimension = self.model.config.hidden_size
        print(f'embedder: {self.name}, source: {self.model_path}, device: {self.device}')

    @classmethod
    def format_query(cls, query):
        return f'Instruct: {cls.query_instruction}\nQuery:{query}'

    def _encode(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        return F.normalize(embeddings.float(), p=2, dim=1)

    def encode_query(self, query):
        return self.encode_queries([query])

    def encode_queries(self, queries):
        return self._encode([self.format_query(query) for query in queries])

    def encode_documents(self, texts):
        return self._encode(texts)


def create_embedder(name, model_path=None, device=None):
    if name == QWEN_NAME:
        return Qwen3Embedder(model_path=model_path, device=device)
    raise ValueError(f'Unknown embedder: {name}. Choose from {EMBEDDER_NAMES}')
