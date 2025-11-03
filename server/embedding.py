from openai import OpenAI


class EmbeddingAPI(object):
    def __init__(self, api_key, api_base, model_name):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def predict(self, text):
        resp = self.client.embeddings.create(
            model= self.model_name,
            input=[text],
            encoding_format="float"
        )
        outputs = resp.data[0].embedding
        return outputs
