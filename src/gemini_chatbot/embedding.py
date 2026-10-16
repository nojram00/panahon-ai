from voyageai.client import Client
from dotenv import load_dotenv

load_dotenv()

class Embed:
    _client = Client()
    
    @classmethod
    def query(cls, query : str):
        return (
            cls._client
            .embed([query], model="voyage-4", input_type="query")
            .embeddings[0]
            )
        
    @classmethod
    def documents(cls, documents : list[str]):
        return (
            cls._client
            .embed(documents, model="voyage-4", input_type='document')
            .embeddings[0]
        )