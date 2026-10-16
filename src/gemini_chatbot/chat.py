from google import genai
from dotenv import load_dotenv

load_dotenv()

class Chat:
    
    _client = genai.Client()
    
    @classmethod
    def chat(cls, query : str):
        interaction = cls._client.interactions.create(
                model='gemini-3.1-flash-lite',
                input=query
            )
        
        return interaction.output_text
    
    @classmethod
    def chat_weather_queries(cls, prompt : str):
        interaction = cls._client.interactions.create(
            model='gemini-3.1-flash-lite',
            system_instruction="You are a weather assistant answering queries based solely on official PAGASA reports provided in the context.",
            input=prompt
        )
        
        return interaction.output_text
    
    @classmethod
    def stream(cls, query : str):
        _stream = cls._client.interactions.create(
            model='gemini-3.1-flash-lite',
            input=query,
            stream=True
        )
        
        for event in _stream:
            yield event

