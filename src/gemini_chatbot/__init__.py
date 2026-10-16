from gemini_chatbot.embedding import Embed

from .chat import Chat
from dotenv import load_dotenv
from .data import ForecastData

load_dotenv()

def main() -> None:
    
    query = "Latest Forecast in Visayas for today?"
    data = ForecastData.fetch_similarities(query)

    formatted_context : list[str] = []
    for doc in data:
        document = doc["document"]
        formatted_context.append(f"""
======================

{document.markdown}

======================        
                      
        """)
        
    prompt = f"""
Use ONLY the context provided below to answer the user's question.

### CONTEXT:
{formatted_context}

### QUESTION:
{query}


    """
    
    result = Chat.chat_weather_queries(prompt)
    
    print(result)
        
