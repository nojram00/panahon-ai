import datetime
from typing import TypedDict, Union, override
from bson import ObjectId
from simple_mongo_2 import SimpleMongo
from dotenv import load_dotenv
from dataclasses import dataclass, asdict, field
from ..embedding import Embed

load_dotenv()

class SimilarityResult(TypedDict):
    document : 'ForecastData'
    similarity_score : float

@dataclass
class ForecastData():
    
    caused_by : str
    impacts : str
    place : str
    weather_condition : str
    _id : ObjectId = field(default_factory=ObjectId)
    date : datetime.datetime = field(default_factory=lambda: datetime.datetime.now())
    
    @property
    def json(self) -> dict[str, Union[str, list[float], list[int]]]:
        return {
            "caused_by" : self.caused_by,
            "impacts" : self.impacts,
            "weather_condition" : self.weather_condition,
            "place" : self.place,
            "date" : self.date.isoformat()
        }
        
    @property
    def markdown(self):
        return f"""
# Weather Condition: {self.date.isoformat()}
{self.weather_condition}

# Caused By
{self.caused_by}

# Affected Places
{self.place}

# Impacts
{self.impacts}
    """
        
    def asmongo(self):
        return asdict(self)
    
    @property
    def embedded(self):
        return Embed.documents([self.markdown])
    
    @staticmethod
    def add_search_index():
        with SimpleMongo() as client:
            pass
            
    
    def save(self):
        data = self.json
        data['embedded'] = self.embedded
        with SimpleMongo() as client:
            collection = client.database('pagasa-weather-forecast').forecast_conditions
            return collection.insert_one(data)

    def update(self, **kwargs):
        with SimpleMongo() as client:
            collection = client.database('pagasa-weather-forecast').forecast_conditions
            return collection.update_one({
                "_id" : self._id
                }, {
                "$set": kwargs
            })
    
    @classmethod 
    def update_by_id(cls, id : ObjectId, **kwargs):
        pass
    
    @classmethod
    def id_find(cls, id : str, **kwargs):
        with SimpleMongo() as client:
            collection = client.database('pagasa-weather-forecast').forecast_conditions
            results = collection.find_one({
                "_id" : ObjectId(id)
            })
            
            if results:
                return cls(
                    caused_by = results['caused_by'],
                    impacts = results['impacts'],
                    place = results['place'],
                    weather_condition = results['place'],
                    _id = results['_id'],
                    date = results['data']
                )
            else:
             raise
                    
    @classmethod
    def all(cls):
        with SimpleMongo() as client:
            collection = client.database('pagasa-weather-forecast').forecast_conditions
            results = collection.find({})
            
            data = [cls(
                _id=result['_id'],
                date=result['date'],
                weather_condition=result['weather_condition'],
                caused_by=result['caused_by'],
                impacts=result['impacts'],
                place=result['place']
                ) for result in results]
            
            return data
    
    @classmethod
    def all_embeddings(cls):
        return [doc.embedded for doc in cls.all()]
    
    @classmethod
    def fetch_similarities(cls, query : str, limit : int = 5):
        import numpy as np
        
        documents = cls.all()
        
        if not documents:
            return []
        
        embedded_docs = np.array([doc.embedded for doc in documents])
        embedded_query = np.array(Embed.query(query)).squeeze()
        
        if embedded_docs.ndim > 2:
            embedded_docs = embedded_docs.squeeze()
        
        similarities = np.dot(embedded_docs, embedded_query)
        ranked_indeces = np.argsort(-similarities)
        
        results : list[SimilarityResult] = []
        
        for idx in ranked_indeces[:limit]:
            doc = documents[idx]
            results.append({
                "document": doc,
                "similarity_score" : float(similarities[idx])
            })
        
        return results
    
        
        