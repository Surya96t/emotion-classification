from abc import ABC, abstractmethod
from emotionClassification.utils.config import Config


class BaseModel(ABC):
    """Abstract model class that is inherited to all models"""
    
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
        
    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def build(self):
        pass
    
    @abstractmethod
    def train(self):
        pass    
    
    @abstractmethod
    def evaluate(self):
        pass    
    
    @abstractmethod
    def evaluate_documnet(self):
        pass    