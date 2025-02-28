from abc import ABC, abstractmethod

class BaseRecognizer(ABC):
    @abstractmethod
    def recognize(self, image):
        pass