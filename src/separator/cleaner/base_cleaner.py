from abc import ABC, abstractmethod

class BaseCleaner(ABC):
    @abstractmethod
    def delete_small_components(self, image, size):
        pass