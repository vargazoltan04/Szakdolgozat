from abc import ABC, abstractmethod

class BaseCleaner(ABC):
    def __init__(self, debug):
        self.debug = debug

    @abstractmethod
    def delete_small_components(self, image, size):
        pass