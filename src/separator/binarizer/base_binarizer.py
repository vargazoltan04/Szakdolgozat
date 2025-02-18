from abc import ABC, abstractmethod

class BaseBinarizer(ABC):
    @abstractmethod
    def binarize(self, image, thresh):
        pass