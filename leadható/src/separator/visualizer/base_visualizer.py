from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    def __init__(self, save_path, debug):
        self.save_path = save_path
        self.debug = debug

    @abstractmethod
    def visualize_confusion_matrix(self, true, prediction):
        pass