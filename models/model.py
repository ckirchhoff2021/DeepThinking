from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    def __init__(self):
        self.path = ''
        
    @abstractmethod
    def from_pretrained(self):
        pass

    @abstractmethod
    def chat(self, image, prompt):
        pass

    @abstractmethod
    def generate(self, **kwargs):
        pass
