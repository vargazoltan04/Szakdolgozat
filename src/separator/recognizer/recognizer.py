from abc import ABC, abstractmethod
from separator.recognizer.base_recognizer import BaseRecognizer
from network.model import VGG16
from torchvision import transforms

import torch
import json
import PIL.Image as Image


class Recognizer(BaseRecognizer):
     def __init__(self, output_num, model_path, mapping_json, debug):
          with open(mapping_json, 'r') as json_file:
               self.index_class_mapping = json.load(json_file)

          self.model = VGG16(output_num) 
          self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
          self.model.eval()

          self.transform = transforms.Compose([
               transforms.Grayscale(num_output_channels=1),
               transforms.Resize((64, 64)),       # Resize images to 64x64
               transforms.ToTensor(),             # Convert images to tensor
          ])

          self.debug = debug

     def recognize(self, letter):
          _, predicted_class = torch.max(self.model(self.transform(Image.fromarray(letter.char)).unsqueeze(0)), 1)
          return self.index_class_mapping[str(predicted_class.item())]