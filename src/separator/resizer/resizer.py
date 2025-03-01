from abc import ABC, abstractmethod
from separator.resizer.base_resizer import BaseResizer
import numpy as np
import cv2

class Resizer(BaseResizer):
    def resize(self, char, scale):
          original_height, original_width = char.char.shape

          if original_width == 0 or original_height == 0 and char.row_num == 0:
               print(f"Figyelmeztetés: Üres betű észlelve! Kihagyva. ({original_width}x{original_height})")
               return None # Kihagyjuk ezt a betűt
          
          target_height = 64
          target_width = 64

          new_width = int(original_width * scale)
          new_height = int(original_height * scale)

          if new_height < 15 and new_width < 15:
               scale = min(15 / original_width, 15 / original_height) + 1
               new_height = int(original_height * scale)
               new_width = int(original_width * scale)
          result = np.full((target_width, target_height), 255, dtype=np.uint8)
          result_bin = np.full((target_width, target_height), 255, dtype=np.uint8)
          if new_width >= 15 or new_height >= 15:
               resized_image = cv2.resize(char.char, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
               resized_image_bin = cv2.resize(char.bin_char, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
               x_center = (target_width - resized_image.shape[1]) // 2
               y_center = (target_height - resized_image.shape[0]) // 2
               result[y_center:y_center + resized_image.shape[0], 
                    x_center:x_center + resized_image.shape[1]] = resized_image
               
               result_bin[y_center:y_center + resized_image.shape[0], 
                    x_center:x_center + resized_image.shape[1]] = resized_image_bin

          char.char = result
          char.bin_char = result_bin
          return char
          #_, self.char = cv2.threshold(self.char, 128, 255, cv2.THRESH_BINARY)