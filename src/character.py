import utility
import numpy as np
import cv2

class character:
    def __init__(self, char, row_num, char_num):
        self.char = char
        self.row_num = row_num
        self.char_num = char_num

    def save_letter(self, filename):
        cv2.imwrite(filename + f"row{self.row_num}_letter{self.char_num}.png", self.char)

    def is_correct_letter(self):
        horizontal_projection_letter = utility.horizontal_projection(self.char)
        horizontal_projection_letter = np.array(horizontal_projection_letter).reshape(1, -1).astype(np.uint8)
        vertical_projection_letter = utility.vertical_projection(self.char)
        vertical_projection_letter = np.array(vertical_projection_letter).reshape(1, -1).astype(np.uint8)
            
        num_labels_horizontal, labels_horizontal = cv2.connectedComponents(horizontal_projection_letter)
        num_labels_vertical, labels_vertical = cv2.connectedComponents(vertical_projection_letter)
        
        letter_inversed = cv2.bitwise_not(self.letter)
        num_labels, labels = cv2.connectedComponents(letter_inversed)

        print(num_labels)
        if num_labels_vertical == 2 and num_labels_horizontal == 2 and num_labels >= 3:
            return False
        
        return True
    
    