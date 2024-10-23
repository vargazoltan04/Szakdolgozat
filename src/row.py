import character
import utility
import numpy as np
import cv2

class row:
    def __init__(self, row, row_num):
        self.row = row
        self.row_num = row_num
        self.letters: character = []

    def save_row(self, filename):
        cv2.imwrite(filename + f"{self.row_num}.png", self.row)
    
    def save_letters(self, filename):
        for i in range(len(self.letters)):
            self.letters[i].save_letter(filename)

    def letter_segmentation(self):
        vertical_projection = utility.vertical_projection(self.row)
        min_points = utility.find_local_minimum_points(vertical_projection)

        for i in range(1, len(min_points)):
            letter = self.row[:, min_points[i-1] + 1:min_points[i] + 1]
            non_white_cols = np.any(letter < 255, axis=0)
            letter_trimmed = letter[:, non_white_cols]
            char: character = character.character(letter_trimmed, self.row_num, i - 1)
            self.letters.append(char)

        return self
    
    def correct_letter_mistake(self):
        pass
