from separator import character
import util.util as util
import numpy as np
import cv2

class row:
    def __init__(self, row, row_num):
        self.row = row
        _, self.bin_row = cv2.threshold(self.row, 128, 255, cv2.THRESH_BINARY)
        self.row_num = row_num
        self.letters: character = []
        self.offset = 0

    def save_row(self, filename):
        cv2.imwrite(filename + f"{self.row_num}.png", self.row)
    
    def save_letters(self, filename):
        for i in range(len(self.letters)):
            self.letters[i].save_letter(filename)

    def letter_segmentation(self):
        vertical_projection = util.vertical_projection(self.bin_row)
        min_points = util.find_local_minimum_points(vertical_projection)

        offset = 0
        for i in range(1, len(min_points)):
            letter_bin = self.bin_row[:, min_points[i-1] + 1:min_points[i] + 1]
            letter = self.row[:, min_points[i-1] + 1:min_points[i] + 1]
            non_white_cols = np.any(letter_bin < 255, axis=0)
            letter_trimmed = letter[:, non_white_cols]
            char: character = character.character(letter_trimmed, self.row_num, offset + i - 1)

            if char.is_correct_letter():
                char.resize()
                self.letters.append(char)
            else:
                sep_letters = char.separate_incorrect_letters()
                for letter in sep_letters:
                    letter.resize()
                self.letters += sep_letters
                offset += len(sep_letters) - 1


                #self.letters.extend(sep_letters)

            
                

        return self
    
    def correct_letter_mistake(self):
        pass
