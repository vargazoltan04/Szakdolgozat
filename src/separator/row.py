from separator import character
import util.util as util
import numpy as np
import cv2

class row:
    def __init__(self, row, bin_row, avg, row_num):
        self.row = row
        self.bin_row = bin_row
        self.row_num = row_num
        self.letters: character = []

        self.avg = avg
        self.offset = 0

    def save_row(self, filename):
        cv2.imwrite(filename + f"{self.row_num}.png", self.bin_row)
    
    def save_letters(self, filename):
        for i in range(len(self.letters)):
            self.letters[i].save_letter(filename)

    def letter_segmentation(self):
        vertical_projection = util.vertical_projection(self.bin_row)
        min_points = util.find_local_minimum_points(vertical_projection)

        min_points.append(len(vertical_projection))
        min_points.insert(0, 1)
        offset = 0
        for i in range(1, len(min_points)):
            start, end = min_points[i - 1], min_points[i]

            if start >= end:  # Ha az intervallum érvénytelen, ugorjuk át
                continue

            letter = self.row[:, start:end]  # Kivágja a betű képét
            letter_bin = self.bin_row[:, start:end]

            non_white_cols = np.any(letter_bin < 255, axis=0)   #Megkeresi az üres oszlopokat

            if not np.any(non_white_cols):  # Ha minden oszlop fehér, lépjünk tovább
                continue

            letter_trimmed = letter[:, non_white_cols]  #Levágja az üres oszlopokat
            letter_trimmed_bin = letter_bin[:, non_white_cols]
  
            white_cols_count = len(non_white_cols[non_white_cols == False]) ## A szóközökhoz kell. Ha nagyobb mint az átlag, van szóköz
            char: character = character.character(letter_trimmed, letter_trimmed_bin, white_cols_count > self.avg, self.row_num, offset + i - 1)
            #print("dsadas", char.row_num, char.char_num, char.char.shape)

            if char.char.shape[0] == 0 or char.char.shape[1] == 0:
                continue

            if char.bin_char.shape[0] == 0 or char.bin_char.shape[1] == 0:
                continue

            if char.is_correct_letter():
                self.letters.append(char)
            else:
                sep_letters = char.separate_incorrect_letters()

                #for letter in sep_letters:
                #    letter.resize()
                
                self.letters += sep_letters
                offset += len(sep_letters) - 1


        return self
    
    def correct_letter_mistake(self):
        pass
