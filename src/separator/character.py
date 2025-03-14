import util.util as util
import numpy as np
import cv2

class character:
    def __init__(self, char, space_after, row_num, char_num):
        inverted = cv2.bitwise_not(char) ##Megkeresi a betűnek a befoglaló téglalapját 
        coords = cv2.findNonZero(inverted)
        x, y, w, h = cv2.boundingRect(coords)

        print(f"row_num: {row_num} \t| char_num: {char_num} \t| char: {char.shape} \t| inverted: {inverted.shape}")
        self.char = char[y:y+h, x:x+w]  #kivágja a betűt, csak a lényeg marad meg
        self.inverted = cv2.bitwise_not(self.char) #Újra invertálja, hogy a méretarányok megmaradjanak

        self.row_num = row_num
        self.char_num = char_num
        self.space_after = space_after

    def save_letter(self, path):
        path = f"{path}/row{self.row_num}_letter{self.char_num}.png"
        util.create_path(path)
        if self.char.shape[0] == 0 or self.char.shape[1] == 0:
            return

        cv2.imwrite(path, self.char)

    def is_correct_letter(self):
        horizontal_projection_letter = util.horizontal_projection(self.char)
        horizontal_projection_letter = np.array(horizontal_projection_letter).reshape(1, -1).astype(np.uint8)
        vertical_projection_letter = util.vertical_projection(self.char)
        vertical_projection_letter = np.array(vertical_projection_letter).reshape(1, -1).astype(np.uint8)
        
        num_labels_horizontal, labels_horizontal = cv2.connectedComponents(horizontal_projection_letter)
        num_labels_vertical, labels_vertical = cv2.connectedComponents(vertical_projection_letter)
        
        num_labels, labels = cv2.connectedComponents(self.inverted)

        if num_labels_vertical == 2 and num_labels_horizontal == 2 and num_labels >= 3:
            return False
        
        return True
    
    def separate_incorrect_letters(self):
        #print(self.row_num, self.char_num)
        output = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.inverted, connectivity=8)

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            #print(f"x: {x} y: {y} w: {w} h: {h}")
            #cv2.rectangle(self.char, (x, y), (x+w, y+h), (255, 0, 0), 1)

            temp_im = self.char[y:y+h, x:x+w]

            temp_char = character(temp_im, False, self.row_num, self.char_num + (i - 1))
            output.append(temp_char)
        

        if self.char.dtype != np.uint8:
            self.char = np.clip(self.char, 0, 255).astype(np.uint8)
        #print(f"{self.row_num} : {self.char_num}")
        return output


