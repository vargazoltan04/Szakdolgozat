from separator import row
import util.util as util
import cv2
import numpy as np

class ocr:
    def __init__(self, image_path, save_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.bin_image = None
        self.save_path = save_path
        self.rows: row = []

    def show(self, windowName):
        cv2.imshow(windowName, self.image)
        return self
    
    def saveim(self, filename):
        cv2.imwrite(self.save_path + filename, self.image)
        return self
    
    def saveim_bin(self, filename):
        cv2.imwrite(self.save_path + filename, self.bin_image)
        return self
    
    def save_rows(self, filename):
        for i in range(len(self.rows)):
            self.rows[i].save_row(self.save_path + filename)

        return self
    
    def save_letters(self, filename):
        for i in range(len(self.rows)):
            self.rows[i].save_letters(filename)
        
        return self
    
    def binarize(self, threshold=128):
        _, self.bin_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)

        return self
    
    def delete_small_components(self, min_size):
        image_inverted = cv2.bitwise_not(self.bin_image)
        num_labels, labels = cv2.connectedComponents(image_inverted)
        sizes = np.bincount(labels.ravel())
        for label in range(0, num_labels):
            if sizes[label] < min_size:
                labels[labels == label] = 0

        self.image = np.where(labels > 0, self.image, self.image).astype(np.uint8)
        return self
    
    #Képen a sorok szegmentálása
    #Úgy éri el, hogy a vízszintes vetületen ahol van 0 érték, ott van sortörés
    def row_segmentation(self):
        #vízszintes vetület
        #végigiterálok a sorokon, és kiszámolom hogy soronként hány fekete pixel van
        horizontal_projection = util.horizontal_projection(self.bin_image)

        #lokális minimumpontok megkeresése (sorközök), és piros vonalak behúzása a képre
        #Ez nem fontos, de látszik a képen, ha hiba van rajta, mivel vizuális. 
        min_points = util.find_local_minimum_points(horizontal_projection)

        min_points.append(len(horizontal_projection))
        #Többi sor, illetve fehér sorok törlése
        for i in range(1, len(min_points)):
            row_image_bin = self.bin_image[min_points[i-1]:min_points[i], :]
            row_image = self.image[min_points[i-1]:min_points[i], :]

            non_white_rows = np.any(row_image_bin < 255, axis=1)
            row_image_trimmed = row_image[non_white_rows]
            row_im: row = row.row(row_image_trimmed, i - 1)
            self.rows.append(row_im)

        return self
    
    def letter_segmentation(self):
        for i in range(len(self.rows)):
            self.rows[i].letter_segmentation()

        return self

