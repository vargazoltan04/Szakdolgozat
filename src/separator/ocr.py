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
            row_image_inverted_bin = cv2.bitwise_not(row_image_bin)

            letter_pixels = cv2.findNonZero(row_image_inverted_bin)
            x, y, w, h = cv2.boundingRect(letter_pixels)
            row_image_trimmed = row_image_bin[y:y+h, x:x+w]

            row_im: row = row.row(row_image_trimmed, 0, i - 1)
            self.rows.append(row_im)

        self.calculate_spaces_length()
        return self
    
    def calculate_spaces_length(self):
        sum = 0
        count = 0
        for r in self.rows:
            image = r.row
            projection = util.vertical_projection(image)

            sum_in_row = 0
            for i in projection:
                if i == 0:
                    sum_in_row += 1

                if i != 0 and sum_in_row > 0:
                    sum += sum_in_row
                    sum_in_row = 0
                    count += 1

        avg = (sum // count) * 1.4
        for r in self.rows:
            r.avg = avg 



    def letter_segmentation(self):
        for i in range(len(self.rows)):
            self.rows[i].letter_segmentation()

        return self

