from separator import row
import util.util as util
import cv2
import numpy as np

class ocr:
    def __init__(self, binarizer, image_path, save_path):
        self.binarizer = binarizer
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.bin_image = None
        self.save_path = save_path
        self.rows: row = []

    def run(self):
        self.bin_image = self.binarizer.binarize(self.image, 128)

        return self

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

        #self.bin_image = cv2.adaptiveThreshold(
        #    self.image, 
        #    maxValue=255, 
        #    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #    thresholdType=cv2.THRESH_BINARY,
        #    blockSize=11,
        #    C=10
        #)
        return self
    
    def delete_small_components(self, min_size):
        image_inverted = cv2.bitwise_not(self.bin_image)
        num_labels, labels = cv2.connectedComponents(image_inverted)
        sizes = np.bincount(labels.ravel())
        for label in range(0, num_labels):
            if sizes[label] < min_size:
                labels[labels == label] = 0

        self.image = np.where(labels > 0, self.image, 255).astype(np.uint8)
        self.bin_image = np.where(labels > 0, self.bin_image, 255).astype(np.uint8)
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
            row_image = self.image[min_points[i-1]:min_points[i], :]        #Kivágja a képből a sornak a képét
            row_image_bin = self.bin_image[min_points[i-1]:min_points[i], :]    #Ugyanaz

            row_image_bin_inverted = cv2.bitwise_not(row_image_bin)

            letter_pixels = cv2.findNonZero(row_image_bin_inverted) #Megkeresi a sorban a szöveg befoglaló téglalapját
            x, y, w, h = cv2.boundingRect(letter_pixels)
        
            row_image_trimmed = row_image[y:y+h, x:x+w] #Kivágja a szöveget belőle
            row_image_bin_trimmed = row_image_bin[y:y+h, x:x+w]

            row_im: row = row.row(row_image_trimmed, row_image_bin_trimmed, 0, i - 1)
            self.rows.append(row_im)

        self.calculate_spaces_length()
        return self
    
    def calculate_spaces_length(self):
        sum = 0
        count = 0
        for r in self.rows:
            image = r.bin_row
            projection = util.vertical_projection(image)

            sum_in_row = 0
            for i in range(0, len(projection) - 1):
                if projection[i] == 0:
                    sum_in_row += 1

                if projection[i] == 0 and projection[i+1] > 0:
                    sum += sum_in_row
                    sum_in_row = 0
                    count += 1


        avg = (sum // count) * 1.5
            
        for r in self.rows:
            r.avg = avg 



    def letter_segmentation(self):
        for i in range(len(self.rows)):
            self.rows[i].letter_segmentation()

        return self
    
    def resize(self):
        min_scale = float('inf')
        for row in self.rows:
            for letter in row.letters:
                original_height, original_width = letter.char.shape

                if original_width == 0 or original_height == 0:
                    print(f"Figyelmeztetés: Üres betű észlelve! Kihagyva. ({original_width}x{original_height})")
                    continue  # Kihagyjuk ezt a betűt

                scale = min(45 / original_width, 45 / original_height)
                if scale < min_scale:
                    min_scale = scale

        for row in self.rows:
            for letter in row.letters:
                if letter.char.shape[0] == 0 or letter.char.shape[1] == 0:
                    continue
                
                letter.resize(min_scale)

        return self