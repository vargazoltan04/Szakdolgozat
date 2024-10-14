import cv2
import matplotlib.pyplot as plt
import scipy.signal
import utility
import numpy as np

#Képen a sorok szegmentálása
#Úgy éri el, hogy a vízszintes vetületen ahol van 0 érték, ott van sortörés
def row_segmentation(image):
    rows = []
    #vízszintes vetület
    #végigiterálok a sorokon, és kiszámolom hogy soronként hány fekete pixel van
    horizontal_projection = utility.horizontal_projection(image)

    #lokális minimumpontok megkeresése (sorközök), és piros vonalak behúzása a képre
    #Ez nem fontos, de látszik a képen, ha hiba van rajta, mivel vizuális. 
    min_points = utility.find_local_minimum_points(horizontal_projection)

    min_points.append(len(horizontal_projection))
    #Többi sor, illetve fehér sorok törlése
    for i in range(1, len(min_points)):
        row_image = image[min_points[i-1]:min_points[i], :]
        non_white_rows = np.any(row_image < 255, axis=1)
        row_image_trimmed = row_image[non_white_rows]
        rows.append(row_image_trimmed)

    return rows


#plt.plot(horizontal_projection)
#plt.title("Horizontal projection")
#plt.xlabel("Index")
#plt.ylabel("Value")
#plt.show()

