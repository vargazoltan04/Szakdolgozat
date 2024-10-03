import cv2
import matplotlib.pyplot as plt
import scipy.signal
import utility
import numpy as np

#Kép beolvasás
image_color = cv2.imread("../images/binarized_image/test_binary.png")
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

height, width = image.shape

#vízszintes vetület
#végigiterálok a sorokon, és kiszámolom hogy soronként hány fekete pixel van
horizontal_projection = []
for row in range(height):
    row_data = image[row, :]

    count_black_pixels = 0
    for pixel in row_data:
        if pixel == 0:
            count_black_pixels += 1

    horizontal_projection.append(count_black_pixels)

#lokális minimumpontok megkeresése (sorközök), és piros vonalak behúzása a képre
#Ez nem fontos, de látszik a képen, ha hiba van rajta, mivel vizuális. 
min_points = utility.find_local_minimum_points(horizontal_projection)
for i in range(len(min_points) - 1):
    cv2.line(image_color, (0, min_points[i]), (width, min_points[i]), (0,0,255), 1)

#Többi sor, illetve fehér sorok törlése
#Ez azért van külön az első sortól, mert mivel a 
for i in range(1, len(min_points)):
    row_image = image[min_points[i-1]:min_points[i], :]
    non_white_rows = np.any(row_image < 255, axis=1)
    row_image_trimmed = row_image[non_white_rows]
    cv2.imwrite(f"../images/rows/row{i}.png", row_image_trimmed)



cv2.imshow("rows", image_color)

plt.plot(horizontal_projection)
plt.title("Horizontal projection")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

