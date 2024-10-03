import cv2
import matplotlib.pyplot as plt
import scipy.signal
import utility
import numpy as np

row_color = cv2.imread("../images/rows/row1.png")
row = cv2.cvtColor(row_color, cv2.COLOR_BGR2GRAY)
height, width = row.shape

vertical_projection = []
for col in range(width):
    col_data = row[:, col]

    count_black_pixels = 0
    for pixel in col_data:
        if pixel == 0:
            count_black_pixels += 1

    vertical_projection.append(count_black_pixels)

#Piros vonalak behúzása azért, hogy vizuálisabb legyen
min_points = utility.find_local_minimum_points(vertical_projection)
for i in range(len(min_points) - 1):
    cv2.line(row_color, (min_points[i], 0), (min_points[i], height), (0,0,255), 1)

cv2.imshow("row", row_color)
letters = []
for i in range(1, len(min_points)):
    letter = row[:, min_points[i-1] + 1:min_points[i] + 1]
    #cv2.imwrite(f"../images/letters/letter{i}.png", letter)
    non_white_cols = np.any(letter < 255, axis=0)
    letter_trimmed = letter[:, non_white_cols]
    cv2.imwrite(f"../images/letters/letter{i}.png", letter_trimmed)


plt.plot(vertical_projection)
plt.title("Horizontal projection")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()