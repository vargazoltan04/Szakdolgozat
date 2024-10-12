import cv2
import matplotlib.pyplot as plt
import scipy.signal
import utility
import numpy as np

rowIndex = 1
while True:
    row_color = cv2.imread(f"../images/rows/row{rowIndex}.png")
    print(rowIndex)
    if row_color is None:
        break

    row = cv2.cvtColor(row_color, cv2.COLOR_BGR2GRAY)
    height, width = row.shape

    vertical_projection = utility.vertical_projection(row)
    min_points = utility.find_local_minimum_points(vertical_projection)

    cv2.imshow("row", row_color)
    letters = []
    for i in range(1, len(min_points)):
        letter = row[:, min_points[i-1] + 1:min_points[i] + 1]
        #cv2.imwrite(f"../images/letters/letter{i}.png", letter)
        non_white_cols = np.any(letter < 255, axis=0)
        letter_trimmed = letter[:, non_white_cols]
        cv2.imwrite(f"../images/letters/row{rowIndex}_letter{i}.png", letter_trimmed)

    rowIndex = rowIndex + 1

    #Piros vonalak behúzása azért, hogy vizuálisabb legyen
    #for i in range(len(min_points) - 1):
    #    cv2.line(row_color, (min_points[i], 0), (min_points[i], height), (0,0,255), 1)

    #plt.plot(vertical_projection)
    #plt.title("Horizontal projection")
    #plt.xlabel("Index")
    #plt.ylabel("Value")
    #plt.show()