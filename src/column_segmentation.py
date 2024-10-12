import cv2
import matplotlib.pyplot as plt
import scipy.signal
import utility
import numpy as np

rowIndex = 6
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

        horizontal_projection_letter = utility.horizontal_projection(letter_trimmed)
        horizontal_projection_letter = np.array(horizontal_projection_letter).reshape(1, -1).astype(np.uint8)
        vertical_projection_letter = utility.vertical_projection(letter_trimmed)
        vertical_projection_letter = np.array(vertical_projection_letter).reshape(1, -1).astype(np.uint8)
         
        num_labels_horizontal, labels_horizontal = cv2.connectedComponents(horizontal_projection_letter)
        num_labels_vertical, labels_vertical = cv2.connectedComponents(vertical_projection_letter)
        

        letter_trimmed_inversed = cv2.bitwise_not(letter_trimmed)
        num_labels, labels = cv2.connectedComponents(letter_trimmed_inversed)

        cv2.imshow(f"{i}", letter_trimmed)
        if num_labels_vertical == 2 and num_labels_horizontal == 2 and num_labels >= 3:
            print(str(i) + ": " + str(True))
        
        cv2.imwrite(f"../images/letters/row{rowIndex}_letter{i}.png", letter_trimmed)
    rowIndex = rowIndex + 1
    break
    #Piros vonalak behúzása azért, hogy vizuálisabb legyen
    #for i in range(len(min_points) - 1):
    #    cv2.line(row_color, (min_points[i], 0), (min_points[i], height), (0,0,255), 1)

    #plt.plot(vertical_projection)
    #plt.title("Horizontal projection")
    #plt.xlabel("Index")
    #plt.ylabel("Value")
    #plt.show()

cv2.waitKey(0)  