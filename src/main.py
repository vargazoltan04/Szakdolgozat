import preprocessing as pp
import row_segmentation as rs
import column_segmentation as cs
import cv2

image = cv2.imread("../images/input/test01.png")
cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.imshow("input", image)

image_binary = pp.binarize(image)
cv2.imwrite("../images/binarized_image/binary.png", image_binary)

min_size = 10
image_cleaned = pp.delete_small_components(image_binary, min_size)
cv2.imwrite("../images/binarized_image/binary_cleaned.png", image_cleaned)

rows = rs.row_segmentation(image_cleaned)
for i in range(len(rows)):
    cv2.imshow("asd", rows[i])
    cv2.imwrite(f"../images/rows/row{i}.png", rows[i])

letters = []
for i in range(len(rows)):
    letters = letters + cs.letter_segmentation(rows[i])

f = open("../temp/asd.txt", 'w')
for i in range(len(letters)):
    cv2.imwrite(f"../images/letters/letter{i}.png", letters[i])

    f.write(str(i) + ": " + str(cs.is_correct_letter(letters[i])) + "\n")
    #if not cs.is_correct_letter(letters[i]):
    #    print(str(i) + ": Error discovered")
cv2.waitKey(0)