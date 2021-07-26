import cv2
import numpy as np
import string

size = 26

img = cv2.imread("0data.png", 0)
print(img.shape)
height, width = img.shape
y = int(width/size)
x = int(height/10)

endy = 0
for j in range(0, 26):
    starty = endy
    endy = (j+1)*y
    character = string.ascii_lowercase[j:j+1]
    endx = 0
    for i in range(0, 10):
        startx = endx
        endx = (i+1)*x
        subimg = img[startx:endx, starty:endy]
        name = str(character) + "" + str(i) + ".png"
        cv2.imwrite(name,subimg)

