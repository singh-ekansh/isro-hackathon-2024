import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('multiple_pic.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('1_pic.png', 0)

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
plt.imshow(res, cmap='gray')
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + 5, top_left[1] + 5)
cv2.rectangle(img_rgb, top_left, bottom_right, 255, 2)

cv2.imshow('Matched image', img_gray)
cv2.waitKey()
cv2.destroyAllWindows()