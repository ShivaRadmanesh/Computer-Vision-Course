import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread("localH1_4.jpg")
img2 = cv2.imread("localH1_8.jpg")
img3 = cv2.imread("localH1_16.jpg")
img4 = cv2.imread("localH1_64.jpg")
mix = np.vstack((img1, img2, img3, img4))
mix.astype("uint8")


cv2.imwrite("mix1.jpg", mix)
