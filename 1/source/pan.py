import cv2
import numpy as np
from math import floor, ceil

img1 = cv2.imread('image/Car1.jpg', 0)
img2 = cv2.imread('image/Car2.jpg', 0)


r1, c1 = img1.shape
r2, c2 = img2.shape

# # tie point for Car1
# tp11 = (408, 528)
# tp12 = (301, 723)  # pre : 723 -> 302
# tp13 = (452, 623)  
# tp14 = (449, 563)

# # tie points for Car2
# tp21 = (428, 163)
# tp22 = (322, 337)
# tp23 = (472, 202)
# tp24 = (469, 140)

# tie point for Car1
tp11 = (379, 834)
tp12 = (318, 763)  
tp13 = (379, 880)  # 380 ---> cool
tp14 = (451, 564)
# tp14 = (382, 949)

# tie points for Car2
tp21 = (398, 413)
tp22 = (338, 349)
tp23 = (399, 457)  # ----> coold
tp24 = (471, 142)
# tp24 = (403, 520)

ref = np.array(([ tp11[0], tp12[0], tp13[0], tp14[0]], [tp11[1], tp12[1], tp13[1], tp14[1]]))

print(ref)

input_mat = np.array((
                    [tp21[0], tp22[0], tp23[0], tp24[0]], 
                    [tp21[1], tp22[1], tp23[1], tp24[1]],
                    [tp21[0] * tp21[1], tp22[0] * tp22[1], tp23[0] * tp23[1], tp24[0] * tp24[1]],
                    [1, 1, 1, 1]
))

c = np.matmul(ref, np.linalg.inv(input_mat))

res_img = np.zeros(shape=(r1 +r2, c1 + c2))
res_img[0: r1, 0:c1] = img1.astype('uint8')
# print(res_img.shape)
count = 0

for i in range(r2):
    for j in range(c2):
        pmat = np.array(([i, j, i * j, 1]))
        new_cord = np.matmul(c, pmat)
        x = round(new_cord[0])
        y = round(new_cord[1])
        # print(f"x = {x}, y = {y}\n")
        if x < r1 + r2 and x >= 0 and y < c1 + c2 and y >= 0:
            if (res_img[x, y] == 0):
                res_img[x, y] = img2[i, j]
                count += 1

print("count: ", count)
print(res_img.shape)
cv2.imshow('img', res_img.astype('uint8'))
cv2.waitKey(0)

# cv2.ImageRe