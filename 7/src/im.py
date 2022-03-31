import cv2
org = cv2.imread('images/Original.bmp')
ref = cv2.imread('images/Reference.bmp')
attack1_1 = cv2.imread('images/Attack 1/1.bmp')
attack1_2 = cv2.imread('images/Attack 1/2.bmp')
attack1_3 = cv2.imread('images/Attack 1/3.bmp')
attack1_4 = cv2.imread('images/Attack 1/4.bmp')

attack2_1 = cv2.imread('images/Attack 2/1.bmp')
attack2_2 = cv2.imread('images/Attack 2/2.bmp')
attack2_3 = cv2.imread('images/Attack 2/3.bmp')
attack2_4 = cv2.imread('images/Attack 2/4.bmp')

out_1 = cv2.imread('output/7_1_1/1.bmp')
out_2 = cv2.imread('output/7_1_1/2.bmp')
out_3 = cv2.imread('output/7_1_1/3.bmp')
out_4 = cv2.imread('output/7_1_1/4.bmp')


cv2.imwrite('output/im/attack1_1.png', attack1_1)
cv2.imwrite('output/im/attack1_2.png', attack1_2)
cv2.imwrite('output/im/attack1_3.png', attack1_3)
cv2.imwrite('output/im/attack1_4.png', attack1_4)

cv2.imwrite('output/im/attack2_1.png', attack2_1)
cv2.imwrite('output/im/attack2_2.png', attack2_2)
cv2.imwrite('output/im/attack2_3.png', attack2_3)
cv2.imwrite('output/im/attack2_4.png', attack2_4)

cv2.imwrite('output/im/ref.png', ref)
cv2.imwrite('output/im/org.png', org)

cv2.imwrite('output/im/1.png', out_1)
cv2.imwrite('output/im/2.png', out_2)
cv2.imwrite('output/im/3.png', out_3)
cv2.imwrite('output/im/4.png', out_4)