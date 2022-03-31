import numpy as np
import cv2

img = cv2.imread('image/HE1.jpg',0)
#print img
img_size=img.shape
#print img_size

img_mod = np.zeros_like(img)
window_size = 51

for i in range(0,img_size[0]-window_size):
    for j in range(0,img_size[1]-window_size):
        kernel = img[i:i+window_size,j:j+window_size]
        rank = 0
        for k in range(0,window_size):
            for l in range(0,window_size):
            
                center = int(window_size /2)
                if(kernel[center, center] >= kernel[k, l]):
                    rank = rank + 1
        img_mod[i + center, j + center] = ((rank * 255 )/(window_size* window_size))
im = np.array(img_mod, dtype = np.uint8)
cv2.imwrite('target1_51.jpg',im)