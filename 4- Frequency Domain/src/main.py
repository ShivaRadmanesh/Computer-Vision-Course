import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

#--------------------------------------------------------------------------------------------------
# this method cumpute the fourier transform of a filter
# returns the magnitude of the fourier transfomed filter
# m, n are the size of the image that filter is going to apply to
def dft_filter(filterr, m , n):
    pad = padding(filterr, m, n)
    fft = np.fft.fft2(pad)
    shift = np.fft.fftshift(fft)
    mag = np.abs(shift)
    mag = normal(mag)
    return mag

#---------------------------------------------------------------------------------------------------
def normal(img):
    min_val = np.amin(img)
    max_val = np.amax(img)
    output = np.zeros_like(img)
    
    for i in range(np.size(img, 0)):
        for j in range(np.size(img, 1)):
            new_val = math.floor(((img[i, j] - min_val)*255) / (max_val - min_val))
            output[i, j] = new_val
    return output

#---------------------------------------------------------------------------------------------------
def padding(img, p, q):
    my_img = np.zeros((p, q))
    if(np.ndim(img) == 2):
        my_img[0:np.size(img, 0), 0:np.size(img, 1)] = img
    elif(np.ndim(img) == 1):
        my_img[0, 0:np.size(img, 0)] = img
    
    return my_img
#---------------------------------------------------------------------------------------------------
def freq_filter(my_filter, img):
    m, n = img.shape
    p = 2*m
    q = 2*n

    pad_filter = padding(my_filter, p, q)
    pad_img = padding(img, p, q)

    fft_filter = np.fft.fft2(pad_filter)
    fft_img = np.fft.fft2(pad_img)

    shift_filter = np.fft.fftshift(fft_filter)
    shift_img = np.fft.fftshift(fft_img)

    filter_mag = np.abs(shift_filter)
    img_mag = np.abs(shift_img)
    img_phase = shift_img - img_mag
    output = shift_filter * shift_img
    output = output + img_phase
    output = np.fft.ifftshift(output)
    ifft = np.fft.ifft2(output)
    mag = np.abs(ifft)
    new_img = normal(mag)
    unpad = new_img[0:m, 0:n]


    return unpad
#--------------------------------------------------------------------------------------------------
def filter_a(img, t):
    m, n = img.shape
    fft = np.fft.fft2(img)
    mag = np.abs(fft)
    phase = fft - mag

    for k in range(m):
        for l in range(n):
            if((t*n) < k and (t*n) < l and k < ((1-t)*n) and l < ((1-t)*n)):
                fft[k, l] = 0


    output = phase + mag
    # output = np.fft.ifftshift(output)
    output = np.fft.ifft2(fft)
    output = np.abs(output)
    output = normal(output)
    return output
#---------------------------------------------------------------------------------------------------
def filter_b(img, t):
    for k in range(m):
        for l in range(n):
            
            if(0 <= k and 0 <= l and k <= (t*m) and l <=  (t*n)):
                fft[k, l] = 0
            
            elif(0 <= k and k <= (t*m) and ((1 -t)*n) <= l and l <= (n-1)):
                fft[k, l] = 0
            
            
            elif(((1 -t)*m) <= k and k <= (m-1) and 0 <= l and l <= (t*n)):
                fft[k, l] = 0
            
            elif(((1 -t)*m) <= k and l <= (n-1)):
                fft[k, l] = 0
            
    output = np.fft.ifft2(fft)
    output = np.abs(output)
    output = normal(output)
    return output

#--------------------------------------------------------------------------------------------------- 

print("sag")  

barbara = cv2.imread('image/Barbara.bmp', 0)
#4-1-1

#cv2.imwrite("output/4_1/barbara.jpg", barbara)
a = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
b = np.array([[-1, -1, -1], [-1,8, -1], [-1, -1, -1]])
c = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
ah = np.array([1, 2, 1]) / 4
av = np.array([[1], [2], [1]]) / 4


"""
a_magnitude = dft_filter(a, m, n)
b_magnitude = dft_filter(b, m, n)
c_magnitude = dft_filter(c, m, n)
ah_magnitude = dft_filter(ah, m, n)
av_magnitude = dft_filter(av, m, n)

cv2.imwrite("output/4_1/a_magnitude.jpg", a_magnitude)
cv2.imwrite("output/4_1/b_magnitude.jpg", b_magnitude)
cv2.imwrite("output/4_1/c_magnitude.jpg", c_magnitude)
cv2.imwrite("output/4_1/ah_magnitude.jpg", ah_magnitude)
cv2.imwrite("output/4_1/av_magnitude.jpg", av_magnitude)
"""


"""
barbara_a = freq_filter(a, barbara)
barbara_b = freq_filter(b, barbara)
barbara_c = freq_filter(c, barbara)
barbara_ah = freq_filter(ah, barbara)
barbara_av = freq_filter(av, barbara)

cv2.imwrite("output/4_1/barbara_a.jpg", barbara_a)
cv2.imwrite("output/4_1/barbara_b.jpg", barbara_b)
cv2.imwrite("output/4_1/barbara_c.jpg", barbara_c)
cv2.imwrite("output/4_1/barbara_ah.jpg", barbara_ah)
cv2.imwrite("output/4_1/barbara_av.jpg", barbara_av)
"""
lena = cv2.imread('image/Lena.bmp', 0)
baboon = cv2.imread('image/Baboon.bmp', 0)
f16 = cv2.imread('image/F16.bmp', 0)
#4-1-2
"""
cv2.imwrite('output/4_1_2/lena.jpg', lena)
cv2.imwrite('output/4_1_2/baboon.jpg', baboon)
cv2.imwrite('output/4_1_2/f16.jpg', f16)

images = [lena, baboon, f16]
images_name = ["lena", 'baboon', 'f16']

#no shift, no log
for i in range(len(images)):
    fft = np.fft.fft2(images[i])
    mag = np.abs(fft)
    mag = normal(mag)
    cv2.imwrite('output/4_1_2/' + images_name[i] + '_n_n.jpg', mag)

#shift, no log
for i in range(len(images)):
    fft = np.fft.fft2(images[i])
    shift = np.fft.fftshift(fft)
    mag = np.abs(shift)
    mag = normal(mag)
    cv2.imwrite('output/4_1_2/' + images_name[i] + '_s_n.jpg', mag)

#no shift, log
for i in range(len(images)):
    fft = np.fft.fft2(images[i])
    mag = np.log(np.abs(fft))
    mag = normal(mag)
    cv2.imwrite('output/4_1_2/' + images_name[i] + '_n_l.jpg', mag)

#shift , log
for i in range(len(images)):
    fft = np.fft.fft2(images[i])
    shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(shift))
    mag = normal(mag)
    cv2.imwrite('output/4_1_2/' + images_name[i] + '_s_l.jpg', mag)
"""
img = lena
t = 1/8
"""
m, n = img.shape
fft = np.fft.fft2(img)
mag = np.abs(fft)
phase = fft - mag

for k in range(m):
    for l in range(n):
        if((t*n) < k and (t*n) < l and k < ((1-t)*n) and l < ((1-t)*n)):
           fft[k, l] = 0


output = phase + mag
# output = np.fft.ifftshift(output)
output = np.fft.ifft2(fft)
output = np.abs(output)
output = normal(output)
"""




m, n = img.shape
fft = np.fft.fft2(img)
"""

for k in range(m):
    for l in range(n):
        
        if(0 <= k and 0 <= l and k <= (t*m) and l <=  (t*n)):
            fft[k, l] = 0
        
        elif(0 <= k and k <= (t*m) and ((1 -t)*n) <= l and l <= (n-1)):
            fft[k, l] = 0
        
        
        elif(((1 -t)*m) <= k and k <= (m-1) and 0 <= l and l <= (t*n)):
            fft[k, l] = 0
        
        elif(((1 -t)*m) <= k and l <= (n-1)):
            fft[k, l] = 0
        
output = np.fft.ifft2(fft)
output = np.abs(output)
output = normal(output)

# o = filter_a(img, 1/16)
cv2.imwrite("output/4_2_2/i_8.jpg", output)
"""
a4 = filter_a(img, 1/4)
a8 = filter_a(img, 1/8)
cv2.imwrite("output/4_2_2/a4.jpg", a4)
cv2.imwrite("output/4_2_2/a8.jpg", a8)