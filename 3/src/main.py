import cv2
import numpy as np
import math
from skimage.util import random_noise
#--------------------------------------------------------------------
# this function applys the box filter on the input image
# the size of the filter is filter_dim*filter_dim
# returns the results of applying the the filter on original_img
def box_filter(original_img, filter_dim):
    img = np.pad(original_img, (filter_dim -1, filter_dim-1), 'reflect')

    box_filter = np.ones([filter_dim, filter_dim], dtype= float)
    box_filter = np.true_divide(box_filter, filter_dim*filter_dim)
    output = filtering(img, box_filter)
    output = unpad(output, filter_dim)
    output = normal(output)
    return output

#---------------------------------------------------------------------
# this method applys the input filter matrix on the input image
# my_filter is the matrix of filter
# returns the result of the input my_filter on img
def filtering(img, my_filter):
    filter_dim = np.size(my_filter, 0)
    output = np.zeros_like(img, dtype = int)
    filter_center = int(filter_dim/2)

    for i in range(np.size(img, 0)-filter_dim+1):
        for j in range(np.size(img, 1)-filter_dim+1):
            img_part = img[i:i+filter_dim, j:j+filter_dim]
            value = np.sum(np.multiply(img_part, my_filter))
            output[i+filter_center, j+filter_center] = value
    return output

#---------------------------------------------------------------------
# this method removes the padding
# takes the image with padding(as img) and size of the filter(as filter_dim)
# returns a image with the original size
def unpad(img, filter_dim):
    original_img = img[filter_dim-1 : np.size(img, 0)-filter_dim+1, filter_dim-1 : np.size(img, 1)-filter_dim+1]
    return original_img

#---------------------------------------------------------------------
# this method applys box filter several times
# n is the number of times that method applys box filter on the image
# filter_dim is the filter size
def several_time_boxfilter(original_img, filter_dim, n):
    new_img = original_img
    for i in range(n):
        new_img = box_filter(new_img, filter_dim)
    return new_img

#---------------------------------------------------------------------
def gaussian_noise(img, var):
    
    mean = 0
    sigma = var**0.5
    image = img
    
    row,col = image.shape
    print(image.shape)
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss.dtype = 'uint8'
    print(gauss.shape,"gav")
    gauss = gauss.reshape(row,col)
    

    noisy = np.add(image, gauss)
    
    
    print(noisy.shape)
    return noisy
#--------------------------------------------------------------------
def salt_and_pepper(image, amount):
    row,col = image.shape
    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))for i in image.shape]
    out[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))for i in image.shape]
    out[coords] = 0
    out.dtype = 'uint8'
    return out
#---------------------------------------------------------------------
# this method applys median filter on the input noisy image
# size of the filter is filter_dim*filter_dim
def medain_filter(noisy, filter_dim):
    img = np.pad(noisy, (filter_dim -1, filter_dim-1), 'reflect')
    output = np.zeros_like(img)
    filter_center = int(filter_dim/2)
    for i in range(np.size(img, 0)-filter_dim+1):
        for j in range(np.size(img, 1)-filter_dim+1):
            img_part = img[i:i+filter_dim, j:j+filter_dim]
            value = np.median(img_part)
            output[i+filter_center, j+filter_center] = value
    output = unpad(output, filter_dim)
    output = normal(output)
    return output
#---------------------------------------------------------------------
# this method applys differnce filter on the input image
# 3 filters are available:
# enter "a" as dif_filter for (1/2)*[1, 0, -1]
# enter "b" as dif_filter for (1/6)*[[1, 0, -1],[1, 0, -1],[1, 0, -1]]
# enter "c" as dif_filter for (1/8)*[[1, 0, 1],[2, 0, -2],[1, 0, -1]]
def difference_filter(img):
    dif_a = np.array([1, 0, -1])
    dif_a = np.true_divide(dif_a, 2)

    dif_b = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    dif_b = np.true_divide(dif_b, 6)

    dif_c = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dif_c = np.true_divide(dif_c, 8)

    out_a = filtering(img, dif_a)
    out_b = filtering(img, dif_b)
    out_c =filtering(img, dif_c)

    out_a = normal(out_a)
    out_b = normal(out_b)
    out_c = normal(out_c)
        
    return out_a, out_b, out_c
#----------------------------------------------------------------------
def robert_filter(img):
    robert_v = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]])
    
    robert_h = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]])

    img = np.pad(img, (2, 2), 'reflect')

    vertical = filtering(img, robert_v)
    horizontal = filtering(img, robert_h)

    output = np.sqrt(np.square(vertical) + np.square(horizontal))
    output = unpad(output, 3)
    output = normal(output)
    vertical = normal(vertical)
    horizontal = normal(horizontal)

    return output, vertical, horizontal
#----------------------------------------------------------------------
def normal(img):
    min_val = np.amin(img)
    max_val = np.amax(img)
    print(min_val, max_val)
    output = np.zeros_like(img)
    
    for i in range(np.size(img, 0)):
        for j in range(np.size(img, 1)):
            new_val = math.floor(((img[i, j] - min_val)*255) / (max_val - min_val))
            output[i, j] = new_val
    return output
#-----------------------------------------------------------------------
def mse(true, pred):
    MSE = np.square(np.subtract(true,pred)).mean() 
    return MSE

#-----------------------------------------------------------------------
original_img = cv2.imread("image/Lena.bmp", 0)

original_img.dtype = "uint8"
cv2.imwrite("output/lena_grayscale.jpg", original_img)

print("sag")
#3.1.3
"""
boxFilter1 = box_filter(original_img, 3)
boxFilter5 = several_time_boxfilter(original_img, 3, 5)
boxFilter10 = several_time_boxfilter(original_img, 3, 10)
boxFilter20 = several_time_boxfilter(original_img, 3, 20)
boxFilter50 = several_time_boxfilter(original_img, 3, 50)


boxFilter1.dtype = "uint8"
cv2.imwrite("output/boxFilter1.jpg", boxFilter1)

boxFilter5.dtype = "uint8"
cv2.imwrite("output/boxFilter5.jpg", boxFilter5)

boxFilter10.dtype = "uint8"
cv2.imwrite("output/boxFilter10.jpg", boxFilter10)

boxFilter20.dtype = "uint8"
cv2.imwrite("output/boxFilter20.jpg", boxFilter20)

boxFilter50.dtype = "uint8"
cv2.imwrite("output/boxFilter50.jpg", boxFilter50)
"""
"""
salt2 = salt_and_pepper(original_img, 0.2)
#new.dtype = "uint8"
salt1 = salt_and_pepper(original_img, 0.1)
salt05 = salt_and_pepper(original_img, 0.05)
"""
"""
cv2.imwrite("output/noisy_salt2.jpg", salt2)
cv2.imwrite("output/noisy_salt1.jpg", salt1)
cv2.imwrite("output/noisy_salt05.jpg", salt05)
"""

"""
x = medain_filter(salt1, 5)
cv2.imwrite("output/salt01_med5.jpg", x)
"""
"""
dif_a, dif_b, dif_c = difference_filter(original_img)
dif_a.astype('uint8')
dif_b.astype("uint8")
dif_c.astype("uint8")

cv2.imwrite("output/dif_a.jpg", dif_a)
cv2.imwrite("output/dif_b.jpg", dif_b)
cv2.imwrite("output/dif_c.jpg", dif_c)
"""

"""
out, v, h = robert_filter(original_img)

out.astype('uint8')
v.astype('uint8')
h.astype('uint8')

cv2.imwrite("output/robert_a.jpg.jpg", v)
cv2.imwrite('output/robert_b.jpg', h)
cv2.imwrite("output/robert_mix.jpg", out)
"""


gauss01 = random_noise(original_img, mode='gaussian', seed=None, clip=True, var = 0.01)
gauss01 = np.multiply(gauss01, 255)

gauss1 = random_noise(original_img, mode='gaussian', seed=None, clip=True, var = 0.1)
gauss1 = np.multiply(gauss1, 255)

gauss05 = random_noise(original_img, mode='gaussian', seed=None, clip=True, var = 0.05)
gauss05 = np.multiply(gauss05, 255)

salt1 = random_noise(original_img, mode="s&p" , seed=None, clip=True, amount = 0.1)
salt1 = np.multiply(salt1, 255)

salt2 = random_noise(original_img, mode="s&p" , seed=None, clip=True, amount = 0.2)
salt2 = np.multiply(salt2, 255)

salt05 = random_noise(original_img, mode='s&p' , seed=None, clip=True, amount = 0.05)
salt05 = np.multiply(salt05, 255)

"""
cv2.imwrite('output/salt1.jpg', salt1)
cv2.imwrite('output/salt2.jpg', salt2)
cv2.imwrite('output/salt05.jpg', salt05)
cv2.imwrite("output/gauss01.jpg", gauss01)
cv2.imwrite("output/gauss05.jpg", gauss05)
cv2.imwrite("output/gauss1.jpg", gauss1)
"""
"""
salt1_med3 = medain_filter(salt1, 3)
salt1_med5 = medain_filter(salt1, 5)
salt1_med7 = medain_filter(salt1, 7)
salt1_med9 = medain_filter(salt1, 9)

salt2_med3 = medain_filter(salt2, 3)
salt2_med5 = medain_filter(salt2, 5)
salt2_med7 = medain_filter(salt2, 7)
salt2_med9 = medain_filter(salt2, 9)

salt05_med3 = medain_filter(salt05, 3)
salt05_med5 = medain_filter(salt05, 5)
salt05_med7 = medain_filter(salt05, 7)
salt05_med9 = medain_filter(salt05, 9)
"""
"""
cv2.imwrite("output/salt1_med3.jpg", salt1_med3)
cv2.imwrite("output/salt1_med5.jpg", salt1_med5)
cv2.imwrite("output/salt1_med7.jpg", salt1_med7)
cv2.imwrite("output/salt1_med9.jpg", salt1_med9)

cv2.imwrite("output/salt2_med3.jpg", salt2_med3)
cv2.imwrite("output/salt2_med5.jpg", salt2_med5)
cv2.imwrite("output/salt2_med7.jpg", salt2_med7)
cv2.imwrite("output/salt2_med9.jpg", salt2_med9)

cv2.imwrite("output/salt05_med3.jpg", salt05_med3)
cv2.imwrite("output/salt05_med5.jpg", salt05_med5)
cv2.imwrite("output/salt05_med7.jpg", salt05_med7)
cv2.imwrite("output/salt05_med9.jpg", salt05_med9)
"""

gauss1_med3 = medain_filter(gauss1, 3)
gauss1_med5 = medain_filter(gauss1, 5)
gauss1_med7 = medain_filter(gauss1, 7)
gauss1_med9 = medain_filter(gauss1, 9)

gauss01_med3 = medain_filter(gauss01, 3)
gauss01_med5 = medain_filter(gauss01, 5)
gauss01_med7 = medain_filter(gauss01, 7)
gauss01_med9 = medain_filter(gauss01, 9)

gauss05_med3 = medain_filter(gauss05, 3)
gauss05_med5 = medain_filter(gauss05, 5)
gauss05_med7 = medain_filter(gauss05, 7)
gauss05_med9 = medain_filter(gauss05, 9)


"""
cv2.imwrite('output/gauss1_med3.jpg', gauss1_med3)
cv2.imwrite('output/gauss1_med5.jpg', gauss1_med5)
cv2.imwrite('output/gauss1_med7.jpg', gauss1_med7)
cv2.imwrite('output/gauss1_med9.jpg', gauss1_med9)

cv2.imwrite('output/gauss01_med3.jpg', gauss01_med3)
cv2.imwrite('output/gauss01_med5.jpg', gauss01_med5)
cv2.imwrite('output/gauss01_med7.jpg', gauss01_med7)
cv2.imwrite('output/gauss01_med9.jpg', gauss01_med9)

cv2.imwrite('output/gauss05_med3.jpg', gauss05_med3)
cv2.imwrite('output/gauss05_med5.jpg', gauss05_med5)
cv2.imwrite('output/gauss05_med7.jpg', gauss05_med7)
cv2.imwrite('output/gauss05_med9.jpg', gauss05_med9)

"""

gauss1_box3 = box_filter(gauss1, 3)
gauss1_box5 = box_filter(gauss1, 5)
gauss1_box7 = box_filter(gauss1, 7)
gauss1_box9 = box_filter(gauss1, 9)

gauss01_box3 = box_filter(gauss01, 3)
gauss01_box5 = box_filter(gauss01, 5)
gauss01_box7 = box_filter(gauss01, 7)
gauss01_box9 = box_filter(gauss01, 9)

gauss05_box3 = box_filter(gauss05, 3)
gauss05_box5 = box_filter(gauss05, 5)
gauss05_box7 = box_filter(gauss05, 7)
gauss05_box9 = box_filter(gauss05, 9)

"""
cv2.imwrite('output/gauss1_box3.jpg', gauss1_box3)
cv2.imwrite('output/gauss1_box5.jpg', gauss1_box5)
cv2.imwrite('output/gauss1_box7.jpg', gauss1_box7)
cv2.imwrite('output/gauss1_box9.jpg', gauss1_box9)

cv2.imwrite('output/gauss01_box3.jpg', gauss01_box3)
cv2.imwrite('output/gauss01_box5.jpg', gauss01_box5)
cv2.imwrite('output/gauss01_box7.jpg', gauss01_box7)
cv2.imwrite('output/gauss01_box9.jpg', gauss01_box9)

cv2.imwrite('output/gauss05_box3.jpg', gauss05_box3)
cv2.imwrite('output/gauss05_box5.jpg', gauss05_box5)
cv2.imwrite('output/gauss05_box7.jpg', gauss05_box7)
cv2.imwrite('output/gauss05_box9.jpg', gauss05_box9)
"""
"""
filter_dim = 9
a = 0.2
img = original_img

smooth = cv2.GaussianBlur(img, (filter_dim, filter_dim), 2)
# smooth = normal(smooth)
sub = smooth - img
result = img + (a*(smooth - img))
result = normal(result)
result.astype('uint8')
cv2.imwrite("output/sooth3.jpg", result)
cv2.imwrite("output/sooth4.jpg", sub)
cv2.imwrite("output/sooth5.jpg", smooth)
# cv2.imwrite('output/smooth3.jpg', smooth3)
"""

img = cv2.imread("image/myimage1.jpg", 0)
# print(img)
medi = medain_filter(img, 7)
medi = medain_filter(img, 7)
medi = medain_filter(img, 7)
medi = medain_filter(img, 7)
box = box_filter(img, 11)
blur = cv2.GaussianBlur(img, (9, 9), 1)
sub = img - box
addi = img + (0.1*sub)
addi = normal(addi)
add = medi + (0.1*sub)
add = normal(add)
meda = medain_filter(addi, 7)
meda = medain_filter(meda, 7)
meda = medain_filter(meda, 7)
meda = medain_filter(meda, 7)
meda = medain_filter(meda, 7)
meda = medain_filter(meda, 7)
meda = medain_filter(meda, 7)


cv2.imwrite("output/test.jpg", add)
cv2.imwrite('output/test1.jpg', img)
cv2.imwrite("output/test_noise.jpg", meda)
cv2.imwrite('output/test_addi.jpg', addi)

"""
boxFilter400 = several_time_boxfilter(original_img, 3, 400)
cv2.imwrite('output/boxFilter400.jpg', boxFilter400)
"""
print(mse(original_img, gauss05_box3), '05_3')
print(mse(original_img, gauss05_box5), '05_5')
print(mse(original_img, gauss05_box7), '05_7')
print(mse(original_img, gauss05_box9), '05_9')

print(mse(original_img, gauss01_box3), '01_3')
print(mse(original_img, gauss01_box5), '01_5')
print(mse(original_img, gauss01_box7), '01_7')
print(mse(original_img, gauss01_box9), '01_9')

print(mse(original_img, gauss1_box3), '1_3')
print(mse(original_img, gauss1_box5), '1_5')
print(mse(original_img, gauss1_box7), '1_7')
print(mse(original_img, gauss1_box9), '1_9')