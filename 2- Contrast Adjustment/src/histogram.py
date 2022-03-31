import numpy as np
import cv2
import matplotlib.pyplot as plt 


#---------------------------------------------
# this method compute the histogram of a grayscale image (assuming 256 levels of gray)
# returns a numpy array of histogram
def histogram(img):
    histogram = np.zeros(256, dtype = int)
    
    for i in range(np.size(img, 0)):
        for j in range(np.size(img, 1)):
            gray_level = img[i, j]
            histogram[gray_level] += 1

    return histogram
#----------------------------------------------
# this method draws a stem plot of the input histogram
# saves the stem plot as an images <plotnamr.jpg>
def write_plot(h, plot_name):

    r = np.array(([i for i in range(256)]))
    plt.figure(plot_name)
    plt.stem(r, h, markerfmt = "None")
    plt.xlabel("intensity value")
    plt.ylabel("number of pixels")
    plt.title('Histogram of the image')
    plt.savefig(plot_name + ".jpg")
#-----------------------------------------------
def histeq(img):
#this method performs histogram equalization on the input image
#returns histogran equalized image as the output
# n is the histogram of the image, p is normalized histogram, and n is the total number of pixels
    n = histogram(img)
    r = np.array(([i for i in range(256)]))
    size = np.size(img, 0) * np.size(img, 1)
    p = np.true_divide(n, size)
    cdf = np.zeros_like(p)

    for i in range(np.size(r, 0)):
        for j in range(i + 1):
            cdf[i] += p[j]

    out = 255 * cdf
    discrete_out = np.round(out)
   
    his_equ = np.zeros_like(img)

    for i in range(np.size(img, 0)):
        for j in range(np.size(img, 1)):
            origine_intensity = img[i][j]
            his_equ[i, j] = discrete_out[origine_intensity]
    
    return his_equ
#------------------------------------------------
#this method performs local histogram equalization on the input image
#returns the locaal histogram eualized image as the output
#fraction indicates that the window size is what fraction of the original image
def local_histeq(img, fraction):
    window_x = int( np.size(img, 0) * fraction)
    window_y = int( np.size(img, 1) * fraction)
    local_equ = np.zeros_like(img)  

    for i in range(0, np.size(img, 0), window_x):
        for j in range(0, np.size(img, 1), window_y):
            window = img[i : min(i + window_x , np.size(img, 0)), j :  min(j + window_y, np.size(img, 1))];
            equ = histeq(window)
            # equ = cv2.equalizeHist(window)
            local_equ[i : min(i + window_x  , np.size(img, 0)), j :  min(j + window_y, np.size(img, 1))] = equ
    return local_equ
#--------------------------------------------------
img1 = cv2.imread("image/CameraMan.bmp", 0)
n = histogram(img1)
#write_plot(histogram, "stem_plot")
r = np.array(([i for i in range(256)]))
'''
size = np.size(img, 0) * np.size(img, 1)
p = np.true_divide(n, size)
cdf = np.zeros_like(p)

for i in range(np.size(r, 0)):
    for j in range(i + 1):
        cdf[i] += p[j]

out = 255 * cdf
discrete_out = np.round(out)
# print(discrete_out)
his_equ = np.zeros_like(img)

for i in range(np.size(img, 0)):
    for j in range(np.size(img, 1)):
        origine_intensity = img[i][j]
        his_equ[i, j] = discrete_out[origine_intensity]
'''
"""
his_equ = histeq(img1)
his_equ.astype("uint8")
cv2.imwrite("equ.jpg", his_equ)
      


fraction = 1/8

#size of window
window_x = int( np.size(img, 0) * fraction)
window_y = int( np.size(img, 1) * fraction)

# print(img)
# print("sag")
# print(img[0 : 3, 0:3])

print(window_x, window_y)
local_equ = np.zeros_like(img)
for i in range(0, np.size(img, 0), window_x):
    for j in range(0, np.size(img, 1), window_y):
        
        window = img[i : min(i + window_x , np.size(img, 0)), j :  min(j + window_y, np.size(img, 1))];
        # print(np.shape(window))
        print(i, j)
        equ = histeq(window)
        # equ = cv2.equalizeHist(window)
        local_equ[i : min(i + window_x  , np.size(img, 0)), j :  min(j + window_y, np.size(img, 1))] = equ
"""
#print(img)
#print("sag")
#a = copy.deepcopy(img)
#print(a)
img = cv2.imread("image/HE1.jpg", 0)
"""
local1 = local_histeq(img, 1)
cv2.imwrite("localH4_1.jpg", local1)

local2 = local_histeq(img, 1/2)
cv2.imwrite("localH4_2.jpg", local2)

local4 = local_histeq(img, 1/4)
cv2.imwrite("localH4_4.jpg", local4)

local8 = local_histeq(img, 1/8)
cv2.imwrite("localH4_8.jpg", local8)

local16 = local_histeq(img, 1/16)
cv2.imwrite("localH4_16.jpg", local16)
"""
local64 = local_histeq(img, 1/64)
cv2.imwrite("localH1_64.jpg", local64)


"""
plt.figure("fig")
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig.suptitle('Histogram of the image')
ax1.stem(r, h1, markerfmt = "None")
ax2.stem(r, h2, markerfmt = "None")
ax1.set(ylabel="number of pixels")
ax2.set(xlabel="intensity value", ylabel="number of pixels")
ax1.set_title('Histogram of the original image')
ax2.set_title('Histogram of the equalized image')
plt.savefig("fig.jpg")
"""
