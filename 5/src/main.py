import cv2
import numpy as np
import math
from sklearn.cluster import KMeans


def quantize(img, level):
    coef = 256 / level
    quantized = np.floor(np.true_divide(img, coef)) * coef
    quantized = quantized.astype("uint8")
    return quantized

def rgb_quantize(img, rlevel, glevel, blevel):
    blue = img[:,:, 0]
    green = img[:,:, 1]
    red = img[:,:, 2]

    q_red = quantize(red, rlevel)
    q_blue = quantize(blue, blevel)
    q_green = quantize(green, glevel)
    quantized_image = np.zeros_like(img)
    quantized_image[:,:,0] = q_blue
    quantized_image[:,:,1] = q_green
    quantized_image[:,:,2] = q_red
    return quantized_image

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def normal(img):
    min_val = np.amin(img)
    max_val = np.amax(img)
    output = np.zeros_like(img)
    
    for i in range(np.size(img, 0)):
        for j in range(np.size(img, 1)):
            new_val = math.floor(((img[i, j] - min_val)*255) / (max_val - min_val))
            output[i, j] = new_val
    return output
#---------------------------------------------------------------
def get_intensity(img):
    img = np.float32(img)/255
    img_blue = img[:,:, 0]
    img_green = img[:,:, 1]
    img_red = img[:,:, 2]
    intensity = (img_blue + img_green + img_red) / 3
    intensity = intensity * 255
    return(normal(intensity))
#--------------------------------------------------------------
def get_saturation(img):
    img = np.float32(img)/255
    img_blue = img[:,:, 0]
    img_green = img[:,:, 1]
    img_red = img[:,:, 2]

    saturation = 1 - (3 / (img_red + img_green + img_blue + 0.001) * np.minimum(np.minimum(img_red, img_green), img_blue))
    return normal(saturation)
#--------------------------------------------------------------
def get_hue(img):
    img = np.float32(img)/255
    hue = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b = img[i, j, 0]
            g = img[i, j, 1]
            r = img[i, j, 2]

            
            teta = np.arccos((0.5*(2*r-b-g)) / ((((r-g)**2) + (r-b)*(g-b))**0.5) + 0.000001)
            teta = np.degrees(teta)
            

            if(b <= g):
                hue[i, j] = teta
            else:
                hue[i,j] = 360 - teta
            
            if(math.isnan(hue[i, j])):
                hue[i, j] = 0
    return normal(hue)
#----------------------------------------------------------------
def color_quantization(img, k):
# Defining input data for clustering
  data = np.float32(img).reshape((-1, 3))
# Defining criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying cv2.kmeans function
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

pepper = cv2.imread("image/Pepper.bmp")
cv2.imwrite("output/pepper.jpg", pepper)
girl = cv2.imread("image/Girl.bmp")
cv2.imwrite("output/girl.jpg", girl)

#5.2.1
quantized = []
levels = [64, 32, 16, 8]
for i in range(len(levels)):
    level = levels[i]
    quantized.append(rgb_quantize(pepper, level, level, level))
    cv2.imwrite("output/5_2_1/pepper" + str(level) + ".jpg", quantized[i])

for i in range(len(quantized)):
    mse  = np.mean( (pepper - quantized[i]) ** 2 )
    psnr_val = psnr(pepper, quantized[i])
    print(str(levels[i]) + "mse: " + str(mse))
    print(str(levels[i]) + "psnr: " + str(psnr_val))

#5.2.2
quantized_522 = rgb_quantize(pepper, 8, 8, 4)
cv2.imwrite("output/5_2_2/quantized.jpg", quantized_522)
mse  = np.mean( (pepper - quantized_522) ** 2 )
psnr_val = psnr(pepper, quantized_522)
print("522 " + "mse: " + str(mse))
print("522 " + "psnr: " + str(psnr_val))
"""
#5.1.1
img = pepper

img_intensity = get_intensity(pepper)
cv2.imwrite("output/5_1_1/intensity.jpg", img_intensity)

img_saturation = get_saturation(pepper)
cv2.imwrite("output/5_1_1/saturation.jpg", img_saturation)

img_hue = get_hue(pepper)
cv2.imwrite("output/5_11/hue.jpg", img_hue)
"""
#5.2.3
quantized_8 = rgb_quantize(girl, 2, 2, 2)
cv2.imwrite("output/5_2_3/222.jpg", quantized_8)

levels_16 = [(2, 2, 4), (2, 4, 2), (4, 2, 2)]
quantized_16 = []
for i in range(len(levels_16)):
    l = levels_16[i]
    q = rgb_quantize(girl, l[0], l[1], l[2])
    quantized_16.append(q)
    cv2.imwrite("output/5_2_3/" + str(l[0]) + str(l[1]) + str(l[2]) + ".jpg", q)

levels_32 = [(2, 4, 4), (4, 2, 4), (4, 4, 2), (8, 2, 2), (2, 8, 2), (2, 2, 8)]
quantized_32 = []
for i in range(len(levels_32)):
    l = levels_32[i]
    q = rgb_quantize(girl, l[0], l[1], l[2])
    quantized_32.append(q)
    cv2.imwrite("output/5_2_3/" + str(l[0]) + str(l[1]) + str(l[2]) + ".jpg", q)


mse8  = np.mean( (pepper - quantized_8) ** 2 )
psnr_val8 = psnr(pepper, quantized_8)
print("(2, 2, 2)" + "mse: " + str(mse8))
print("(2, 2, 2)" + "psnr: " + str(psnr_val8))

for i in range(len(quantized_16)):
    mse  = np.mean( (pepper - quantized_16[i]) ** 2 )
    psnr_val = psnr(pepper, quantized_16[i])
    print(str(levels_16[i]) + "mse: " + str(mse))
    print(str(levels_16[i]) + "psnr: " + str(psnr_val))

for i in range(len(quantized_32)):
    mse  = np.mean( (pepper - quantized_32[i]) ** 2 )
    psnr_val = psnr(pepper, quantized_32[i])
    print(str(levels_32[i]) + "mse: " + str(mse))
    print(str(levels_32[i]) + "psnr: " + str(psnr_val))
# 5.2.3 second solution(kmeans)
colorNum = [3, 8, 16, 32, 128]
kmeans_quantized = []
for i in range (len(colorNum)):
    q = color_quantization(girl, colorNum[i])
    kmeans_quantized.append(q)
    cv2.imwrite("output/5_2_3/kmeans" + str(colorNum[i]) + ".jpg", q)

for i in range(len(colorNum)):
    mse  = np.mean( (pepper - kmeans_quantized[i]) ** 2 )
    psnr_val = psnr(pepper, kmeans_quantized[i])
    print("kmeans" + str(colorNum[i]) + "  mse: " + str(mse))
    print("kmeans" + str(colorNum[i]) + "  psnr: " + str(psnr_val))