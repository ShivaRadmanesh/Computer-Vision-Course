import cv2 
import numpy as np
import math
import pywt

def avg_downsample(img, rate):
    avg_downsampled =np.zeros([int(np.size(img, 0)/rate), int(np.size(img, 1)/rate)], dtype = np.uint8)

    for i in range(np.size(avg_downsampled, 0)):
        for j in range(np.size(avg_downsampled, 1)):

            summ = 0
            #calculate the average
            for x in range(i*rate, i*rate + rate):
                for y in range(j*rate, j*rate + rate):
                    summ += img[x, y]
                    
            avg = round(summ / (rate*rate))
            avg_downsampled[i, j] = avg

    return avg_downsampled

#---------------------------------------------------

def replication_upsample(img, rate):
    upsampled = np.zeros([np.size(img, 0)* rate, np.size(img, 1)* rate], dtype = np.uint8)

    for i in range(np.size(upsampled, 0)):
        for j in range(np.size(upsampled, 1)):

            upsampled[i, j] = img[math.floor(i/2), math.floor(j/2)]

    return upsampled

#---------------------------------------------------------------------
# this method applys the input filter matrix on the input image
# my_filter is the matrix of filter
# returns the result of the input my_filter on img
def filtering(original_img, my_filter):
    filter_dim = np.size(my_filter, 0)
    filter_center = int(filter_dim/2)
    img = np.pad(original_img, (filter_dim -1, filter_dim-1), 'reflect')
    output = np.zeros_like(img, dtype=int)

    for i in range(np.size(img, 0)-filter_dim+1):
        for j in range(np.size(img, 1)-filter_dim+1):
            img_part = img[i:i+filter_dim, j:j+filter_dim]
            value = np.sum(np.multiply(img_part, my_filter))
            output[i+filter_center, j+filter_center ] = value
    output = unpad(output, filter_dim)
    # output = normal(output)
    return output
#---------------------------------------------------------------------
# this method removes the padding
# takes the image with padding(as img) and size of the filter(as filter_dim)
# returns a image with the original size
def unpad(img, filter_dim):
    original_img = img[filter_dim-1 : np.size(img, 0)-filter_dim+1, filter_dim-1 : np.size(img, 1)-filter_dim+1]
    return original_img
#---------------------------------------------------------------------
def normal(img):
    min_val = np.amin(img)
    max_val = np.amax(img)
    output = np.zeros_like(img)
    
    for i in range(np.size(img, 0)):
        for j in range(np.size(img, 1)):
            new_val = math.floor(((img[i, j] - min_val)*255) / (max_val - min_val))
            output[i, j] = new_val
    return output
#----------------------------------------------------------------------
def downsample(img, rate):
    downsampled =np.zeros([int(np.size(img, 0)/rate), int(np.size(img, 1)/rate)], dtype = np.uint8)

    for i in range(np.size(downsampled, 0)):
        for j in range(np.size(downsampled, 1)):
            downsampled[i, j] = img[i*rate, j*rate]

    return downsampled
#-----------------------------------------------------------------------

def wavelet_transform (img, level):
    # img = cv2.imread('./images/Lena.bmp', cv2.IMREAD_GRAYSCALE)
    coeffs = pywt.wavedec2(img, 'haar', level=level)

    

    [cA, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3)] = coeffs

    output = pywt.waverec2(coeffs, 'haar').astype('uint8')

    
    coeffs[0]= normal(coeffs[0])
    for i in range(1,len(coeffs)):
        coeffs[i] = [normal(d) for d in coeffs[i]]
    
    arr = pywt.coeffs_to_array(coeffs)


    return arr[0], output
#------------------------------------------------------------------------
def wavelet_compression(img, l, gamma):
    coeff = pywt.wavedec2(img, 'haar', level=l)

    coeff[0]= gamma * np.sign(coeff[0]) * np.floor(coeff[0] / gamma)
    for i in range(1,len(coeff)):
            coeff[i] = [gamma * np.sign(d) * np.floor(d / gamma) for d in coeff[i]]

    out = pywt.waverec2(coeff, 'haar').astype('uint8')

    coeff[0]= normal(coeff[0])
    for i in range(1,len(coeff)):
        coeff[i] = [normal(d) for d in coeff[i]]

    array = pywt.coeffs_to_array(coeff)
    return normal(array[0]), normal(out)
#------------------------------------------------------------------------
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#_________________6.1.1_________________

lena = cv2.imread("image/Lena.bmp", 0)
cv2.imwrite("output/lena.jpg", lena)

gaussian_pyramid = []
laplacian_pyramid = []
lena_shape = lena.shape 
level = math.log2(lena_shape[0])
gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

current_img = lena
last_level = 0
for i in range(int(level) + 1):
    if(i < int(level)):
        g = filtering(current_img, gaussian)
        l = current_img - g
        gaussian_pyramid.append(current_img)
        laplacian_pyramid.append(l)
        current_img = avg_downsample(g, 2)
    else:
        last_level = current_img

for i in range(len(gaussian_pyramid)):
    cv2.imwrite('output/6_1_1/G' + str(i) + '.jpg', gaussian_pyramid[i])
    cv2.imwrite('output/6_1_1/L' + str(i) + '.jpg', normal(laplacian_pyramid[i]))

#________________6.1.2 -> recunstruction________________

laplacian = laplacian_pyramid
level_num = int(level)

current = gaussian_pyramid[3]

for i in range(len(laplacian) - 6):
    gauss = replication_upsample(current, 2)
    # dim = (current.shape[0]*2, current.shape[1]*2)
    # current = current.astype('float64')
    # gauss = cv2.resize(current, dim, interpolation = cv2.INTER_LINEAR)
    lap = laplacian[level_num - i -7]
    current = gauss + lap
    current = normal(current)  
    
# cv2.imwrite('output/6_1_1/reconstructed_replication.jpg', current)
#_________________________6.1.3_______________________________
wave_img, rec = wavelet_transform(lena, 3)
cv2.imwrite('output/6_1_3/wave_img.jpg', wave_img)
cv2.imwrite('output/6_1_3/reconstructed.jpg', rec)

#_________________________6.1.4_______________________________

arr_img, out_img = wavelet_compression(lena, 3, 2)

cv2.imwrite('output/6_1_4/rec.png', out_img)
cv2.imwrite('output/6_1_4/wave.png', arr_img)

