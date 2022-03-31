import cv2
import numpy as np
import math

def rotate(img, degree):
    affine = np.array([[math.cos(math.radians(degree)), math.sin(math.radians(degree)), 0],
                     [-(math.sin(math.radians(degree))), math.cos(math.radians(degree)), 0],
                     [0, 0, 1]])
    rotated = np.zeros_like(img)
    half_x = np.size(img, 0) / 2
    half_y = np.size(img, 1) / 2

    for v in range(np.size(img, 0)):
        for w in range(np.size(img, 1)):
            x, y = mapped_coordinate(v - half_x, w - half_y, affine)
            x = round(x + half_x)
            y = round(y + half_y)
            if(x < np.size(img, 0) and y < np.size(img, 1) and x > 0 and y > 0):
                rotated[x, y] = img[v, w]
    return rotated

#---------------------------------------------------

def mapped_coordinate(v, w, affine):
    in_coordinate = np.array([v, w, 1])
    out_coordinate  = np.matmul(in_coordinate, affine)
    x = out_coordinate[0]
    y = out_coordinate[1]
    return x, y

#---------------------------------------------------

def bilinear(img, rotated, degree):
    affine = np.array([[math.cos(math.radians(degree)), math.sin(math.radians(degree)), 0],
                     [-(math.sin(math.radians(degree))), math.cos(math.radians(degree)), 0],
                     [0, 0, 1]])
    half_x = np.size(img, 0) / 2
    half_y = np.size(img, 1) / 2
    invert_affine = np.linalg.inv(affine)
    for x in range(np.size(rotated, 0)):
        for y in range(np.size(rotated, 1)):
            if  rotated[x, y] == 0 :
                v, w = mapped_coordinate(x - half_x , y - half_y, invert_affine)
                v = v + half_x
                w = w + half_y
                if(v < np.size(img, 0) - 1 and w < np.size(img, 1) - 1 and v > 0 and w > 0):
                    rotated[x, y] = bilinear_val(img, v, w)
    return rotated

# ---------------------------------------------------

def bilinear_val(img, v, w):
    ax = math.floor(v)  #ax = bx = floor(v)
    ay = math.floor(w)  #ay = dx = floor(w)
    bx = math.floor(v)
    by = math.ceil(w)   #by = cy = ceil(w)
    dx = math.ceil(v)  #cx = dx = ceil(v)
    dy = math.floor(w)
    cx = math.ceil(v)
    cy = math.ceil(w)

    dr1 = abs(v - ax)
    dr2 = abs(v - dx)
    dc1 = abs(w - ay)
    dc2 = abs(w - by)
    value = (img[ax, ay] * dr2 * dc2) + (img[bx, by] * dc1 * dr2) + (img[dx, dy] * dr1 * dc2) + (img[cx, cy] * dr1 * dc1)
    return value               

#-------------------------------------------------

def downsample(img, rate):
    downsampled =np.zeros([int(np.size(img, 0)/rate), int(np.size(img, 1)/rate)], dtype = np.uint8)

    for i in range(np.size(downsampled, 0)):
        for j in range(np.size(downsampled, 1)):
            downsampled[i, j] = img[i*rate, j*rate]

    return downsampled

#---------------------------------------------------

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

#-----------------------------------------------

def bilinear_upsample(img, rate):
    upsampled = np.zeros([np.size(img, 0)* rate, np.size(img, 1)* rate], dtype = np.uint8)

    for x in range(np.size(upsampled, 0)):
        for y in range(np.size(upsampled, 1)):
            v = x/rate
            w = y/rate
            if (v < np.size(img, 0)-1 and w < np.size(img, 1)-1):
                if(math.floor(v) == v or math.floor(w) == w):
                    upsampled[x, y] = img[int(v), int(w)]
                else:
                    upsampled[x, y] = bilinear_val(img, v, w)
    return upsampled
#-----------------------------------------------

def quantize(img, level):
    coef = 256 / level
    quantized = np.floor(np.true_divide(img, coef)) * coef
    quantized = quantized.astype("uint8")
    return quantized
path = "image/Elaine.bmp"
#img = cv2.imread(path, 0)
#rotated = np.zeros_like(img)

affine_30 = np.array([[math.cos(math.radians(30)), math.sin(math.radians(30)), 0],
                     [-(math.sin(math.radians(30))), math.cos(math.radians(30)), 0],
                     [0, 0, 1]])
"""
affine_90 = np.array([[math.cos(math.radians(90)), math.sin(math.radians(90)), 0],
                     [-(math.sin(math.radians(90))), math.cos(math.radians(90)), 0],
                     [0, 0, 1]])                     


for v in range(np.size(img, 0)):
    for w in range(np.size(img, 1)):
        in_coordinate = np.array([v, w, 1])
        out_coordinate  = np.matmul(in_coordinate, affine_30)
        x = round(out_coordinate[0])
        y = round(out_coordinate[1])
        if(x < np.size(img, 0) and y < np.size(img, 1) and x > 0 and y > 0):
            rotated[x, y] = img[v, w]
"""        
#rotated = rotate(img, 45)
#cv2.imwrite("45.bmp", rotated)
#interpolated = bilinear(img, rotated, 45)

"""
invert_affine = np.linalg.inv(affine_30)
for x in range(np.size(rotated, 0)):
    for y in range(np.size(rotated, 1)):
        if  rotated[x , y] == 0 :
            v, w = mapped_coordinate(x , y, invert_affine)
            if(v < np.size(img, 0) - 1 and w < np.size(img, 1) - 1 and v > 0 and w > 0):
                ax = math.floor(v)  #ax = bx = floor(v)
                ay = math.floor(w)  #ay = dx = floor(w)
                bx = math.floor(v)
                by = math.ceil(w)   #by = cy = ceil(w)
                dx = math.ceil(v)  #cx = dx = ceil(v)
                dy = math.floor(w)
                cx = math.ceil(v)
                cy = math.ceil(w)

                dr1 = abs(v - ax)
                dr2 = abs(v - dx)
                dc1 = abs(w - ay)
                dc2 = abs(w - by)
               #value = img[round(v), round(w)]
                value = (img[ax, ay] * dr2 * dc2) + (img[bx, by] * dc1 * dr2) + (img[dx, dy] * dr1 * dc2) + (img[cx, cy] * dr1 * dc1)
                rotated[x, y] = value
"""                
#cv2.imwrite("interpolated45.bmp", interpolated)
#goldhill_path = "image/Goldhill.bmp"
#img = cv2.imread(goldhill_path, 0)

"""
#downsmple without average filter
downsampled =np.zeros([int(np.size(img, 0)/rate), int(np.size(img, 1)/rate)], dtype = np.uint8);

for i in range(np.size(downsampled, 0)):
    for j in range(np.size(downsampled, 1)):
        downsampled[i, j] = img[i*rate, j*rate]
"""
#downsampled = downsample(img , 2)
#cv2.imwrite("downsmpled.bmp", downsampled)
#cv2.imshow("imagw", downsampled)
#cv2.waitKey(0)

"""
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
"""
#avg_downsampled = avg_downsample(img, 2)
#cv2.imwrite("avg_downsmpled.bmp", avg_downsampled)
#cv2.imshow("imagw", avg_downsampled)
#cv2.waitKey(0)
#rate = 2
#upsample with repplication
"""
upsampled = np.zeros([np.size(downsampled, 0)* rate, np.size(downsampled, 1)* rate], dtype = np.uint8)

for i in range(np.size(upsampled, 0)):
    for j in range(np.size(upsampled, 1)):

        upsampled[i, j] = downsampled[math.floor(i/2), math.floor(j/2)]

#cv2.imshow("imagw", upsampled)
#cv2.waitKey(0)
"""
#upsampled = replication_upsample(downsampled, 2)
#cv2.imwrite("rep_upsampled.bmp", upsampled)

#upsample with interpolation
"""
upsampled = np.zeros([np.size(downsampled, 0)* rate, np.size(downsampled, 1)* rate], dtype = np.uint8)

for x in range(np.size(upsampled, 0)):
    for y in range(np.size(upsampled, 1)):
        v = x/rate
        w = y/rate
        if (v < np.size(downsampled, 0)-1 and w < np.size(downsampled, 1)-1):
            if(math.floor(v) == v and math.floor(w) == w):
                upsampled[x, y] = downsampled[int(v), int(w)]
            else:
                upsampled[x, y] = bilinear_val(downsampled, v, w)
           # upsampled[x, y] = value

            #print(dr1, dr2, dc1, dc2, value)
"""

#upsampled = bilinear_upsample(downsampled, rate)
#cv2.imwrite("bilinear_upsampled.bmp", upsampled)

#barbara_path = 'image/Barbara.bmp'
#img = cv2.imread(path, 0)
#out = rotate(img, 30)
#out = bilinear(img, out, 30)
#cv2.imwrite("center.bmp", out)

#equ = cv2.equalizeHist(img)
#print(type(equ))


#cv2.imshow('image', res) 
#cv2.waitKey(0)
#level2 = np.double(img)
#level2 = np.floor(np.true_divide(level2, 64)) * 64
#print(level2)
#level2 = quantize(img, 128)
#level2 = np.double(level2)
#out = np.zeros(level2.shape, np.double)
#normalized = cv2.normalize(level2, out, 1.0, 0.0, cv2.NORM_MINMAX)
#level2 = quantize(img, 4)
#cv2.imshow("image",level2)
#cv2.waitKey(0)
"""

img = cv2.imread("image/Barbara.bmp", 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ)) 
q8 = quantize(img, 8)
eq8 = quantize(equ, 8)

q16 = quantize(img, 16)
eq16 = quantize(equ, 16)

q32 = quantize(img, 32)
eq32 = quantize(equ, 32)

q64 = quantize(img, 64)
eq64 = quantize(equ, 64)

q128 = quantize(img, 128)
eq128 = quantize(equ, 128)

cv2.imwrite("q8.jpg", q8)
cv2.imwrite("eq8.jpg", eq8)

cv2.imwrite("q16.jpg", q16)
cv2.imwrite("eq16.jpg", eq16)

cv2.imwrite("q32.jpg", q32)
cv2.imwrite("eq32.jpg", eq32)

cv2.imwrite("q64.jpg", q64)
cv2.imwrite("eq64.jpg", eq64)

cv2.imwrite("q128.jpg", q128)
cv2.imwrite("eq128.jpg", eq128)


print(np.square(np.subtract(img,q8)).mean(), "q8")
print(np.square(np.subtract(img,eq8)).mean(), "eq8")

print(np.square(np.subtract(img,q16)).mean(), "q16")
print(np.square(np.subtract(img,eq16)).mean(), "eq16")

print(np.square(np.subtract(img,q32)).mean(), "q32")
print(np.square(np.subtract(img,eq32)).mean(), "eq32")

print(np.square(np.subtract(img,q64)).mean(), "q64")
print(np.square(np.subtract(img,eq64)).mean(), "eq64")

print(np.square(np.subtract(img,q128)).mean(), "q128")
print(np.square(np.subtract(img,eq128)).mean(), "eq128")
"""
img = cv2.imread("image/Goldhill.bmp", 0)
downsampled = downsample(img, 2)
avg_downsampled = avg_downsample(img, 2)
#print(np.size(avg_downsampled, 0))
#cv2.imwrite('downsampled.jpg', downsampled)

replicated = replication_upsample(downsampled, 2)
replicated_avg = replication_upsample(avg_downsampled, 2)

bilinear_upsampled = bilinear_upsample(downsampled, 2)
bilinear_avg = bilinear_upsample(avg_downsampled, 2)

cv2.imwrite("replicated.jpg", replicated)
cv2.imwrite("replicated_avg.jpg", replicated_avg)
cv2.imwrite("bilinear_upsampled.jpg", bilinear_upsampled)
cv2.imwrite("bilinear_avg.jpg", bilinear_avg)


print(np.square(np.subtract(img,replicated)).mean(), "replicated")
print(np.square(np.subtract(img,replicated_avg)).mean(), "replicated_avg")

print(np.square(np.subtract(img,bilinear_upsampled)).mean(), "bilinear upsampled")
print(np.square(np.subtract(img,bilinear_avg)).mean(), "bilinear avg")
