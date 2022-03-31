clc;
clear all;
close all;
img = imread('CameraMan.bmp')
adjust1 = imadjust(img)
his_equ = histeq(img)


adjust_limited = imadjust(img,[0.3 0.7],[])
adjust_lower = imadjust(img, [],[],0.5)
adjust_higher = imadjust(img, [],[], 1.5)

imwrite(adjust1, "adjust1.jpg")
imwrite(his_equ, "his_equ.jpg")
imwrite(adjust_limited, 'limited.jpg')
imwrite(adjust_lower, 'lower.jpg')
imwrite(adjust_higher, 'higher.jpg')
imhist(adjust_limited)
