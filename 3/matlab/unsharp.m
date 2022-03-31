clear all;
close all;
img = imread('lena.bmp');
img = rgb2gray(img);
%img = im2double(img);

smooth3 = imgaussfilt(img, 2, 'FilterSize',[3,3]);
res3_05 = img + (0.5*(smooth3 - img));
res3_05 = normal(res3_05);

res3_01 = img + (0.1*(smooth3 - img));
res3_01 = normal(res3_01);

res3_001 = img + (0.01*(smooth3 - img));
res3_001 = normal(res3_001);

res3_1 = img + (1*(smooth3 - img));
res3_1 = normal(res3_1);

res3_0 = img ;
res3_0 = normal(res3_0);

res3_08 = img + (0.8*(smooth3 - img));
res3_08 = normal(res3_08);

res3_03 = img + (0.3*(smooth3 - img));
res3_03 = normal(res3_03);

res3_09 = img + (0.9*(smooth3 - img));
res3_09 = normal(res3_09);

imwrite(res3_05, "output/res3_05.jpg")
imwrite(res3_01, "output/res3_01.jpg")
imwrite(res3_001, "output/res3_001.jpg")
imwrite(res3_1, "output/res3_1.jpg")
imwrite(res3_0, "output/res3_0.jpg")
imwrite(res3_08, "output/res3_08.jpg")
imwrite(res3_03, "output/res3_03.jpg")
imwrite(res3_09, "output/res3_09.jpg")
fprintf("sag")