%% Clean all up

close all % close all figures
clc % clean Command Window
clear % clear Workspace

%% Read Image 

I = imread('matlab_teste1.jpg');
imshow(I);

%% RGB Color Space

l_red = I(:,:,1);
l_green = I(:,:,2);
l_blue = I(:,:,3);

figure;
subplot(2,2,1)
imshow(I), title('Original');
subplot(2,2,2)
imshow(l_red), title('Red Plane');
subplot(2,2,3)
imshow(l_green), title('Green Plane');
subplot(2,2,4)
imshow(l_blue), title('Blue Plane');

%% Thresholding each layer  global VS adaptive

%figure; 
tg_red = imbinarize(l_red, 'global');
ta_red = imbinarize(l_red, 0.66);
imshowpair(tg_red, ta_red, 'montage');
title('RED');
%%
figure; 
tg_green = imbinarize(l_green, 'global');
ta_green = imbinarize(l_green, 'adaptive');
imshowpair(tg_green, ta_green, 'montage');
title('GREEN');

figure;
tg_blue = imbinarize(l_blue, 'global');
ta_blue = imbinarize(l_blue, 'adaptive');
imshowpair(tg_blue, ta_blue, 'montage');
title('BLUE');

%% Conclusion: global

I_thr = tg_red & tg_green & tg_blue;

figure;
imshow(I_thr), title('Thresholed Image');

%% Fill holes

% importante to work with black as a background

I_fillled = imfill(I_thr, 'holes');

figure;
imshow(I_fillled), title('Image with holes filled');

%% Remove extra structures

SE = strel('disk',200);
I_cleaned = imopen(I_fillled,SE);


%figure;
imshow(I_cleaned), title('Image without non wished structures');

%% invention
close all
II = I;

for i = 1:size(I,3)
    II(:,:,i) = I(:,:,i).*uint8(I_cleaned);
end
    
imshow(II)
    
    





    