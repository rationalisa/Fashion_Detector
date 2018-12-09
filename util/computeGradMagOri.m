function [gradMag, gradOri, dx, dy] = computeGradMagOri(I,f)

% dx = imfilter(im2double(I),f);
% dy = imfilter(im2double(I),f');
dx = convn(im2double(I),f,'same'); % faster than imfilter
dy = convn(im2double(I),f','same'); % faster than imfilter

gradMag = (dx.*dx + dy.*dy).^(.5);
[gradMag,maxNdx] = max(gradMag,[],3);
[X,Y] = meshgrid(1:size(gradMag,2),1:size(gradMag,1));
ind = sub2ind(size(dx),Y,X,maxNdx);
dx = dx(ind);
dy = dy(ind);
gradOri = atan2(dy,dx);