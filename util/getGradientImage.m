function I1 = getGradientImage(I)
    [GX, GY] = gradient(I);
%     I1 = sum(abs(GX), 3) + sum(abs(GY), 3);
%     I1 = I1.^2;
    I1 = max(GX.^2, [], 3) + max(GY.^2, [], 3);
    I1 = I1.^(.5);
end