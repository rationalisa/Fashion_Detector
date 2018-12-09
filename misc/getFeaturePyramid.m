function pyramid = getFeaturePyramid(I,scales,sBin)

for ii=1:numel(scales)
    thisI = imresize(I,scales(ii));
    [nr,nc,nd] = size(thisI);
    % make divisible by sBin
    thisI = thisI(1:end-mod(nr,sBin),1:end-mod(nc,sBin),:);
    
    if nd==1
        thisI = cat(3,thisI,thisI,thisI);
    end
    feats{ii} = single(features(im2double(thisI),sBin));
    gradientLevs{ii} = getGradientImage(im2double(thisI));
end
pyramid = struct('features', {feats}, 'scales', single(scales), 'sBin', sBin, 'gradimg', {gradientLevs});
