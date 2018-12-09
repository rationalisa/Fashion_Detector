function [hardNegFeats] = mineHardNegInImg(I,model,param)

pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
    
% w = model.SupportVectors'*model.Alpha;
% decision = w'*pyramid.featMat; % much faster than svmpredict
% decision
[~,score] = predict(model,pyramid.featMat');
Scores = score(:,1);
hardNegFeats = pyramid.featMat(:,(Scores'>=-1.01));