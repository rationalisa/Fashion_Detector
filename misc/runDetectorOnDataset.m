function [keepInfo] = runDetectorOnDataset(model,param,imgs,imgdir)

clear keepInfo;
n = 0;
for matchDec=1:numel(param.decRange)
    n = n + 1;
    count = 1;
    for ii=1:numel(imgs{matchDec})
        imname = [imgdir imgs{matchDec}(ii).name];       
        I = imread(imname);

        pyramid = getFeaturePyramid(im2double(I),param.scalesDet,param.sBin);
        pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 

        % w = model.SupportVectors'*model.Alpha;
        % decision = w'*pyramid.featMat; % much faster than svmpredict
        [~,score] = predict(model,pyramid.featMat');
        decision = score(:,2);

        [maxVal,maxNdx] = max(decision'); % keep only max single detection
        
        if maxVal > -1            
            keepInfo{n}(count).decision = maxVal;
            keepInfo{n}(count).name = imname;
            keepInfo{n}(count).scale = pyramid.featScale(maxNdx);
            keepInfo{n}(count).pos = pyramid.featPos(:,maxNdx).*param.sBin+1;
            count = count + 1;            
        end
    end
end

