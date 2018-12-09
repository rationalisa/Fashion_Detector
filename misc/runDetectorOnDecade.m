function [keepInfo] = runDetectorOnDecade(model,param,imgs,posDecInd)

% n iterates over decades
clear keepInfo;
for n=1:numel(posDecInd)   
    count = 1;
    for ii=1:numel(imgs{posDecInd(n)})
        imname = [imgs{posDecInd(n)}(ii).name]; 
        I = imread(imname);

        pyramid = getFeaturePyramid(im2double(I),param.scalesDet,param.sBin);
        pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
 
        [~,score] = predict(model,pyramid.featMat');
        % score   
        if size(score,2)==1
            decision=score(:,1)
        else
            decision = score(:,2);
        end
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

