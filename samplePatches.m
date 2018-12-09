function samplePatches(param)

for queryDec = 1:numel(param.decRange)
    queryImgNames = param.trainImages{queryDec};
    for queryImg=1:10:numel(queryImgNames)
        savename = [param.sampledir 'queryDec=' num2str(queryDec) '_queryImg=' num2str(queryImg) '_detector.mat'];
        if lock(savename)==1
            continue;
        end
        
        imname = [param.trainimgdir queryImgNames(queryImg).name];    
        I = imread(imname);

        pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
        detector = sampleRandomPatchesInImage(pyramid,param.initPatchSize,param.numPatchesPerScale);
        for ii=1:numel(detector)    
            detector(ii).hog = detector(ii).hog-mean(detector(ii).hog);
            detector(ii).hog = detector(ii).hog/sqrt(detector(ii).hog'*detector(ii).hog);
        end
        save('-v7',savename,'detector');
        unlock(savename);
    end
end

