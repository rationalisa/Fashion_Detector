function computeNearestNeighbors(param)
 
for queryDec = 1:numel(param.decRange)
    queryImgNames = param.trainImages{queryDec};
        
    for matchDec = 1:numel(param.decRange)
        matchImgNames = param.trainImages{matchDec};

        for queryImg = 1:10:numel(queryImgNames)
            savename = [param.matchdir 'queryDec=' num2str(queryDec) '_queryImg=' num2str(queryImg) '_matchDec=' num2str(matchDec) '.mat'];
            if lock(savename)==1
                continue;
            end
        
            detectname = [param.sampledir 'queryDec=' num2str(queryDec) '_queryImg=' num2str(queryImg) '_detector.mat'];
            load([detectname],'detector');

            matchYpos = zeros(numel(matchImgNames),numel(detector),'uint16');
            matchXpos = zeros(numel(matchImgNames),numel(detector),'uint16');
            matchScales = zeros(numel(matchImgNames),numel(detector),'single');
            matchScores = zeros(numel(matchImgNames),numel(detector),'single');

            for ii=1:numel(matchImgNames) 
                imname = [param.trainimgdir matchImgNames(ii).name];    
                I = imread(imname);

                pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
                pyramid = pyramid2Mat(pyramid,param.initPatchSize,param.normalizeFeats);    
                for jj=1:numel(detector)                
                    match = retrievekBestMatches(detector(jj),pyramid,param.initPatchSize,1);

                    matchYpos(ii,jj) = match.imPos(1);
                    matchXpos(ii,jj) = match.imPos(2);
                    matchScales(ii,jj) = single(match.scale);
                    matchScores(ii,jj) = single(match.score);
                end
            end

            save('-v7',savename,'matchScores','matchScales','matchYpos','matchXpos');
            unlock(savename);
        end
    end
end
