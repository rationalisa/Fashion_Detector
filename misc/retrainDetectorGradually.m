function [contentDetector,iterOut] = retrainDetectorGradually(contentCluster,model,param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% determine neg years to remove outliers from cluster
yearDist = 1996:2019;
posYears = zeros(numel(contentCluster),1);
for ii=1:numel(contentCluster)
    posYears(ii) = contentCluster(ii).year;
end     
matchHist = histc(posYears,yearDist);
matchHist = conv(matchHist,ones(5,1),'same');
negYears = yearDist(matchHist<=1);

posYears(ismember(posYears,negYears)) = [];
posYearInd = zeros(1,numel(yearDist));
posYearInd(ismember(yearDist,unique(posYears))) = 1;
posYearInd = reshape(posYearInd,[8,numel(param.decRange)]); % 8 decades
posDecInd = round(mean(posYearInd,1)); % mark decades that contain positives
if isempty(find(posDecInd==0))
    [~,minInd]=min(mean(posYearInd,1))
    posDecInd(minInd) = 0;
end
% can happen if cluster is between two decades; include both initially..
% if this doesn't work, just take max decade only

if isempty(find(posDecInd==1))
    maxInd = find(mean(posYearInd,1)>0.25);
    posDecInd(maxInd) = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set images for iterative training
clear iter;
for ii=1:(numel(param.decRange))    
    nn = 1;
    for jj=1:3:numel(param.trainImages{ii})
        iter(1).posImgs{ii}(nn).name = [param.trainimgdir param.trainImages{ii}(jj).name];        
        try 
            iter(2).posImgs{ii}(nn).name = [param.trainimgdir param.trainImages{ii}(jj+1).name];
            iter(3).posImgs{ii}(nn).name = [param.trainimgdir param.trainImages{ii}(jj+2).name];
        catch
        end
        nn = nn + 1;
    end 
end

d = dir([param.bgDir '*.jpg']);
randNdx = randperm(numel(d));
nn = 1;
for ii=1:numel(d)/450
    for jj=1:450
        negIter(ii).negImgs(jj).name = [param.bgDir d(randNdx(nn)).name];
        % negIter(ii)
        nn = nn + 1;
    end    
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start = 1;
negIt = 1;
num = 1;
origPosFeats = [];

numInstPerDec = 5;
perRoundImgs = 220;
initImgs = 10;
numRounds = 2;
 
% exand model over adjacent decades
while(~isempty(find(posDecInd==0)))
    donePosDecInd = posDecInd;
    posDecInd = conv(posDecInd,ones(1,3),'same');
    posDecInd(posDecInd>1) = 1; 
    if start == 1
        % at first iteration, include instances from pos decade
        thisPosDecInd = find(posDecInd==1);
        start = 0;
    else
        % only augment instances from unseen decades
        thisPosDecInd = find((posDecInd-donePosDecInd)==1);
    end
    % loop over iterations
    for it = 1:numel(iter)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % augment pos features from detections across different time periods
        [thisContentDetections] = runDetectorOnDecade(model,param,iter(it).posImgs,thisPosDecInd);

        % could also force initial instances to be part of the positives
        numDec = numel(thisContentDetections);
        posFeats = zeros(numInstPerDec*numDec,(param.patchSize/param.sBin)^2*31,'single');

        clear posData;
        count = 1;
        for ii=1:numDec
            decScores = zeros(numel(thisContentDetections{ii}),1);
            for jj=1:numel(thisContentDetections{ii})
                decScores(jj) = thisContentDetections{ii}(jj).decision;
            end
            [~,sortNdx] = sort(decScores,'descend');
            if numInstPerDec>numel(sortNdx)
                numInstPerDec=numel(sortNdx);
            end
            for jj=1:numInstPerDec
                imname = thisContentDetections{ii}(sortNdx(jj)).name;    
                I = imread(imname);            
                y = thisContentDetections{ii}(sortNdx(jj)).pos(1);
                x = thisContentDetections{ii}(sortNdx(jj)).pos(2);
                scale_ = thisContentDetections{ii}(sortNdx(jj)).scale;

                thisI = imresize(I,scale_); 
                if size(I,3)==1
                    thisI = cat(3,thisI,thisI,thisI);
                end

                % if we want coarser HoG (i.e., with bigger bin size), we may not be able to construct the feature
                try
                    thisI = thisI(y-param.sBin:y+param.patchSize-1+param.sBin,x-param.sBin:x+param.patchSize-1+param.sBin,:);
                catch
                    continue;
                end
                thisHoG = single(features(im2double(thisI),param.sBin));
                posFeats(count,:) =  vec(thisHoG(:,:,1:31))'; 

                posData(count).imname = imname;
                posData(count).scale = scale_;
                posData(count).y = y;
                posData(count).x = x;
                posData(count).decade = thisPosDecInd(ii);
                
                count = count + 1;           
            end
        end
        iterOut(num).dets = posData;
        num = num + 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if it == numel(iter)
            % if last iteration, append new pos features to existing ones
            origPosFeats = [origPosFeats; posFeats];
        else
            % otherwise, update model
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % train detector using background patches from natural images
            tic;
            % append orig patches 
            posFeats = [posFeats; origPosFeats];
            if negIt<=size(negIter,2)                
                [contentDetector] = hardNegativeMining_inputImgs(posFeats,perRoundImgs,initImgs,numRounds,param,negIter(negIt).negImgs);
                fprintf([num2str(toc) 's for ' num2str(numRounds) ' rounds of hard negative mining\n']);
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            model = contentDetector(end).model;
            negIt = negIt + 1;
        end        
    end
end