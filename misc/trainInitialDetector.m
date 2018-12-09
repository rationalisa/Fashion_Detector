function [contentDetector] = trainInitialDetector(contentCluster,param)
     
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get features from cluster patches
posFeats = zeros(numel(contentCluster),(param.patchSize/param.sBin)^2*31,'single');
count = 1;
for ii=1:numel(contentCluster)
    % don't consider as positive, an outlier cluster patch
    if isempty(find(negYears==contentCluster(ii).year,1))
        imname = contentCluster(ii).imname;   
        I = imread(imname);            
        [nRow,nCol,nDim] = size(I);
        y = contentCluster(ii).y -param.sBin; % since initPatchSize is smaller (by 2*sBin) than patchSize
        x = contentCluster(ii).x -param.sBin;
        thisI = imresize(I,contentCluster(ii).scale); 
        if nDim==1
            thisI = cat(3,thisI,thisI,thisI);
        end
        % if we want coarser HoG (i.e., with bigger bin size), we may not be able to construct the feature
        if x<=param.sBin
            z=param.sBin-x+1;
            x=x+z;
        end
        if y<=param.sBin
            z=param.sBin-y+1;
            y=y+z;
        end
        if y+param.patchSize-1+param.sBin>size(thisI,1)
            y=size(thisI,1)-param.sBin-param.patchSize
        end
        if x+param.patchSize-1+param.sBin>size(thisI,2)
            x=size(thisI,2)-param.sBin-param.patchSize
        end
        % try
        thisI = thisI(y-param.sBin:y+param.patchSize-1+param.sBin,x-param.sBin:x+param.patchSize-1+param.sBin,:);
        %catch
        %    fprintf('error')
        %    continue;
       % end
        thisHoG = single(features(im2double(thisI),param.sBin));
        posFeats(count,:) =  vec(thisHoG(:,:,1:31))'; 
        count = count + 1;
    end
end
posFeats(count:end,:) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train detector using background patches from natural images
perRoundImgs = 255;
initImgs = 10;
numRounds = 2;
tic;
size(posFeats)
[contentDetector] = hardNegativeMining(posFeats,perRoundImgs,initImgs,numRounds,param);
fprintf([num2str(toc) 's for ' num2str(numRounds) ' rounds of hard negative mining\n']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
