function [iter] = hardNegativeMining(posFeats,perRoundImgs,initImgs,numRounds,param)
% hard negative mining

d = dir([param.bgDir '*.jpg']);

negFeats = zeros(initImgs*4000,(param.patchSize/param.sBin)^2*31,'single');
count = 1;
for ii=1:initImgs
    imname = [param.bgDir d(ii).name];
    I = imread(imname); 

    pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
    pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
    
    M = size(pyramid.featMat,2);
    negFeats(count:count+M-1,:) = pyramid.featMat';

    count = count + M;
end
negFeats = negFeats(1:count-1,:);

% train initial linear svm model
trainFeats = double([posFeats; negFeats]);
trainLabels = [ones(size(posFeats,1),1); -1*ones(size(negFeats,1),1)];
%size(posFeats)
%size(negFeats)
model = fitcsvm(trainFeats, trainLabels); %, 's', 0, 't', 0, 'c', 0.1
iter(1).model = model;
iter(1).count = count;
% model.SupportVectors
y = model.SupportVectorLabels;
hardNegFeats = single(full(model.SupportVectors(find(y==-1),:)));
% size(model.SupportVectors)
% size(model.SupportVectors(find(y==-1),:))

% hard negative mining iterations
fprintf(['hard negative mining...\n']);
for ii=1:numRounds
    % test linear svm    
    thishardNegFeats = zeros(initImgs*4000,(param.patchSize/param.sBin)^2*31,'single');
    count = 1;
    for jj=1:perRoundImgs
        imname = [param.bgDir d(initImgs+(ii-1)*perRoundImgs+jj).name];
        I = imread(imname);

        thisNegFeats = mineHardNegInImg(I,model,param);
        M = size(thisNegFeats,2);
        thishardNegFeats(count:count+M-1,:) = thisNegFeats';
        count = count + M;
    end
    thishardNegFeats = thishardNegFeats(1:count-1,:);
    hardNegFeats = [hardNegFeats; thishardNegFeats];
    
    % train linear svm
    trainFeats = double([posFeats; hardNegFeats]);
    trainLabels = [ones(size(posFeats,1),1); -1*ones(size(hardNegFeats,1),1)];
    model = fitcsvm(trainFeats, trainLabels); % '-s 0 -t 0 -c 0.1'
    iter(ii+1).model = model;
    iter(ii+1).count = count;
    fprintf(['done with round ' num2str(ii) ' of ' num2str(numRounds) '\n']);
end
