% clear;

addpath(genpath(pwd));

param = setParameters;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% view detections

% loop over all trained detectors
%for mm=1:numel(param.decRange)
%    for nn=1:param.numClustersPerDecade      
        nn=1;
        mm=1;
        load([param.detectordir '/trainDets_' num2str(mm) '_' num2str(nn) '.mat']);
        
        % display detections in each decade

        count=30;

        decNum= zeros(count*3,1);
        decYear= zeros(count*3,1);
        fe = zeros(initImgs*4000,(param.patchSize/param.sBin)^2*31,'single');
        for ii=1:numel(detections)
            thisDets = detections{ii}; 
            thisdecScores = zeros(count,1);
            %feats = zeros(numel(thisDets),)
            %Feats = zeros(numel(thisDets)*4000,(param.patchSize/param.sBin)^2*31);
            for jj=1:numel(thisDets)
                thisdecScores(jj) = thisDets(jj).decision;
            end
                
                
            [sortVal,sortNdx] = sort(thisdecScores,'descend');
            icount=1;
            for kk=1:count
                if sortVal(kk)>-1
                    I = imread(thisDets(sortNdx(kk)).name);
                    pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
                    pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
    
                    M = size(pyramid.featMat,2);
                    fe(icount:icount+M-1,:) = pyramid.featMat';

                    icount = icount + M;
                    
                    decYear(kk+icount*(ii-1))= getYear(thisDets(sortNdx(kk)).name);
                    decNum(kk+icount*(ii-1)) = ii;
                else
                    break;
                end
            end
            fe=fe(1:count-1,:);
            
                % I = imread(thisDets(jj).name);
                % feats=single(features(im2double(I),param.sBin));
                % size(feats)
                
                %M = size(pyramid.featMat,2);
                %negFeats(count:count+M-1,:) = pyramid.featMat';

                %count = count + M;
        end 
            %pause;
    %end
    
    fprintf(['training linear svm...\n']);
    trainFeats = fe;
    trainLabels = decNum;
    model = fitcsvm(trainFeats, trainLabels);
    
    %for nn=1:param.numClustersPerDecade 
        load([param.detectordir '/testDets_' num2str(mm) '_' num2str(nn) '.mat']);
        testcount=0;
        for ii=1:numel(detections)
            testcount=testcount+numel(detections{ii});
        end
        testdecNum= zeros(testcount,1);
        testdecYear= zeros(testcount,1);
        testfe=cell(testcount,1);
        preC=0;
        
        for ii=1:numel(detections)
            thisDets = detections{ii}; 
            for jj=1:numel(thisDets)
                % thisdecScores(jj) = thisDets(jj).decision;
                I = imread(thisDets(jj).name);
                pyramid = getFeaturePyramid(im2double(I),param.scalesDet,param.sBin);
                testfe{jj+preC}=pyramid.features;
                testdecYear(jj+preC)= getYear(thisDets(jj).name);
                testdecNum(jj+preC) = ii;
            end
            preC=preC+numel(thisDets)
        end

%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

