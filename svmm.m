% clear;

addpath(genpath(pwd));

param = setParameters;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% view detections

% loop over all trained detectors
% for mm=1:numel(param.decRange)
    mm=1;
    list=zeros(param.numClustersPerDecade,1);
    for nn=1:param.numClustersPerDecade      
        load([param.detectordir '/trainDets_' num2str(mm) '_' num2str(nn) '.mat']);
        
        % display detections in each decade

        count=30;

        fe1 = zeros(count*4000,(param.patchSize/param.sBin)^2*31,'single');
        
        thisDets1 = detections{1}; 
        thisdecScores1 = zeros(count,1);
        for jj=1:numel(thisDets1)
            thisdecScores1(jj) = thisDets1(jj).decision;
        end
                
                
        [sortVal,sortNdx] = sort(thisdecScores1,'descend');
        icount=1;
        for kk=1:count
                if sortVal(kk)>-1
                    I = imread(thisDets1(sortNdx(kk)).name);
                    pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
                    pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
    
                    M = size(pyramid.featMat,2);
                    fe1(icount:icount+M-1,:) = pyramid.featMat';

                    icount = icount + M;
                    
                    %decYear(kk+icount*(ii-1))= getYear(thisDets(sortNdx(kk)).name);
                    %decNum(kk+icount*(ii-1)) = ii;
                else
                    break;
                end
          end
          fe1=fe1(1:count-1,:);
            
          
        fe2 = zeros(count*4000,(param.patchSize/param.sBin)^2*31,'single');
        thisDets2 = detections{2}; 
        thisdecScores2 = zeros(count,1);
        for jj=1:numel(thisDets2)
            thisdecScores2(jj) = thisDets2(jj).decision;
        end
                        
        [sortVal,sortNdx] = sort(thisdecScores2,'descend');
        icount=1;
        for kk=1:count
                if sortVal(kk)>-1
                    I = imread(thisDets2(sortNdx(kk)).name);
                    pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
                    pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
    
                    M = size(pyramid.featMat,2);
                    fe2(icount:icount+M-1,:) = pyramid.featMat';

                    icount = icount + M;
                    
                    %decYear(kk+icount*(ii-1))= getYear(thisDets(sortNdx(kk)).name);
                    %decNum(kk+icount*(ii-1)) = ii;
                else
                    break;
                end
          end
          fe2=fe2(1:count-1,:);
          
        fe3 = zeros(count*4000,(param.patchSize/param.sBin)^2*31,'single');
        
        thisDets3 = detections{3}; 
        thisdecScores3 = zeros(count,1);
        for jj=1:numel(thisDets3)
            thisdecScores3(jj) = thisDets3(jj).decision;
        end
                
                
        [sortVal,sortNdx] = sort(thisdecScores3,'descend');
        icount=1;
        for kk=1:count
                if sortVal(kk)>-1
                    I = imread(thisDets3(sortNdx(kk)).name);
                    pyramid = getFeaturePyramid(im2double(I),param.scales,param.sBin);
                    pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
    
                    M = size(pyramid.featMat,2);
                    fe3(icount:icount+M-1,:) = pyramid.featMat';

                    icount = icount + M;
                    
                    %decYear(kk+icount*(ii-1))= getYear(thisDets(sortNdx(kk)).name);
                    %decNum(kk+icount*(ii-1)) = ii;
                else
                    break;
                end
          end
          fe3=fe3(1:count-1,:);
          
          trainFeats = double([fe1; fe2; fe3]);
          trainLabels = [ones(size(fe1,1),1); 2*ones(size(fe2,1),1);3*ones(size(fe3,1),1)];
          t = templateSVM('BoxConstraint',0.00001)
          model =  fitcecoc(trainFeats,trainLabels,'Learners',t); %, 's', 0, 't', 0, 'c', 0.1
            
        load([param.detectordir '/testDets_' num2str(mm) '_' num2str(nn) '.mat']);
        testcount=0;
        for ii=1:numel(detections)
            testcount=testcount+numel(detections{ii});
        end
        accuCount=0;
        for ii=1:numel(detections)
            thisDets = detections{ii}; 
            for jj=1:numel(thisDets)
                I = imread(thisDets(jj).name);
                pyramid = getFeaturePyramid(im2double(I),param.scalesDet,param.sBin);
                pyramid = pyramid2Mat(pyramid,param.patchSize,param.normalizeDet); 
 
                [label,score] = predict(model,pyramid.featMat');
                if(abs(mean(label)-ii)<=0.5)
                    accuCount=accuCount+1;
                end
            end
        end
        list(nn)=accuCount/testcount
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

