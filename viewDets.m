% clear;

addpath(genpath(pwd));

param = setParameters;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%view mined instances used to train detector

%loop over all clusters

% for ii=1:numel(param.decRange)
%     for jj=1:param.numClustersPerDecade        
%         load([param.detectordir '/models_' num2str(ii) '_' num2str(jj) '.mat']);
%             
%         figure(1); clf;
%         set(gcf,'Color',[1 1 1]);
%         nn = 1;
%         for kk=3:3:numel(iter)
%             for ll=1:numel(iter(kk).dets)
%                 I = imread(iter(kk).dets(ll).imname);
%                 I = imresize(I,iter(kk).dets(ll).scale);
%                 x = iter(kk).dets(ll).x;
%                 y = iter(kk).dets(ll).y;
%                 I = I(y:y+79,x:x+79,:);
%                 
%                 year = getYear(iter(kk).dets(ll).imname);
%                 subplot(5,8,nn); imshow(I); title(year);
%                 nn =  nn + 1;
%             end
%         end
%         pause;
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% view detections

% loop over all trained detectors
for mm=1:numel(param.decRange)
    for nn=1:param.numClustersPerDecade      

        load([param.detectordir '/trainDets_' num2str(mm) '_' num2str(nn) '.mat']);
%         load([param.detectordir '/testDets_' num2str(mm) '_' num2str(nn) '.mat']);

        % display detections in each decade
        for ii=1:numel(detections)
            thisDets = detections{ii};

            decScores = zeros(numel(thisDets),1);
            for jj=1:numel(thisDets)
                decScores(jj) = thisDets(jj).decision;
            end
            [sortVal,sortNdx] = sort(decScores,'descend');
%             numel(find(sortVal>0))    

            figure(2); clf;
            % display the top 49 detections in this decade
            for jj=1:9
                if sortVal(jj)>-1
                    I = imread(thisDets(sortNdx(jj)).name);
                    I = imresize(I,thisDets(sortNdx(jj)).scale);
                    x = thisDets(sortNdx(jj)).pos(2);
                    y = thisDets(sortNdx(jj)).pos(1);
                    I = I(y:y+param.patchSize-1,x:x+param.patchSize-1,:);
                    subplot(3,3,jj); imshow(I); %title(thisDets(sortNdx(jj)).decision);
                    if jj==1
                        title(['det(' num2str(mm) ',' num2str(nn) ') decade: ' num2str(ii)]);
                    end
                else
                    break;
                end
            end
            
            pause;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

