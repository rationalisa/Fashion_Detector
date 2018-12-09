function trainAndFireDetectors(param)

load([param.clusterdir 'decCluster.mat'],'decCluster');
for ii=1:size(decCluster,1)
    for jj=1:size(decCluster,2)
        if lock([param.detectordir 'testDets_' num2str(ii) '_' num2str(jj) '.mat'])==1
            continue;
        end
        
        initDetector = trainInitialDetector(decCluster{ii,jj},param);             
        save('-v7',[param.detectordir 'models_' num2str(ii) '_' num2str(jj) '.mat'],'initDetector');
        [finalDetector,iter] = retrainDetectorGradually(decCluster{ii,jj},initDetector(end).model,param);       
        save('-v7',[param.detectordir 'models_' num2str(ii) '_' num2str(jj) '.mat'],'finalDetector','iter');
        
        detections = runDetectorOnDataset(finalDetector(end).model,param,param.trainImages,param.trainimgdir);      
        save('-v7',[param.detectordir 'trainDets_' num2str(ii) '_' num2str(jj) '.mat'],'detections');
        
        detections = runDetectorOnDataset(finalDetector(end).model,param,param.testImages,param.testimgdir);      
        save('-v7',[param.detectordir 'testDets_' num2str(ii) '_' num2str(jj) '.mat'],'detections');
                
        unlock([param.detectordir 'testDets_' num2str(ii) '_' num2str(jj) '.mat']);
    end
end