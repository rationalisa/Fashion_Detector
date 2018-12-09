function [trainImages,testImages] = processData(param)
% process and store image info

%%%%%%%%%%%%%%%%%%
% training set
d = dir([param.trainimgdir '*.jpg']);
d = d(1:param.numTrainImages);
decs = zeros(numel(d),1);
years = zeros(numel(d),1);
for ii=1:numel(d)
    years(ii) = getYear([param.trainimgdir d(ii).name]);
    decs(ii) = ceil((years(ii)-param.decRange(1))/8);   
end
decs(decs==0) = 1;

clear trainImages;
for ii=1:numel(param.decRange)
    ndx = find(decs==ii);
    for jj=1:numel(ndx)
        trainImages{ii}(jj,1).name = d(ndx(jj)).name;
        trainImages{ii}(jj,1).year = years(ndx(jj));
    end
end
%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%
% testing set
d = dir([param.testimgdir '*.jpg']);

decs = zeros(numel(d),1);
years = zeros(numel(d),1);
for ii=1:numel(d)
    years(ii) = getYear([param.testimgdir d(ii).name]);
    decs(ii) = ceil((years(ii)-param.decRange(1))/8);         
end
decs(decs==0) = 1;

clear testImages;
for ii=1:numel(param.decRange)
    ndx = find(decs==ii);
    for jj=1:numel(ndx)
        testImages{ii}(jj,1).name = d(ndx(jj)).name;
        testImages{ii}(jj,1).year = years(ndx(jj));
    end
end
%%%%%%%%%%%%%%%%%%


