function pyramid = pyramid2Mat(pyramid,patchSize,normalize,contrastThresh)
% converts pyramid to a 2D matrix, removing low-contrast features

if nargin<4
    contrastThresh = 0.015;
end

w = patchSize/pyramid.sBin;
h = fspecial('gaussian', patchSize, patchSize/3);

%numel(pyramid.features)
numFeats = zeros(1,numel(pyramid.features));
for ii=1:numel(pyramid.features)
    [fr,fc,fd] = size(pyramid.features{ii});
    numFeats(ii) = (fr-w+1)*(fc-w+1);
end

featMat = zeros(w*w*fd,sum(numFeats),'single');
featPos = zeros(2,sum(numFeats),'uint16');
featScale = zeros(1,sum(numFeats),'single');

count = 1;
for ii=1:numel(pyramid.features)
    [fr,fc,~] = size(pyramid.features{ii});
    for nn=1:fr-w+1
        for mm=1:fc-w+1
            pG = extractPatch(pyramid.gradimg{ii},nn*pyramid.sBin+1,mm*pyramid.sBin+1,patchSize,1);
            % remove low-contrast patches
            tot = sum(sum(pG.*h));
            if tot<contrastThresh
                continue;
            end
            %vec(pyramid.features{ii}(nn:nn+w-1,mm:mm+w-1,1:31))
            size(featMat);
            featMat(:,count) = vec(pyramid.features{ii}(nn:nn+w-1,mm:mm+w-1,1:31));
            featPos(:,count) = uint16([nn mm]');
            featScale(count) = pyramid.scales(ii);
            count = count + 1;
        end
    end
end

if normalize==1
    X = bsxfun(@minus, featMat(:,1:count-1), mean(featMat(:,1:count-1),1));
    pyramid.featMat = bsxfun(@times, X, 1./sqrt(sum(X.*X,1)));
else
    pyramid.featMat = featMat(:,1:count-1);
end
pyramid.featPos = featPos(:,1:count-1);
pyramid.featScale = featScale(:,1:count-1);

