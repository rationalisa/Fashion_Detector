function feat = sampleRandomPatchesInImage(pyramid,patchSize,numPatchesPerScale)

if nargin<5
    contrastThresh = 0.015;
end
h = fspecial('gaussian', patchSize, patchSize/3);

count = 1;
for ii=1:numel(pyramid.scales)
    [nr,nc,z] = size(pyramid.features{ii});
    coords = ceil(bsxfun(@times,[nr-patchSize/pyramid.sBin+1 nc-patchSize/pyramid.sBin+1],rand(numPatchesPerScale,2)));
    coords = unique(coords,'rows');
    
    [r,c] = find(coords<=0);
    coords(r,:) = [];
    
    for jj=1:size(coords,1)
        coord = coords(jj,:);
        
        % remove low-contrast patches
        pG = extractPatch(pyramid.gradimg{ii},coord(1)*pyramid.sBin,coord(2)*pyramid.sBin,patchSize,1);
        tot = sum(sum(pG.*h));
        if tot<contrastThresh              
            continue;
        end

        rows = coord(1):coord(1)+patchSize/pyramid.sBin-1;
        cols = coord(2):coord(2)+patchSize/pyramid.sBin-1;

        % sampled patch may not fit in image
        try
            %class(pyramid.features{ii})
            feat(count).hog = vec(pyramid.features{ii}(rows,cols,1:31));
            feat(count).coord = uint16(coord);
            feat(count).scale = single(pyramid.scales(ii));

            count = count + 1;
        catch
            continue;
        end
    end
end