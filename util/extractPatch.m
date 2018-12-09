function p = extractPatch(I,row,col,patchSize,isgray)

p = I(row:row+patchSize-1,col:col+patchSize-1,:);

if nargin<5
    isgray = 0;
end
if isgray==0 && size(p,3)==1
    p = cat(3,p,p,p);
end