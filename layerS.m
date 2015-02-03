function S=layerS(X,bases,patchsz,scparam,stridecv,padcv)
%% Concatinate the features
[row,col,num_bases]=size(X);
tmpt = zeros(row+2*padcv,col+2*padcv,num_bases);
tmpt(padcv+1:row+padcv,padcv+1:col+padcv,:) = X;
X = tmpt;
rowsel = [1:stridecv:size(X,1)-patchsz+1]';
colsel = 1:stridecv:size(X,2)-patchsz+1;
rs = reshape(repmat(rowsel,1,size(colsel,2)),[],1)';
cs = reshape(repmat(colsel,size(rowsel,1),1),1,[]);
mask = sub2ind([size(X,1)-patchsz+1,size(X,2)-patchsz+1],rs,cs);
descrs = zeros(patchsz^2*num_bases,size(mask,2),'single');
for i=1:num_bases
     tmp = im2col(X(:,:,i),[patchsz,patchsz],'sliding');
     descrs((i-1)*patchsz^2+1: i*patchsz^2, :) = tmp(:,mask);
end

%% Normalize the features
descrs=patch_normalize(descrs);

%% Project to the principal components space 

%% calculate S1 map

S = descrs'*bases;
%S = mexLasso(descrs,bases,scparam)';
%sparsity_S = sum(S(:)~=0)/length(S(:));
%fprintf('sparsity_S = %g\n', full(sparsity_S));    

% convert back to 3D arrays
S=reshape(single(full(S)),size(rowsel,1),size(colsel,2),size(bases,2));

