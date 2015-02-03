function X= sampleimages(datadir,samples, winsize,imgchl)
% INPUT variables:
% samples            total number of patches to take
% winsize            patch width in pixels
%
% OUTPUT variables:
% X                  the image patches as column vectors

% list all image files in the folder
filelist=dir(fullfile(datadir,'*.jpg') );

% Number of images
dataNum = length(filelist);
getsample = round(samples/dataNum);
% Initialize the matrix to hold the patches
X = cell(1,dataNum);
parfor i=(1:dataNum)
    
    % Load the image.
    I = imread(fullfile(datadir,filelist(i).name));    
    if size(I,3) == 3
        I = rgb2gray(I);
    end;
    
    % Transform to double precision
    I = single(I);
 
    % Sample patches in random locations
    sizex = size(I,2);
    sizey = size(I,1);
    posx = floor(rand(1,getsample)*(sizex-winsize-1))+1;
    posy = floor(rand(1,getsample)*(sizey-winsize-1))+1;
    if imgchl == 1
        tmpt = zeros(winsize^2,getsample,'single');
    else
        tmpt = zeros(winsize^2*imgchl,getsample,'single');
    end
    for j=1:getsample
        %fprintf(['i=' num2str(i) ' j=' num2str(j) '\n']);
        tmpt(:,j) = reshape( I(posy(1,j):posy(1,j)+winsize-1, ...
            posx(1,j):posx(1,j)+winsize-1,:),[],1);
    end
    X{i} = patch_normalize(tmpt);
end
X = cell2mat(X);




