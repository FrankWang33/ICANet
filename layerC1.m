function layerC1(datadir,C1respdir,bases,stridecv,padcv,stridepl,ratio,sz,MinImSize,scparam)
if ~exist(C1respdir,'dir'), mkdir(C1respdir); end

% list all image files in the folder
filelist=dir( fullfile(datadir,'*.jpg') );

%bs = reshape(bases,[sz,sz,size(bases,1)/sz/sz,size(bases,2)])
parfor i=1:length(filelist);
    filename=filelist(i).name;
    
    X=imread(fullfile(datadir,filename));
    if size(X,3) == 3
        X = rgb2gray(X);
    end;
    % Transform to double precision
    X = single(X);
    
    X = imresize( X, MinImSize/min(size(X)) );
    
    % calculate S1 map
    matfile= regexprep(filename, '.jpg', '.mat');
    fullname=fullfile(C1respdir,matfile);
    S1map=layerS(X,bases,sz,scparam,stridecv,padcv);
    %S1map = vl_nnconv(X,bs,[],'stride',stridecv,'pad',[padcv,padcv,padcv,padcv]);
    %S1map(S1map<0) = 0;
    % calculate C1 map
    C = vl_nnpool(S1map,ratio,'stride',stridepl,'pad',[0,0,0,0],'method','max');
    savefile(fullname,C) 
end


function savefile(fullname,C)
save(fullname,'C');





