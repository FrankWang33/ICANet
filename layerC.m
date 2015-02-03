function layerC(C1respdir,C2respdir,bases,stridecv,padcv,stridepl,ratio,sz,scparam)
if ~exist(C2respdir,'dir'), mkdir(C2respdir); end

% list all mat files in the folder
filelist=dir( fullfile(C1respdir,'*.mat') );
%basenum = size(bases,2);
%bs = reshape(bases,[sz,sz,size(bases,1)/sz/sz,basenum]);
for i=1:length(filelist)
    filename=filelist(i).name;
    fullname=fullfile(C2respdir,filename);
    fullname1=fullfile(C1respdir,filename);
    temp=load(fullname1, 'C');
    % calculate S2 map
    Smap=layerS(temp.C,bases,sz,scparam,stridecv,padcv);
    %Smap = vl_nnconv(temp.C,bs,[],'stride',stridecv,'pad',[padcv,padcv,padcv,padcv]);
    %Smap(Smap<0) = 0;
    % calculate C2 map
    %C=Smap_pooling(Smap, ratio, ratio, mod(size(Smap,1),ratio),mod(size(Smap,2),ratio), 'max');
    if ratio == 1
        C = Smap;
    else
        C = vl_nnpool(Smap,ratio,'stride',stridepl,'pad',[0,0,0,0],'method','max');
    end
    
    savefile(fullname,C);
end

function savefile(fullname,C)
save(fullname,'C');