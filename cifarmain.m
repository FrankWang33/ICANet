clear;
addpath('spams-matlab\build');
global CORENUM
CORENUM = 8;
%Initialize Matlab Parallel Computing Environment
if isempty(gcp('nocreate'))
    parpool('local',CORENUM);
else
    disp('Parallel computing mode already initialized'); 
end
datadir = 'data/cifar10';

layer = 3;
imgchannel = 3;
samplesz = [600000,600000,600000];
patchsz = [5,5,5];
conv_stride = [1,1,1];
conv_padding = [2,2,2];
nbases = [32,32,64];
ratio = [3,3,3];
pool_stride = [2,2,2];
skip_layer = [false,false,false,false,false,false];

scparam.lambda=0.15;
scparam.m = 600000;
scparam.iter=200;
scparam.epsilon = 1e-5;
batchsz = 100;

param = cell(1,layer);
for k = 1:layer
    info.samplesz = samplesz(k);
    info.patchsz = patchsz(k);
    info.conv_stride = conv_stride(k);
    info.conv_padding = conv_padding(k);
    info.nbases = nbases(k);
    info.ratio = ratio(k);
    info.pool_stride = pool_stride(k);
    info.scparam = scparam;
    info.scparam.numFeatures = info.nbases;
    if k==1
        info.datadir = datadir;
        info.scparam.n = info.patchsz^2*3;
    else
        info.datadir = param{k-1}.Crespdir;
        info.scparam.n = info.patchsz^2 * param{k-1}.nbases;
    end
    info.Crespdir = ['respdir/cifar10/C' num2str(k)];
    info.Afile = ['Results/S' num2str(k) 'bases' ...
        '_nBases_' num2str(info.nbases) '_patchsz_' num2str(info.patchsz) '.mat'];
    param{k} = info;
end

for k = 1:layer
    if skip_layer(2*k-1)
        load(param{k}.Afile,'A');
    else
        fprintf('-----------------------------------\n\n');
        fprintf(['Learning S' num2str(k) 'bases...\n']);
        fprintf('-----------------------------------\n');

        fprintf('Sampling & Normalizing data...\n')
        tic;
        if k==1
            X = sampleimages(param{k}.datadir,param{k}.samplesz,param{k}.patchsz,imgchannel);
        else
            X = sample3D(param{k}.datadir,param{k}.samplesz,param{k}.patchsz,param{k-1}.nbases);
        end
        toc;
        fprintf('Starting SC...\n')
        tic;
        Theta = randn(param{k}.scparam.numFeatures,param{k}.scparam.n)*0.01;  % 1/sqrt(params.n);
        Theta = Theta ./ repmat(sqrt(sum(Theta.^2,2)), 1, size(Theta,2)); 
        Theta = Theta(:);
        for j=1:param{k}.scparam.iter
            for l=1:param{k}.scparam.m/batchsz
                [~, grad] = softICACost(Theta,X(:,(l-1)*batchsz+1:l*batchsz),param{k}.scparam);
                Theta = Theta - 0.01*grad;
            end
        end
        toc;
        A = reshape(Theta,param{k}.scparam.numFeatures,param{k}.scparam.n)';
        save(param{k}.Afile,'A'); 
    end
    
    if ~skip_layer(2*k)
        fprintf('\n-----------------------------------\n');
        fprintf(['Calculating C' num2str(k) '  layer response...\n']);
        fprintf('-----------------------------------\n');
        load(param{k}.Afile);
        tic;
        if k==1
            layerC1(param{k}.datadir,param{k}.Crespdir,A,param{k}.conv_stride,param{k}.conv_padding,param{k}.pool_stride,param{k}.ratio,param{k}.patchsz);
        else
            layerC(param{k}.datadir,param{k}.Crespdir,A,param{k}.conv_stride,param{k}.conv_padding,param{k}.pool_stride,param{k}.ratio,param{k}.patchsz);
        end
        toc;
    end
end
delete(gcp('nocreate'));
