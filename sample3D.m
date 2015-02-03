function X = sample3D(datadir,samples, winsize,numbases)

% This is how many patches to take per map
database = dir(fullfile(datadir,'*.mat'));
num_files = length(database);
getsample = round(samples/num_files);

% Initialize the matrix to hold the patches
X = cell(1,num_files);
parfor i=1:num_files
  % Load the map.
  tmp=loadfile(fullfile(datadir,database(i).name));
  C = tmp.C;
  % extract patches at random from C map to make data vector X
  [rowsz,colsz]=size(C(:,:,1));  
  
  tt = zeros(winsize^2*numbases,getsample,'single');
  for j=1:getsample
    r=ceil((rowsz-winsize+1)*rand);
    c=ceil((colsz-winsize+1)*rand);
    tt(:,j) = reshape(C(r:r+winsize-1,c:c+winsize-1,:),...
        numbases*winsize^2,1);
  end
  X{i} = tt;
end
X = cell2mat(X);

function C = loadfile(filename)
C=load(filename,'C');


