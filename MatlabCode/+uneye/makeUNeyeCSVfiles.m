function makeUNeyeCSVfiles(Exp, dataDir)
% saves out csv files for training with UNeye

if nargin < 2
    dataDir = '~/Dropbox/Projects/FreeViewing/Data/uneye';
end

fname = strrep(Exp.FileTag, '.mat', '');

X = Exp.vpx.smo(:,2);
Y = Exp.vpx.smo(:,3);
L = Exp.vpx.Labels;

nTrials = 800;

remainder = mod(numel(X),nTrials);
padsize = nTrials - remainder;


X = [X; nan(padsize,1)];
X = reshape(X, [], nTrials)';
csvwrite(fullfile(dataDir, [fname 'Xmat.csv']), X)

Y = [Y; nan(padsize,1)];
Y = reshape(Y, [], nTrials)';
csvwrite(fullfile(dataDir, [fname 'Ymat.csv']), Y)

if min(L)==1
    L = L - 1; % for python
end

L = [L; 3*ones(padsize,1)];
L = reshape(L, [], nTrials)';
csvwrite(fullfile(dataDir, [fname 'Lmat.csv']), L)