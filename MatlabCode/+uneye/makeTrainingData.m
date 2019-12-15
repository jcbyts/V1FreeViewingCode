

dataDir = '~/Dropbox/Projects/FreeViewing/Data';
fname = fullfile(dataDir, 'L20191205');

Exp = load(fullfile([fname '.mat']));

X = Exp.vpx.smo(:,2);
Y = Exp.vpx.smo(:,3);
L = Exp.vpx.Labels;
t = Exp.vpx.smo(:,1);
ix = (t < 417);

X = X(ix);
Y = Y(ix);
L = L(ix);

nTrials = 800;


remainder = mod(numel(X),nTrials);
padsize = nTrials - remainder;


X = [X; nan(padsize,1)];
X = reshape(X, [], nTrials)';
csvwrite(fullfile(dataDir, 'Xtrain.csv'), X)

Y = [Y; nan(padsize,1)];
Y = reshape(Y, [], nTrials)';
csvwrite(fullfile(dataDir, 'Ytrain.csv'), Y)

if min(L)==1
    L = L - 1; % for python
end

L = [L; 3*ones(padsize,1)];
L = reshape(L, [], nTrials)';
csvwrite(fullfile(dataDir, 'Ltrain.csv'), L)