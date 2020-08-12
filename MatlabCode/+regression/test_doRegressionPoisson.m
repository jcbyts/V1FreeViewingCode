% addpath ~/Dropbox/MatlabCode/Repos/ncclabcode/
addpath ~/Dropbox/MatlabCode/Repos/ncclabcode/nlfuns
% requires ncclabcode

% 1 bilinear covariate
ntb  = 15;
nsb  = 10;
% 1 linear covariate
nlin = 20;

wspace = exp(- ((1:nsb)-nsb/2).^2/15);
wtime  = cos( .5*(1:ntb));
wlin  = exp(-cos(1:nlin));
wst = wtime(:)*wspace(:)';

wtrue = [wlin(:); wst(:)]/10;

nk       = numel(wtrue);

nsamples = 500;
X = randn(nsamples, nk);
nlfun=@expfun;
lambda=nlfun(X*wtrue);

Y=poissrnd(lambda);
%% try linear regression
wridge=doRegression(X,Y,'RIDGE');
wsard=doRegression(X,Y,'SMOOTH ARD');

figure(1); clf
plot(1:nk, wtrue, 'k', 1:nk,wridge.khat, 1:nk, wsard.khat)
legend({'true', 'ridge', 'Smooth ARD'})

%% try poisson regression
clear dspec
dspec.model.regressionMode='RIDGE';
dspec.binSize=1;

%% Try autoregression
model=doRegressionPoisson(X,Y, dspec);

figure(1); clf
plot(1:nk, wtrue, 'k', 1:nk,wridge.khat, 1:nk, model.khat)
legend({'true', 'ridge', 'poisson ridge'})

%% Try bilinear
k=1;
dspec.covar(k).label='wlin';
dspec.covar(k).desc='wlin';
dspec.covar(k).edim=nlin;

for i=1:nsb
    k=k+1;
    dspec.covar(k).label=sprintf('space%d', i);
    dspec.covar(k).desc=sprintf('space%d', i);
    dspec.covar(k).edim=ntb;
end

dspec.edim=sum([dspec.covar.edim]);
dspec.model.bilinearMode='ON';
dspec.model.bilinearRank=1;
dspec.model.bilinearCovariate='space';

blin=doRegressionPoisson(X,Y, dspec);

figure(1); clf
plot(1:nk, wtrue, 'k', 1:nk,wridge.khat, 1:nk, model.khat, 1:nk, blin.khat)
legend({'true', 'ridge', 'poisson ridge', 'blinear'})

