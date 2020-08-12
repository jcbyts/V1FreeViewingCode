

%% load data
Exp = io.dataFactoryGratingSubspace(5);

osp = Exp.osp;

%% plot basis
binSize = .25e-3; % 1/4 ms bins

t = (0:binSize*1e3:200)';
nlags = numel(t);
numBasis = 14;
shortestLag = .5;
nlStretch = 1.6; % 2 equals doubling
B = raised_cosine(t, numBasis, shortestLag, nlStretch);
B(1:2,1) = 1;
% B = orth(B);

figure(1); clf
subplot(131)
plot(t, B)
xlim([1 10])
subplot(132)
imagesc(B)
subplot(133)
plot(t, sum(B,2))
ylim([0 1.5])
%% one unit


binEdges = min(osp.st):binSize:max(osp.st);

% cc = 1;
cc = cc + 1;

cids = osp.cids; % cluster ids
if cc > numel(cids)
    cc = 1;
end

% bin spike counts
sptimes = osp.st(osp.clu==cids(cc));
[spcnt, ~, id] = histcounts(sptimes, binEdges);

NT = numel(spcnt);
Xd = conv2(spcnt(:), B, 'full');
Xd = Xd(1:NT,:);

% offset by one bin (so it doesn't perfectly predict itself)
n = 2;
y = [spcnt(n:end)'; zeros(n-1,1)];

Xd = [ones(NT, 1) Xd];

disp('fitting')
% tic
% w = glmfit(Xd, y, 'poisson');
% toc


tic
lambda = 1e-3; % ridge parameter
C = blkdiag(0, lambda*eye(size(Xd,2)-1));
w0 = (Xd'*Xd + C)\(Xd'*y); % ridge regression

w = w0;
nlfun = @(x) x;


% % 
% % poisson maximum likelihood
% % nlfun = @nlfuns.expfun;
% fun = @(w) regression.neglogli_bernoulli(w, Xd, y);%, @nlfuns.expfun, binSize);
% opts = optimset('GradObj', 'on', 'Hessian', 'off', 'Display', 'iter', 'UseParallel', true);
% w = fminunc(fun, w0, opts);
% toc
toc

% get autocorrelation in units of excess firing rate
[xc, lags] = xcorr(y, nlags);
xc = xc / sum(y) / binSize;
ix = lags > 0;
xc = xc(ix);
lags = lags(ix) * binSize * 1e3;

figure(1); clf
plot(lags, xc); hold on

h = nlfun(B*w(2:end)+w(1));
plot(t, h/binSize)
% xlim([0 50])


%%
figure(1); clf

plot(nlfun(Xd*w), y, '.')

