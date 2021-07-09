function [mae, stimhat, yhat, wts, gfun] = von_mises_ori_decoder(R, stim, iitrain, iitest, varargin)
% [mae, yhat, wts, gfun] = von_mises_decoder(R, stim, iitrain, iitest, varargin)

ip = inputParser();
ip.addParameter('nBasis', numel(unique(stim)))
ip.addParameter('Beta', [])
ip.addParameter('lambda', 1e2)
ip.parse(varargin{:});

nneur = size(R,2);
ntrials = size(R,1);

nBasis = ip.Results.nBasis;

% create von mises basis
gfun = decoding.vm_ori_basis_deg(nBasis, ip.Results.Beta, false);

% stimulus evaluated on basis
ss = gfun(stim);
ss = zscore(ss, [], 2);

rrd = [ones(ntrials,1) R]; % design matrix

nd = nneur + 1; % dimensionality (# neurons +1 )

lam = ip.Results.lambda; % ridge regression param

% fast ridge regression
wts = (rrd(iitrain,:)'*rrd(iitrain,:) + lam*speye(nd))\(rrd(iitrain,:)'*ss(iitrain,:)); % linear regression

xd = 0:179; % basis will be interpolated over these values
B = gfun(xd);

yhat = (rrd(iitest,:)*wts)*B'; % prediction on interpolated basis

circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;
[~, stimhat] = max(yhat,[],2);
stimhat = xd(stimhat);
mae = median(abs(circdiff(stimhat(:), stim(iitest))));
