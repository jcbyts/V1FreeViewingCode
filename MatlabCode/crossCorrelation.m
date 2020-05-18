function [xc, lags, err] = crossCorrelation(sptimes1, varargin)
% [xc, lags, err] = crossCorrelation(sptimes1, varargin)

args = varargin;
if nargin > 1 && isnumeric(varargin{1})
    sptimes2 = varargin{1};
    args(1) = [];
    isauto = false;
else
    sptimes2 = sptimes1;
    isauto = true;
end

ip = inputParser();
ip.addParameter('binSize', 1e-3)
ip.addParameter('numLags', 200)
ip.parse(args{:});

binSize = ip.Results.binSize;
numLags = ip.Results.numLags;

binEdges = min(min(sptimes1),min(sptimes2)):binSize:max(max(sptimes1), max(sptimes2));

% bin spike times
[spcnt, ~, id] = histcounts(sptimes1, binEdges);
if ~isauto
    [~, ~, id] = histcounts(sptimes2, binEdges);
end

% get cross-correlation in units of excess firing rate
lags = -numLags:numLags;
ix = id + lags;
ix(any(ix<1 | ix > numel(spcnt),2),:) = [];
I = spcnt(ix);
if isauto
    I(:,lags==0)= 0; % remove zero-lag spike
end

mu = mean(I);
mu0 = mean(spcnt);

% binomial confidence intervals
n = size(I,1);
binoerr = 2*sqrt( (mu - mu.^2)/n);

xc = (mu - mu0) / binSize; % in excess spikes/sec

% push error through same nonlinearity ( baseline subtract / binsize)
err = ((mu + binoerr) - mu0)/binSize - xc;