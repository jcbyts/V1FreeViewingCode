function [Y, binfun] = binNeuronSpikeTimesFast(sp, eventTimes, binSize)
% BIN SPIKE TIMES THE FASTEST WAY POSSIBLE IN MATLAB
% 
% Inputs:
%   sp [struct]: Kilosort output struct
%   has fields:
%       st [T x 1]: spike times
%       clu [T x 1]: unit id
% Outpus:
%   Y [nBins x nNeurons]
%
% Example Call:
%   Y = binNeuronSpikeTimesFast(Exp.osp, eventTimes, binsize)

NT = numel(eventTimes);
NC = max(sp.cids);

eventTimes = [eventTimes(:); eventTimes(end) + binSize];
Y = zeros(NT, NC);
for cc = sp.cids(:)'
    cnt = histcounts(sp.st(sp.clu==cc), eventTimes);
    cnt(diff(eventTimes) > .1) = 0;
    Y(:,cc) = cnt;
end

% conversion from time to bins
binfun = @(x) (x==0) + ceil(x / binSize);
return

% bin spike times
% bst = binfun(sp.st);
% 
% % bin frame times
% bft = binfun(eventTimes);
% 
% % create binned spike times
% Y = sparse(bst, double(sp.clu), ones(numel(bst), 1), max(max(bst),max(bft)),double(max(sp.clu)));
% 
% % index in with binned frame times
% Y = Y(bft,:);