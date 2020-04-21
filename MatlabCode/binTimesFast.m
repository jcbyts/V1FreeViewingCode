function [Y, binfun] = binTimesFast(st, eventTimes, binSize)
% BIN TIMES THE FASTEST WAY POSSIBLE IN MATLAB
% 
% Inputs:
%   st [T x 1]: spike times
%   eventTimes [N x 1]: event times
%   binSize [1 x 1]: size of bin
% Outpus:
%   Y [nBins x nNeurons]
%
% Example Call:
%   Y = binNeuronSpikeTimesFast(Exp.osp, eventTimes, binsize)

% conversion from time to bins
binfun = @(x) (x==0) + ceil(x / binSize);

% bin spike times
bst = binfun(st);
bst(bst<1) = [];

% bin frame times
bft = binfun(eventTimes);

% create binned spike times
Y = sparse(bst, ones(numel(bst),1), ones(numel(bst), 1), max(max(bst),max(bft)),1);
% Y = sparse(bst, ones(numel(bst), 1), ones(numel(bst), 1));

% index in with binned frame times
Y = Y(bft,:);