function [S, opts] = running_vs_spikePC(D, opts)
% [S, opts] = running_vs_spikePC(D, opts)
% plot spiking PC binned by trials -- this way we can ignore smoothing and get rid of nans

defopts = struct();
defopts.prewin = .1;
defopts.postwin = .1;
defopts.spike_rate_thresh = 1;
defopts.ndim = 5;
defopts.plot = true;

if nargin < 2
    opts = struct();
end

opts = mergeStruct(defopts, opts);

onsets = D.GratingOnsets-opts.prewin;
tstart = min(onsets);
opts.winsize = mode(D.GratingOffsets - D.GratingOnsets) + opts.prewin + opts.postwin;

% get running speed during grating
treadgood = find(~isnan(D.treadTime));
[~, ~, id1] = histcounts(D.GratingOnsets, D.treadTime(treadgood));
[~, ~, id2] = histcounts(D.GratingOffsets, D.treadTime(treadgood));
bad = (id1 == 0 | id2 == 0);
runningspeed = nan(numel(D.GratingOnsets), 1);
id1 = treadgood(id1(~bad));
id2 = treadgood(id2(~bad));
runningspeed(~bad) = arrayfun(@(x,y) nanmean(D.treadSpeed(x:y)), id1, id2);

onsets = onsets - tstart;
st = D.spikeTimes - tstart;
iix = st > min(onsets); %#ok<*NANMIN> 

cids = unique(D.spikeIds);
if isfield(D, 'unit_area')
    cids = cids(strcmp(D.unit_area, 'VISp'));
    iix = iix & ismember(D.spikeIds, cids);
end

r = binNeuronSpikeTimesFast(struct('st', st(iix), 'clu', D.spikeIds(iix)+1), onsets, opts.winsize);
clist = unique(D.spikeIds(iix)+1);
r = r(:,clist);
cids = mean(r) ./ opts.winsize > opts.spike_rate_thresh;
r = r(:,cids);

NC = size(r,2);
S.cids = clist(cids)-1;
S.robs = r;

% pca on spikes
C = cov(r);
[u,s] = svd(C);

rpc = r*u(:,1:opts.ndim).*sign(sum(u(:,1:opts.ndim)));
rpc = (rpc - min(rpc)) ./ range(rpc); % normalize

iix = ~isnan(runningspeed);
[rho, pval] = corr(rpc(iix,1), runningspeed(iix), 'Type', 'Spearman');

S.rpc = rpc;
S.runningspeed = runningspeed;
S.rho = rho;
S.pval = pval;
S.rhounit = nan(NC,1);
S.pvalunit = nan(NC,1);

for cc = 1:NC
    [rho, pval] = corr(r(iix,cc), runningspeed(iix), 'Type', 'Spearman');
    S.rhounit(cc) = rho;
    S.pvalunit(cc) = pval;
end

if opts.plot
    figure(1); clf
    plot(rpc + (0:(opts.ndim-1)), 'k')
    hold on
    plot(runningspeed/max(runningspeed) - 1)
    xlabel('Trials')
end