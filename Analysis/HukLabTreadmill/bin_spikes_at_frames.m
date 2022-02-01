function [Robs, cids] = bin_spikes_at_frames(Stim, D, spike_rate_thresh, trialidx)
% [Robs, cids] = bin_spikes_at_frames(Stim, D, spike_rate_thresh (optional), trialidx (optional))

if nargin < 4 || isempty(trialidx)
    trialidx = 1:numel(Stim.grating_onsets);
end

if nargin < 3 || isempty(spike_rate_thresh)
    spike_rate_thresh = 1;
end

bin_times = reshape(Stim.trial_time(:) + Stim.grating_onsets(trialidx)', [], 1);
tstart = bin_times(1)-.1;

bin_times = bin_times - tstart;
[bsorted, ind] = sort(bin_times);

% if ~isfield(D, 'units')
%     rfcids = [];
% else
%     rfcids = find(arrayfun(@(x) x.maxV > 10 & x.area > .5 , D.units));
% end


spikeTimes = D.spikeTimes - tstart;
spikeIds = D.spikeIds + 1;
cids = unique(spikeIds);
if isfield(D, 'unit_area')
    cids = intersect(cids, find(strcmp(D.unit_area, 'VISp')));
end

% rfcids = cids(rfcids);

iix = ismember(spikeIds, cids);
iix = spikeTimes > min(bin_times(:)) & iix;
sp = struct('st', spikeTimes(iix), 'clu', spikeIds(iix), 'cids', cids);

Robs = binNeuronSpikeTimesFast(sp, bsorted, Stim.bin_size*1.05);

rate_cids = find((mean(Robs) / Stim.bin_size) > spike_rate_thresh);

cids = rate_cids; %, rfcids);
NC = numel(cids);
fprintf('%d units meet the firing rate or RF criterion \n', NC)

Robs = Robs(:, cids);
Robs = Robs(ind,:);
