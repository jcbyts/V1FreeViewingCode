function sp = clip_waveforms(Exp, S, varargin)
% sp = io.clip_waveforms(Exp, S, varargin)
% CLIP out waveforms from wideband signal

ip = inputParser();
ip.addParameter('Fs', 30000)
ip.addParameter('Win', -10:40)
ip.parse(varargin{:})

dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
spfname = fullfile(dataPath, 'spikes', sprintf('%s_%s.mat', strrep(Exp.FileTag, '.mat', ''), 'kilowf'));

if exist(spfname, 'file')
    sp = load(spfname);
    return
end
    
% load Ops
ops = io.loadOps(S.rawFilePath);

% load wideband signal (in muV)
[data, timestamps] = io.loadRaw(ops, [], true);

% highpass filter
Fs = ip.Results.Fs;
[b,a] = butter(6, (300/Fs)*2, 'high');

data = filter(b,a,data');

sp = Exp.osp;

cids = sp.cids;
win = ip.Results.Win; % window around spike to bin
if numel(win)==2
    win = win(1):win(2);
end

% preallocate
NC = numel(cids);
numChan = size(data,2);
numLags = numel(win);

WFmed = zeros(NC, numLags, numChan);
WFciHi = zeros(NC, numLags, numChan);
WFciLow = zeros(NC, numLags, numChan);

for cc = 1:NC % loop over units
fprintf('Unit %d/%d\n', cc, NC)
% for this unit: get all "clean spike times". These are times that do not
% collide with other neighboring units. We want Kilosort to include those
% because we care about simultaneous spikes, but we don't want them
% included in our waveform estimates
spikeIx = sp.clu==cids(cc); % this unit
spikeTimes = sp.st(spikeIx); % this unit spike times

% get rid of bursts of spikes (that can mess up our waveform estimate)
spikeTimes(diff(spikeTimes)<2e-3)=[]; % remove spikes within close succession

% find all other spike times from neighboring units
otherSpikes = ~spikeIx;
otherSpikeTimes = sp.st;
otherSpikeDepths = sp.spikeDepths(otherSpikes);
otherSpikeDistance = sqrt((otherSpikeDepths - sp.clusterDepths(cc)).^2);
closeSpikeTimes = otherSpikeTimes(otherSpikeDistance < 35); % spikes that are less than 35 microns away

% make histcount bins out of our units spike times
spikeTimesAsBins = [spikeTimes(:)'-1e-3; spikeTimes(:)'+1e-3];

% count the number of other spikes that occur within a 2ms window of our
% units spikes
cnt = histcounts(closeSpikeTimes, spikeTimesAsBins(:));
cnt = cnt(1:2:end);

% only take clean spike counts
spikeTimes = spikeTimes(cnt==0);

% bin spike counts at sample rate
[~, ~, id] = histcounts(spikeTimes, timestamps);

% get template
idx = id + win;

idx(any(idx < 1 | idx > size(data,1),2),:) = [];

numSpikes = size(idx,1);

for ch = 1:numChan
    x = reshape(data(idx,ch), size(idx));
    bnds = prctile(x, [16 50 84], 1);
    
    WFciLow(cc,:,ch) = bnds(1,:);
    WFmed(cc,:,ch) = bnds(2,:);
    WFciHi(cc,:,ch) = bnds(3,:);
end

end

sp.WFciLow = WFciLow;
sp.WFmed = WFmed;
sp.WFciHi = WFciHi;
sp.WFtax = win/Fs*1e3;

save(spfname, '-v7.3', '-struct', 'sp')