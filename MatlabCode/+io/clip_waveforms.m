function sp = clip_waveforms(Exp, S, varargin)
% sp = io.clip_waveforms(Exp, S, varargin)
% CLIP out waveforms from wideband signal

ip = inputParser();
ip.addParameter('Fs', 30000)
ip.addParameter('Win', -10:40)
ip.parse(varargin{:})

if ~isfield(Exp.osp, 'sorter')
    Tag = 'kilo';
else
    Tag = lower(Exp.osp.sorter);
end

dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
spfname = fullfile(dataPath, 'spikes', sprintf('%s_%s.mat', strrep(Exp.FileTag, '.mat', ''), [Tag 'wf']));

if exist(spfname, 'file')
    sp = load(spfname);
    return
end

% load Ops
ops = io.loadOps(S.rawFilePath);

if numel(ops) > 1
    warning('more than one ops! ignoring everything but the first. fix this.')
    ops = ops(1);
end

ops = io.convertOpsToNewDirectory(ops, S.rawFilePath);
disp('saving high-pass filtered data')
preprocess.save_filtered_data(ops, true);

disp('Done')

info = load(fullfile(ops.root, 'ephys_info.mat'));
Fs = info.sampleRate;
% % load wideband signal (in muV)
% [data, timestamps] = io.loadRaw(ops, [], true);
% 
% % very rarely, the timing of all samples will not be unique. When this
% % happens, it involves only one sample and has to do with the recording
% % being paused (*I think*). A simple, but hacky solution is to only take
% % unique timestamps.
% [timestamps, ia] = unique(timestamps);
% data = data(:,ia);
% clear ia
% 
% % highpass filter
% Fs = ip.Results.Fs;
% [b,a] = butter(6, (300/Fs)*2, 'high');
% 
% % causal filtering only
% data = filter(b,a,data');

sp = Exp.osp;
clear Exp

cids = sp.cids;
win = ip.Results.Win; % window around spike to bin
if numel(win)==2
    win = win(1):win(2);
end

% preallocate
NC = numel(cids);
numChan = ops.Nchan;
numLags = numel(win);

WFmed = zeros(NC, numLags, numChan);
WFciHi = zeros(NC, numLags, numChan);
WFciLow = zeros(NC, numLags, numChan);

% open binary file
fHP = strrep(ops.fbinary, '.dat', '_HP.dat');
% fid = fopen(ops.fbinary);
fid = fopen(fHP);

for cc = 1:NC % loop over units
    fprintf('Unit %d/%d\n', cc, NC)
    % for this unit: get all "clean spike times". These are times that do not
    % collide with other neighboring units. We want Kilosort to include those
    % because we care about simultaneous spikes, but we don't want them
    % included in our waveform estimates
    spikeIx = sp.clu==cids(cc); % this unit
    spikeTimes = sp.st(spikeIx); % this unit spike times
    
    % get rid of bursts of spikes (that can mess up our waveform estimate)
    spikeTimes = sort(spikeTimes);
    spikeTimes(diff(spikeTimes)<=2e-3)=[]; % remove spikes within close succession
    
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
    
    % cleanup for memory saving
    clear closeSpikeTimes spikeTimesAsBins otherSpikeTimes spikeIx otherSpikes otherSpikeDepths otherSpikeDistance
    
    % only take clean spike counts
    spikeTimes = spikeTimes(cnt==0);
    
    clear cnt
    
    % bin spike counts at sample rate
    id = io.convertTimeToSamples(spikeTimes, info.sampleRate, info.timestamps(:), info.fragments(:));
    
    clear spikeTimes
    fseek(fid, 0, 'bof');
    bufferSize = [numChan numLags];
    
    numSpikes = numel(id);
    X = zeros(numSpikes, numLags, numChan);
    for ispike = 1:numSpikes
        fseek(fid, max(id(ispike)-20, 1)*2*ops.Nchan, 'bof');
        data = double(fread(fid, bufferSize, '*int16'))*info.bitVolts;
        X(ispike, :,:) = data';
    end
    disp('Done')
   
           
    bnds = prctile(X, [16 50 84], 1);
    
    WFciLow(cc,:,:) = bnds(1,:,:);
    WFmed(cc,:,:) = bnds(2,:,:);
    WFciHi(cc,:,:) = bnds(3,:,:);
   
end

sp.WFciLow = WFciLow;
sp.WFmed = WFmed;
sp.WFciHi = WFciHi;
sp.WFtax = win/Fs*1e3;

save(spfname, '-v7.3', '-struct', 'sp')