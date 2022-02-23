function [stim, robs, behavior, opts] = bin_ssunit(D, unitId, varargin)
% BIN SUPER SESSION UNIT SPIKE TRAINS ALIGNED TO STIMULUS ONSET
% [stim, robs, behavior, opts] = bin_ssunit(D, unitID, varargin)
% INPUTS:
%   D:      supersession
%   unitID: the id of the unit to analyze
% OUTPUTS:
%   stim:      {StimDir, StimSpeed, StimFreq};
%   robs:      Binned spikes;
%   behavior:  {runSpeed, GratPhase, PupilArea};
%   opts:      the parameters of the analysis
%
% Optional Arguments:
% plot
% win [seconds before (pass in negative), seconds after]
% binsize (in seconds)

ip = inputParser();
ip.addParameter('plot', true);
ip.addParameter('win', [-.1 .1])
ip.addParameter('binsize', 10e-3)
ip.parse(varargin{:})


%% bin spikes
binsize = ip.Results.binsize; % 10 ms bins
if isnan(unitId)
    unitIx = true(size(D.spikeIds));
else
    unitIx = D.spikeIds == unitId;
end

sessNums = unique(D.sessNumSpikes(unitIx));

ix = ismember(D.sessNumGratings, sessNums);

StimOnset  = D.GratingOnsets(ix);
StimOffset = D.GratingOffsets(ix);
StimDir    = D.GratingDirections(ix);
StimSpeed = D.GratingSpeeds(ix);
StimFreq = D.GratingFrequency(ix);

treadSessIx = ismember(D.sessNumTread, sessNums) & ~isnan(D.treadTime);
treadTime = D.treadTime(treadSessIx);
treadSpeed = D.treadSpeed(treadSessIx);

frameIx = D.frameTimes >= (min(StimOnset) - 1) & D.frameTimes < (max(StimOffset) + 1);
frameTime = D.frameTimes(frameIx);
framePhase = D.framePhase(frameIx);

% resample time with new binsize
newTreadTime = treadTime(1):binsize:treadTime(end);
newTreadSpeed = interp1(treadTime, treadSpeed, newTreadTime);

newFrameTime = newTreadTime;
newFramePhase = interp1(frameTime, framePhase, newFrameTime);

pupil = nan(size(newFrameTime));
eyeIx = ismember(D.sessNumEye, sessNums) & ~isnan(D.eyePos(:,3));
eyeTime = D.eyeTime(eyeIx);
eyePupil = D.eyePos(eyeIx,3);
if ~isempty(eyePupil)
    pupil = interp1(eyeTime,eyePupil,newFrameTime);
end

treadTime = newTreadTime;
treadSpeed = newTreadSpeed;
framePhase = newFramePhase;

% find index into onsets and offsets
[~, ~, idOn] = histcounts(StimOnset, treadTime);
[~, ~, idOff] = histcounts(StimOffset, treadTime);

% check stim duration and only include gratings that were fully shown
stimDuration = idOff - idOn;
durBins = mode(stimDuration);

validStim = find( abs(stimDuration-durBins)<2 & ~isnan(StimDir));
idOff = idOff(validStim);
idOn = idOn(validStim);
StimDur = mode(StimOffset(validStim) - StimOnset(validStim));
StimDir = StimDir(validStim);
StimSpeed = StimSpeed(validStim);
StimFreq = StimFreq(validStim);

win = [ip.Results.win(1) StimDur+ip.Results.win(2)];
bins = win(1):binsize:win(2);

nbins = numel(bins)-1;

blags = floor(bins/binsize); blags = blags(1:nbins);

NT = numel(validStim);

opts = struct();
opts.NTrials = NT;
opts.NLags = nbins;

% get running speed aligned to stim onset
runSpeed = nan(NT, nbins);
GratPhase = nan(NT, nbins);
PupilArea = nan(NT, nbins);
nt = numel(treadSpeed);

for i = 1:NT
    iix = blags + idOn(i);
    valid = iix > 0 & iix < nt;
    runSpeed(i,valid) = treadSpeed(iix(valid));
    GratPhase(i,valid) = framePhase(iix(valid));
    PupilArea(i,valid) = pupil(iix(valid));
end

SpikeTimes = D.spikeTimes(unitIx);
SpikeIds = D.spikeIds(unitIx);
UnitList = unique(SpikeIds);

NC = numel(UnitList); % number of neurons
assert(NC == 1 | isnan(unitId), "You requested one unit and we have more than that")

opts.unitId = unitId;
opts.stimDuration = StimDur;

plotDuringImport = ip.Results.plot;

% count spike times aligned to STIM onset
% fprintf('Counting spikes aligned to stim onset \n')    
% count spikes
[scnt, bins] = decoding.binSpTimes(SpikeTimes, StimOnset(validStim), win, binsize);

    
if plotDuringImport
    figure(10); clf
    % visualize tuning
    [~, ind] = sort(StimDir);
        
    smcnt = imgaussfilt(scnt(ind,:), [5 3], 'FilterSize', [11 3]); % smooth along trials
    
    if binsize > 1e-3
        imagesc(bins, StimDir(ind), smcnt); colormap(1-gray)
    else % if 1ms bins, plot raster
        [iTrial,j] = find(scnt(ind,:));
        plot.raster(bins(j), iTrial, 1);
    end
    
    
    drawnow
end

if plotDuringImport
    title(sprintf('Unit %d', unitId))
    xlabel('Time from stim onset', 'Color', 'k')
    ylabel('Trials (sorted by Direction)', 'Color', 'k')
end

% output
stim = {StimDir, StimSpeed, StimFreq};
robs = scnt;
behavior = {runSpeed, GratPhase, PupilArea};
opts.lags = bins;
opts.binsize = binsize;
