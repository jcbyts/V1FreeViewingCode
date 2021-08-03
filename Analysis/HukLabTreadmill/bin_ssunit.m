function [stim, robs, behavior, opts] = bin_ssunit(D, unitId, varargin)
% BIN SUPER SESSION UNIT SPIKE TRAINS ALIGNED TO STIMULUS ONSET
% [stim, robs, behavior, opts] = bin_ssunit(D, unitID, varargin)
% Optional Arguments:
% plot
% win
% binsize
ip = inputParser();
ip.addParameter('plot', true);
ip.addParameter('win', [-.1 .1])
ip.addParameter('binsize', 10e-3)
ip.parse(varargin{:})


%% bin spikes
binsize = ip.Results.binsize; % 10 ms bins

unitIx = D.spikeIds == unitId;
sessNums = unique(D.sessNumSpikes(unitIx));

ix = ismember(D.sessNumGratings, sessNums);

StimOnset  = D.GratingOnsets(ix);
StimOffset = D.GratingOffsets(ix);
StimDir    = D.GratingDirections(ix);

treadTime = D.treadTime(~isnan(D.treadTime));
treadSpeed = D.treadSpeed(~isnan(D.treadTime));

treadSessIx = treadTime > (StimOnset(1) - 5) & treadTime < (StimOffset(end) + 5);
treadTime = treadTime(treadSessIx);
treadSpeed = treadSpeed(treadSessIx);

% resample time with new binsize
newTreadTime = treadTime(1):binsize:treadTime(end);
newTreadSpeed = interp1(treadTime, treadSpeed, newTreadTime);

% figure(1); clf
% plot(treadTime, treadSpeed); hold on
% plot(newTreadTime, newTreadSpeed);

treadTime = newTreadTime;
treadSpeed = newTreadSpeed;

% find index into onsets and offsets
[~, ~, idOn] = histcounts(StimOnset, treadTime);
[~, ~, idOff] = histcounts(StimOffset, treadTime);

% check stim duration and only include gratings that were fully shown
stimDuration = idOff - idOn;
durBins = mode(stimDuration);
validStim = find(stimDuration==durBins);

StimDur = mode(StimOffset(validStim) - StimOnset(validStim));
win = [ip.Results.win(1) StimDur+ip.Results.win(2)];
bins = win(1):binsize:win(2);

nbins = numel(bins)-1;

blags = floor(bins/binsize); blags = blags(1:nbins);

NT = numel(validStim);

opts = struct();
opts.NTrials = NT;
opts.NLags = nbins;

% get running speed aligned to stim onset
runSpeed = zeros(NT, nbins);
for i = 1:NT
    iix = blags + idOn(i);
    runSpeed(i,iix>0) = treadSpeed(iix(iix>0));
end


SpikeTimes = D.spikeTimes(unitIx);
SpikeIds = D.spikeIds(unitIx);
UnitList = unique(SpikeIds);

NC = numel(UnitList); % number of neurons
assert(NC == 1, "You requested one unit and we have more than that")

%%

opts.unitId = unitId;
opts.stimDuration = StimDur;

plotDuringImport = ip.Results.plot;

spksb = zeros(NT, nbins);

% count spike times aligned to STIM onset
fprintf('Counting spikes aligned to stim onset \n')

    
% count spikes
[scnt, bins] = decoding.binSpTimes(SpikeTimes, StimOnset(validStim), win, binsize);

    
if plotDuringImport
    figure(10); clf
    % visualize tuning
        
    [~, ind] = sort(StimDir(validStim));
        
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
stim = StimDir(validStim);
robs = scnt;
behavior = runSpeed;
opts.lags = bins;
