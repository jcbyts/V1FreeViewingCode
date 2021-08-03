function [stim, robs, behavior, opts] = bin_population(D, sessionId, varargin)
% BIN POPULATION SPIKE TRAINS ALIGNED TO STIMULUS ONSET
% [stim, robs, behavior, opts] = bin_population(D, sessionId, varargin)
% Arguments:
%   D - Data structure (output of io.dataFactory)
%   sessionId - Session ID (which session number to use)
% Optional arguments (as key/value pairs)
%   'plot'      - whether to plot the results (default: true)
%   'win'       - padding to use for binning before stim onset and after stim offset (default: [-.1, .1])
%   'binsize'   - bin size in seconds (default: .01)
% 
% Outputs:
% stim      - NT x 1 vector of stimulus directions
% robs      - binned spike trains aligned to response onset
% behavior  - running speed (same size as robs)
% opts      - options used for binning / additional analyses
% 
% written by Jake, 2021

ip = inputParser();
ip.addParameter('plot', true);
ip.addParameter('win', [-.1 .1])
ip.addParameter('binsize', 10e-3)
ip.parse(varargin{:})


%% bin spikes
binsize = ip.Results.binsize; % (in seconds)

ix = D.sessNumGratings == sessionId; % index into the grating times that correspond to requested session

StimOnset  = D.GratingOnsets(ix);
StimOffset = D.GratingOffsets(ix);
StimDir    = D.GratingDirections(ix);

treadTime = D.treadTime(~isnan(D.treadTime));
treadSpeed = D.treadSpeed(~isnan(D.treadTime));

% resample time with new binsize
newTreadTime = treadTime(1):binsize:treadTime(end);
newTreadSpeed = interp1(treadTime, treadSpeed, newTreadTime);

% figure(1); clf
% plot(treadTime, treadSpeed); hold on
% plot(newTreadTime, newTreadSpeed);

treadTime = newTreadTime;
treadSpeed = newTreadSpeed;

[~, ~, idOn] = histcounts(StimOnset, treadTime);
[~, ~, idOff] = histcounts(StimOffset, treadTime);

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

runSpeed = zeros(NT, nbins);
for i = 1:NT
    iix = blags + idOn(i);
    runSpeed(i,iix>0) = treadSpeed(iix(iix>0));
end

ix = D.sessNumSpikes == sessionId;
SpikeTimes = D.spikeTimes(ix);
SpikeIds = D.spikeIds(ix);
UnitList = unique(SpikeIds);

NC = numel(UnitList); % number of neurons
fprintf("Session %d) %d Grating Trials, %d Units %02.2f sec duration\n", sessionId, NT, NC, StimDur)

opts.NCells = NC;
opts.stimDuration = StimDur;

plotDuringImport = ip.Results.plot;

spksb = zeros(NT, nbins, NC);

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

if plotDuringImport
    fig(1) = figure(1); clf
    
    ax1 = axes('Position', [.08 .08 .84 .84]);
    ax1.XColor = 'w';
    ax1.YColor = 'w';
    ax = decoding.tight_subplot(sx, sy, .01);
    
end

% count spike times aligned to motion onset
fprintf('Counting spikes aligned to stim onset \n')
for k = 1:(sx*sy)
    if plotDuringImport
        figure(fig(1))
        set(fig(1), 'currentaxes', ax(k))
    end
    
    if k > NC
        axis off
        continue
    end
    
    % count spikes
    st = SpikeTimes(SpikeIds==UnitList(k));
    [scnt, bins] = decoding.binSpTimes(st, StimOnset(validStim), win, binsize);
    spksb(:,:,k) = scnt;
    
    if plotDuringImport
        % visualize tuning
        [~, ind] = sort(StimDir(validStim));
        smcnt = imgaussfilt(scnt(ind,:), 30, 'FilterSize', [31 1]); % smooth along trials
        
        if binsize > 1e-3
            imagesc(smcnt); colormap(1-gray)
            axis off
        else % if 1ms bins, plot raster
            [iTrial,j] = find(scnt(ind,:));
            plot.raster(bins(j), iTrial, 1);
            axis off
        end
        
        
        drawnow
    end
end

if plotDuringImport
    xlabel(ax1, 'Time from stim onset', 'Color', 'k')
    ylabel(ax1, 'Trials (sorted by Direction)', 'Color', 'k')
end

% output
stim = StimDir(validStim);
robs = spksb;
behavior = runSpeed;
opts.lags = bins;
opts.binsize = binsize;
