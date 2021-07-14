function Dstat = decode_running(D, sessionId, varargin)
% do orientation / direction decoding for the huklab treadmill data
% Dstat = decode_running(D, sessionId, varargin)
% Inputs:
%   D            <struct> super session struct
    % sessionId <numeric> session number
% Optional Arguments:
%   binSize:        bin size (seconds)
%   Latency:        estimated stim onset (for decoding window, in seconds)
%   Decoder:        SVM, GLM
%   slidingWin:     boxcar width for sliding window analyses (seconds)
%   runThreshold:   threshold in cm/s to consider running
%   plot:           boolean for saving plots (default: true)

ip = inputParser();
ip.addParameter('binSize', 5e-3)
ip.addParameter('Latency', .04)
ip.addParameter('Decoder', 'svm')
ip.addParameter('slidingWin', 80e-3)
ip.addParameter('runThreshold', 1)
ip.addParameter('plot', true)
ip.parse(varargin{:})


ix = D.sessNumGratings == sessionId;

StimOnset = D.GratingOnsets(ix);
StimOffset = D.GratingOffsets(ix);
StimDir = D.GratingDirections(ix);
StimDur = median(StimOffset - StimOnset);

NT = numel(StimDir);
Dstat = struct();
Dstat.NTrials = NT;

treadTime = D.treadTime(~isnan(D.treadTime));
treadSpeed = D.treadSpeed(~isnan(D.treadTime));
[~, ~, idOn] = histcounts(StimOnset, treadTime);
[~, ~, idOff] = histcounts(StimOnset, treadTime);

runSpeed = zeros(NT, 1);
for i = 1:NT
    runSpeed(i) = mean(treadSpeed(idOn(i):idOff(i)));
end

ix = D.sessNumSpikes == sessionId;
SpikeTimes = D.spikeTimes(ix);
SpikeIds = D.spikeIds(ix);
UnitList = unique(SpikeIds);

NC = numel(UnitList); % number of neurons
fprintf("Session %d) %d Grating Trials, %d Units %02.2f sec duration\n", sessionId, NT, NC, StimDur)

Dstat.NCells = NC;
Dstat.stimDuration = StimDur;

%% preprocess some of the data

% parameters for counting spikes
StimDur = max(StimDur, .1);
win = [-0.21 StimDur];
binsize = ip.Results.binSize; % 10 ms bins

bins = win(1):binsize:win(2);
nbins = numel(bins)-1;

plotDuringImport = ip.Results.plot;

stimid = unique(StimDir); % unique stimuli shown
nstm = numel(stimid);

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
    [scnt, bins] = decoding.binSpTimes(st, StimOnset, win, binsize);
    spksb(:,:,k) = scnt;
    
    if plotDuringImport
        % visualize tuning
        [~, ind] = sort(StimDir);
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
    xlabel(ax1, 'Time from motion onset', 'Color', 'k')
    ylabel(ax1, 'Trials (sorted by Direction)', 'Color', 'k')
end


%% Do some decoding

% window to train decoder on
tidx = bins > ip.Results.Latency & bins < StimDur;

rr = squeeze(mean(spksb(:,tidx,:),2)); % sum over time

ntrials = size(rr,1);

decoderTot = nan(ntrials, 1);

runTrials = runSpeed > ip.Results.runThreshold;

fprintf('Leave one out cross-validation\n')
clear B
for iTrial = 1:ntrials
%     fprintf('Trial %d / %d\n', iTrial, ntrials)
    
    iitrain = setdiff(1:ntrials, iTrial);
    iitest = iTrial;
    
    switch ip.Results.Decoder
        case 'svm'
            model = fitcsvm(rr(iitrain,:), runTrials(iitrain));
            yhat = predict(model, rr(iitest,:));
        case 'glm'
            model = fitglm(rr, runTrials, 'Distribution', 'binomial');
            yhat = predict(model, rr(iitest,:));
            yhat = yhat > .5;
    end

    decoderTot(iitest) = yhat;
    
end
fprintf('Done\n')

Dstat.decoderTot = decoderTot;
Dstat.runTrials = runTrials;

% some summaries
Dstat.accCi = bootci(100, @mean, Dstat.runTrials==Dstat.decoderTot);
Dstat.acc = mean(Dstat.runTrials==Dstat.decoderTot);
Dstat.chance = max(mean(Dstat.runTrials), mean(1-Dstat.runTrials));
fprintf('Decoding accuracy: %02.2f [%02.2f,%02.2f], chance=%02.2f\n', Dstat.acc, Dstat.accCi(1), Dstat.accCi(2), Dstat.chance);
