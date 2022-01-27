function stat = tuning_empirical(D, varargin)
% Empirical (bootstrap-based) tuning curve analyses
% stat = tuning_empirical(D, varargin)
% Inputs:
%   D - big supersession struct
% Optional Arguments:
%   binsize
%   win

ip = inputParser();
ip.addParameter('binsize', 10e-3)
ip.addParameter('win', [-.25 1.1])
ip.addParameter('runningthresh', 1)
ip.addParameter('nboot', 100)
ip.addParameter('seed', [])
ip.parse(varargin{:});

stat = struct();

%% bin spikes in window aligned ton onset
binsize = ip.Results.binsize;
win = ip.Results.win;
nboot = ip.Results.nboot;
D.spikeIds(D.spikeIds==0) = 1;
spikeIds = unique(D.spikeIds);
NC = numel(spikeIds);

% mapping from spike Ids to unit number
mapSpikeIds = zeros(max(D.spikeIds), 1);
mapSpikeIds(spikeIds) = (1:NC)';

stat.unitId2Num = mapSpikeIds;
stat.num2unitId = spikeIds;

% bin spike times
binnedSpikeTimes = (D.spikeTimes==0) + ceil(D.spikeTimes/binsize);

% fastest way to bin spikes using sparse.m
spbn = sparse(binnedSpikeTimes, mapSpikeIds(D.spikeIds), ones(numel(binnedSpikeTimes), 1));

% lags (in time and bins)
lags = win(1):binsize:win(2);
blags = ceil(lags/binsize);

numlags = numel(blags);
balign = ceil(D.GratingOnsets/binsize);

% handle exceptions where the grating times excede the recording duration
validix = (balign + blags(end)) < size(spbn,1);
if sum(validix) ~= numel(validix)
    warning('tuning_empirical: some grating trials are excluded. not sure the code can handle this error')
end

balign = balign(validix);

numStim = numel(balign);

treadTime = D.treadTime;
treadSpeed = D.treadSpeed;
iiTread = ~(isnan(D.treadTime) | isnan(D.treadSpeed));
iiTread = iiTread & D.treadTime >=0;
treadTime = treadTime(iiTread);
treadSpeed = treadSpeed(iiTread);

% binning treadmill speed
treadBins = ceil(treadTime/binsize);
treadSpd = nan(size(spbn,1), 1); % make same size as spike bins

% interpolate nan's
treadSpd(treadBins(~isnan(treadSpeed))) = treadSpeed(~isnan(treadSpeed));
treadSpd = (repnan(treadSpd, 'pchip'));

% Do the binning here
disp('Binning spikes')
spks = zeros(numStim, NC, numlags);
tspd = zeros(numStim, numlags);
dfilt = false(numStim, NC); % this is the "data filter" it says which Gratings correspond to the selected unit
dfiltT = false(numel(treadTime), NC); % "data filter" for tread times

for cc = 1:NC
    sessionsWithUnit = unique(D.sessNumSpikes(D.spikeIds==spikeIds(cc)));
    dfilt(:,cc) = ismember(D.sessNumGratings(validix), sessionsWithUnit);
    dfiltT(:,cc) = ismember(D.sessNumTread(iiTread), sessionsWithUnit);
end

% bin spikes aligned to grating onset
for i = 1:numlags
    spks(:,:,i) = spbn(balign + blags(i),:);
    tspd(:,i) = treadSpd(balign + blags(i));
end

% update dfilt and dfiltT
fprintf('Restricting range based on unit stability\n')
for cc = 1:NC

    unitix = find(dfilt(:,cc));
    
    fprintf('Unit %d) n = %d (initial), ', cc, numel(unitix))
    
    dur = median(D.GratingOffsets(unitix) - D.GratingOnsets(unitix));
    dur = max(dur, .1);
    win = [0.04 dur];
    
    % --- find stable region of firing rate
    spk = squeeze(spks(unitix,cc,:));
    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);
    R = imboxfilt(R, 21);
    goodix = getStableRange(R, 'plot', false); % key function: finds stable region of firing
    goodix = unitix(goodix);
    
    goodwin = [D.GratingOnsets(goodix(1)) D.GratingOffsets(goodix(end))];
    
    badix = setdiff(1:numStim, goodix);
    dfilt(badix,cc) = false;
    
    dfiltT(treadTime < goodwin(1),cc) = false;
    dfiltT(treadTime > goodwin(2),cc) = false;
    
    fprintf('n = %d (cleaned)\n', sum(dfilt(:,cc)))
end
    

fprintf("Done\n")

stat.dataFilterGratings = dfilt;
stat.dataFilterTread = dfiltT;

%% plot distribution of running speeds
thresh = ip.Results.runningthresh;
figure(10); clf
runningSpd = mean(tspd,2);
runningTrial = runningSpd > thresh;
stationaryTrial = abs(runningSpd)<thresh;

histogram(runningSpd(~runningTrial), 'binEdges', [-5:.1:25], 'FaceColor', [.5 .5 .5])
hold on
clr = [1 .2 .2];
histogram(runningSpd(runningTrial), 'binEdges', -5:.1:25, 'FaceColor', clr)
plot.fixfigure(gcf,10,[4 4]);
xlabel('Running Speed (cm/s)')
ylabel('Trial Count') 
text(1, mean(ylim), sprintf('Running Trials (n=%d)', sum(runningTrial)), 'Color', clr, 'FontSize', 14)
text(1, mean(ylim)+.1*range(ylim), sprintf('Stationary Trials (n=%d)', sum(~runningTrial)), 'Color', .5*[1 1 1], 'FontSize', 14)
set(gca, 'YScale', 'linear')


%% sanity check alignment
% figure(1);clf
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% 
% bss = binnedSpikeTimes(D.spikeIds == spikeIds(cc));
% iix = min(bss):max(bss);
% subplot(2,1,1)
% plot(iix*binsize, imgaussfilt(full(spbn(iix,cc)), 10)/binsize); hold on
% 
% cnt = histcounts( D.spikeTimes(D.spikeIds == spikeIds(cc)), treadTime(dfiltT(:,cc)));
% rate = cnt(:) ./ diff(treadTime(dfiltT(:,cc)));
% tt = treadTime(dfiltT(:,cc));
% plot(tt(1:end-1), imgaussfilt(rate, 10))
% 
% subplot(2,1,2)
% plot(iix*binsize, treadSpd(iix)); hold on
% plot(treadTime(dfiltT(:,cc)), treadSpeed(dfiltT(:,cc)), '--')

%% Stim aligned psth
ths = unique(D.GratingDirections);
nd = numel(ths);
nlags = numel(lags);

stat.psths.bins = lags;
stat.psths.running = zeros(numlags, nd, 3, NC);
stat.psths.stationary = zeros(numlags, nd, 3, NC);
for cc = 1:NC % cells
    fprintf("Unit %d/%d\n", cc, NC)
    for i = 1:nd % directions
        iix = D.GratingDirections==ths(i) & dfilt(:,cc);
        
        runinds = find(iix & runningTrial);
        statinds = find(iix & stationaryTrial);
        
        runmu = zeros(nlags, nboot);
        statmu = zeros(nlags, nboot);
        for iboot = 1:nboot
            bootinds = randsample(runinds, numel(runinds), true);
            runmu(:,iboot) = squeeze(mean(spks(bootinds,cc,:),1));
            bootinds = randsample(statinds, numel(runinds), true);
            statmu(:,iboot) = squeeze(mean(spks(bootinds,cc,:),1));
        end
        
        stat.psths.running(:,i,:,cc) = prctile(runmu, [2.5 50 97.5], 2);
        stat.psths.stationary(:,i,:,cc) = prctile(statmu, [2.5 50 97.5], 2);
    end
end

%%
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% cc = 40;
% figure(1); clf
% for t = 1:nd
%     subplot(1,nd,t)
%     plot(stat.psths.bins, imgaussfilt(stat.psths.running(:,t,1,cc), 2), 'r'); hold on
%     plot(stat.psths.bins, imgaussfilt(stat.psths.running(:,t,3,cc), 2), 'r')
%     
%     plot(stat.psths.bins, imgaussfilt(stat.psths.stationary(:,t,1,cc), 2), '--b'); hold on
%     plot(stat.psths.bins, imgaussfilt(stat.psths.stationary(:,t,3,cc), 2), '--b')
%     
%     ylim([0 .5])
% end
% % imagesc(stat.psths.running(:,:,2,cc)')
% % title(cc)

%% Tuning curve to running
% nSpdBins = 15;
% binEdges = prctile(abs(treadSpeed), linspace(0, 100, nSpdBins+1));

binEdges = [0 .5 1 2 8 20];
nSpdBins = numel(binEdges) - 1;

spdBins = (binEdges(1:end-1) + binEdges(2:end))/2;

numSpdSamples = zeros(nSpdBins, NC);
rateSpdMu = nan(nSpdBins, NC);
rateSpdSe = nan(nSpdBins, NC);

numSpdSamplesStim = zeros(nSpdBins, NC);
rateSpdMuStim = nan(nSpdBins, NC);
rateSpdSeStim = nan(nSpdBins, NC);

stimIdx = getTimeIdx(treadTime, D.GratingOnsets, D.GratingOffsets);

fprintf('Tuning curve to running speed\n')
for cc = 1:NC
    fprintf('Unit %d/%d\n', cc, NC)
    
    for ibin = 1:nSpdBins
        iispd = treadSpeed > binEdges(ibin) & min(treadSpeed, binEdges(end)) < binEdges(ibin+1);
        
        % raw speed modulation
        iix = dfiltT(:,cc) & iispd;
        numSpdSamples(ibin, cc) = sum(iix);
        
        if sum(iix)==0
            continue
        end
        cnt = histcounts( D.spikeTimes(D.spikeIds == spikeIds(cc)), treadTime(iix));
        rate = cnt(:) ./ diff(treadTime(iix));
        
        rateSpdMu(ibin,cc) = mean(rate);
        rateSpdSe(ibin,cc) = std(rate) / sqrt(numSpdSamples(ibin, cc));
        
        % while stimulus is on
        iix = dfiltT(:,cc) & iispd & stimIdx;
        numSpdSamplesStim(ibin, cc) = sum(iix);
        
        if sum(iix)==0
            continue
        end
        cnt = histcounts( D.spikeTimes(D.spikeIds == spikeIds(cc)), treadTime(iix));
        rate = cnt(:) ./ diff(treadTime(iix));
        
        rateSpdMuStim(ibin,cc) = mean(rate);
        rateSpdSeStim(ibin,cc) = std(rate) / sqrt(numSpdSamplesStim(ibin, cc));
    end
    
end

stat.speedTuning.binEdges = binEdges;
stat.speedTuning.bins = spdBins;
stat.speedTuning.numSpdSamples = numSpdSamples;
stat.speedTuning.rateSpdMu = rateSpdMu;
stat.speedTuning.rateSpdSe = rateSpdSe;

stat.speedTuning.numSpdSamplesStim = numSpdSamplesStim;
stat.speedTuning.rateSpdMuStim = rateSpdMuStim;
stat.speedTuning.rateSpdSeStim = rateSpdSeStim;

%% running onset aligned psth
% detect epochs of running. Align to the onsets of running after a short
% period of stationary behavior

nb = 41; % smoothing window

stat.running = repmat(struct('spikerate', [], ...
    'runningspeed', [], ...
    'runonsets', [], ...,
    'runoffsets', [], ...
    'goodix', [], ...
    'psthBins', [], ...
    'psthMu', [], ...
    'psthSe', [], ...
    'psthNullCi', [], ...
    'rateStatNull', [], ...
    'rateRun', []), NC, 1);

for cc = 1:NC

    figure(1); clf
    
    unitix = find(dfiltT(:,cc));

    % get spike rate
    cnt = histcounts( D.spikeTimes(D.spikeIds == spikeIds(cc)), treadTime(unitix));
    sprate = cnt(:) ./ diff(treadTime(unitix));
    sprate = imboxfilt(sprate, nb); % smooth with boxcar filter
    
    subplot(2,1,1)
    h = plot(sprate, 'k'); hold on
    h(2) = plot(treadSpeed(unitix), 'r');
    
    xlabel('Time')
    % find running epochs
    isrunning = treadSpeed(unitix) > thresh;
    isstationary = find(abs(treadSpeed(unitix)) < thresh);
    
    % onsets and offsets
    onsets = find(diff(isrunning) == 1);
    offsets = find(diff(isrunning) == -1);
    
    if isempty(onsets)
        continue
    end
        
    if onsets(1) > offsets(1)
        onsets = [1; onsets];
    end
    
    if offsets(end) < onsets(end)
        offsets = [offsets; numel(treadSpeed)];
    end
    
    assert(numel(onsets)==numel(offsets), "onset offset mismatch")
    
    dt = median(diff(treadTime(unitix)));
    PreRunStatDur = ceil(.5/dt); % must've been stationary for .5 second before onset
    RunDur = ceil(.5/dt); % must've been running for at least half a second
    
    % throw out epochs that
    onsetix = find((onsets(2:end) - offsets(1:end-1)) > PreRunStatDur & (offsets(2:end) - onsets(2:end)) > RunDur) + 1;
    
    nr = numel(onsetix);
    for i = 1:nr
        h(3) = fill([onsets(onsetix(i)*[1 1]); offsets(onsetix(i)*[1 1])]', [ylim, fliplr(ylim)]', 'r', 'FaceAlpha', .25, 'EdgeColor', 'none'); hold on
    end
    
    legend(h, {'Spike Rate', 'Running Speed', 'Running Epoch'});
    xlim([1 numel(isrunning)])
    
    %disp(nr)
    rwin = [-200 200];
    
    annull = zeros(diff(rwin)+1, nboot);
    for iboot = 1:nboot
        an = eventTriggeredAverage(sprate, randi(numel(isrunning), [nr 1]), rwin);
        annull(:,iboot) = an;
    end
    
    [an,sd] = eventTriggeredAverage(sprate, onsets(onsetix), rwin);
    rlags = (rwin(1):rwin(2))*dt;
    
    ci = prctile(annull, [2.5 97.5], 2);
    
    subplot(2,1,2)
    plot(rlags, ci, 'k--'); hold on
    plot.errorbarFill(rlags, an, sd/sqrt(nr));
    plot(rlags, an, 'k', 'Linewidth', 2)
    xlabel('Seconds')
    axis tight
    title(cc)
    
    % --- sample from running and not running
    muStat = nan(nboot, 1);
    
    non = numel(onsets);
    dur = offsets - onsets;
    
    for iboot = 1:nboot
        son = randsample(isstationary, non, true);
        nullidx = [];
        for i = 1:non
            nullidx = [nullidx; son(i) + (1:dur(i))'];
        end
        nullidx = unique(nullidx);
        nullidx = min(nullidx, numel(sprate));
        muStat(iboot) = mean(sprate(nullidx));
    end
    
    muRun = mean(sprate(isrunning(1:end-1)));
    
    figure(2); clf
    histogram(muStat); hold on
    plot(muRun*[1 1], ylim, 'k')
    
    
    % store analyses
    stat.running(cc).spikerate = sprate;
    stat.running(cc).runningspeed = treadSpeed(unitix(1:end-1));
    stat.running(cc).runonsets = onsets;
    stat.running(cc).runoffsets = offsets;
    stat.running(cc).goodix = onsetix;
    stat.running(cc).psthBins = rlags;
    stat.running(cc).psthMu = an;
    stat.running(cc).psthSe = sd / sqrt(nr);
    stat.running(cc).psthNullCi = ci;
    stat.running(cc).rateStatNull = muStat;
    stat.running(cc).rateRun = muRun;
    
end


%% empirical tuning curve differences
ths = unique(D.GratingDirections);
nth = numel(ths);

if ~isempty(ip.Results.seed)
    rng(ip.Results.seed) % set random seed for reproducing the exact numbers
end

nboot = ip.Results.nboot; % get nboot from 

TCdiffNull = nan(NC, nboot);
maxFRdiffNull = nan(NC, nboot);
TCdiff = nan(NC,1);
maxFRdiff = nan(NC,1);

TCrun = nan(nth, NC, 3);
TCstat = nan(nth, NC, 3);

for cc = 1:NC
    fprintf('%d/%d\n', cc, NC)
    unitix = dfilt(:,cc);
    dur = median(D.GratingOffsets(unitix) - D.GratingOnsets(unitix));
    dur = max(dur, .1);
    win = [0.04 dur];

    % Get tuning curve
    spk = squeeze(spks(unitix,cc,:));
    theta = D.GratingDirections(unitix);
    
    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);
    
    thetas = unique(theta);
    tfun = @(th,cnt) arrayfun(@(x) mean(cnt(th==x)), thetas);
    
    iirun = runningTrial(unitix);
    iistat = stationaryTrial(unitix);
    
    tuningCurveR = tfun(theta(iirun), R(iirun)); % RUNNING
    tuningCurveS = tfun(theta(iistat), R(iistat)); % STATIONARY
    
    nn = numel(theta);
    nrun = sum(iirun);
    nstat = sum(iistat);
    
    iir = randi(nn, [nrun nboot]);
    iis = randi(nn, [nstat nboot]);
    
    
    % boostrap the null distribution
    tcRnull = zeros(numel(thetas), nboot);
    tcSnull = zeros(numel(thetas), nboot);
    for iboot = 1:nboot
        tcRnull(:,iboot) = tfun(theta(iir(:,iboot)), R(iir(:,iboot)));
        tcSnull(:,iboot) = tfun(theta(iis(:,iboot)), R(iis(:,iboot)));
        TCdiffNull(cc,iboot) = mean( (tcRnull(:,iboot) - tcSnull(:,iboot)).^2);
        maxFRdiffNull(cc,iboot) = max(tcRnull(:,iboot)) - max(tcSnull(:,iboot));
        
    end
    
    % boostrap TC errorbars
    tcR = zeros(numel(thetas), nboot);
    tcS = zeros(numel(thetas), nboot);
    for iboot = 1:nboot
        iiboot = randsample(find(iirun), nrun, true);
        tcR(:,iboot) = tfun(theta(iiboot), R(iiboot));
        iiboot = randsample(find(iistat), nrun, true);
        tcS(:,iboot) = tfun(theta(iiboot), R(iiboot));
    end
    
    
    thix = ismember(ths, thetas);
    
    TCrun(thix, cc, :) = prctile(tcR, [2.5 50 97.5], 2);
    TCstat(thix, cc, :) = prctile(tcS, [2.5 50 97.5], 2);
    
    TCdiff(cc) = mean( (tuningCurveR - tuningCurveS).^2 );
    maxFRdiff(cc) = max(tuningCurveR) - max(tuningCurveS);

end

stat.TCempirical.TCrun = TCrun;
stat.TCempirical.TCstat = TCstat;
stat.TCempirical.thetas = ths;
stat.TCempirical.TCdiff = TCdiff;
stat.TCempirical.maxFRdiff = maxFRdiff;
stat.TCempirical.TCdiffNull = TCdiffNull;
stat.TCempirical.maxFRdiffNull = maxFRdiffNull;


% 
% %% explanatory plot
% cmap = lines;
% 
% figure(2); clf
% subplot(1,3,1)
% plot(thetas, tuningCurveS, 'Linewidth', 2); hold on
% plot(thetas, tuningCurveR, 'Linewidth', 2)
% title('Raw Tuning Curves')
% axis tight
% ylabel('Firing Rate')
% 
% subplot(1,3,2)
% fill([thetas' fliplr(thetas')], [tuningCurveS' fliplr(tuningCurveR')], 'k', 'Linewidth', 2); hold on
% plot(thetas, tuningCurveS, 'Linewidth', 2); hold on
% plot(thetas, tuningCurveR, 'Linewidth', 2)
% title('TC diff')
% axis tight
% xlabel('Direction')
% subplot(1,3,3)
% plot(thetas, tuningCurveS, 'Linewidth', 2); hold on
% plot(thetas, tuningCurveR, 'Linewidth', 2)
% [mS, idS] = max(tuningCurveS);
% [mR, idR] = max(tuningCurveR);
% plot([0 thetas(idS)], mS*[1 1], '-', 'Color', cmap(1,:), 'Linewidth', 2)
% plot(thetas(idS), mS, 'o', 'Color', cmap(1,:), 'Linewidth', 2)
% plot([0 thetas(idR)], mR*[1 1], '-', 'Color', cmap(2,:), 'Linewidth', 2)
% plot(thetas(idR), mR, 'o', 'Color', cmap(2,:), 'Linewidth', 2)
% plot([1 1]*thetas(idS), [mS mR], 'k', 'Linewidth', 2)
% axis tight
% title('Max FR')


%% fit tuning curves
clear fitS fitR

% cc = cc + 1;
for cc = 1:NC
    fprintf('%d/%d\n', cc, NC)
    unitix = dfilt(:,cc);
    dur = median(D.GratingOffsets(unitix) - D.GratingOnsets(unitix));
    dur = max(dur, .1);
    win = [0.04 dur];
    
    % STATIONARY
    iix = ~runningTrial & unitix;
    spk = squeeze(spks(iix,cc,:));
    th = D.GratingDirections(iix);

    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);

    fitS(cc) = tcurve.fit_tuningcurve(th, R, false);
    
    % RUNNING
    iix = runningTrial & unitix;
    spk = squeeze(spks(iix,cc,:));
    th = D.GratingDirections(iix);

    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);

    fitR(cc) = tcurve.fit_tuningcurve(th, R, false);

end

stat.TCfitS = fitS;
stat.TCfitR = fitR;


