%% Step 0: set your paths
% The FREEVIEWING codebase uses matlab preferences to manage paths (so
% different users can have different paths)

addFreeViewingPaths('jakelaptop') % switch to your user
addpath Analysis/HukLabTreadmill/ % the code will always assume you're running from the FreeViewing base directory

%% Step 1: Make sure a session is imported
% You have to add the session to the datasets.xls file that is in the 
% google drive MANUALLY. If you open that file, you'll see the format for 
% everything

% dataFactoryTreadmill is the main workhorse for loading / importing data

% if you call it with no arguments, it will list all sessionIDs that have
% been added to the datasets.xls file
sesslist = io.dataFactoryTreadmill();

%% Step 1.1: Try importing a session

sessionId = 24;
Exp = io.dataFactoryTreadmill(sessionId);

% Note: PLDAPS sessions and MarmoV5 will load into different formats

%% Step 1.2: Create a super session datafile for each subject
% Create a supersession file for the subject you want to analyze

subject = 'brie';
import_supersession(subject);

% TODO: this does not track RF locations yet

%% Step 2: Load a super session file
subject = 'brie';
D = load_subject(subject);

%% Step 2.1: plot one session
% This just demonstrates how to index into individual sessions and get
% relevant data

sessionId = 1; % pick a session id

% index into grating times from that session
gratIx = D.sessNumGratings==sessionId;
gratOnsets = D.GratingOnsets(gratIx);
gratOffsets = D.GratingOffsets(gratIx);
gratDir = D.GratingDirections(gratIx);

% find treadmill times that correspond to this session 
treadIx = D.treadTime > gratOnsets(1) & D.treadTime < gratOffsets(end);
treadTime = D.treadTime(treadIx);


figure(111); clf
% PLOT GRATINGS TIME VS. DIRECTION
subplot(3,1,1)
nGratings = numel(gratOnsets);
for iG = 1:nGratings
    plot([gratOnsets(iG) gratOffsets(iG)], gratDir(iG)*[1 1], 'r', 'Linewidth', 2); hold on
end
ylabel('Direction')
xlim(treadTime([1 end]))
title('Gratings')
ylim([0 360])

% PLOT SPIKE TIMES
spikeIds = unique(D.spikeIds(D.sessNumSpikes==sessionId));
NC = numel(spikeIds);
spikeRate = zeros(numel(treadTime), NC);

bs = diff(treadTime);
for cc = 1:NC
    spikeRate(:,cc) = [0 histcounts(D.spikeTimes(D.spikeIds==spikeIds(cc)), treadTime)./bs'];
end

spikeRate = spikeRate ./ max(spikeRate); % normalize for visualization

treadSpeed = D.treadSpeed(treadIx);
runThresh=3;

isrunning = treadSpeed > runThresh;
onsets = find(diff(isrunning) == 1);
offsets = find(diff(isrunning) == -1);

if onsets(1) > offsets(1)
    onsets = [1; onsets];
end

if offsets(end) < onsets(end)
    offsets = [offsets; numel(treadSpeed)];
end

assert(numel(onsets)==numel(offsets), "onset offset mismatch")

subplot(3,1,2) % plot spike count
imagesc(treadTime, 1:NC, spikeRate'); hold on
for i = 1:numel(onsets)
    fill(treadTime([onsets(i) onsets(i) offsets(i) offsets(i)]), [ylim fliplr(ylim)], 'r', 'FaceColor', 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
end
xlim(treadTime([1 end]))
colormap(1-gray)
title('Spikes')
ylabel('Unit #')   

% PLOT TREADMILL RUNNING SPEED
subplot(3,1,3) % tread speed
plot(treadTime, treadSpeed , 'k'); hold on
xlabel('Time (s)')
ylabel('Speed (cm/s)')

for i = 1:numel(onsets)
    fill(treadTime([onsets(i) onsets(i) offsets(i) offsets(i)]), [ylim fliplr(ylim)], 'r', 'FaceColor', 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
end
xlim(treadTime([1 end]))

title('Treadmill')
trmax = prctile(D.treadSpeed(treadIx), 99);




%% play movie
% play a movie of the session
figure(gcf)
t0 = treadTime(1);
win = 25;
for t = 1:500
    xd = [t0 t0 + win];
    for i = 1:3
        subplot(3,1,i)
        xlim(xd)
    end
    ylim([0 trmax])
    drawnow
    t0 = t0 + win/10;
    if t0 > treadTime(end)
        t0 = treadTime(1);
    end
    pause(0.1)
end

%% Do direction / orientation decoding

% example session to explore
Dstat = decode_stim(D, 1);


%%
Nsess = max(D.sessNumGratings);
clear Dstat
for sessionId = 1:Nsess
    Dstat(sessionId) = decode_stim(D, sessionId, 'runThreshold', 3);
end

%%

%% decoding error
circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;

mR = zeros(Nsess,1);
mRCi = zeros(Nsess,2);
mS = zeros(Nsess,1);
mSCi = zeros(Nsess,2);
nS = zeros(Nsess,1);
nR = zeros(Nsess,1);

figure(1); clf
for iSess = 1:Nsess
    aerr = abs(circdiff(Dstat(iSess).Stim, Dstat(iSess).decoderStimTot));
    
    inds = Dstat(iSess).runTrials;
    nR(iSess) = numel(inds);
    mR(iSess) = median(aerr(inds));
    mRCi(iSess,:) = bootci(1000, @median, aerr(inds));
    
    inds = setdiff(1:Dstat(iSess).NTrials, Dstat(iSess).runTrials);
    nS(iSess) = numel(inds);
    mS(iSess) = median(aerr(inds));
    mSCi(iSess,:) = bootci(1000, @median, aerr(inds));
    
    plot(mS(iSess)*[1 1], mRCi(iSess,:), 'Color', .5*[1 1 1]); hold on
    plot(mSCi(iSess,:), mR(iSess)*[1 1], 'Color', .5*[1 1 1]);
    h = plot(mS(iSess), mR(iSess), 'o');
    h.MarkerFaceColor = h.Color;
    
end

plot(xlim, xlim, 'k')
xlim([0 20])
ylim([0 20])
xlabel('Stationary')
ylabel('Running')
title('Median Decoding Error (degrees)')

%% Decode running
Nsess = max(D.sessNumGratings);
clear Dstat
for sessionId = 1:Nsess
    Dstat(sessionId) = decode_running(D, sessionId, 'Decoder', 'svm', 'runThreshold', 3);
end

%%


figure(1); clf
for s = 1:Nsess
    h = plot(Dstat(s).chance*[1 1], Dstat(s).accCi);
    hold on
    plot(Dstat(s).chance, Dstat(s).acc, 'o', 'Color', h.Color, 'MarkerFaceColor', h.Color);
end
plot(xlim, xlim, 'k')
xlabel('Chance level (Based on % running)')
ylabel('Decoding Accuracy')

%% Bootstrapped empirical analyses and tuning curve fits

fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
fname = fullfile(fpath, [subject 'TCstats.mat']);

if exist(fname, 'file')
    stat = load(fname);
else
    stat = tuning_empirical(D, 'binsize', 10e-3, ...
        'runningthresh', 3, ...
        'nboot', 100, ...
        'seed', 1234);
    save(fname, '-v7.3', '-struct', 'stat')
end



%% plot running tuning?

NC = numel(stat.TCfitR);
runrat = stat.speedTuning.rateSpdMu./stat.speedTuning.rateSpdMu(1,:);
[~, ind] = sort(runrat(end,:));


sx = ceil(sqrt(NC));
sy = round(sqrt(NC));


figure(1); clf
ax = plot.tight_subplot(sx, sy, 0.001, 0.001);
cmap = lines;
cmap(1,:) = .2*[1 1 1];
for i = 1:(sx*sy)
    
    
    set(gcf, 'currentaxes', ax(i))
    
    if i > NC
        axis off
        continue
    end
    
    fprintf('Unit %d/%d\n', i, NC)
    
    cc = ind(i);
    
    plot.errorbarFill(stat.speedTuning.bins, stat.speedTuning.rateSpdMu(:,cc), stat.speedTuning.rateSpdSe(:,cc), 'b', 'FaceColor', cmap(1,:), 'EdgeColor', 'none', 'FaceAlpha', .5); hold on
    plot(stat.speedTuning.bins, stat.speedTuning.rateSpdMu(:,cc), 'o', 'Color', cmap(1,:))
    
%     plot.errorbarFill(stat.speedTuning.bins, stat.speedTuning.rateSpdMuStim(:,cc), stat.speedTuning.rateSpdSeStim(:,cc), 'r')
    plot(xlim, (stat.speedTuning.rateSpdMu(1,cc) + stat.speedTuning.rateSpdSe(1,cc))*[1 1], 'k--')
    plot(xlim, (stat.speedTuning.rateSpdMu(1,cc) - stat.speedTuning.rateSpdSe(1,cc))*[1 1], 'k--')
%     plot(xlim, stat.speedTuning.rateSpdMu(1,cc)*[1 1], 'k')
    xlabel('Speed (cm / s)')
    ylabel('Firing Rate')
    axis off
    
    drawnow
    
end


%% Raw Running Modulation (ignore stimulus entirely)
% Just look at the entire session. Using labeled epochs of running and
% stationary, count the mean firing rate (while accounting for issues with
% stationarity by resampling from the epochs to hopefully match)

figure(1); clf
set(gcf, 'Color', 'w')

rateS = nan(NC, 3);
rateR = nan(NC, 1);
cmap = lines;

nbins = numel(stat.running(1).psthMu);
psthRunOnset = nan(nbins, NC);
psthBins = stat.running(1).psthBins;
numEpochs = nan(NC, 1);
isvalid = find(arrayfun(@(x) ~isempty(x.spikerate), stat.running));

subplot(2,2,1)
for cc = 1:NC
    if isempty(stat.running(cc).spikerate)
        continue
    end
    
    rateS(cc,:) = prctile(stat.running(cc).rateStatNull, [2.5 50 97.5]);
    rateR(cc) = stat.running(cc).rateRun;
    
    plot(rateS(cc,[1 3]),rateR(cc)*[1 1], 'Color', .5*[1 1 1]); hold on
    plot(rateS(cc,2), rateR(cc), 'ow', 'MarkerFaceColor', cmap(1,:))
   
    psthRunOnset(:,cc) = stat.running(cc).psthMu;
    
    numEpochs(cc) = numel(stat.running(cc).goodix);
end

plot(xlim, xlim, 'k')
title('Mean Firing Rate', 'Fontweight', 'normal')
xlabel('Stationary')
ylabel('Running')


suppressed = find(rateR < rateS(:,1));
enhanced = find(rateR > rateS(:,3));
nEnc = numel(enhanced);
nSup = numel(suppressed);
nTot = sum(~isnan(rateR));

fprintf('Found %d/%d enhanced (%02.2f%%)\n', nEnc, nTot, 100*nEnc/nTot)
fprintf('Found %d/%d suppressed (%02.2f%%)\n', nSup, nTot, 100*nSup/nTot)

for cc = suppressed(:)'
    plot(rateS(cc,[1 3]),rateR(cc)*[1 1], 'Color', .5*[1 1 1]); hold on
    plot(rateS(cc,2), rateR(cc), 'ow', 'MarkerFaceColor', cmap(2,:))
end

for cc = enhanced(:)'
    plot(rateS(cc,[1 3]),rateR(cc)*[1 1], 'Color', .5*[1 1 1]); hold on
    plot(rateS(cc,2), rateR(cc), 'ow', 'MarkerFaceColor', cmap(4,:))
end

subplot(2,2,2)
nfun = @(x) x./mean(x(psthBins<0,:));
plot(psthBins, nfun(psthRunOnset(:,enhanced)), 'Color', (1+cmap(4,:))/2); hold on
plot(psthBins, nfun(psthRunOnset(:,suppressed)), 'Color', (1+cmap(2,:))/2); hold on

plot(psthBins, mean(nfun(psthRunOnset(:,enhanced)),2), 'Color', cmap(4,:), 'Linewidth', 2)
plot(psthBins, mean(nfun(psthRunOnset(:,suppressed)),2), 'Color', cmap(2,:), 'Linewidth', 2)
xlabel('Time from running onset (s)')
ylabel('Relative Rate (mean normalized)')
ylim([.5 2])
xlim(psthBins([1 end]))

subplot(2,2,3)
nfun = @(x) (x - mean(x(psthBins<0,:)))./std(x(psthBins<0,:));
plot(psthBins, nfun(psthRunOnset(:,enhanced)), 'Color', (1+cmap(4,:))/2); hold on
plot(psthBins, nfun(psthRunOnset(:,suppressed)), 'Color', (1+cmap(2,:))/2); hold on

plot(psthBins, mean(nfun(psthRunOnset(:,enhanced)),2), 'Color', cmap(4,:), 'Linewidth', 2)
plot(psthBins, mean(nfun(psthRunOnset(:,suppressed)),2), 'Color', cmap(2,:), 'Linewidth', 2)
xlabel('Time from running onset (s)')
ylabel('Normalized Rate (z score)')
ylim([-5 5])
xlim(psthBins([1 end]))

subplot(2,2,4)
nfun = @(x) (x - mean(x(psthBins<0,:)));
plot(psthBins, nfun(psthRunOnset(:,enhanced)), 'Color', (1+cmap(4,:))/2); hold on
plot(psthBins, nfun(psthRunOnset(:,suppressed)), 'Color', (1+cmap(2,:))/2); hold on

plot(psthBins, mean(nfun(psthRunOnset(:,enhanced)),2), 'Color', cmap(4,:), 'Linewidth', 2)
plot(psthBins, mean(nfun(psthRunOnset(:,suppressed)),2), 'Color', cmap(2,:), 'Linewidth', 2)
xlabel('Time from running onset (s)')
ylabel('\Delta Rate (spikes/sec)')
ylim([-5 5])
xlim(psthBins([1 end]))

% Sanity Check: Check that this effect isn't a function of the number of running epochs
figure(2); clf
set(gcf, 'Color', 'w')
plot([1; 1]*numEpochs', (rateR-rateS(:,[1 3]))', '-k', 'Linewidth', 2); hold on
plot([1; 1]*numEpochs(suppressed)', (rateR(suppressed)-rateS(suppressed,[1 3]))', '-', 'Color', cmap(2,:), 'Linewidth', 2);
plot([1; 1]*numEpochs(enhanced)', (rateR(enhanced)-rateS(enhanced,[1 3]))', '-', 'Color', cmap(4,:), 'Linewidth', 2);
plot(xlim, [0 0], 'k--')
xlabel('Num Running Epochs')
ylabel('\Delta Firing Rate')

% display top 5 examples of suppression and enhancement
deltaFR = rateR(isvalid) - rateS(isvalid,2);
[~, ind] = sort(deltaFR);
ind = isvalid(ind);

figure(3); clf
set(gcf, 'Color', 'w')
nExamples = 10;
spacing = 0.05;
ax = plot.tight_subplot(nExamples, 2, spacing, 0.05, 0.05);
sm = 20;


for i = 1:nExamples
    cc = ind(i);
    
    set(gcf, 'currentaxes', ax((i-1)*2+1))
    yyaxis left
    plot(imgaussfilt(stat.running(cc).spikerate, sm))
    ylabel('Firing Rate')
    axis tight
    
    yyaxis right
    plot(imgaussfilt(stat.running(cc).runningspeed, sm))
    ylabel('Running Speed')
    axis tight
    if i==1
        title('Most Suppressed')
    end
end
xlabel('Time')

for i = 1:nExamples
    cc = ind(end-(i-1));

    set(gcf, 'currentaxes', ax((i-1)*2+2))
    yyaxis left
    plot(imgaussfilt(stat.running(cc).spikerate, sm))
    ylabel('Firing Rate')
    axis tight
    
    yyaxis right
    plot(imgaussfilt(stat.running(cc).runningspeed, sm))
    ylabel('Running Speed')
    axis tight
    
    if i==1
        title('Most Enhanced')
    end
end

xlabel('Time')

% PLOT running onset-aligned PSTHs 
figure(4); clf
set(gcf, 'Color', 'w')
ax = plot.tight_subplot(nExamples, 2, spacing, 0.05, 0.05);

for i = 1:nExamples
    cc = ind(i);
    
    set(gcf, 'currentaxes', ax((i-1)*2+1))
    plot.errorbarFill(stat.running(cc).psthBins, stat.running(cc).psthMu, stat.running(cc).psthSe); hold on
    plot(stat.running(cc).psthBins, stat.running(cc).psthMu, 'k', 'Linewidth', 2)
    plot(stat.running(cc).psthBins, stat.running(cc).psthNullCi, 'r--')
    ylabel('Firing Rate')
    axis tight
    
    if i==1
        title('Most Suppressed')
    end
end
xlabel('Time from Running Onset (s)')

for i = 1:nExamples
    cc = ind(end-(i-1));

    set(gcf, 'currentaxes', ax((i-1)*2+2))
    plot.errorbarFill(stat.running(cc).psthBins, stat.running(cc).psthMu, stat.running(cc).psthSe); hold on
    plot(stat.running(cc).psthBins, stat.running(cc).psthMu, 'k', 'Linewidth', 2)
    plot(stat.running(cc).psthBins, stat.running(cc).psthNullCi, 'r--')
    axis tight
    
    if i==1
        title('Most Enhanced')
    end
end
xlabel('Time from Running Onset (s)')

% TUning curve analysis
figure(5); clf
set(gcf, 'Color', 'w')
ax = plot.tight_subplot(nExamples, 2, spacing, 0.05, 0.05);

for i = 1:nExamples
    cc = ind(i);
    
    set(gcf, 'currentaxes', ax((i-1)*2+1))
    plot.errorbarFill(stat.TCfitS(cc).thetas, stat.TCfitS(cc).tuningCurve, stat.TCfitS(cc).tuningCurveSE*2, 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .5, 'EdgeColor', 'none'); hold on
    plot(stat.TCfitS(cc).thetas, stat.TCfitS(cc).tuningCurve, 'o', 'Color', cmap(1,:))
    plot.errorbarFill(stat.TCfitR(cc).thetas, stat.TCfitR(cc).tuningCurve, stat.TCfitR(cc).tuningCurveSE*2, 'k', 'FaceColor', cmap(2,:), 'FaceAlpha', .5, 'EdgeColor', 'none'); hold on
    plot(stat.TCfitR(cc).thetas, stat.TCfitR(cc).tuningCurve, 'o', 'Color', cmap(2,:))
    
    ylabel('Firing Rate')
    axis tight
    
    if i==1
        title('Most Suppressed')
    end
end
xlabel('Direction')

for i = 1:nExamples
    cc = ind(end-(i-1));

    set(gcf, 'currentaxes', ax((i-1)*2+2))
    plot.errorbarFill(stat.TCfitS(cc).thetas, stat.TCfitS(cc).tuningCurve, stat.TCfitS(cc).tuningCurveSE, 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .5, 'EdgeColor', 'none'); hold on
    plot(stat.TCfitS(cc).thetas, stat.TCfitS(cc).tuningCurve, 'o', 'Color', cmap(1,:))
    plot.errorbarFill(stat.TCfitR(cc).thetas, stat.TCfitR(cc).tuningCurve, stat.TCfitR(cc).tuningCurveSE, 'k', 'FaceColor', cmap(2,:), 'FaceAlpha', .5, 'EdgeColor', 'none'); hold on
    plot(stat.TCfitR(cc).thetas, stat.TCfitR(cc).tuningCurve, 'o', 'Color', cmap(2,:))
    ylabel('Firing Rate')
    axis tight
    
    if i==1
        title('Most Enhanced')
    end
end
xlabel('Time from Running Onset (s)')



%% TC diff
ix = ~arrayfun(@(x) isempty(x.tuningCurve), stat.TCfitR);
TCMaxR = arrayfun(@(x) max(x.tuningCurve), stat.TCfitR(ix));
TCMaxS = arrayfun(@(x) max(x.tuningCurve), stat.TCfitS(ix));
figure(1); clf
plot(TCMaxS, TCMaxR, 'o'); hold on
plot(xlim, xlim, 'k')
xlabel('Stationary')
ylabel('Running')
signrank(TCMaxR, TCMaxS)

%%

TCdiffNull = stat.TCempirical.TCdiffNull;
TCdiff = stat.TCempirical.TCdiff;
maxFRdiffNull = stat.TCempirical.maxFRdiffNull;
maxFRdiff = stat.TCempirical.maxFRdiff;

nullLevel = prctile(TCdiffNull, 95, 2);

figure(2); clf
plot(nullLevel, TCdiff, '.'); hold on
plot(xlim, xlim, 'k')
xlabel("95th percentile for null running modulation")
ylabel("Empirical running modulation")
text(min(xlim) + .8*range(xlim), min(ylim) + .2*range(ylim), 'Favors Null')
text(min(xlim) + .2*range(xlim), min(ylim) + .8*range(ylim), 'Reject Null')

sigTCdiff = TCdiff > nullLevel;
fprintf('%d/%d units have TC modulation (%02.2f)%%\n', sum(sigTCdiff), numel(sigTCdiff), mean(sigTCdiff))
title('TC diff')
% max FR
nullLevel = prctile(maxFRdiffNull, [2.5 97.5], 2);

figure(3); clf
set(gcf, 'Color', 'w')
subplot(1,2,1)
plot(nullLevel(:,1), maxFRdiff, '.'); hold on
plot(xlim, xlim, 'k')
xlabel("2.5th percentile for null max FR modulation")
ylabel("Empirical FR modulation")
text(min(xlim) + .8*range(xlim), min(ylim) + .2*range(ylim), 'Reject Null')
text(min(xlim) + .2*range(xlim), min(ylim) + .8*range(ylim), 'Favors Null')

subplot(1,2,2)
plot(nullLevel(:,2), maxFRdiff, '.'); hold on
plot(xlim, xlim, 'k')

xlabel("97.5th percentile for null max FR modulation")
ylabel("Empirical FR modulation")
text(min(xlim) + .8*range(xlim), min(ylim) + .2*range(ylim), 'Favors Null')
text(min(xlim) + .2*range(xlim), min(ylim) + .8*range(ylim), 'Reject Null')

sigFRmod = (maxFRdiff < nullLevel(:,1) | maxFRdiff > nullLevel(:,2));
fprintf('%d/%d units have Max FR modulation (%02.2f)%%\n', sum(sigFRmod), numel(sigFRmod), mean(sigFRmod))

modUnits = union(find(sigFRmod), find(sigTCdiff));
nMod = numel(modUnits);

fprintf('%d units have potential modulation\n', nMod)

iUnit = 1;
%%


% needs: spls, dfilt, lags, ths
% iUnit = iUnit + 1;
% if iUnit > nMod
%     iUnit = 1;
% end
% modUnits = 1:size(dfilt,2);
% nMod = numel(modUnits);
for iUnit = 1:nMod
    cc = modUnits(iUnit);
    
    fprintf('Unit: %d\n', cc)
    
    % find stable region of firing rate
    unitix = dfilt(:,cc);
    dur = median(D.GratingOffsets(unitix) - D.GratingOnsets(unitix));
    dur = max(dur, .1);
    win = [0.04 dur];
    
    nStim = numel(ths);
    FrateR = nan(numel(lags), nStim);
    FrateS = nan(numel(lags), nStim);
    TCR = nan(nStim, 3);
    TCS = nan(nStim, 3);
    
    figure(1); clf
    subplot(4,2,[1 3]) % no running
    
    spkS = [];
    spkR = [];
    thctr = 0;
    for th = 1:nStim
        iix = find(GratingDirections==ths(th) & ~runningTrial & unitix);
        
        nt = numel(iix);
        spk = squeeze(spks(iix,cc,:));
        if binsize == 1e-3
            [ii,jj] = find(spk);
            plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
        else
            spk = imboxfilt(spk, [1 3]);
            imagesc(lags, (1:nt)+thctr, spk); hold on
        end
        if size(spk,2) == 1
            spk = spk';
        end
        
        spkS = [spkS; spk];
        thctr = thctr + nt;
        
        R = sum(spk(:,tix),2);
        if isempty(R)
            continue
        end
        TCS(th,1) = mean(R);
        TCS(th,2:3) = bootci(100, @mean, R)';
        FrateS(:,th) = mean(spk);
    end
    title('Stationary')
    ylabel('Trials (sorted by direction)')
    axis tight
    
    subplot(4,2,[5 7]) % no running
    
    thctr = 0;
    for th = 1:nStim
        iix = find(GratingDirections==ths(th) & runningTrial & unitix);
        
        nt = numel(iix);
        spk = squeeze(spks(iix,cc,:));
        
        if binsize == 1e-3
            [ii,jj] = find(spk);
            plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
        else
            spk = imboxfilt(spk, [1 3]);
            imagesc(lags, (1:nt)+thctr, spk); hold on
        end
        
        if size(spk,2) == 1
            spk = spk';
        end
        
        spkR = [spkR; spk];
        
        thctr = thctr + nt;
        
        R = sum(spk(:,tix),2);
        if isempty(R) || numel(R) < 5
            continue
        end
        TCR(th,1) = mean(R);
        TCR(th,2:3) = bootci(100, @mean, R)';
        FrateR(:,th) = mean(spk);
    end
    title('Running')
    axis tight
    colormap(1-gray)
    ylabel('Trials (sorted by direction)')
    xlabel('Time from Grating Onset')
    
    
    
    vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)];
    clim = [min(vals(:)) max(vals(:))];
    
    subplot(4,2,2)
    m = FrateS';
    m = imboxfilt(m, [1 3]);
    imagesc(lags, ths, m, clim)
    axis tight
    ylabel('Direction')
    xlabel('Time')
    title('PSTH Stationary')
    
    subplot(4,2,4)
    m = FrateR';
    m = imboxfilt(m, [1 3]);
    imagesc(lags, ths, m, clim)
    axis tight
    ylabel('Direction')
    xlabel('Time')
    title('PSTH Running')
    
    colormap(1-gray)
    
    subplot(4,2,6)
    cmap = lines;
    plot(ths, TCS(:,1), 'k', 'Color', cmap(1,:)); hold on
    fill([ths' fliplr(ths')], [TCS(:,2)' fliplr(TCS(:,3)')], 'k', 'EdgeColor', cmap(1,:))
    
    plot(ths, TCR(:,1), 'k', 'Color', cmap(2,:)); hold on
    fill([ths' fliplr(ths')], [TCR(:,2)' fliplr(TCR(:,3)')], 'k', 'EdgeColor', cmap(2,:))
    title('Tuning Curve')
    xlabel('Direction')
    ylabel('Spike Count')
    xlim([0 360])
    set(gca, 'box', 'off')
    
    subplot(4,2,8)
    plot(lags, nanmean(FrateS, 2)/binsize, 'Color', cmap(1,:)); hold on
    plot(lags, nanmean(FrateR, 2)/binsize, 'Color', cmap(2,:))
    axis tight
    
    title('Mean across directions')
    xlabel('Time from Grat Onset')
    ylabel('Firing Rate')
    
    plot.suplabel(sprintf('Unit %d', cc), 't');
    plot.fixfigure(gcf, 10, [6 8]);
    saveas(gcf, fullfile('Figures', 'HuklabTreadmill', sprintf('examplemod%02.0f.png', cc)))
end


%% Some summaries

figure(1); clf
histogram(arrayfun(@(x) x.llrpval, fitS), 100); hold on
histogram(arrayfun(@(x) x.llrpval, fitR), 100);
legend({'Stationary', 'Running'})
xlabel('LL ratio pval')
title('How many cells are "tuned"?')




%% Plot all tuning curves
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));


figure(1); clf
ax = plot.tight_subplot(sx, sy, 0.001, 0.001);
for cc = 1:NC
    if min(fitS(cc).numTrials, fitR(cc).numTrials) < 50
        continue
    end
    
    if fitS(cc).llrpval > 0.05 && fitR(cc).llrpval > 0.05
        continue
    end
    fprintf("Unit %d/%d\n", cc, NC)

    set(gcf, 'currentaxes', ax(cc))
    
    cmap = lines;
    % STATIONARY
    h = errorbar(fitS(cc).thetas, fitS(cc).tuningCurve, fitS(cc).tuningCurveSE, 'o', 'Color', cmap(1,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(1,:);
    h.CapSize = 0;
    hold on
    if fitS(cc).llrpval > 0.05
        plot(xlim, mean(fitS(cc).tuningCurve)*[1 1], 'Color', cmap(1,:))
    else
        plot(linspace(0, 360, 100), fitS(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(1,:))
    end
    
    % RUNNING
    h = errorbar(fitR(cc).thetas, fitR(cc).tuningCurve, fitR(cc).tuningCurveSE, 'o', 'Color', cmap(2,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(2,:);
    h.CapSize = 0;
    hold on
    if fitR(cc).llrpval > 0.05
        plot(xlim, mean(fitR(cc).tuningCurve)*[1 1], 'Color', cmap(2,:))
    else
        plot(linspace(0, 360, 100), fitR(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(2,:))
    end

    set(gca, 'XTick', [], 'YTick', [])
%     set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    text(10, .1*max(ylim), sprintf('%d', cc), 'fontsize', 5)
%     axis off
%     title(cc)
end

%%
ntrials = arrayfun(@(x,y) min(x.numTrials, y.numTrials), fitR, fitS);
figure(2); clf
istuned = arrayfun(@(x,y) (x.llrpval < 0.05) & (y.llrpval < 0.05), fitR, fitS);
istuned = istuned & ntrials > 50;
fprintf('%d units that are tuned\n', sum(istuned))

wrappi = @(x) mod(x/pi, 1)*pi;
wrap2pi = @(x) mod(x/2/pi, 1)*2*pi;

mfr = arrayfun(@(x,y) max([x.tuningCurve; y.tuningCurve]), fitS(istuned), fitR(istuned));

bS = arrayfun(@(x) x.paramsML(4), fitS(istuned));
bR = arrayfun(@(x) x.paramsML(4), fitR(istuned));
bSsd = arrayfun(@(x) x.paramsSD(4), fitS(istuned));
bRsd = arrayfun(@(x) x.paramsSD(4), fitR(istuned));

AS = arrayfun(@(x) x.paramsML(3), fitS(istuned));
AR = arrayfun(@(x) x.paramsML(3), fitR(istuned));
ASsd = arrayfun(@(x) x.paramsSD(3), fitS(istuned));
ARsd = arrayfun(@(x) x.paramsSD(3), fitR(istuned));

thS = arrayfun(@(x) x.paramsML(1), fitS(istuned));
thR = arrayfun(@(x) x.paramsML(1), fitR(istuned));
thSsd = arrayfun(@(x) x.paramsSD(1), fitS(istuned));
thRsd = arrayfun(@(x) x.paramsSD(1), fitR(istuned));

thS = wrap2pi(thS);
thR = wrap2pi(thR);

vS = arrayfun(@(x) x.paramsML(2), fitS(istuned));
vR = arrayfun(@(x) x.paramsML(2), fitR(istuned));

lS = arrayfun(@(x) x.paramsML(end), fitS(istuned));
lR = arrayfun(@(x) x.paramsML(end), fitR(istuned));
lSsd = arrayfun(@(x) x.paramsSD(end), fitS(istuned));
lRsd = arrayfun(@(x) x.paramsSD(end), fitR(istuned));

thS(lS > .5) = wrappi(thS(lS > .5));
thR(lR > .5) = wrappi(thR(lR > .5));


subplot(2,2,1)
errorbar(bS, bR, bSsd, bSsd, bRsd, bRsd, 'o', 'Color', .5*[1 1 1], 'MarkerSize', 5, 'MarkerFaceColor', cmap(1,:), 'CapSize', 0); hold on
xlim([0 20])
ylim([0 20])
plot(xlim, xlim, 'k')
title('Baseline')
xlabel('Stationary')
ylabel('Running')

subplot(2,2,2)
% plot
mfr = max(mfr, 10);
errorbar(AS./mfr, AR./mfr, ASsd./mfr, ASsd./mfr, ARsd./mfr, ARsd./mfr, 'o', 'Color', .5*[1 1 1], 'MarkerSize', 5, 'MarkerFaceColor', cmap(1,:), 'CapSize', 0); hold on
% errorbar(AS, AR, ASsd, ASsd, ARsd, ARsd, 'o', 'Color', .5*[1 1 1], 'MarkerSize', 5, 'MarkerFaceColor', cmap(1,:), 'CapSize', 0); hold on
plot(xlim, xlim, 'k')
title('Amplitude (normalized by max FR)')
xlabel('Stationary')
ylabel('Running')
xlim([0 1])
ylim([0 1])


subplot(2,2,3)
errorbar(thS, thR, thSsd, thSsd, thRsd, thRsd, 'o', 'Color', .5*[1 1 1], 'MarkerSize', 5, 'MarkerFaceColor', cmap(1,:), 'CapSize', 0); hold on
plot(xlim, xlim, 'k')
title('Ori Pref')
xlabel('Stationary')
ylabel('Running')
xlim([0 1]*pi)
ylim([0 1]*pi)

subplot(2,2,4)
errorbar(lS, lR, lSsd, lSsd, lRsd, lRsd, 'o', 'Color', .5*[1 1 1], 'MarkerSize', 5, 'MarkerFaceColor', cmap(1,:), 'CapSize', 0); hold on
title('Lambda')
xlabel('Stationary')
ylabel('Running')
xlim([0 1])
ylim([0 1])
plot(xlim, xlim, 'k')


%% Units that became more direction tuned
figure(10); clf

lrat = max(lS, .1)./max(lR, .1);

tunedList = find(istuned);
idx = tunedList(lrat > 2);
n = numel(idx);
sx = ceil(sqrt(n));
sy = round(sqrt(n));
for i = 1:n
    subplot(sx, sy, i)
    cc = idx(i);
     % STATIONARY
    h = errorbar(fitS(cc).thetas, fitS(cc).tuningCurve, fitS(cc).tuningCurveSE, '-o', 'Color', cmap(1,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(1,:);
    h.CapSize = 0;
    hold on
    if fitS(cc).llrpval > 0.05
        plot(xlim, mean(fitS(cc).tuningCurve)*[1 1], 'Color', cmap(1,:))
    else
        plot(linspace(0, 360, 100), fitS(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(1,:))
    end
    
    % RUNNING
    h = errorbar(fitR(cc).thetas, fitR(cc).tuningCurve, fitR(cc).tuningCurveSE, '-o', 'Color', cmap(2,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(2,:);
    h.CapSize = 0;
    hold on
    if fitR(cc).llrpval > 0.05
        plot(xlim, mean(fitR(cc).tuningCurve)*[1 1], 'Color', cmap(2,:))
    else
        plot(linspace(0, 360, 100), fitR(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(2,:))
    end

    set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    text(10, .8*range(ylim)+min(ylim), sprintf('%d', cc), 'fontsize', 8)
end

plot.suplabel('Direction', 'x');
plot.suplabel('Firing Rate', 'y');
plot.suplabel('Units that became more direction tuned', 't')

%% Units that became less direction tuned
figure(10); clf

lrat = max(lS, .1)./max(lR, .1);

tunedList = find(istuned);
idx = tunedList(lrat < .5);
n = numel(idx);
sx = ceil(sqrt(n));
sy = round(sqrt(n));
for i = 1:n
    subplot(sx, sy, i)
    cc = idx(i);
     % STATIONARY
    h = errorbar(fitS(cc).thetas, fitS(cc).tuningCurve, fitS(cc).tuningCurveSE, '-o', 'Color', cmap(1,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(1,:);
    h.CapSize = 0;
    hold on
    if fitS(cc).llrpval > 0.05
        plot(xlim, mean(fitS(cc).tuningCurve)*[1 1], 'Color', cmap(1,:))
    else
        plot(linspace(0, 360, 100), fitS(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(1,:))
    end
    
    % RUNNING
    h = errorbar(fitR(cc).thetas, fitR(cc).tuningCurve, fitR(cc).tuningCurveSE, '-o', 'Color', cmap(2,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(2,:);
    h.CapSize = 0;
    hold on
    if fitR(cc).llrpval > 0.05
        plot(xlim, mean(fitR(cc).tuningCurve)*[1 1], 'Color', cmap(2,:))
    else
        plot(linspace(0, 360, 100), fitR(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(2,:))
    end

    set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    text(10, .8*range(ylim)+min(ylim), sprintf('%d', cc), 'fontsize', 8)
end

plot.suplabel('Direction', 'x');
plot.suplabel('Firing Rate', 'y');
plot.suplabel('Units that became less direction tuned', 't')


%% Units that increased amplitude
figure(10); clf

amprat = max(AS, .1)./max(AR, .1);

tunedList = find(istuned);
idx = tunedList(amprat < .8);
n = numel(idx);
sx = ceil(sqrt(n));
sy = round(sqrt(n));
for i = 1:n
    subplot(sx, sy, i)
    cc = idx(i);
     % STATIONARY
    h = errorbar(fitS(cc).thetas, fitS(cc).tuningCurve, fitS(cc).tuningCurveSE, '-o', 'Color', cmap(1,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(1,:);
    h.CapSize = 0;
    hold on
    if fitS(cc).llrpval > 0.05
        plot(xlim, mean(fitS(cc).tuningCurve)*[1 1], 'Color', cmap(1,:))
    else
        plot(linspace(0, 360, 100), fitS(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(1,:))
    end
    
    % RUNNING
    h = errorbar(fitR(cc).thetas, fitR(cc).tuningCurve, fitR(cc).tuningCurveSE, '-o', 'Color', cmap(2,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(2,:);
    h.CapSize = 0;
    hold on
    if fitR(cc).llrpval > 0.05
        plot(xlim, mean(fitR(cc).tuningCurve)*[1 1], 'Color', cmap(2,:))
    else
        plot(linspace(0, 360, 100), fitR(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(2,:))
    end

    set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    text(10, .8*range(ylim)+min(ylim), sprintf('%d', cc), 'fontsize', 8)
end

plot.suplabel('Direction', 'x');
plot.suplabel('Firing Rate', 'y');
plot.suplabel('Units that increased amplitude', 't')


%% Units that decreased amplitude
figure(10); clf

amprat = max(AS, .1)./max(AR, .1);

tunedList = find(istuned);
idx = tunedList(amprat < 1.2);
n = numel(idx);
sx = ceil(sqrt(n));
sy = round(sqrt(n));
for i = 1:n
    subplot(sx, sy, i)
    cc = idx(i);
     % STATIONARY
    h = errorbar(fitS(cc).thetas, fitS(cc).tuningCurve, fitS(cc).tuningCurveSE, '-o', 'Color', cmap(1,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(1,:);
    h.CapSize = 0;
    hold on
    if fitS(cc).llrpval > 0.05
        plot(xlim, mean(fitS(cc).tuningCurve)*[1 1], 'Color', cmap(1,:))
    else
        plot(linspace(0, 360, 100), fitS(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(1,:))
    end
    
    % RUNNING
    h = errorbar(fitR(cc).thetas, fitR(cc).tuningCurve, fitR(cc).tuningCurveSE, '-o', 'Color', cmap(2,:));
    h.MarkerSize = 2;
    h.MarkerFaceColor = cmap(2,:);
    h.CapSize = 0;
    hold on
    if fitR(cc).llrpval > 0.05
        plot(xlim, mean(fitR(cc).tuningCurve)*[1 1], 'Color', cmap(2,:))
    else
        plot(linspace(0, 360, 100), fitR(cc).tuningFun(linspace(0, 360, 100)), 'Color', cmap(2,:))
    end

    set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    text(10, .8*range(ylim)+min(ylim), sprintf('%d', cc), 'fontsize', 8)
end

plot.suplabel('Direction', 'x');
plot.suplabel('Firing Rate', 'y');
plot.suplabel('Units that decreased amplitude', 't')
%% plot tuning curves sorted

figure(33); clf

thetas = linspace(0, 360, 100);

S = cell2mat(arrayfun(@(x,y) x.tuningFun(thetas)./y, fitS(istuned)', mfr', 'uni', 0));
R = cell2mat(arrayfun(@(x,y) x.tuningFun(thetas)./y, fitR(istuned)', mfr', 'uni', 0));
n = sum(istuned);

[~, ind] = sort(vS);

figure(10); clf
subplot(1,3,1)
imagesc(thetas, 1:n, S(ind,:))
title('Stationary')
subplot(1,3,2)
imagesc(thetas, 1:n, R(ind,:))
title('Running')
subplot(1,3,3)
imagesc(thetas, 1:n, R(ind,:)-S(ind,:))
title('Difference')

[~, ind] = sort(thR);

figure(11); clf
subplot(1,3,1)
imagesc(thetas, 1:n, S(ind,:))
title('Stationary')
subplot(1,3,2)
imagesc(thetas, 1:n, R(ind,:))
title('Running')
subplot(1,3,3)
imagesc(thetas, 1:n, R(ind,:)-S(ind,:))
title('Difference')

%% Firing rate analysis

istuned = true(numel(fitS), 1);
figure(10); clf

x = arrayfun(@(x) mean(x.tuningCurve), fitS(istuned));
y = arrayfun(@(x) mean(x.tuningCurve), fitR(istuned));

subplot(1,2,1)
plot(x, y, 'o'); hold on
xlabel('Stationary')
ylabel('Running')
title('Mean Firing Rate')
plot(xlim, xlim, 'k')

mxci = bootci(100, @median, x);
myci = bootci(100, @median, y);

fprintf('MEAN FIRING RATE\n')
fprintf('Stationary FR median = %02.2f [%02.2f, %02.2f]\n', median(x), mxci(1), mxci(2))
fprintf('Running FR median = %02.2f [%02.2f, %02.2f]\n', median(y), myci(1), myci(2))

[pval, h, stats] = ranksum(x, y);

fprintf('wilcoxon pval = %02.5f\n', pval)

figure(2); clf
set(gcf, 'Color', 'w')

subplot(1,2,1)

m = geomean(y./x);
mci = bootci(1000, @geomean, y./x);

fprintf('Ratio of FR = %02.2f [%02.2f, %02.2f]\n', m, mci(1), mci(2))

[cnt, bins] = histcounts(y./x, 100);
bins = (bins(1:end-1) + bins(2:end))/2;

bar(bins, cnt, 'FaceColor', .5*[1 1 1]);
hold on

fill(mci([1 1 2 2]), [ylim, fliplr(ylim)], 'r', 'FaceAlpha', .5)

xlabel('mean FR Ratio Running : Stationary')
ylabel('Count')


% MAX FIRING RATE
fprintf('MAX FIRING RATE\n')

figure(10);

subplot(1,2,2)

x = arrayfun(@(x) max(x.tuningCurve), fitS(istuned));
y = arrayfun(@(x) max(x.tuningCurve), fitR(istuned));

plot(x, y, 'o'); hold on
xlabel('Stationary')
ylabel('Running')
title('Max Firing Rate')
plot(xlim, xlim, 'k')

mxci = bootci(100, @median, x);
myci = bootci(100, @median, y);
fprintf('Stationary FR median = %02.2f [%02.2f, %02.2f]\n', median(x), mxci(1), mxci(2))
fprintf('Running FR median = %02.2f [%02.2f, %02.2f]\n', median(y), myci(1), myci(2))

[pval, h, stats] = ranksum(x, y);
fprintf('wilcoxon pval = %02.5f\n', pval)

figure(2);

subplot(1,2,2)
m = geomean(y./x);
mci = bootci(1000, @geomean, y./x);

fprintf('Ratio of FR = %02.2f [%02.2f, %02.2f]\n', m, mci(1), mci(2))

[cnt, bins] = histcounts(y./x, 100);
bins = (bins(1:end-1) + bins(2:end))/2;

bar(bins, cnt, 'FaceColor', .5*[1 1 1]);
hold on

fill(mci([1 1 2 2]), [ylim, fliplr(ylim)], 'r', 'FaceAlpha', .5)

xlabel('Max FR Ratio Running : Stationary')
ylabel('Count')



%%

figure(33); clf

thetas = fitS(1).thetas;

S = cell2mat(arrayfun(@(x,y) x.tuningCurve(:)'./y, fitS(istuned)', mfr', 'uni', 0));
R = cell2mat(arrayfun(@(x,y) x.tuningCurve(:)'./y, fitR(istuned)', mfr', 'uni', 0));
n = sum(istuned);

[~, ind] = sort(vS);

figure(10); clf
subplot(1,3,1)
imagesc(thetas, 1:n, S(ind,:))
title('Stationary')
subplot(1,3,2)
imagesc(thetas, 1:n, R(ind,:))
title('Running')
subplot(1,3,3)
imagesc(thetas, 1:n, R(ind,:)-S(ind,:))
title('Difference')

[~, ind] = sort(thR);

figure(11); clf
subplot(1,3,1)
imagesc(thetas, 1:n, S(ind,:))
title('Stationary')
subplot(1,3,2)
imagesc(thetas, 1:n, R(ind,:))
title('Running')
subplot(1,3,3)
imagesc(thetas, 1:n, R(ind,:)-S(ind,:))
title('Difference')




%%



thetas = linspace(0, 360, 100);
for cc = find(istuned)
    plot3(cc*ones(100,1), thetas, fitS(cc).tuningFun(thetas)./max(fitS(cc).tuningFun(thetas))); hold on
end

%%
figure(10); clf;
plot(AS, mfr, '.'); hold on
plot(AR, mfr, '.')
xlabel('Amplitude')
ylabel('Max Firing Rate')

figure(11); clf
plot(bS, AS, '.'); hold on
plot(bR, AR, '.')

figure(12); clf
% plot(

%% Step over cells, plot PSTH as image
cc = cc + 1;
if cc > NC
    cc = 1;
end
vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)];
clim = [min(vals(:)) max(vals(:))];
figure(1); clf
subplot(1,2,1)

imagesc(lags, ths, psthsRunning(:,:,cc)', clim)
ylabel('Direction')
xlabel('Time')
title('Running')

subplot(1,2,2)
imagesc(lags, ths, psthsNoRunning(:,:,cc)', clim)
ylabel('Direction')
xlabel('Time')
title('No Running')
colormap(plot.viridis)


%% plot Tuning Curves
win = [0.04 .4];
iix = lags > win(1) & lags < win(2);
tdur = lags(find(iix,1,'last'))-lags(find(iix,1,'first'));
tcRun = squeeze(nansum(psthsRunning(iix,:,:)))/tdur;
tcNoRun = squeeze(nansum(psthsNoRunning(iix,:,:)))/tdur;

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));


figure(1); clf
ax = plot.tight_subplot(sx, sy, 0.01, 0.01);
for cc = 1:NC
    fprintf("Unit %d/%d\n", cc, NC)
%     subplot(sx, sy, cc)
%     inds = tcRun(:,cc)>0;
    if tcRun(1,cc) == 0
        inds = 2:numel(ths);
    else
        inds = 1:(numel(ths)-1);
    end
    set(gcf, 'currentaxes', ax(cc))
    plot(ths(inds), tcRun(inds,cc), 'k', 'Linewidth', 2); hold on
    plot(ths(inds), tcNoRun(inds,cc), 'r', 'Linewidth', 2); hold on
    set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    axis off
%     title(cc)
end

% plot.fixfigure(gcf, 12, [14 14])
% a = plot.suplabel('Spike Rate', 'y'); 
% a.FontSize = 20;
% a = plot.suplabel('Direction', 'x');
% a.FontSize = 20;

%%
figure(2); clf
set(gcf, 'Color', 'w')
plot(max(tcNoRun), max(tcRun), 'ow', 'MarkerFaceColor', .5*[1 1 1])
hold on
plot(xlim, xlim, 'k')
xlabel('Max Rate (Stationary)')
ylabel('Max Rate (Running)')
%%
cc = 1;

%%

cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(10); clf
nth = numel(unique(D.GratingDirections));
cmap = parula(nth);
ax = plot.tight_subplot(2, nth, 0.01, 0.01);
    
vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)]/binsize;
clim = [min(vals(:)) max(vals(:))];


for ith = 1:nth
    
    set(gcf, 'currentAxes', ax(ith));
    plot(lags, imgaussfilt(psthsNoRunning(:,ith,cc)/binsize, 2), 'Color', cmap(ith,:), 'Linewidth', 2); hold on
    clr = (cmap(ith,:) + [1 1 1])/2;
    plot(lags, imgaussfilt(psthsRunning(:,ith,cc)/binsize, 2), '-', 'Color', clr, 'Linewidth', 2); hold on
    ylim(clim)
    axis off
    if ith==1
        text(lags(1), .9*clim(2), sprintf('Unit: %d', cc))
        text(lags(1), .8*clim(2), 'Running', 'Color', clr)
        text(lags(1), .7*clim(2), 'No Running', 'Color', cmap(ith,:))
    end
    set(gcf, 'currentAxes', ax(ith+nth));
    [dx, dy] = pol2cart(ths(ith)/180*pi, 1);
    q = quiver(0,0,dx,dy,'Color', cmap(ith,:), 'Linewidth', 5, 'MaxHeadSize', 2); hold on
%     plot([0 dx], [0 dy], 'Color', cmap(ith,:), 'Linewidth', 5); hold on
%     plot(dx, dy, 'o', 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 20)
    
%     R = [cos(pi/2) sin(pi/2); -sin(pi/2) -cos(pi/2)];
    
    
%     for i = [90 270]
%         [dx, dy] = pol2cart((ths(ith) + i)/180*pi, .1);
%         plot(-dx, -dy, 'o', 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 10)
% %     S = [1 0; 0 1];
% %     dxdy = [dx dy] * R*S;
% %     plot(dxdy(1), dxdy(2), 
% %     dxdy = [dx dy] * -R*S;
% %     plot(dxdy(1), dxdy(2), 'o', 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 20)
%     
%     end
    xlim([-1 1]*2)
    ylim([-1 1]*2)
    axis off
end

set(gcf, 'Color', 'w')


%%

[Exp,S] = io.dataFactoryTreadmill(6);
% add unit quality (since some analyses require this field)
Exp.osp.cgs = ones(size(Exp.osp.cids))*2;
io.checkCalibration(Exp);

D = io.get_drifting_grating_output(Exp);

exname = Exp.FileTag;
outdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'processed');
fname = fullfile(outdir, exname);

save(fname, '-v7', '-struct', 'D')

%% copy to server (for python analyses)
old_dir = pwd;

cd(outdir)
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/HuklabTreadmill/processed/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command exname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)

cd(old_dir)

