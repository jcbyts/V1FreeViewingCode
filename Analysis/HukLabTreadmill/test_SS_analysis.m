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
ip.parse(varargin{:});

%% bin spikes in window aligned ton onset
binsize = ip.Results.binsize;
win = ip.Results.win;

spikeIds = unique(D.spikeIds);
NC = numel(spikeIds);

mapSpikeIds = zeros(max(D.spikeIds), 1);
mapSpikeIds(spikeIds) = (1:NC)';

bs = (D.spikeTimes==0) + ceil(D.spikeTimes/binsize);

spbn = sparse(bs, mapSpikeIds(D.spikeIds), ones(numel(bs), 1));
lags = win(1):binsize:win(2);
blags = ceil(lags/binsize);

numlags = numel(blags);
balign = ceil(D.GratingOnsets/binsize);
validix = (balign + blags(end)) < size(spbn,1);
balign = balign(validix);

numStim = numel(balign);

treadTime = D.treadTime;
treadSpeed = D.treadSpeed;
iix = ~(isnan(D.treadTime) | isnan(D.treadSpeed));

treadTime = treadTime(iix);
treadSpeed = treadSpeed(iix);

% binning treadmill speed
treadBins = ceil(treadTime/binsize);

treadSpd = nan(size(spbn,1), 1); % make same size as spike bins

% % interpolate nan's
treadSpd(treadBins(~isnan(treadSpeed))) = treadSpeed(~isnan(treadSpeed));
% treadSpd = abs(repnan(treadSpd, 'pchip')); % absolute value (not forward vs. backwards)

% only use valid gratings
GratingDirections = D.GratingDirections(validix);

% Do the binning here
disp('Binning spikes')
spks = zeros(numStim, NC, numlags);
tspd = zeros(numStim, numlags);
dfilt = false(numStim, NC); % this is the "data filter" it says which Gratings correspond to the selected unit

for cc = 1:NC
   dfilt(:,cc) = ismember(D.sessNumGratings(validix), unique(D.sessNumSpikes(D.spikeIds==spikeIds(cc)))); 
end

for i = 1:numlags
    spks(:,:,i) = spbn(balign + blags(i),:);
    tspd(:,i) = treadSpd(balign + blags(i));
end

disp('Done')
cc = 1;


%% plot distribution of running speeds
thresh = 1;

figure(10); clf
runningSpd = mean(tspd,2);
runningTrial = runningSpd > thresh;

histogram(runningSpd(~runningTrial), 'binEdges', 0:.1:25, 'FaceColor', [.5 .5 .5])
hold on
clr = [1 .2 .2];
histogram(runningSpd(runningTrial), 'binEdges', 0:.1:25, 'FaceColor', clr)
plot.fixfigure(gcf,10,[4 4]);
xlabel('Running Speed (cm/s)')
ylabel('Trial Count') 
text(1, mean(ylim), sprintf('Running Trials (n=%d)', sum(runningTrial)), 'Color', clr, 'FontSize', 14)
text(1, mean(ylim)+.1*range(ylim), sprintf('Stationary Trials (n=%d)', sum(~runningTrial)), 'Color', .5*[1 1 1], 'FontSize', 14)
set(gca, 'YScale', 'linear')


%% Get PSTHs
ths = unique(D.GratingDirections);
fkern = ones(5,1)/5;
spksfilt = spks;
% for i = 1:numStim
%     spksfilt(i,:,:) = filtfilt(fkern, 1, squeeze(spksfilt(i,:,:))')';
% end

nd = numel(ths);
psthsRunning = zeros(numlags, nd, NC);
psthsNoRunning = zeros(numlags, nd, NC);
for cc = 1:NC % cells
    fprintf("Unit %d/%d\n", cc, NC)
    for i = 1:nd % directions
        iix = GratingDirections==ths(i) & dfilt(:,cc);
        psthsRunning(:,i,cc) = squeeze(mean(spksfilt(iix & runningTrial,cc,:),1))';
        psthsNoRunning(:,i,cc) = squeeze(mean(spksfilt(iix & ~runningTrial,cc,:),1))';
    end
end

% % trim filtering artifacts
% psthsRunning(1:10,:,:) = nan;
% psthsRunning(end-10:end,:,:) = nan;
% psthsNoRunning(1:10,:,:) = nan;
% psthsNoRunning(end-10:end,:,:) = nan;

cc = 1;
%% Step through and plot individual cells
% cc = 1
figure(1); clf

    
cc = cc + 1;
if cc > NC
    cc = 1;
end

% cc = 401;
% cc = 318;
% cc = 42;

fprintf('Unit: %d\n', cc)
subplot(1,2,1) % no running

spkS = [];
spkR = [];
thctr = 0;
for th = 1:numel(ths)
    iix = find(GratingDirections==ths(th) & ~runningTrial & dfilt(:,cc));

    nt = numel(iix);
    spk = squeeze(spks(iix,cc,:));
    if binsize == 1e-3
        [ii,jj] = find(spk);
        plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
    else
        spk = imboxfilt(spk, [1 3]);
        imagesc(lags, (1:nt)+thctr, spk); hold on
    end
    spkS = [spkS; spk];
    thctr = thctr + nt;
end
title('No Running')
axis tight

subplot(1,2,2)

thctr = 0;
for th = 1:numel(ths)
    iix = find(GratingDirections==ths(th) & runningTrial & dfilt(:,cc));
    
    nt = numel(iix);
    spk = squeeze(spks(iix,cc,:));
    
    if binsize == 1e-3
        [ii,jj] = find(spk);
        plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
    else
        spk = imboxfilt(spk, [1 3]);
        imagesc(lags, (1:nt)+thctr, spk); hold on
    end
    spkR = [spkR; spk];

    thctr = thctr + nt;
end
title('Running')
axis tight
colormap(1-gray)

figure(2); clf

vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)];
clim = [min(vals(:)) max(vals(:))];

subplot(1,2,1)

m = psthsRunning(:,:,cc)';
m = imboxfilt(m, [1 3]);
imagesc(lags, ths, m, clim)
ylabel('Direction')
xlabel('Time')
title('Running')

subplot(1,2,2)
m = psthsNoRunning(:,:,cc)';
m = imboxfilt(m, [1 3]);
imagesc(lags, ths, m, clim)
ylabel('Direction')
xlabel('Time')
title('No Running')
colormap(1-gray)


figure(3); clf
iix = lags > 0 & lags < .2;
NT = size(spkS,1);
plot((1:NT)/NT, mean(spkS(:,iix),2), '-o', 'Linewidth', 2); hold on
NT = size(spkR,1);
plot((1:NT)/NT, mean(spkR(:,iix),2), '-o', 'Linewidth', 2)


%% empirical tuning curve differences
rng(1234) % set random seed for reproducing the exact numbers
nboot = 1000;
TCdiffNull = nan(NC, nboot);
maxFRdiffNull = nan(NC, nboot);
TCdiff = nan(NC,1);
maxFRdiff = nan(NC,1);

for cc = 1:NC
    fprintf('%d/%d\n', cc, NC)
    unitix = dfilt(:,cc);
    dur = median(D.GratingOffsets(unitix) - D.GratingOnsets(unitix));
    dur = max(dur, .1);
    win = [0.04 dur];
    
    % find stable region of firing rate
    spk = squeeze(spks(unitix,cc,:));
    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);
    R = imboxfilt(R, 21);
    goodix = getStableRange(R, 'plot', true);
    
    unitinds = find(unitix);
    badix = setdiff(1:numel(R), goodix);
    badinds = unitinds(badix);
    unitix(badinds) = false;
    
    % STATIONARY
    spk = squeeze(spks(unitix,cc,:));
    theta = D.GratingDirections(unitix);

    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);
    
    thetas = unique(theta);
    tfun = @(th,cnt) arrayfun(@(x) mean(cnt(th==x)), thetas);
    
%     tuningCurve = arrayfun(@(x) mean(R(theta==x)), thetas);

    iirun = runningTrial(unitix);
    tuningCurveR = tfun(theta(iirun), R(iirun));
    tuningCurveS = tfun(theta(~iirun), R(~iirun));
    
    clf
    plot(thetas, tuningCurveR); hold on
    plot(thetas, tuningCurveS);
    
    nn = numel(theta);
    nrun = sum(iirun);
    nstat = sum(~iirun);
    
    iir = randi(nn, [nrun nboot]);
    iis = randi(nn, [nstat nboot]);
    
    for iboot = 1:nboot
        tcR = tfun(theta(iir(:,iboot)), R(iir(:,iboot)));
        tcS = tfun(theta(iis(:,iboot)), R(iis(:,iboot)));
        TCdiffNull(cc,iboot) = mean( (tcR - tcS).^2);
        maxFRdiffNull(cc,iboot) = max(tcR) - max(tcS);
        
    end
    
    TCdiff(cc) = mean( (tuningCurveR - tuningCurveS).^2 );
    maxFRdiff(cc) = max(tuningCurveR) - max(tuningCurveS);

end

%% explanatory plot
cmap = lines;

figure(2); clf
subplot(1,3,1)
plot(thetas, tuningCurveS, 'Linewidth', 2); hold on
plot(thetas, tuningCurveR, 'Linewidth', 2)
title('Raw Tuning Curves')
axis tight
ylabel('Firing Rate')

subplot(1,3,2)
fill([thetas' fliplr(thetas')], [tuningCurveS' fliplr(tuningCurveR')], 'k', 'Linewidth', 2); hold on
plot(thetas, tuningCurveS, 'Linewidth', 2); hold on
plot(thetas, tuningCurveR, 'Linewidth', 2)
title('TC diff')
axis tight
xlabel('Direction')
subplot(1,3,3)
plot(thetas, tuningCurveS, 'Linewidth', 2); hold on
plot(thetas, tuningCurveR, 'Linewidth', 2)
[mS, idS] = max(tuningCurveS);
[mR, idR] = max(tuningCurveR);
plot([0 thetas(idS)], mS*[1 1], '-', 'Color', cmap(1,:), 'Linewidth', 2)
plot(thetas(idS), mS, 'o', 'Color', cmap(1,:), 'Linewidth', 2)
plot([0 thetas(idR)], mR*[1 1], '-', 'Color', cmap(2,:), 'Linewidth', 2)
plot(thetas(idR), mR, 'o', 'Color', cmap(2,:), 'Linewidth', 2)
plot([1 1]*thetas(idS), [mS mR], 'k', 'Linewidth', 2)
axis tight
title('Max FR')









%% TC diff
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
    
    tix = lags > win(1) & lags < win(2);
    
    spk = squeeze(spks(unitix,cc,:));
    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);
    R = imboxfilt(R, 21);
    goodix = getStableRange(R, 'plot', false);
    
    unitinds = find(unitix);
    badix = setdiff(1:numel(R), goodix);
    badinds = unitinds(badix);
    unitix(badinds) = false;
    
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
        if numel(R) < 2
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
%% fit tuning curves
clear fitS fitR

% cc = cc + 1;
for cc = 1:NC
    fprintf('%d/%d\n', cc, NC)
    unitix = dfilt(:,cc);
    dur = median(D.GratingOffsets(unitix) - D.GratingOnsets(unitix));
    dur = max(dur, .1);
    win = [0.04 dur];
    
    % find stable region of firing rate
    spk = squeeze(spks(unitix,cc,:));
    tix = lags > win(1) & lags < win(2);
    R = sum(spk(:,tix),2);
    R = imboxfilt(R, 21);
    goodix = getStableRange(R, 'plot', false);
    
    unitinds = find(unitix);
    badix = setdiff(1:numel(R), goodix);
    badinds = unitinds(badix);
    unitix(badinds) = false;
    
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


