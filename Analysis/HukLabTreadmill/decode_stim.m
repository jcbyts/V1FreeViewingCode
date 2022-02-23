function Dstat = decode_stim(D, sessionId, varargin)
% do orientation / direction decoding for the huklab treadmill data
% Dstat = decode_stim(D, sessionId, varargin)
% Inputs:
%   D            <struct> super session struct
    % sessionId <numeric> session number
% Optional Arguments:
%   binSize:        bin size (seconds)
%   Latency:        estimated stim onset (for decoding window, in seconds)
%   Decode:         decoding target {'Orientation', 'Direction'}
%   slidingWin:     boxcar width for sliding window analyses (seconds)
%   runThreshold:   threshold in cm/s to consider running
%   plot:           boolean for saving plots (default: true)

ip = inputParser();
ip.addParameter('binSize', 5e-3)
ip.addParameter('Latency', .04)
ip.addParameter('Decode', 'Orientation')
ip.addParameter('slidingWin', 80e-3)
ip.addParameter('runThreshold', 3)
ip.addParameter('plot', true)
ip.addParameter('figDir', fullfile('Figures', 'HuklabTreadmill', 'decoding'))
ip.parse(varargin{:})

figDir = ip.Results.figDir;
if ~exist(figDir, 'dir')
    mkdir(figDir)
end
%% useful functions
circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;
circmean = @(x) angle(sum(exp(1i*x/180*pi)))/pi*180;

% wrap in radians
wrappi = @(x) mod(x/pi, 1)*pi;
wrap2pi = @(x) mod(x/2/pi, 1)*2*pi;

% in degrees
wrap180 = @(x) mod(x/180, 1)*180;
wrap360 = @(x) mod(x/360, 1)*360;

crossValidate = true;
trainStationaryOnly = false;
instantaneousDecoder = false; % retrain decoder at each timelag
cumulativeDecoder = false;

%%
[StimDir, spksb, runningSpeed, Dstat] = bin_population(D, sessionId, 'plot', false);

% sx = ceil(sqrt(Dstat.NCells));
% sy = round(sqrt(Dstat.NCells));
% set(gcf, 'PaperSize', [sy sx]*2, 'PaperPosition', [0 0 sy sx]*2)
% % plot.fixfigure(gcf, 10, [sy sx]*2, 'offsetAxes', false);
% saveas(gcf, fullfile(figDir, 'binned_spikes.pdf'))

% throw out spurious running trials
toofast = find(any(runningSpeed > 60,2) | isnan(StimDir));

StimDir(toofast,:) = [];
spksb(toofast,:,:) = [];
runningSpeed(toofast,:) = [];
Dstat.NTrials = Dstat.NTrials-numel(toofast);
% figure(1); clf
% imagesc(runningSpeed)

runSpeed = mean(runningSpeed,2);

NC = Dstat.NCells; % number of neurons
NT = Dstat.NTrials;
nbins = Dstat.NLags;
binsize = Dstat.binsize;
win = Dstat.lags([1 end]);
fprintf("Session %d) %d Grating Trials, %d Units %02.2f sec duration\n", sessionId, NT, NC, Dstat.stimDuration)

%% Do some decoding

bins = Dstat.lags;

% window to train decoder on
tidx = bins > ip.Results.Latency & bins < Dstat.stimDuration;

decodeOrientation = strcmpi(ip.Results.Decode, 'orientation');

if decodeOrientation
    fprintf('Decoding Orientation\n')
    st = wrap180(StimDir);
    limval = 180;
else
    fprintf('Decoding Direction\n')
    st = wrap360(StimDir);
    limval = 360;
end

rr = squeeze(mean(spksb(:,tidx,:),2)); % sum over time
spks = spksb;

rproj = rr;
ntrials = size(rr,1);

nBasis = min(numel(unique(st(~isnan(st))))+1, 40);

decoderStimTot = nan(ntrials, 1);
xd = 0:limval;

yprob = nan(ntrials, numel(xd), nbins);
decoderStim = nan(ntrials, nbins);

nSlidingBins = ceil(ip.Results.slidingWin/binsize);

runTrials = find(runSpeed > ip.Results.runThreshold);

fprintf('Leave one out cross-validation\n')
clear B
for iTrial = 1:ntrials
    %     fprintf('Trial %d / %d\n', iTrial, ntrials)
    if crossValidate
        if trainStationaryOnly
            iitrain = setdiff(1:ntrials, [runTrials(:); iTrial]);
        else
            iitrain = setdiff(1:ntrials, iTrial);
        end
    else
        if trainStationaryOnly
            iitrain = setdiff(1:nTrials, runTrials);
        else
            iitrain = 1:ntrials;
        end
    end
    
    iitest = iTrial;
    
    if ~instantaneousDecoder
        if decodeOrientation
            [~, stimhat, yhat, wts, gfun] = decoding.von_mises_ori_decoder(rproj, st, iitrain, iitest, 'nBasis', nBasis, 'Beta', 12, 'lambda', 1);
        else
            [~, stimhat, yhat, wts, gfun] = decoding.von_mises_decoder(rproj, st, iitrain, iitest, 'nBasis', nBasis, 'Beta', 12, 'lambda', 1);
        end
        
        if ~exist('B', 'var')
            B = gfun(xd);
            B = B./max(B);
        end
    end
    
    for ibin = 1:nbins
        
        % sliding window
        if cumulativeDecoder
            binix = 1:ibin;
            if ibin > find(bins>0,1)
                binix = find(bins>0,1):ibin;
            end
        else
            binix = ibin + (-nSlidingBins:0);
        end
        binix(binix<1) = [];
        binix(binix>nbins) = [];
        
        
        
        if instantaneousDecoder
            
            rr = squeeze(mean(spks(:,binix,:),2)); % sum over time
            if decodeOrientation
                [~, stimhat, yhat, wts, gfun] = decoding.von_mises_ori_decoder(rr, st, iitrain, iitest, 'nBasis', nBasis, 'Beta', 12, 'lambda', 10e3);
            else
                [~, stimhat, yhat, wts, gfun] = decoding.von_mises_decoder(rr, st, iitrain, iitest, 'nBasis', nBasis, 'Beta', 12, 'lambda', 10e3);
            end
            
            if ~exist('B', 'var')
                B = gfun(xd);
                B = B./max(B);
            end
            
        else
            rr = squeeze(mean(spks(iTrial,binix,:),2)); % sum over time
            rrd = [1 rr'];
            yhat = (rrd*wts)*B';
        end
        
        
        yprob(iTrial,:, ibin) = yhat;
        [~, stimid] = max(yhat);
        decoderStim(iTrial,ibin) = xd(stimid);
        decoderStimTot(iTrial) = stimhat;
    end
end
fprintf('Done\n')

Dstat.decoderStim = decoderStim;
Dstat.decoderStimTot = decoderStimTot;
Dstat.yprob = yprob;
Dstat.Stim = st;
Dstat.runTrials = runTrials;
Dstat.wts = wts;

%% plot decoder schematic and performance
figure(1); clf
set(gcf, 'Color', 'w')

iTrial = randsample(1:NT,1);

[~, ind] = sort(wts(2:end,2));
subplot('Position', [.05 .15 .1 .75])
imagesc(bins, 1:NC, squeeze(spks(iTrial,:,:))'); hold on
plot([0 0], ylim, 'r--')
colormap(1-gray)
set(gca, 'XTick', [.2 .4], 'XTickLabel', '', ...
    'YTick', [], 'Box', 'off', 'YColor', 'w', 'TickDir', 'out')
annotation(gcf, 'TextBox', [.025 .2 .1 .8], 'String', 'A', 'Linestyle', 'none')
xlabel('Time')
ylabel('Neuron', 'Color', 'k')
axis xy
title('Spike trains')

subplot('Position', [.15 .15 .05 .75])
cmap = lines;
for i = 1:2:nBasis
    xx = repmat([0 1], NC, 1)';
    yy = [(1:NC)' repmat(i/nBasis*NC, NC, 1)]';
    spacing = 1;
    cstr = 1 - exp(wts(:,i)) ./ max(exp(wts(:,i)));
    clr = cstr(:).*cmap(i,:) + 2*[1 1 1]; clr = clr/3;
    for j = 1:spacing:NC
        plot(xx(:,j), yy(:,j), 'Color', clr(j,:)); hold on
    end
    spacing = 4;
    plot(xx(:,1:spacing:end), yy(:,1:spacing:end), 'Color', cmap(i,:)); hold on
end
ylim([1 NC])
axis off
title(sprintf('Linear\nRegression'))

subplot('Position', [.2 .15 .05 .75])
for iBas = 1:nBasis
    if iBas <=i
        plot(B(:,iBas)./max(B(:))*2 + iBas/nBasis*NC, 'Color', cmap(iBas,:), 'Linewidth', 2); hold on
    end
end
% plot(bsxfun(@plus, (B./max(B(:)))*2, (1:nBasis)/nBasis*nneur))
ylim([1 NC])
axis off
title(sprintf('Super\nNeurons'))


subplot('Position', [.25 .15 .05 .75])

Bproj = ([1 rproj(iTrial,:)]*wts).*B;

Bproj = Bproj ./ max(Bproj(:));

for iBas = 1:nBasis
    if iBas <=i
        plot(Bproj(:,iBas)*2 + iBas/nBasis*NC, 'Color', cmap(iBas,:), 'Linewidth', 2); hold on
        plot([0 limval], [0 0]'+ iBas/nBasis*NC, 'k--')
    end
end

% plot(bsxfun(@plus, (B./max(B(:)))*2, (1:nBasis)/nBasis*nneur))
ylim([1 NC])
axis off
title(sprintf('Scaled\nResponse'))
axis off

subplot('Position', [.34 .15 .05 .75])
rtmp = sum(Bproj,2);
rtmp = exp(rtmp)./sum(exp(rtmp));
plot(rtmp,xd, 'k', 'Linewidth', 2); hold on
plot(interp1(xd, rtmp, st(iTrial)), st(iTrial), 'or', 'MarkerFaceColor', 'r')
[mx, id] = max(rtmp);
plot(mx, xd(id), 'ob', 'MarkerFaceColor', 'b')
ylabel('Direction')
xlabel('Probability')
% plot(xlim, st(iTrial)*[1 1] ,'k')
ylim([0 limval])
set(gca, 'XTick', [0 0.01])
set(gca, 'YTick', [0:90:limval], 'Box', 'off', 'YColor', 'k', 'TickDir', 'out')
title('MAP estimate')


subplot('Position', [.45 .15 .2 .75])
cmap = lines;
tridx = 1:ntrials;
y = decoderStimTot;
ix1 = (y-st) < -250;
ix2 = (y-st) > 250;
y(ix1) = y(ix1) + limval;
y(ix2) = y(ix2) - limval;

statTrials = setdiff(tridx, runTrials);
plot(st(statTrials), y(statTrials), 'ow', 'MarkerFaceColor', cmap(1,:), 'MarkerSize', 5); hold on
plot(st(runTrials), y(runTrials), 'ow', 'MarkerFaceColor', cmap(2,:), 'MarkerSize', 5); hold on

plot([-5 st(iTrial)], [xd(id) xd(id)], 'b--')
plot([st(iTrial) st(iTrial)], [0 xd(id)], 'r--')
xlim([-5 limval+5])
ylim([0 limval])
plot(xlim, xlim, 'k')
set(gca, 'XTick', 0:90:limval)
set(gca, 'YTick', 0:90:limval,'TickDir', 'out', 'Box', 'off')

maeS = median(abs(circdiff(decoderStimTot(statTrials), st(statTrials))));
maeR = median(abs(circdiff(decoderStimTot(runTrials), st(runTrials))));
title(sprintf('Median Error: %0.2f, %0.2f degrees', maeS, maeR))
xlabel('True Direction')
ylabel('Decoded Direction')

annotation(gcf, 'TextBox', [.4 .2 .1 .8], 'String', 'B', 'Linestyle', 'none')

% replace blanks with the pre-sac stim
% shift the activation map to center on the true stimulus direction

if ip.Results.plot
    set(gcf, 'PaperSize', [10 3], 'PaperPosition', [0 0 12 3])
    saveas(gcf, fullfile(figDir, sprintf('decoderschematic%02.2f.pdf', sessionId)))
end

%% initialize
yprobShift = yprob;

% loop over trials and shift the yprob map up and down depending on which
% stimulus was shown
for iTrial = 1:ntrials
    rtmp = squeeze(yprob(iTrial,:,:));
    rtmp = exp(rtmp) ./ sum(exp(rtmp));
    
    % round because it must be an integer amount
    shiftAmount = -ceil(st(iTrial))+90;
    
    yprobShift(iTrial,:,:) = circshift(rtmp, shiftAmount);
end


iTrial = 1;

changeAmount = zeros(size(yprob, 1), 1);

Dstat.yprobShift = yprobShift;

%% inspect individual trials
if (0)
    iTrial =iTrial + 1;
    if iTrial > NT
        iTrial = 1;
    end
    
    figure(1); clf
    subplot(1,2,1)
    imagesc(bins, xd, squeeze(yprob(iTrial,:,:))); hold on;
    plot([0 win(2)], [1 1]*st(iTrial), 'r')
    subplot(1,2,2)
    imagesc(bins, xd, squeeze(yprobShift(iTrial,:,:))); hold on;
    plot([0 win(2)], [1 1]*180, 'r')
    plot([win(1) 0], [1 1]*180 - changeAmount(iTrial), 'r')
    colorbar
end

%% Plot the average probability map in a sliding window

DirectionBins = linspace(-180, 180, 40);
DecoderError = circdiff(decoderStim,st);

cmap = decoding.cbrewer('seq', 'YlGnBu', 100);
cmap = cmap.^1.2;
cmap = flipud(cmap);

% fix colormap range for all plots
clim = [min(min(mean(yprobShift,1))) 1.1*max(max(mean(yprobShift,1)))];

conditions = {ismember(1:ntrials, runTrials)', ...
    ismember(1:ntrials, statTrials)'};
labels = {'Running', 'Stationary'};

nConds = numel(conditions);

circmeanWeighted = @(x, y) angle(sum(exp(1i*x/180*pi) .* (y./sum(y)))) / pi*180;

figure(10); clf
set(gcf, 'Color', 'w')
for iCond = 1:nConds
    subplot(1,nConds,iCond)
    c = conditions{iCond};
    
    rtmp = squeeze(mean(yprobShift(c,:,:),1));
    imagesc(bins, xd, rtmp, clim);
    
    hold on
    [~, id] = max(rtmp,[],1);
    plot(bins, xd(id), 'k*')
    
    
    plot([0 win(2)], 90+[0 0],'r', 'Linewidth', 2)
    
    
    colormap(cmap)
    xlim([-.1 .3])
    title(labels{iCond})
    xlabel('Time from stim onset (seconds)')
    ylabel('\Delta From True Direction (degrees)')
end

if ip.Results.plot
    set(gcf, 'PaperSize', [12 2.5], 'PaperPosition', [0 0 12 2.5])
    saveas(gcf, fullfile(figDir, sprintf('decoderprobability%02.0f.pdf', sessionId)))
    
    colorbar
    set(gcf, 'PaperSize', [12 3], 'PaperPosition', [0 0 12 3])
    saveas(gcf, fullfile(figDir, sprintf('decoderprobabilitywcolorbar%02.0f.pdf', sessionId)))
end

Dstat.DecodeError = DecoderError;
%%
% Make the same plots but with sliding window on decoding error
figure(13); clf
set(gcf, 'Color', 'w')
for iCond = 1:nConds
    subplot(1,nConds,iCond)
    c = conditions{iCond};
    
    ErrorMap = nan(numel(DirectionBins)-1, nbins);
    for ibin = 1:nbins
        ErrorMap(:,ibin) = histcounts(DecoderError(c,ibin), DirectionBins);
    end
    
    Dstat.(labels{iCond}).ErrorMap = ErrorMap;
    
    rtmp = imgaussfilt(ErrorMap, 1);
    
    imagesc(bins, DirectionBins(1:end-1), rtmp)
    hold on
    [~, id] = max(rtmp,[],1);
    plot(bins, DirectionBins(id), 'k*')
    
    if strcmp(labels{iCond}, 'Blank')
        plot([win(1) 0], 0*[1 1], 'r')
    else
        plot([0 win(2)], [0 0],'r')
        plot([win(1) 0], -nanmean(changeAmount(c))*[1 1], 'r')
    end
    colormap(cmap)
    xlim([-.1 .3])
    title(labels{iCond})
    xlabel('Time from motion onset (seconds)')
    ylabel('\Delta From True Direction (degrees)')
end

if ip.Results.plot
    set(gcf, 'PaperSize', [12 3], 'PaperPosition', [0 0 12 3])
    saveas(gcf, fullfile(figDir, sprintf('decodererror%02.0f.pdf', sessionId)))
end



%%


nConds = numel(conditions);

% plot percent correct
delta = median(abs(diff(unique(round(st)))))/2;

correct = abs(DecoderError) <= delta;

figure(11); clf
set(gcf, 'Color', 'w')
cmap = lines(nConds);
clear h
for iCond = 1:nConds
    c = conditions{iCond};
    mu = mean(correct(c,:));
    serr = std(correct(c,:))./sqrt(sum(c));
    Dstat.(labels{iCond}).bins = bins;
    Dstat.(labels{iCond}).percentCorrect = mu;
    Dstat.(labels{iCond}).percentCorrectSE = serr;
    
    plot.errorbarFill(bins, mu, serr, 'k', 'EdgeColor', cmap(iCond,:), 'FaceColor', cmap(iCond,:), 'FaceAlpha', .5); hold on
    h(iCond) = plot(bins, mu, 'Color', cmap(iCond,:));
end
plot(xlim, (1/numel(unique(StimDir)))*[1 1], 'k--')
xlabel('Time from Grating Onset')
ylabel('Proportion Correct')
legend(h, labels)
xlim([-.1 .3])

correct = abs(DecoderError);

figure(12); clf
set(gcf, 'Color', 'w')
cmap = lines(nConds);
for iCond = 1:nConds
    c = conditions{iCond};
    
    mu = imgaussfilt(mean(correct(c,:)),1);
    serr = std(correct(c,:))./sqrt(sum(c));
    
    Dstat.(labels{iCond}).medianAbsoluteError = mu;
    Dstat.(labels{iCond}).medianAbsoluteErrorSE = serr;
    
    plot.errorbarFill(bins, mu, serr, 'k', 'EdgeColor', cmap(iCond,:), 'FaceColor', cmap(iCond,:), 'FaceAlpha', .5, 'Linestyle', 'none'); hold on
    h(iCond) = plot(bins, mu, 'Color', cmap(iCond,:));
end
plot(xlim, 60*[1 1], 'k--')
xlim([-.1 .3])
xlabel('Time from Grating Onset')
ylabel('Mean Absolute Error')
legend(h, labels)
set(gca, 'Box', 'off', 'TickDir', 'out')
% decoding.offsetAxes(gca)
if ip.Results.plot
    set(gcf, 'PaperSize', [4 4], 'PaperPosition', [0 0 4 4])
    saveas(gcf, fullfile(figDir, sprintf('decoders_mae%02.0f.pdf', sessionId)))
end