
spikeSorting = 'kilowf';

close all
sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', spikeSorting, 'cleanup_spikes', 0);

eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);


% get visually driven units
W = io.get_waveform_stats(Exp.osp);
spkS = io.get_visual_units(Exp, 'plotit', true, 'waveforms', W);

%
figure
plot.plotWaveforms(W)
% check PSTH to RSVP stimulus

validTrials = io.getValidTrials(Exp, 'FixRsvpStim');

% bin spikes and eye pos
binsize = 5e-3; % 1 ms bins for rasters
win = [-.5 2]; % -100ms to 2sec after fixation onset

% trial length
n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials));
bad = n < 100;
validTrials(bad) = [];
tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1), Exp.D(validTrials)));

bins = min(Exp.osp.st):binsize:max(Exp.osp.st);
[~, ~, evbins] = histcounts(tstart, bins);

%
cids = Exp.osp.cids;
% Exp.osp.clusterDepths
NC = numel(cids);
cmap = hsv(NC);
figure; clf
subplot(NC, 5,unique((((1:NC)-1)*5)' + (1:2)), 'align')
plot.plotWaveforms(W, 1, 'cmap', cmap, 'overloadSU', true)

for cc = 1:NC
    subplot(NC, 5, (cc-1)*5 + (4:5), 'align')
    cnt = histcounts(Exp.osp.st(Exp.osp.clu==cids(cc)), bins);
    [av, ~, xax] = eventTriggeredAverage(cnt(:), evbins, ceil(win/binsize));


    x = xax*binsize;
    m = sgolayfilt(av, 1, 9);
    fill([x(:); flipud(x(:))], [m(:); 0*m(:)], 'k', 'FaceColor', cmap(cc,:))
    xlim(xax([1 end])*binsize)
    axis off
    
    subplot(NC, 5, (cc-1)*5 + 3, 'align')
    imagesc( (spkS(cc).srf-mean(spkS(cc).srf(:)))/std(spkS(cc).srf(:)), [0 3.5])
    colormap gray
    axis off
    axis xy
end