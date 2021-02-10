spikeSorting = 'jrclustwf';
sessId = 45;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', spikeSorting, 'cleanup_spikes', 0);

%%

validTrials = io.getValidTrials(Exp, 'FixRsvpStim');

% bin spikes and eye pos
binsize = 16e-3; % 1 ms bins for rasters
win = [-.5 2]; % -100ms to 2sec after fixation onset

% trial length
n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials));
% bad = n < 100;
% validTrials(bad) = [];
% n = n(~bad);
tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1), Exp.D(validTrials)));

bins = min(Exp.osp.st):binsize:max(Exp.osp.st);
[~, ~, evbins] = histcounts(tstart, bins);

%
cids = Exp.osp.cids;
% Exp.osp.clusterDepths
NC = numel(cids);
cmap = hsv(NC);
% figure; clf
% subplot(NC, 5,unique((((1:NC)-1)*5)' + (1:2)), 'align')
% plot.plotWaveforms(W, 1, 'cmap', cmap, 'overloadSU', true)

%%

D = load('Data/logan_20191231_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat');

figure(1); clf;

ctrX = Exp.S.centerPix(1);
ctrY = Exp.S.centerPix(2);
ppd = Exp.S.pixPerDeg;
eyeRad = hypot(D.eyeAtFrame(:,1)-ctrX, D.eyeAtFrame(:,2)-ctrY)/ppd;

plot(eyeRad);

trends = [find(diff(D.frameTimes)>1); numel(D.frameTimes)];
trstartIdx = [1; find(diff(D.frameTimes)>1)+1];
nTrials = numel(trstartIdx);
fixstart = nan(nTrials,1);
fixstops = nan(nTrials,1);
for iTrial = 1:nTrials
    tix = trstartIdx(iTrial):trends(iTrial);
    try
        fixstart(iTrial) = find(eyeRad(tix)<1, 1, 'first') + tix(1);
        fixstops(iTrial) = find(eyeRad(tix)<1, 1, 'last') + tix(1);
    end
end

good = ~isnan(fixstart);
fixstart = fixstart(good);
fixstops = fixstops(good);
fixdur = fixstops-fixstart;



cc = 1
%%
figure(1); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end

cnt = histcounts(Exp.osp.st(Exp.osp.clu==cids(cc)), bins);
[av, ~, xax, wfs] = eventTriggeredAverage(cnt(:), evbins, ceil(win/binsize));

subplot(3,1,1:2)
[~, ind] = sort(n);
imagesc(wfs(ind,:))
title(cc)

subplot(3,1,3)
x = xax*binsize;
m = sgolayfilt(av, 1, 3);

fill([x(:); flipud(x(:))], [m(:); 0*m(:)], 'k', 'FaceColor', cmap(cc,:))
xlim(xax([1 end])*binsize)
axis off
   

soff = 50;
[~, ~, xax, wfs0] = eventTriggeredAverage(D.Robs(:,cc), fixstart, [-soff 240]);
figure(2); clf
[~, ind0] = sort(fixdur);
subplot(3,1,1:2)
for iTrial = 1:numel(fixstart)
    if fixdur(iTrial)>0
        wfs0(iTrial,(fixdur(iTrial)+soff):end) = nan;
    else
        wfs0(iTrial,:) = nan;
    end
end

imagesc(wfs0(ind0,:))
title(cc)

av = nanmean(wfs0);
subplot(3,1,3)
x = xax*binsize;
m = sgolayfilt(av, 1, 3);

fill([x(:); flipud(x(:))], [m(:); 0*m(:)], 'k', 'FaceColor', cmap(cc,:))
% xlim(xax([1 end])*binsize)




