
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

addpath Analysis/202001_K99figs_01  
addpath Analysis/manuscript_freeviewingmethods/

%% load data
sessId = 45;
sorter = 'jrclustwf';
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', sorter);

%% get supporting data
% get spatial RFs
srf = spatial_RF_single_session(Exp);
% get grating RFs
grf = grating_RF_single_session(Exp);

%% ffrf firing rate
fftrf0 = fixrate_by_fftrf(Exp, srf, grf, 'plot', false, 'makeMovie', true);

%%
cc = 1;
NC = numel(fftrf);
%%

cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(1); clf
subplot(1,3,1)
imagesc(fftrf(cc).rf.kx, fftrf(cc).rf.ky, fftrf(cc).rf.Ifit)

subplot(1,3,2)
bar(fftrf(cc).xproj.bins, fftrf(cc).xproj.cnt)

subplot(1,3,3)
plot.errorbarFill(fftrf(cc).lags, fftrf(cc).rateHi, fftrf(cc).stdHi/sqrt(fftrf(cc).nHi), 'k', 'FaceColor', 'b'); hold on
plot.errorbarFill(fftrf(cc).lags, fftrf(cc).rateLow, fftrf(cc).stdHi/sqrt(fftrf(cc).nLow), 'k', 'FaceColor', 'r');
xlim([-.05, .25])
%%

vis = ~arrayfun(@(x) isnan(x.sta(1)), srf.fine);
gr = arrayfun(@(x) x.r2test, grf)>0;
vis = find(vis & gr);

for c = 1:numel(vis)
    cc = vis(c);
    figure(1); clf
    subplot(1,3,1)
    imagesc(fftrf(cc).rf.kx, fftrf(cc).rf.ky, fftrf(cc).rf.Ifit)

subplot(1,3,2)
bar(fftrf(cc).xproj.bins, fftrf(cc).xproj.cnt)

subplot(1,3,3)
plot.errorbarFill(fftrf(cc).lags, fftrf(cc).rateHi, fftrf(cc).stdHi/sqrt(fftrf(cc).nHi), 'k', 'FaceColor', 'b'); hold on
plot.errorbarFill(fftrf(cc).lags, fftrf(cc).rateLow, fftrf(cc).stdHi/sqrt(fftrf(cc).nLow), 'k', 'FaceColor', 'r');
xlim([-.05, .25])

pause
end


%%
vis = ~arrayfun(@(x) isnan(x.sta(1)), srf.fine);
gr = arrayfun(@(x) x.r2test, grf)>0;
rateHi = cell2mat(arrayfun(@(x) x.rateHi(:)', fftrf, 'uni', 0));

vis = find(vis & gr & mean(rateHi,2)>5);
% vis = find(mean(rateHi,2)>1);

lags = fftrf(1).lags;
rateHi = cell2mat(arrayfun(@(x) x.rateHi(:)', fftrf(vis), 'uni', 0));
rateLow = cell2mat(arrayfun(@(x) x.rateLow(:)', fftrf(vis), 'uni', 0));

iix = lags > -0.05 & lags < 0.01;
n = std((rateHi(:,iix) + rateLow(:,iix)) / 2 , [], 2);
n = mean((rateHi(:,iix) + rateLow(:,iix)) / 2, 2);

figure(1); clf
nrateHi = rateHi./n;
nrateLow = rateLow./n;

errorbar(lags, nanmean(nrateHi), nanstd(nrateHi)./sqrt(numel(vis)), 'b');hold on
errorbar(lags, nanmean(nrateLow), nanstd(nrateLow)./sqrt(numel(vis)), 'r');
xlim([-.05 .25])

%%
nv = numel(vis);
figure(1); clf
rdiff = nrateHi - nrateLow;


C = cov([nrateHi; nrateLow]);

[u,s] = svd(C);

subplot(1,2,1)
imagesc(C)
subplot(1,2,2)
plot(u(:,1:2))

x = rdiff*u(:,1:2);

plot(x(1:nv,1), x(1:nv,2), 'o'); hold on
% plot(x(nv+1:end,1), x(nv+1:end,2), 'o');







