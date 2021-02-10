
%% First, get example cell

exIx = 45;
spat = cell2mat(arrayfun(@(x) x.fine(:), Srf(exIx), 'uni', 0));
grat = cell2mat(arrayfun(@(x) x.rfs(:), Sgt(exIx), 'uni', 0));
Frf = cell2mat(arrayfun(@(x) x.rfs(:), fftrf(exIx), 'uni', 0)');


cc = 1;
NC = numel(Frf);
%% plot spatial RF
% cc = 1;
cc = cc + 1;
if cc > NC
    cc = 1;
end
iEx = exIx;
ppd = Srf(iEx).coarse.details(cc).xax(end)/BIGROI(3);
figure(2); clf
subplot(311, 'align')
Imap = Srf(iEx).coarse.details(cc).srf;
Imap = Imap / max(Imap(:));

xax = Srf(iEx).coarse.details(cc).xax/ppd;
yax = Srf(iEx).coarse.details(cc).yax/ppd;


imagesc(xax, yax, Imap)
colormap(plot.viridis)
axis xy
hold on
title(cc)

if ~isnan(Srf(iEx).fine(cc).ROI)
    ROI = Srf(iEx).fine(cc).ROI;
    
    plot(ROI([1 3]), ROI([2 2]), 'r', 'Linewidth', 1)
    plot(ROI([1 3]), ROI([4 4]), 'r', 'Linewidth', 1)
    plot(ROI([1 1]), ROI([2 4]), 'r', 'Linewidth', 1)
    plot(ROI([3 3]), ROI([2 4]), 'r', 'Linewidth', 1)
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
    
    subplot(312, 'align')
    Imap = Srf(iEx).fine(cc).srf;
    Imap = Imap / max(Imap(:));
    xax = Srf(iEx).fine(cc).xax;
    yax = Srf(iEx).fine(cc).yax;
    mu = Srf(iEx).fine(cc).gfit.mu;
    C = Srf(iEx).fine(cc).gfit.cov;
    
    imagesc(xax, yax, Imap)
    
    hold on
    
    plot.plotellipse(mu, C, 2, 'r', 'Linewidth', 2);
    axis xy
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
    
    subplot(313, 'align')
    
    [~, peak_id] = max(Srf(iEx).fine(cc).srf(:));
    [~, min_id] = min(Srf(iEx).fine(cc).srf(:));
    sta = Srf(iEx).fine(cc).sta / max(Srf(iEx).fine(cc).sta(:));
    tk = sta(:,peak_id);
    tmin = sta(:,min_id);
    
    %     [u,s,v] = svd(Srf(iEx).fine(cc).sta);
    %     tk = u(:,1);
    %     tk = tk*sign(sum(tk));
    %
    tk = flipud(tk);
    tmin = flipud(tmin);
    plot((1:numel(tk))*(1e3/120), tk, '-ok', 'MarkerFaceColor', 'k', 'MarkerSize', 2); hold on
    plot((1:numel(tk))*(1e3/120), tmin, '-o', 'Color', .5*[1 1 1], 'MarkerFaceColor', .5*[1 1 1], 'MarkerSize', 2);
    xlabel('Lags (ms)')
    
    plot.fixfigure(gcf, 7, [1 3], 'OffsetAxes', false, 'FontName', 'Arial');
    
    colormap(plot.viridis)
    exname = sesslist{iEx};
    %     saveas(gcf, fullfile(figDir, sprintf('fig04_SRF_%s_%d.pdf', exname, cc)));
end




%%
I = Sgt(iEx).rfs(cc).srf;

subplot(1,2,1)
imagesc(I)

subplot(1,2,2)
imagesc([I; rot90(I, 2)])

% I = zeros(size(I));

% I = [rot90(I, 2) I]; % mirror symmetric
I = [I; rot90(I, 2)];
yax = [-flipud(Sgt(iEx).rfs(cc).cpds); Sgt(iEx).rfs(cc).cpds];
xax = Sgt(iEx).rfs(cc).oris;

figure(1); clf
[xx,yy] = meshgrid(xax, yax);
[ori, cpd] = cart2pol(xx,yy);

iix = ori==0 | (ori-pi).^2 < .001;
iix = iix & (cpd - 4.0).^2 < 0.001;
% I(iix) = 1;


% contourf(cos(ori - pi/2)); colormap gray
subplot(121)
imagesc(xax, yax, I, [-1 1]);
%%
contourf(xx, yy, I, 'Linestyle', 'none')


contourf(xx, yy, I, 'Linestyle', 'none')

params = [Sgt(exIx).rfs(cc).rffit.oriBandwidth/180*pi, ...
    Sgt(exIx).rfs(cc).rffit.oriPref/180*pi, ...
    Sgt(exIx).rfs(cc).rffit.sfPref,...
    Sgt(exIx).rfs(cc).rffit.sfBandwidth,...
    Sgt(exIx).rfs(cc).rffit.amp, ...
    Sgt(exIx).rfs(cc).rffit.base];
Ifit = prf.parametric_rf(params, [ori(:), cpd(:)]);

% hold on
subplot(122)
contourf(xx,yy,reshape(Ifit, size(I)), 'r', 'Linestyle', 'none')
% scatteredInterpolant(xx(:), yy(:), 100*ones(size(xx(:))), I(:) + min(I(:)))


%%
NC = 34;
% cc = 18;
cc = cc + 1;
if cc > NC
    cc = 1;
end


figure(1); clf
set(gcf, 'Color', 'w')

subplot(2,2,1)
Imap = Srf(iEx).fine(cc).srf;
Imap = Imap / max(Imap(:));
xax = Srf(iEx).fine(cc).xax;
yax = Srf(iEx).fine(cc).yax;
mu = Srf(iEx).fine(cc).gfit.mu;
C = Srf(iEx).fine(cc).gfit.cov;

imagesc(xax, yax, Imap)


% hold on
% 
% plot.plotellipse(mu, C, 2, 'r', 'Linewidth', 2);
axis xy
% xlabel('Azimuth (d.v.a)')
% ylabel('Elevation (d.v.a)')


subplot(2,2,2)
imagesc(Frf(cc).rf.kx, Frf(cc).rf.ky, Frf(cc).rf.Ifit)
xlim([-1 1]*15)
ylim([-1 1]*15)
xlabel('SF_x')
ylabel('SF_y')
set(gca, 'XTick', -15:5:15, 'YTick', -15:5:15)

subplot(2,2,3)
bar(Frf(cc).xproj.bins, Frf(cc).xproj.cnt); hold on
plot(Frf(cc).xproj.levels(1)*[1 1], ylim, 'r')
plot(Frf(cc).xproj.levels(2)*[1 1], ylim, 'b')

subplot(2,2,4)
cmap = lines;
plot.errorbarFill(Frf(cc).lags, Frf(cc).rateHi, Frf(cc).stdHi/sqrt(Frf(cc).nHi), 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .8, 'EdgeColor', cmap(1,:)); hold on
plot.errorbarFill(Frf(cc).lags, Frf(cc).rateLow, Frf(cc).stdHi/sqrt(Frf(cc).nLow), 'k', 'FaceColor', cmap(5,:), 'FaceAlpha', .8, 'EdgeColor', cmap(5,:));
xlim([-.05, .25])
xlabel('Time from fixation onset (s)')
ylabel('Firing Rate')
title(cc)
plot.fixfigure(gcf, 14, [8 8], 'OffsetAxes', false)
saveas(gcf, sprintf('Figures/example_%d.pdf', cc))
%%
hasRF = ~arrayfun(@(x) isempty(x.fine), Srf(:));
hasGrf = ~arrayfun(@(x) isempty(x.rfs), Sgt(:));
hasFrf = ~arrayfun(@(x) isempty(x.rfs), fftrf(:));
exIx = hasRF & hasGrf & hasFrf;
% exIx = 1:5;
spat = cell2mat(arrayfun(@(x) x.fine(:), Srf(exIx), 'uni', 0));
Frf = cell2mat(arrayfun(@(x) x.rfs(:), fftrf(exIx), 'uni', 0)');

%%

% vis = ~arrayfun(@(x) isnan(x.sta(1)), srf.fine);
% gr = arrayfun(@(x) x.r2test, grf)>0;
rateHi = cell2mat(arrayfun(@(x) x.rateHi(:)', Frf, 'uni', 0));

% vis = find(vis & gr & mean(rateHi,2)>5);
vis = find(mean(rateHi,2)>1);

lags = Frf(1).lags;
rateHi = cell2mat(arrayfun(@(x) x.rateHi(:)', Frf(vis), 'uni', 0));
rateLow = cell2mat(arrayfun(@(x) x.rateLow(:)', Frf(vis), 'uni', 0));

iix = lags > -0.05 & lags < 0.01;
n = std((rateHi(:,iix) + rateLow(:,iix)) / 2 , [], 2);
n = mean((rateHi(:,iix) + rateLow(:,iix)) / 2, 2);
n = max(n, 1);

figure(1); clf
nrateHi = rateHi./n;
nrateLow = rateLow./n;

cmap = lines;
plot.errorbarFill(lags, nanmean(nrateHi), nanstd(nrateHi)./sqrt(numel(vis)), 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .8, 'EdgeColor', cmap(1,:));hold on
plot.errorbarFill(lags, nanmean(nrateLow), nanstd(nrateLow)./sqrt(numel(vis)), 'r', 'FaceColor', cmap(5,:), 'FaceAlpha', .8, 'EdgeColor', cmap(5,:));
xlim([-.05 .25])
plot(xlim, [1 1], 'k--')
xlabel('Time from fixation onset (s)')
ylabel('Relative Rate')

plot.fixfigure(gcf, 14, [4 4])

saveas(gcf, 'Figures/natImgFixPsthPop.pdf')

%%
figure(1); clf
rdiff = nrateHi - nrateLow;
C = cov(rdiff);
[u,s] = svd(C);

subplot(1,2,1)
imagesc(C)

xproj = rdiff*u(:,2);
[~, ind] = sort(xproj, 'ascend');
[~, ind] = sort(mean(rdiff(:,lags > .15 & lags < .25),2), 'descend');

subplot(1,2,2)
imagesc(rdiff(ind,:))
colormap parula

%%
figure(2); clf
idx = ind(1:50);
errorbar(lags, nanmean(nrateHi(idx,:)), nanstd(nrateHi(idx,:))./sqrt(numel(idx)), 'b');hold on
errorbar(lags, nanmean(nrateLow(idx,:)), nanstd(nrateLow(idx,:))./sqrt(numel(idx)), 'r');
xlim([-.05 .25])

%%

idx = vis(ind(1:50));
cid = cid + 1;
cc = vis(cid);
% for cc = idx(:)'
    figure(2); clf
    plot.errorbarFill(Frf(cc).lags, Frf(cc).rateHi, Frf(cc).stdHi/sqrt(Frf(cc).nHi), 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .8, 'EdgeColor', cmap(1,:)); hold on
    plot.errorbarFill(Frf(cc).lags, Frf(cc).rateLow, Frf(cc).stdHi/sqrt(Frf(cc).nLow), 'k', 'FaceColor', cmap(5,:), 'FaceAlpha', .8, 'EdgeColor', cmap(5,:));
    xlim([-.05, .25])
    xlabel('Time from fixation onset (s)')
    ylabel('Firing Rate')
    title(cc)
%     pause
% end
%% 
plot.fixfigure(gcf, 12, [5 3])
saveas(gcf, sprintf('Figures/niFix_example%d.pdf', cc))

%%

figure(3); clf
plot(rdiff*u(:,1), rdiff*u(:,2), '.')
