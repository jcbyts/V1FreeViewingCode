
%% path
figdir = 'Figures/HuklabTreadmill/manuscript/';
%% load session
subject = 'gru';
D = load_subject(subject);


%% Unit by Unit simple analysis
thresh = 3;
unitList = unique(D.spikeIds);
%%
% gru: 10,22, 94, 97, 126, 164, 356, 357, 358, 460
cc = 460;

unitId = unitList(cc);
[stimconds, robs, B, opts] = bin_ssunit(D, unitId, 'win', [-.2 .2], 'plot', false, 'binsize', 1e-3);

stimDir = stimconds{1};
stimFreq = stimconds{3};
runspeed = nanmean(B{1},2);

directions = unique(stimDir)';
freqs = unique(stimFreq)';

nd = numel(directions);
nf = numel(freqs);

figure(1); clf
set(gcf, 'Color', 'w')
ax = plot.tight_subplot(nf+1, nd+1, 0.005, 0.005);

xax = linspace(-1, 1, 100);
yax = linspace(-1,1, 100);
[xx, yy] = meshgrid(xax, yax);

axis(ax(1), 'off')

for idir = 1:nd
    set(gcf, 'currentaxes', ax(1+idir))
    theta = directions(idir);
    
    u = cosd(theta);
    v = sind(theta);

    imagesc(xax, yax, sind(360 * (u*xx + v*yy) ), [-2 2]); hold on
    h = quiver(-u/2,-v/2,u,v,1,'filled', 'Color', 'r');
    h.LineWidth = 2;
    h.MaxHeadSize = 5;
    xlim(xax([1 end]))
    ylim(yax([1 end]))
    axis off
    axis xy
end

for ifreq = 1:nf
    set(gcf, 'currentaxes', ax((ifreq)*(nd+1) + 1))
    imagesc(xax, yax, sind(360*freqs(ifreq)*xx), [-2 2])
    axis off

end

n = zeros(nd, nf);
for idir = 1:nd
    for ifreq = 1:nf
        trialix = stimDir == directions(idir) & stimFreq == freqs(ifreq);
        n(idir, ifreq) = sum(trialix);
    end
end

cmap = plot.coolwarm(10);
for idir = 1:nd
    for ifreq = 1:nf
        set(gcf, 'currentaxes', ax((ifreq)*(nd+1) + idir + 1))
        
        trialix = stimDir == directions(idir) & stimFreq == freqs(ifreq);
        r = robs(trialix,:);
        rs = runspeed(trialix);
        ind = rs > thresh;
        [i,j] = find(r(ind,:));
        plot.raster(opts.lags(j), i, 1, 'Color', cmap(end,:)); hold on
        offset = max(i);
        ind = rs < thresh;
        [i,j] = find(r(ind,:));
        if isempty(offset)
            ylim([0 median(n(:))])
            axis off
            continue
        end
        i = i + offset;
        plot.raster(opts.lags(j), i, 1, 'Color', 'k'); hold on
        axis off
        ylim([0 median(n(:))])
    end
end

% plot.fixfigure(gcf, 6, [3 1])
set(gcf, 'PaperSize', [3 1], 'PaperPosition', [0 0 3 1])
saveas(gcf, fullfile(figdir, sprintf('example_raster_%s_%d.pdf', subject, cc)))


%%
x = stimDir(:)==directions;
psth = (x'*robs) ./ sum(x)';

figure(1); clf
bs =opts.lags(2)-opts.lags(1);

subplot(2,3,1)
plot(opts.lags, psth'/bs)
title(numel(stimDir))

subplot(2,3,2)
[ds, ind] = sort(stimDir);
imagesc(opts.lags, ds, robs(ind,:))

subplot(2,3,5)
plot(ds, sum(robs(ind,:),2), '.')

subplot(2,3,3)
[rs, ind] = sort(nanmean(B{1},2));
imagesc(robs(ind(~isnan(rs)),:))
colormap(1-gray)

subplot(2,3,6)
plot(rs(~isnan(rs)), sum(robs(ind(~isnan(rs)),:),2), '.')
iix = abs(rs) < 3;
ci = prctile(sum(robs(ind(iix),:),2), [5 95]);
hold on
plot(xlim, ci(1)*[1 1], '--r')
plot(xlim, ci(2)*[1 1], '--r')