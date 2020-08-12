
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

switch user
    case 'jakework'
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\NIMclass
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\sNIMclass
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\L1General'))    
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\minFunc_2012'))  
    case 'jakelaptop'
        addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
        addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
        addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))    
        addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))  
end

%% load data
sessId = 12;
[Exp, S] = io.dataFactory(sessId);


%% plot rasters
stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);
binsize = 4e-3;

[~, ~, ~, lagsrast, spksNI, fixdurNI, validSaccadesNI] = get_saccade_relative_rate(Exp, S, validTrials, 'binsize', binsize);




%%
fixdur = [fixdurNI]; %; fixdurG];
spks = [spksNI]; %; spksG];
[~, ind] = sort(fixdur, 'ascend');

%%
win = [-.1 .5];
figure(1); clf
set(gcf, 'Color', 'w')
NC = size(spksNI, 2);
for cc = 1:NC
    subplot(6, ceil(NC/6), cc, 'align')
    [i, j] = find(spksNI(ind,cc,:));
    plot.raster(lagsrast(j), i, 10); hold on
    plot([0 0], ylim, 'r')
    plot(fixdur(ind), 1:numel(ind), 'g')
    xlim(win)
    ylim([1 numel(ind)])
    title(sprintf('Unit: %d', cc))
    axis off
%     ylim(ind(end)-[400 0])
end

%%
% cc=1;
% cc = 18;
% cc = 42;
cc = mod(cc + 1, NC); cc = max(cc, 1);
figure(1); clf
% subplot(4,1,1:3)
rast = squeeze(spks(ind,cc,:));
[i, j] = find(rast);
plot.raster(lagsrast(j), i, 20); hold on
axis off
set(gcf, 'PaperSize', [4 5], 'PaperPosition', [0 0 4 5])
saveas(gcf, fullfile('Figures', 'K99', sprintf('raster%d.png', cc)))
axis on
plot([0 0], ylim, 'r')
plot(fixdur(ind), 1:numel(ind), 'g')
xlim(win)
ylim([500 numel(ind)])
title(sprintf('Unit: %d', cc))
ylabel('Fixation #')
plot.fixfigure(gcf, 10, [4 5])
% saveas(gcf, fullfile('Figures', 'K99', sprintf('raster%d.pdf', cc)))

% get saccade size
sstart = Exp.slist( validSaccadesNI,4);
sstop = Exp.slist( validSaccadesNI,5);
dx = Exp.vpx.smo(sstop,2)-Exp.vpx.smo(sstart,2);
dy = Exp.vpx.smo(sstop,3)-Exp.vpx.smo(sstart,3);
sacSize = hypot(dx, dy);

% sort by saccade Size
sacSizes = [1 2 4 6 8 10];
ns = numel(sacSizes)-1;
figure(2); clf
cmap = jet(ns);
for i = 1:ns
    ix = sacSize > sacSizes(i) & sacSize < sacSizes(i+1) & fixdurNI > .1;
    disp(sum(ix))
    m = mean(squeeze(spks(ix,cc,:)));
    m = imgaussfilt(m, 2.5);
    h(i) = plot(lagsrast, m/binsize, 'Color', cmap(i,:)); hold on
end
xlim([-.05 .15])
xlabel('Time from saccade offset')
ylabel('Firing Rate')
title(cc)
legend(h, arrayfun(@(x) sprintf('%02.2f', x), sacSizes(1:ns), 'uni', 0), 'Location', 'Best')

% sort by saccade direction
sacTheta = wrapTo360(cart2pol(dx,dy)/pi*180);
sacThetas = [-45:45:(360+45)];
ns = numel(sacThetas)-1;
figure(3); clf
cmap = jet(ns);
for i = 1:ns
    ix = sacTheta > sacThetas(i) & sacTheta < sacThetas(i+1) & fixdurNI > .1;
%     disp(sum(ix))
    m = mean(squeeze(spks(ix,cc,:)));
    m = imgaussfilt(m, 2.5);
    h(i) = plot(lagsrast, m/binsize, 'Color', cmap(i,:)); hold on
end
xlim([-.05 .15])
xlabel('Time from saccade offset')
ylabel('Firing Rate')
title(cc)
legend(h, arrayfun(@(x) sprintf('%02.2f', x), sacThetas(1:ns), 'uni', 0), 'Location', 'Best')
%%

figure(2); clf
thresh = 0.2;
binsize = 1e-3;

iix = fixdurNI > thresh;
sp = squeeze(spksNI(iix,cc,:))/binsize;
sp = filter(ones(10,1)/10, 1, sp')';
mFR = mean(sp);
sFR = std(sp)/sqrt(size(sp,1));

cmap = lines;
plot.errorbarFill(lagsrast, mFR, sFR, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .5); hold on

iix = fixdurG > thresh;
sp = squeeze(spksG(iix,cc,:))/binsize;
sp = filter(ones(10,1)/10, 1, sp')';
mFR = mean(sp);
sFR = std(sp)/sqrt(size(sp,1));

cmap = lines;
plot.errorbarFill(lagsrast, mFR, sFR, 'k', 'FaceColor', cmap(4,:), 'EdgeColor', cmap(4,:), 'FaceAlpha', .5)

xlabel('Time from fixation onset')
ylabel('Firing rate')
legend('BackImage', 'Gabor Noise')
plot.fixfigure(gcf, 10, [4 2])
saveas(gcf, fullfile('Figures', 'K99', sprintf('sacpsth%d.pdf', cc)))

%%
figure(2); clf
fd = fixdurNI(fixdurNI>0.05);
mean(fd>0.1 & fd < 0.2)
histogram(fd, 0:.01:.5, 'FaceColor', .5*[1 1 1], 'EdgeColor', .5*[1 1 1])
xlim(win)
ylim([0 200])
plot.fixfigure(gcf, 10, [4 2])
saveas(gcf, fullfile('Figures', 'K99', 'fixduration.pdf'))
%% plot response variability early, late
iix = fixdurNI > thresh;

lagix = lagsrast > 0 & lagsrast < 0.1;
w = hanning(sum(lagix));
% w = w./sum(w);
spE = squeeze(spksNI(iix,cc,lagix))*w;

lagix = lagsrast > 0.1 & lagsrast < 0.2;
w = hanning(sum(lagix));
% w = w./sum(w);
spL = squeeze(spksNI(iix,cc,lagix))*w;

figure(1); clf
histogram(spE-spL)
% plot(spE, spL, '.'); hold on
% plot(xlim, xlim, 'k')

%%
clf
C = hist3([spE(:,cc), spL(:,cc)]/.1, {0:1:50, 0:1:50});
imagesc(C)
axis xy
% plot(xlim, xlim, 'k')




%%
cc = mod(cc + 1, NC); cc = max(cc, 1);


figure(2); clf
% backimage
iix = fixdurNI > thresh;
mFR = squeeze(mean(spksNI(iix,:,:)))'/binsize;
mFR = filter(ones(10,1)/10, 1, mFR);
tot = mean(mFR);
% plot(lagsrast, mean(mFR(:,cc)./tot,2)); hold on
plot(lagsrast, mFR(:,cc)); hold on

% gabor
iix = fixdurG > thresh;
mFR = squeeze(mean(spksG(iix,:,:)))'/binsize;
mFR = filter(ones(10,1)/10, 1, mFR);
% plot(lagsrast, mean(mFR./tot,2)); hold on
plot(lagsrast, mFR(:,cc)); hold on
% 
% grating
iix = fixdurGr > thresh;
mFR = squeeze(mean(spksGr(iix,:,:)))'/binsize;
mFR = filter(ones(10,1)/10, 1, mFR);
% plot(lagsrast, mean(mFR./tot,2)); hold on
plot(lagsrast, mFR(:,cc)); hold on
% % grating
% iix = fixdurF > thresh;
% mFR = squeeze(mean(spksF(iix,:,:)))'/binsize;
% mFR = filter(ones(10,1)/10, 1, mFR);
% plot(lagsrast, mFR(:,cc)); hold on
title(cc)
%%
k = mod(k + 1, NC); k = max(k, 1);
figure(2); clf
set(gcf, 'Color', 'w')
% plot(lags, squeeze(var(spks(fixdur > .25,k,:)))'/binsize); hold on
thresh = .25;
mFR = squeeze(mean(spks(fixdur > thresh,:,:)))'/binsize;
mFR = filter(ones(5,1)/5, 1, mFR);

plot(lags, mFR); hold on
plot(thresh*[1 1], ylim, 'k--')
xlabel('Time from fixation onset')
ylabel('Spike Rate')

% plot(squeeze(mean(spks(fixdur < .2,:,:),1))');
figure(3); clf
set(gcf, 'Color', 'w')
% plot(lags, squeeze(var(spks(fixdur > .25,k,:)))'/binsize); hold on
thresh = .25;
mFR = squeeze(mean(spks(fixdur > thresh,:,:)))'/binsize;
mFR = filter(ones(5,1)/5, 1, mFR);
% mTot = full(mean(Y(:,S.cids)))/binsize;
mTot = mean(mFR(lags < .250 & lags > .1,:));
mTot(mTot < 1) = nan;
plot(lags, mFR./mTot); hold on
plot(thresh*[1 1], ylim, 'k--')
plot(xlim, [1 1], 'k--')
xlabel('Time from fixation onset')
ylabel('Relative Rate')

figure(1); clf
plot(mean(mFR(lags > .15,:)), mean(mFR(lags < .1,:)), '.'); hold on
plot(xlim, xlim, 'k')

%%

labels = Exp.vpx.Labels;
good = labels == 1 | labels ==2;
[bw, num] = bwlabel(good);
ar = nan(num, 1);
for i = 1:num
    ar(i) = sum(bw==i);
end

%%
   
[~, ind] = sort(ar, 'descend');

longest = bw == ind(1);


t = Exp.vpx.smo(:,1);
x = sgolayfilt(Exp.vpx.smo(:,2), 2, 7);
y = Exp.vpx.smo(:,3);

figure(1); clf
plot(t(longest), x(longest), 'k')

