
[Exp,S] = io.dataFactoryTreadmill(4);
% add unit quality (since some analyses require this field)
Exp.osp.cgs = ones(size(Exp.osp.cids))*2;
io.checkCalibration(Exp);

%%
c = 1;
I = squeeze(A.z.Shapes(c,:,:));
figure(1); clf, plot(I')

%%


% get visually driven units
[spkS, W] = io.get_visual_units(Exp, 'plotit', true, 'numTemporalBasis', 10); %, 'ROI', BIGROI);


%%
% plot spatial RFs to try to select a ROI
unit_mask = 0;
NC = numel(spkS);
hasrf = find(~isnan(arrayfun(@(x) x.x0, spkS)));
figure(2); clf
set(gcf, 'Color', 'w')
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

xax = spkS(1).xax/Exp.S.pixPerDeg;
yax = spkS(1).yax/Exp.S.pixPerDeg;
cmap = lines(NC);
for cc = 1:NC
    I = spkS(cc).srf;
    I = I / max(abs(I(:)));
    unit_mask = unit_mask + I;
end
unit_mask = unit_mask / max(unit_mask(:));

for cc = 1:NC %hasrf(:)'
    subplot(sx,sy,cc,'align')
    
    
    I = spkS(cc).srf;
    I = I - mean(I);
    I = I / max(abs(I(:)));

    imagesc(xax, yax, I, [-1 1]); hold on
    colormap parula
    axis xy
%     xlim([-4 4])
%     ylim([-4 4])
    
    plot([0 0], ylim, 'w')
    plot(xlim,[0 0], 'w')
%     xlim([0 40])
%     ylim([-40 0])
%     [~, h] = contour(xax, yax, rf, [.75:.05:1], 'Color', cmap(cc,:)); hold on
end

figure(1); clf
ppd = Exp.S.pixPerDeg;
imagesc(xax*ppd, yax*ppd,unit_mask); axis xy

%%
eyePos = Exp.vpx.smo(:,2:3);
figure(1); clf
plot(eyePos(:,1), eyePos(:,2), '.')

%% Forward correlation spatial RF mapping
BIGROI = [-10 -8 10 8];

stat = spat_rf_helper(Exp, 'ROI', BIGROI, ...
    'win', [0 40],...
    'binSize', .3, 'plot', true, 'debug', false, 'eyePos', eyePos);

%% saccade modulation
frf = fixrate_by_stim(Exp, 'stimSets', {'DriftingGrating', 'BackImage'});

%% Grating tuning

validTrials = io.getValidTrials(Exp, 'DriftingGrating');

tonsets = [];
toffsets = [];
directions = [];
speeds = [];
treadTime = [];
treadSpeed = [];
figure(1); clf

for iTrial = validTrials(:)'
    pos = Exp.D{iTrial}.treadmill.locationSpace(:,4); % convert to meters
    ttime = Exp.D{iTrial}.treadmill.locationSpace(:,1);
    
    dpos = diff(imgaussfilt(pos, 5)) ./ diff(ttime);
    spd = [0; abs(dpos(:))];
    
    
    bad = isnan(Exp.D{iTrial}.treadmill.locationSpace(:,2));
    bad = imboxfilt(double(bad), 101)>0;
    spd(bad) = nan;
    plot(spd); hold on
    
    ttime(bad) = nan;
    
    ylim([0 5])
    
    treadTime = [treadTime; ttime];
    treadSpeed = [treadSpeed; spd];
    
    % Fields:
    % time, orientation, cpd, phase, direction, speed, contrast
    contrast = Exp.D{iTrial}.PR.NoiseHistory(:,7);
    
    onsets = find(diff(contrast)>0)+1;
    offsets = find(diff(contrast)<0)+1;
    
%     figure(1); clf
%     plot(contrast); hold on
%     plot(onsets, .5*ones(numel(onsets), 1), 'o')
%     plot(offsets, .5*ones(numel(offsets), 1), '+')
    
    if contrast(1) > 0
        onsets = [1; onsets];
    end
    
    if contrast(end) > 0
        offsets = [offsets; numel(contrast)];
    end
    
    
    tonsets = [tonsets; Exp.D{iTrial}.PR.NoiseHistory(onsets,1)];
    toffsets = [toffsets; Exp.D{iTrial}.PR.NoiseHistory(offsets,1)];
    directions = [directions;  Exp.D{iTrial}.PR.NoiseHistory(onsets,5)];
    speeds = [speeds; Exp.D{iTrial}.PR.NoiseHistory(onsets,6)];
    
end

tonsets = Exp.ptb2Ephys(tonsets);
toffsets = Exp.ptb2Ephys(toffsets);
treadTime = Exp.ptb2Ephys(treadTime);

ths = unique(directions);
nth = numel(ths);
nt = numel(directions);


%%
st = Exp.osp.st;
clu = Exp.osp.clu;


figure(1); clf
plot.raster(st, clu, 1); hold on
plot((tonsets*[1 1])', (ones(numel(tonsets),1)*ylim)', 'r', 'Linewidth', 2)
plot((toffsets*[1 1])', (ones(numel(toffsets),1)*ylim)', 'g', 'Linewidth', 2)

%% bin spikes in window aligned ton onset
binsize = 1e-3;
win = [-.5 2.5];

rem = find(tonsets+win(2) > max(st), 1);
tonsets(rem:end) = [];
directions(rem:end) = [];

NC = max(clu);
bs = (st==0) + ceil(st/binsize);

spbn = sparse(bs, clu, ones(numel(bs), 1));
lags = win(1):binsize:win(2);
blags = ceil(lags/binsize);

numlags = numel(blags);
balign = ceil(tonsets/binsize);
numStim = numel(balign);

% binning treadmill speed
treadBins = ceil(treadTime/binsize);
treadSpd = nan(size(spbn,1), 1);
treadSpd(treadBins(~isnan(treadSpeed))) = treadSpeed(~isnan(treadSpeed));
treadSpd = max(repnan(treadSpd, 'v5cubic'),0);

% Do the binning here
disp('Binning spikes')
spks = zeros(numStim, NC, numlags);
tspd = zeros(numStim, numlags);
for i = 1:numlags
    spks(:,:,i) = spbn(balign + blags(i),:);
    tspd(:,i) = treadSpd(balign + blags(i));
end

disp('Done')
cc = 1

%%
figure(1); clf
plot(mean(tspd,2))



%%
thresh = 1;

figure(10); clf
runningSpd = mean(tspd,2);
runningTrial = runningSpd > thresh;

histogram(runningSpd(~runningTrial), 'binEdges', 0:.1:25, 'FaceColor', [.5 .5 .5])
hold on
clr = [1 .2 .2];
histogram(runningSpd(runningTrial), 'binEdges', 0:.1:25, 'FaceColor', clr)
plot.fixfigure(gcf,10,[4 4]);
xlabel('Running Speed (units?)')
ylabel('Trial Count') 
text(.1, mean(ylim), sprintf('Running Trials (n=%d)', sum(runningTrial)), 'Color', clr, 'FontSize', 14)
set(gca, 'YScale', 'linear')

% plot.formatFig(gcf, [1 1], 'default')
%%
% cc = 1
ths = unique(directions);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf
% ax = plot.tight_subplot(sx, sy, 0.01, 0.01);
% 
% for cc = 1:NC
%     set(gcf, 'currentaxes', ax(cc))
    
cc = cc + 1;
if cc > NC
    cc = 1;
end


subplot(1,2,1) % no running

thctr = 0;
for th = 1:numel(ths)
    iix = find(directions==ths(th) & ~runningTrial);
%     sum(
    nt = numel(iix);
    spk = squeeze(spks(iix,cc,:));
    [ii,jj] = find(spk);
    plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
    thctr = thctr + nt;
end
title('No Running')
axis tight

subplot(1,2,2)

thctr = 0;
for th = 1:numel(ths)
    iix = find(directions==ths(th) & runningTrial);
%     sum(
    nt = numel(iix);
    spk = squeeze(spks(iix,cc,:));
    [ii,jj] = find(spk);
    plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
    thctr = thctr + nt;
end
title('Running')
axis tight

% end



%%
fkern = ones(20,1)/20;
spksfilt = spks;
for i = 1:numStim
    spksfilt(i,:,:) = filtfilt(fkern, 1, squeeze(spksfilt(i,:,:))')';
end

nd = numel(ths);
psthsRunning = zeros(numlags, nd, NC);
psthsNoRunning = zeros(numlags, nd, NC);
for i = 1:nd
    iix = directions==ths(i);
    psthsRunning(:,i,:) = squeeze(mean(spksfilt(iix & runningTrial,:,:),1))';
    psthsNoRunning(:,i,:) = squeeze(mean(spksfilt(iix & ~runningTrial,:,:),1))';
end
cc = 1;

% trim filtering artifacts
psthsRunning(1:10,:,:) = nan;
psthsRunning(end-10:end,:,:) = nan;
psthsNoRunning(1:10,:,:) = nan;
psthsNoRunning(end-10:end,:,:) = nan;

%%
cc = cc + 1;
if cc > NC
    cc = 1;
end
vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)];
clim = [min(vals(:)) max(vals(:))];
figure(1); clf
subplot(1,2,1)

imagesc(lags, ths, psthsRunning(:,:,cc)', clim)
xlabel('Direction')
ylabel('Time')
title('Running')

subplot(1,2,2)
imagesc(lags, ths, psthsNoRunning(:,:,cc)', clim)
xlabel('Direction')
ylabel('Time')
title('No Running')
colormap(plot.viridis)


%% plot Tuning Curves
tdur = lags(end)-lags(1);
tcRun = squeeze(nansum(psthsRunning))/tdur;
tcNoRun = squeeze(nansum(psthsNoRunning))/tdur;
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(1); clf
for cc = 1:NC
    subplot(sx, sy, cc)
    plot(ths, tcRun(:,cc), 'k', 'Linewidth', 2); hold on
    plot(ths, tcNoRun(:,cc), 'r', 'Linewidth', 2); hold on
    set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    title(cc)
end

plot.fixfigure(gcf, 12, [14 14])
a = plot.suplabel('Spike Rate', 'y'); 
a.FontSize = 20;
a = plot.suplabel('Direction', 'x');
a.FontSize = 20;
%%
cc = 1;

%%

cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(1); clf

cmap = parula(nth);
ax = plot.tight_subplot(2, nth, 0.01, 0.01);
    
vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)]/1e-3;
clim = [min(vals(:)) max(vals(:))];


for ith = 1:nth
    
    set(gcf, 'currentAxes', ax(ith));
    plot(lags, imgaussfilt(psthsNoRunning(:,ith,cc)/1e-3, 25), 'Color', cmap(ith,:), 'Linewidth', 2); hold on
    clr = (cmap(ith,:) + [1 1 1])/2;
    plot(lags, imgaussfilt(psthsRunning(:,ith,cc)/1e-3, 25), '-', 'Color', clr, 'Linewidth', 2); hold on
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

xlabel('Time from grating onset (s)')
ylabel('Firing Rate')

%%

ith = 1;
figure(2); clf
[dx, dy] = pol2cart(ths(ith)/180*pi, 1);
    q = quiver(0,0,dx,dy,'Color', cmap(ith,:), 'Linewidth', 5, 'MaxHeadSize', 2); hold on
    
    R = [cos(pi/2) sin(pi/2); -sin(pi/2) -cos(pi/2)];
    S = [.1 0; 0 .1];
    dxdy = [dx dy] * R*S;
    plot(dxdy(1), dxdy(2), 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 20)
    dxdy = [dx dy] * -R*S;
    plot(dxdy(1), dxdy(2), 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 20)


%%






%%
% hold on

% plot(tonsets, 'o')

% bins = 
% histcounts(





% D.z.Times
%%

itrial = itrial - 1;
figure(1); clf

plot(exy(itrial,:,1), exy(itrial,:,2)); hold on
[u,v] = pol2cart(directions(itrial)/180*pi, speeds(itrial));
quiver(0,0,u,v, .5)
xlim([-1 1]*.5)
ylim([-1 1]*.5)

%%
load 
% spd = sqrt(exy(:,:,1).^2 + exy(:,:,2).^2);
% for i = 1:nt
%     spd(i,2:end) = diff(spd(i,:));
% end

% imagesc(spd)
% 
% imagesc(spd(all(spd < .5,2),:)')





