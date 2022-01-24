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

sessionId = 46;
Exp = io.dataFactoryTreadmill(sessionId);

% get directory to save QA figures in
dataPath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
exname = strrep(Exp.FileTag, '.mat', '');
figDir = fullfile(dataPath, 'imported_sessions_qa', exname);

%%
io.checkCalibration(Exp);

%% get spatial map
RFs = get_spatial_rfs(Exp);

%%

BIGROI = [-30 -10 30 10];

eyePos = Exp.vpx.smo(:,2:3);

binSize = .5;
stat2 = spat_rf_helper(Exp, 'ROI', BIGROI, ...
    'win', [-5 12],...
    'binSize', binSize, 'plot', false, 'debug', false, 'spikesmooth', 0);

%%
stat2 = spat_rf_helper(Exp, 'ROI', BIGROI, ...
    'win', [0 12],...
    'stat', stat2, ...
    'binSize', binSize, 'plot', false, 'debug', false, 'spikesmooth', 0);

%%
% hasrf = find(RFs.sig_pixels > 5 & RFs.sig_pixels < 10);
hasrf = find(stat2.area > binSize & stat2.maxV > 5);

fprintf('%d / %d units have RFs\n', numel(hasrf), numel(stat2.cids))
N = numel(hasrf);
sx = round(sqrt(N));
sy = ceil(sqrt(N));
dims = size(RFs.spatrfs);

figure(3); clf
set(gcf, 'Color', 'w')
ax = plot.tight_subplot(sx, sy, 0.01, 0.05);

for cc = 1:(sx*sy)
    set(gcf, 'currentaxes', ax(cc))
    if cc > N
        axis off
        continue
    end
        
    I = reshape(stat2.spatrfs(:,:,hasrf(cc)), [], 1);
    I = reshape(zscore(I), dims(1:2));
    
    imagesc(stat2.xax, stat2.yax, I); hold on
    try
    plot(stat2.contours{hasrf(cc)}(:,1), stat2.contours{hasrf(cc)}(:,2), 'k')
    end
    axis xy
    grid on
    
end

colormap(plot.coolwarm)

%%
s = io.getUnitLocations(Exp.osp);

N = numel(hasrf);

figure(4); clf
set(gcf, 'Color', 'w')
cmap = lines(4);

for cc = 1:N
    contours = stat2.contours{hasrf(cc)};
    contours = contours + .1*randn(size(contours));
    depth = s.y(stat2.cids(hasrf(cc)));
    [~, id] = min( abs(s.x(stat2.cids(hasrf(cc)))-unique(s.xcoords)));
    
    plot3(contours(:,1), contours(:,2), ones(numel(contours(:,1)),1)*depth, 'Color', cmap(id,:)); hold on
end
zlabel('depth')
ylabel('degrees')
xlabel('degrees')


%%
[xx,yy] = meshgrid(xax, yax);
pow = 20;


for ifac = 1:numel(factors)
    
    i = factors(ifac);
    figure(i); clf
    
    I = reshape(wnmf(:,i), opts.dims);
    I = I ./ max(I(:));
    I(I < 0.1) = 0;

    Iw = I.^pow/sum(I(:).^pow);
    x = xx(:)'*Iw(:);
    y = yy(:)'*Iw(:);

    imagesc(xax, yax, I)
    hold on
    plot(x, y, 'ok')

    % initial parameter guess
    par0 = [x y 1];

    xy = get_rf_contour(xx, yy, I, 'thresh', .05, 'plot', false);

    ROI = [min(xy(:,1)) min(xy(:,2)) max(xy(:,1)) max(xy(:,2))];
    
    % expand ROI
    width = (ROI(3) - ROI(1))*.5;
    height = (ROI(4) - ROI(2))*.5;
    ROI(1) = ROI(1) - width;
    ROI(3) = ROI(3) + width;
    ROI(2) = ROI(2) - width;
    ROI(4) = ROI(4) + width;
    
    plot(ROI([1 3]), ROI([2 2]), 'r', 'Linewidth', 2)
    plot(ROI([1 3]), ROI([4 4]), 'r', 'Linewidth', 2)
    plot(ROI([1 1]), ROI([2 4]), 'r', 'Linewidth', 2)
    plot(ROI([3 3]), ROI([2 4]), 'r', 'Linewidth', 2)

    title(sprintf('ROI %d', i))
    xlabel('Azimuth (pixels)')
    ylabel('Elevation (pixels)')
    
    colormap(plot.coolwarm)
    
    drawnow
    
    stat(ifac) = spat_rf_helper(Exp, 'ROI', ROI, ...
        'win', [0 12],...
        'binSize', .3, ...
        'plot', false, ...
        'debug', false, ...
        'spikesmooth', 0);
    
end

%%
nfac = numel(factors);
NC = numel(stat(1).rffit);
amp = zeros(NC, nfac);
r2 = zeros(NC, nfac); 
sig = [stat.sig];

for ifac = 1:nfac

    hasfit = arrayfun(@(x) x.warning==0, stat(ifac).rffit);

    amp(hasfit,ifac) = arrayfun(@(x) x.amp, stat(ifac).rffit(hasfit));
    r2(hasfit, ifac) = arrayfun(@(x) x.r2, stat(ifac).rffit(hasfit));
end

fac = zeros(NC,1);
r2C = zeros(NC,1);
ampC = zeros(NC,1);
for cc = 1:NC
    
    if all(amp(cc,:)==0) || all(r2(cc,:)==0)
        continue
    end
    
    [~, idmax] = max(amp(cc,:));
    [~, idr2] = max(r2(cc,:));
    
    if idmax == idr2
        r2C(cc) = r2(cc,idr2);
        fac(cc) = idr2;
        ampC(cc) = amp(cc,idmax);
    end
    
end
    
    
    
%%
figure(1); clf
plot(ampC, '.')
%%
hasfit = ampC > 1 & r2C > 0.15;

fitunits = find(hasfit);
N = numel(fitunits);
fprintf('%d units have RFs\n', N)

sx = round(sqrt(N));
sy = ceil(sqrt(N));


figure(3); clf
ax = plot.tight_subplot(sx, sy, 0.01, 0.01);
ecc = zeros(N,1);
ar = zeros(N,1);

for cc = 1:N
    ifac = fac(fitunits(cc));
    I = stat(ifac).spatrf(:,:,fitunits(cc));
    
    set(gcf, 'currentaxes', ax(cc))
%     imagesc(xax, yax, reshape(wnmf(:,ifac), opts.dims)); hold on
    imagesc(stat(ifac).xax, stat(ifac).yax, I); hold on
    plot.plotellipse(stat(ifac).rffit(fitunits(cc)).mu, stat(ifac).rffit(fitunits(cc)).C, 1, 'Color', 'g');
    
    ecc(cc) = stat(ifac).rffit(fitunits(cc)).ecc;
    ar(cc) = stat(ifac).rffit(fitunits(cc)).ar;
    
%     xlim([-30 30])
%     ylim([-15 15])
%     plot(gridx, gridy, 'w'); 
%     plot(gridy, gridx, 'w')
%     xlim(xax([1 end]))
%     ylim(yax([1 end]))
    
end

%%
figure(4); clf
plot(ecc, ar, 'o')
hold on
plot(xlim, .1*xlim, 'r')

%%
dotTrials = io.getValidTrials(Exp, 'Dots');
if ~isempty(dotTrials)
    
    
    BIGROI = [-1 -1 1 1]*5;
    
    % eyePos = eyepos;
    eyePos = Exp.vpx.smo(:,2:3);
    % eyePos(:,1) = -eyePos(:,1);
    % eyePos(:,2) = -eyePos(:,2);
    
    stat = spat_rf_helper(Exp, 'ROI', BIGROI, ...
        'win', [0 12],...
        'binSize', .3, 'plot', true, 'debug', false, 'spikesmooth', 0);
    
    
    
    NC = numel(stat.cgs);
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    figure(11); clf
    ax = plot.tight_subplot(sx, sy, 0.02);
    for cc = 1:NC
        set(gcf, 'currentaxes', ax(cc))
        imagesc(stat.xax, stat.yax, stat.spatrf(:,:,cc));
        hold on
        plot(xlim, [0 0], 'r')
        plot([0 0], ylim, 'r')
        axis xy
        title(sprintf('Unit: %d', Exp.osp.cids(cc)))
    end
      
    
    plot.suplabel('Coarse RFs', 't');
    plot.fixfigure(gcf, 10, [sy sx]*2, 'offsetAxes', false)
    saveas(gcf, fullfile(figDir, 'Dot_RFs.pdf'))
        
    
end

%%

Exp.spikeTimes = Exp.osp.st;
Exp.spikeIds = Exp.osp.clu;

% convert to D struct format
D = io.get_drifting_grating_output(Exp);

% Note: PLDAPS sessions and MarmoV5 will load into different formats

%% Step 2.1: plot one session
% This just demonstrates how to index into individual sessions and get
% relevant data

sessionId = 1; % pick a session id

% index into grating times from that session
gratIx = D.sessNumGratings==sessionId;
gratIx = gratIx & D.GratingOnsets > min(D.spikeTimes) & D.GratingOffsets < max(D.spikeTimes);
D.sessNumGratings(~gratIx) = 10;

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

subplot(3,1,2) % plot spike count


imagesc(treadTime, 1:NC, imgaussfilt(spikeRate', [5 20])); hold on
colormap(1-gray)
xlim(treadTime([1 end]))

% PLOT TREADMILL RUNNING SPEED
subplot(3,1,3) % tread speed
plot(treadTime, treadSpeed , 'k'); hold on
xlabel('Time (s)')
ylabel('Speed (cm/s)')

xlim(treadTime([1 end]))

title('Treadmill')
trmax = prctile(D.treadSpeed(treadIx), 99);

% plot.fixfigure(gcf, 12, [8 11])
% saveas(gcf, fullfile(figDir, 'grating_sess_birdseye.pdf'))

%% play movie
vidObj = VideoWriter(fullfile(figDir, 'session.mp4'), 'MPEG-4');
vidObj.Quality = 100;
vidObj.FrameRate = 5;
open(vidObj);


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
        break
    end
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);
end

close(vidObj);


%% Unit by Unit simple analysis
thresh = 3;
nboot = 100;

unitList = unique(D.spikeIds);
NC = numel(unitList);

corrRho = zeros(NC,1);
corrPval = zeros(NC,1);

frBaseR = zeros(NC,3);
frBaseS = zeros(NC,3);

frStimR = zeros(NC,3);
frStimS = zeros(NC,3);

for cc = 1:NC
    unitId = unitList(cc);
    [stimDir, robs, runSpd, opts] = bin_ssunit(D, unitId, 'win', [-.2 .2]);
    
    goodIx = getStableRange(sum(robs,2), 'plot', false);
    
    stimDir = stimDir(goodIx);
    robs = robs(goodIx,:);
    runSpd = runSpd{1}(goodIx,:);
    
    iix = opts.lags < 0;
    frbase = sum(robs(:,iix),2) / (max(opts.lags(iix)) - min(opts.lags(iix)));
    
    spd = mean(runSpd,2);
    
    [corrRho(cc), corrPval(cc)] = corr(spd, frbase, 'type', 'Spearman');
    
    runTrials = find(spd > thresh);
    statTrials = find(abs(spd) < 1);
    mixTrials = [runTrials; statTrials];
    
    nrun = numel(runTrials);
    nstat = numel(statTrials);
    
    n = min(nrun, nstat);
    
    frBaseR(cc,:) = prctile(mean(frbase(runTrials(randi(nrun, [n nboot])))), [2.5 50 97.5]);
    frBaseS(cc,:) = prctile(mean(frbase(statTrials(randi(nstat, [n nboot])))), [2.5 50 97.5]);
    
    iix = opts.lags > 0.04 & opts.lags < opts.lags(end)-.15;
    frstim = sum(robs(:,iix),2) / (max(opts.lags(iix)) - min(opts.lags(iix)));
    
    frStimR(cc,:) = prctile(mean(frstim(runTrials(randi(nrun, [n nboot])))), [2.5 50 97.5]);
    frStimS(cc,:) = prctile(mean(frstim(statTrials(randi(nstat, [n nboot])))), [2.5 50 97.5]);
    
end

%% plot some outcomes

incBaseIx = find(frBaseR(:,2) > frBaseS(:,3));
decBaseIx = find(frBaseR(:,2) < frBaseS(:,1));

incStimIx = find(frStimR(:,2) > frStimS(:,3));
decStimIx = find(frStimR(:,2) < frStimS(:,1));

figure(1); clf
set(gcf, 'Color', 'w')
ms = 4;
cmap = lines;
subplot(1,2,1)
plot(frBaseS(:,[2 2])', frBaseR(:,[1 3])', 'Color', .5*[1 1 1]); hold on
plot(frBaseS(:,[1 3])', frBaseR(:,[2 2])', 'Color', .5*[1 1 1])
plot(frBaseS(:,2), frBaseR(:,2), 'o', 'Color', cmap(1,:), 'MarkerSize', ms, 'MarkerFaceColor', cmap(1,:))
plot(frBaseS(incBaseIx,2), frBaseR(incBaseIx,2), 'o', 'Color', cmap(2,:), 'MarkerSize', ms, 'MarkerFaceColor', cmap(2,:))
plot(frBaseS(decBaseIx,2), frBaseR(decBaseIx,2), 'o', 'Color', cmap(3,:), 'MarkerSize', ms, 'MarkerFaceColor', cmap(3,:))

xlabel('Stationary')
ylabel('Running')
title('Baseline Firing Rate')

set(gca, 'Xscale', 'log', 'Yscale', 'log')
plot(xlim, xlim, 'k')

subplot(1,2,2)
plot(frStimS(:,[2 2])', frStimR(:,[1 3])', 'Color', .5*[1 1 1]); hold on
plot(frStimS(:,[1 3])', frStimR(:,[2 2])', 'Color', .5*[1 1 1])
plot(frStimS(:,2), frStimR(:,2), 'o', 'Color', cmap(1,:), 'MarkerSize', ms, 'MarkerFaceColor', cmap(1,:))
plot(frStimS(incStimIx,2), frStimR(incStimIx,2), 'o', 'Color', cmap(2,:), 'MarkerSize', ms, 'MarkerFaceColor', cmap(2,:))
plot(frStimS(decStimIx,2), frStimR(decStimIx,2), 'o', 'Color', cmap(3,:), 'MarkerSize', ms, 'MarkerFaceColor', cmap(3,:))

xlabel('Stationary Firing Rate')
ylabel('Running Firing Rate')
title('Stim-driven firing rate')

set(gca, 'Xscale', 'log', 'Yscale', 'log')
plot(xlim, xlim, 'k')

nIncBase = numel(incBaseIx);
nDecBase = numel(decBaseIx);

nIncStim = numel(incStimIx);
nDecStim = numel(decStimIx);

modUnits = unique([incBaseIx; decBaseIx; incStimIx; decStimIx]);
nModUnits = numel(modUnits);

fprintf('%d/%d (%02.2f%%) increased baseline firing rate\n', nIncBase, NC, 100*nIncBase/NC)
fprintf('%d/%d (%02.2f%%) decreased baseline firing rate\n', nDecBase, NC, 100*nDecBase/NC)

fprintf('%d/%d (%02.2f%%) increased stim firing rate\n', nIncStim, NC, 100*nIncStim/NC)
fprintf('%d/%d (%02.2f%%) decreased stim firing rate\n', nDecStim, NC, 100*nDecStim/NC)

fprintf('%d/%d (%02.2f%%) total modulated units\n', nModUnits, NC, 100*nModUnits/NC)

[pvalStim, ~, sStim] = signrank(frStimS(:,2), frStimR(:,2));
[pvalBase, ~, sBase] = signrank(frBaseS(:,2), frBaseR(:,2));

fprintf('Wilcoxon signed rank test:\n')
fprintf('Baseline rates: p = %02.10f\n', pvalBase)
fprintf('Stim-driven rates: p = %02.10f\n', pvalStim)

good = ~(frBaseR(:,2)==0 | frBaseS(:,2)==0);

m = geomean(frBaseR(good,2)./frBaseS(good,2));
ci = bootci(nboot, @geomean, frBaseR(good,2)./frBaseS(good,2));

fprintf("geometric mean baseline firing rate ratio (Running:Stationary) is %02.3f [%02.3f, %02.3f] (n=%d)\n", m, ci(1), ci(2), sum(good)) 

m = geomean(frStimR(:,2)./frStimS(:,2));
ci = bootci(nboot, @geomean, frStimR(:,2)./frStimS(:,2));

fprintf("geometric mean stim-driven firing rate ratio (Running:Stationary) is %02.3f [%02.3f, %02.3f] (n=%d)\n", m, ci(1), ci(2), NC)




%% Do direction / orientation decoding

% example session to explore
Dstat = decode_stim(D, 1);

%% Bootstrapped empirical analyses and tuning curve fits

fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');

D.sessNumTread = ones(size(D.treadTime))*1;
stat = tuning_empirical(D, 'binsize', 10e-3, ...
    'runningthresh', 3, ...
    'nboot', 500, ...
    'seed', 1234);



%% Plot all tuning curves
fitS = stat.TCfitS;
fitR = stat.TCfitR;
NC = numel(fitS);

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));


figure(1); clf
ax = plot.tight_subplot(sx, sy, 0.02);
for cc = 1:NC
    if min(fitS(cc).numTrials, fitR(cc).numTrials) < 50
        continue
    end
%     
%     if fitS(cc).llrpval > 0.05 && fitR(cc).llrpval > 0.05
%         continue
%     end
    fprintf("Unit %d/%d\n", cc, NC)

    set(gcf, 'currentaxes', ax(cc))
    
    cmap = lines;
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

    set(gca, 'XTick', [], 'YTick', [])
%     set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    text(10, .1*max(ylim), sprintf('%d', cc), 'fontsize', 5)
%     axis off
    title(sprintf('Unit: %d', Exp.osp.cids(cc)))
end

plot.fixfigure(gcf, 10, [sx sy]*2, 'offsetAxes', false);
saveas(gcf, fullfile(figDir, 'tuning_curves.pdf'))
