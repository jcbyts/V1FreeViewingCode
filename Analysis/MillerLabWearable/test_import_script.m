%% Step 0: set your paths
% The FREEVIEWING codebase uses matlab preferences to manage paths (so
% different users can have different paths)

addFreeViewingPaths('jakelaptop') % switch to your user
addpath Analysis/HukLabTreadmill/ % the code will always assume you're running from the FreeViewing base directory

%% 
% dataPath = '/Volumes/GoogleDrive/My Drive/Test data (Suki 211105 chair Intan)/';
dataPath = '~/Downloads/Suki211105chairIntan/';
Exp = io.basic_marmoview_import(dataPath);

%% load spikes
% Load spikes file
spks = load(fullfile(dataPath, 'SpikeTimes.mat'));

if isfield(spks, 'SpikeTimes') && iscell(spks.SpikeTimes)
    nUnits = numel(spks.SpikeTimes);
    st = [];
    clu = [];

    for kunit = 1:nUnits
        stmp = double(spks.SpikeTimes{kunit});
        st = [st; stmp];
        clu = [clu; kunit*ones(numel(stmp),1)];
    end

else

    % get name of marmoview file
    mvfiles = dir(fullfile(dataPath, '*z.mat')); % marmoview files
    fname = mvfiles(1).name; % one file

    id = strfind(fname,'_');
    subj = fname(id(1)+1:id(2)-1); % subject
    dat = datenum(fname(id(2)+1:id(3)-1), 'ddmmyy'); % date (as number)
    dateStr = [subj '_' datestr(dat, 'yymmdd')]; % tag to find session id in the spikes file

    recId = find(contains(spks.z.RecId, dateStr)); % spikes file id that corresponds to this session

    st = [];
    clu = [];

    for rId = recId(:)'

        figure(rId); clf

        unitlist = find(cellfun(@(x) ~isempty(x), spks.z.Times{rId}));
        nUnits = numel(unitlist);
        sx = ceil(sqrt(nUnits));
        sy = round(sqrt(nUnits));
        ax = plot.tight_subplot(sx, sy, 0.05);

        for iunit = 1:nUnits
            kunit = unitlist(iunit);

            set(gcf, 'currentaxes', ax(iunit))
            plot(squeeze(spks.z.Shapes(:,kunit,:))', 'Linewidth', 2)
            title(sprintf('Unit: %d', kunit))

            stmp = double(spks.z.Times{rId}{kunit}) / spks.z.Sampling;
            st = [st; stmp];
            clu = [clu; kunit*ones(numel(stmp),1)];
        end
    end

end

Exp.sp = struct('st', st, 'clu', clu, 'cids', unique(clu));
Exp.osp = Exp.sp;




%% Import eye position
disp('IMPORT EYE POSITION')

% Import eye position signals
[Exp, fig] = io.import_eye_position(Exp, dataPath, 'fid', 1);

%%

% if ~isempty(fig)
%     saveas(fig, fullfile(figDir, 'eye_time_sync.pdf'))
% end

disp('CLEANUP CALIBRATION')
fig = io.checkCalibration(Exp);

%%
% saveas(fig, fullfile(figDir, 'eye_calibration_from_file.pdf'))

if any(isnan(S.CalibMat))
    disp('Running Manual Calibration')
    Exp.FileTag = S.processedFileName;
    C = calibGUI(Exp, 'datasetFile', fullfile(datashare, 'datasets.xls'));
    keyboard
    S.CalibMat = C.cmat;
end
% redo calibration offline using FaceCal data
[eyePos, eyePos0] = io.getEyeCalibrationFromRaw(Exp, 'cmat', S.CalibMat);

Exp.vpx.smo(:,2:3) = eyePos0;
fig = io.checkCalibration(Exp);
fig.Position(1) = 50;
title('Before Auto Refine')

Exp.vpx.smo(:,2:3) = eyePos;
fig = io.checkCalibration(Exp);
fig.Position(1) = fig.Position(1) + 100;
title('After Auto Refine')

f = uifigure('Position', [500 200 400 170]);
selection = uiconfirm(f, 'Use Auto Refine Calibration?', 'Which Calibration', ...
    'Options', {'Before', 'After'});
close(f)

if strcmp(selection, 'After')
    Exp.vpx.smo(:,2:3) = eyePos;
    fig = io.checkCalibration(Exp);
    title('After Auto Refine')
    saveas(fig, fullfile(figDir, 'eye_calibration_redo_autorefine.pdf'))
else
    Exp.vpx.smo(:,2:3) = eyePos0;
    fig = io.checkCalibration(Exp);
    title('Before Auto Refine')
    saveas(fig, fullfile(figDir, 'eye_calibration_redo_before.pdf'))
end

close all
% 
%%

% trial = 15;
% eyepos = Exp.D{trial}.eyeData(:,2:3);
% eyepos = Exp.vpx.raw(:,2:3);
% eyepos(:,2) = 1 - eyepos(:,2);
% 
% x = (eyepos(:,1)-Exp.D{trial}.c(1)) / (Exp.D{trial}.dx*Exp.S.pixPerDeg);
% y = (eyepos(:,2)-Exp.D{trial}.c(2)) / (Exp.D{trial}.dy*Exp.S.pixPerDeg);
% 
% figure(1); clf
% plot(x, y)


%%
% 


% Saccade processing:
% Perform basic processing of eye movements and saccades
Exp = saccadeflag.run_saccade_detection_cloherty(Exp, ...
    'ShowTrials', false,...
    'accthresh', 2e4,...
    'velthresh', 10,...
    'velpeak', 10,...
    'isi', 0.02);

% track invalid sampls
Exp.vpx.Labels(isnan(Exp.vpx.raw(:,2))) = 4;

% 
% x = cell2mat(trialnum);
% y = cell2mat(matches);
% x = [x ones(numel(x), 1)];
% w = (x'*x)\(x'*y);
% hold on
% plot(xlim, (xlim*w(1) + w(2)))

%% get spatial map

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

isrunning = treadSpeed > runThresh;
onsets = find(diff(isrunning) == 1);
offsets = find(diff(isrunning) == -1);

if onsets(1) > offsets(1)
    onsets = [1; onsets];
end

if offsets(end) < onsets(end)
    offsets = [offsets; numel(treadSpeed)];
end

assert(numel(onsets)==numel(offsets), "onset offset mismatch")

subplot(3,1,2) % plot spike count
imagesc(treadTime, 1:NC, spikeRate'); hold on
for i = 1:numel(onsets)
    fill(treadTime([onsets(i) onsets(i) offsets(i) offsets(i)]), [ylim fliplr(ylim)], 'r', 'FaceColor', 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
end
xlim(treadTime([1 end]))
colormap(1-gray)
title('Spikes')
ylabel('Unit #')   

% PLOT TREADMILL RUNNING SPEED
subplot(3,1,3) % tread speed
plot(treadTime, treadSpeed , 'k'); hold on
xlabel('Time (s)')
ylabel('Speed (cm/s)')

for i = 1:numel(onsets)
    fill(treadTime([onsets(i) onsets(i) offsets(i) offsets(i)]), [ylim fliplr(ylim)], 'r', 'FaceColor', 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
end
xlim(treadTime([1 end]))

title('Treadmill')
trmax = prctile(D.treadSpeed(treadIx), 99);

plot.fixfigure(gcf, 12, [8 11])
saveas(gcf, fullfile(figDir, 'grating_sess_birdseye.pdf'))

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
    runSpd = runSpd(goodIx,:);
    
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
Dstat = decode_stim(D, 1, 'figDir', figDir);

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
