function checkCalibration(Exp)
% Check the existing gaze calibration
%% organize the data
fprintf('Correcting eye pos by reanalyzing FaceCal\n')

validTrials = io.getValidTrials(Exp, 'FaceCal');

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
validIx = getTimeIdx(eyeTime, tstart, tstop);

xy = Exp.vpx.smo(validIx,2:3);
spd = Exp.vpx.smo(validIx,7);

trialTargets = cellfun(@(x) x.PR.faceconfig(:,1:2), Exp.D(validTrials), 'uni', 0);
targets = unique(cell2mat(trialTargets),'rows');
ntargs = size(targets,1);

n = sum(validIx);
ix = true(n,1);
ix = all(abs(zscore(xy(ix,:)))<1,2); % remove outliers
ix = ix & ( spd / median(spd) < 2); % find fixations

%% check original
[~, id] = calibration_loss([1 1 0 0 0], xy(ix,:), targets);

figure; clf
set(gcf, 'Color', 'w')
cmap = jet(ntargs);
inds = find(ix);
for j = 1:ntargs
    plot(xy(inds(id==j),1), xy(inds(id==j),2), '.', 'Color', cmap(j,:)); hold on
    plot(targets(j,1), targets(j,2), 'ok', 'MarkerFaceColor', cmap(j,:), 'Linewidth', 2); hold on    
end
xlabel('Space (deg)')
ylabel('Space (deg)')
title('Calibration')