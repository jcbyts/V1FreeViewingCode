function D = get_drifting_grating_output(Exp)
% Output for GLM analyses from the drifting grating protocol

%% Grating tuning

if isfield(Exp, 'GratingOnsets')
    D = Exp;
else
    validTrials = io.getValidTrials(Exp, 'DriftingGrating');
    
    tonsets = [];
    toffsets = [];
    directions = [];
    speeds = [];
    freq = [];
    treadTime = [];
    treadSpeed = [];
    % figure(1); clf
    
    for iTrial = validTrials(:)'
        pos = Exp.D{iTrial}.treadmill.locationSpace(:,4); % convert to meters
        ttime = Exp.D{iTrial}.treadmill.locationSpace(:,1);
        
        dpos = diff(imgaussfilt(pos, 5)) ./ diff(ttime);
        spd = [0; abs(dpos(:))];
        
        
        bad = isnan(Exp.D{iTrial}.treadmill.locationSpace(:,2));
        bad = imboxfilt(double(bad), 101)>0;
        spd(bad) = nan;
        
        ttime(bad) = nan;
        
        ylim([0 5])
        
        treadTime = [treadTime; ttime];
        treadSpeed = [treadSpeed; spd];
        
        % Fields:
        % time, orientation, cpd, phase, direction, speed, contrast
        contrast = Exp.D{iTrial}.PR.NoiseHistory(:,7);
        
        onsets = find(diff(contrast)>0)+1;
        offsets = find(diff(contrast)<0)+1;
        
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
        freq = [freq; Exp.D{iTrial}.PR.NoiseHistory(onsets,3)];
        
    end
    
    tonsets = Exp.ptb2Ephys(tonsets);
    toffsets = Exp.ptb2Ephys(toffsets);
    treadTime = Exp.ptb2Ephys(treadTime);
    
    % --- spike times / saccade times / eye pos
    st = Exp.spikeTimes;
    clu = Exp.spikeIds;
    eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    eyePos = Exp.vpx.smo(:,2:3);
    
    % build struct
    D = struct();
    D.spikeTimes = st;
    D.spikeIds = clu;
    D.treadTime = treadTime;
    D.treadSpeed = treadSpeed;
    D.GratingOnsets = tonsets;
    D.GratingOffsets = toffsets;
    D.GratingDirections = directions;
    D.GratingFrequency = freq;
    D.GratingSpeeds = speeds;
    D.eyeTime = eyeTime;
    D.eyePos = eyePos;
    D.eyeLabels = Exp.vpx.Labels;
    ix = hypot(D.eyePos(:,1), D.eyePos(:,2)) > 20;
    D.eyePos(ix,:) = nan;
    
end

% --- cut out breaks in data (from other trial types)
breaks = diff(D.GratingOnsets);
breakStart = D.GratingOnsets(breaks>10) + 1;
breakStop = D.GratingOnsets(find(breaks>10)+1) - 1;

plotGratingData(D)
t = breakStart;
plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'k', 'Linewidth', 2)
t = breakStop;
plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'k--', 'Linewidth', 2)
drawnow

% remove spike times during brea
lastOffset = D.GratingOffsets(end) + 2;
firstOnset = D.GratingOnsets(1) - 2;
ix = getTimeIdx(D.spikeTimes, breakStart, breakStop) | D.spikeTimes > lastOffset | D.spikeTimes < firstOnset;
fprintf("Removing %d spikes that occured outside the stimulus trials\n", sum(ix));
D.spikeTimes(ix) = [];
D.spikeIds(ix) = [];

ix = getTimeIdx(D.eyeTime, breakStart, breakStop) | D.eyeTime > lastOffset | D.eyeTime < firstOnset;
fprintf("Removing %d eye pos samples that occured outside the stimulus trials\n", sum(ix));
D.eyeTime(ix) = [];
D.eyePos(ix,:) = [];
D.eyeLabels(ix) = [];

ix = getTimeIdx(D.treadTime, breakStart, breakStop) | D.treadTime > lastOffset | D.treadTime < firstOnset;
fprintf("Removing %d eye pos samples that occured outside the stimulus trials\n", sum(ix));
D.treadTime(ix) = [];
D.treadSpeed(ix) = [];

plotGratingData(D)
t = breakStart;
plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'k', 'Linewidth', 2)
t = breakStop;
plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'k--', 'Linewidth', 2)
drawnow

nBreaks = numel(breakStop);
fprintf('Adjusting times to remove gaps\n')
for iBreak = 1:nBreaks
    
    breakDur = breakStop(iBreak) - breakStart(iBreak);
    
    plotGratingData(D)
    plot(breakStart(iBreak)*[1 1], ylim, 'b', 'Linewidth', 2)
    plot(breakStop(iBreak)*[1 1], ylim, 'b', 'Linewidth', 2)
    drawnow
    
    % adjust all subsequent times
    D.spikeTimes(D.spikeTimes > breakStop(iBreak)) = D.spikeTimes(D.spikeTimes > breakStop(iBreak)) - breakDur;
    D.GratingOnsets(D.GratingOnsets > breakStop(iBreak)) = D.GratingOnsets(D.GratingOnsets > breakStop(iBreak)) - breakDur;
    D.GratingOffsets(D.GratingOffsets > breakStop(iBreak)) = D.GratingOffsets(D.GratingOffsets > breakStop(iBreak)) - breakDur;
    D.eyeTime(D.eyeTime > breakStop(iBreak)) = D.eyeTime(D.eyeTime > breakStop(iBreak)) - breakDur;
    D.treadTime(D.treadTime > breakStop(iBreak)) = D.treadTime(D.treadTime > breakStop(iBreak)) - breakDur;
    
    if iBreak < nBreaks
        breakStop((iBreak + 1):end) = breakStop((iBreak + 1):end) - breakDur;
        breakStart((iBreak + 1):end) = breakStart((iBreak + 1):end) - breakDur;
    end
end

function plotGratingData(D)
    figure(111); clf
    subplot(3,1,1)
    plot.raster(D.spikeTimes, D.spikeIds, 1); hold on
    t = D.GratingOnsets;
    plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'r', 'Linewidth', 2); hold on
    t = D.GratingOffsets;
    plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'g', 'Linewidth', 2)
    title('Spikes + Gratings')
    
    subplot(3,1,2)
    plot(D.eyeTime, D.eyePos(:,1), 'k')
    hold on
    plot(D.eyeTime, D.eyePos(:,2), 'Color', .5*[1 1 1])
    plot(D.eyeTime(D.eyeLabels==2), D.eyePos(D.eyeLabels==2, 1), 'r.')
    title('Eye Position')
    
    subplot(3,1,3)
    plot(D.treadTime, D.treadSpeed, 'k', 'Linewidth', 2); hold on
    title('Treadmill')




