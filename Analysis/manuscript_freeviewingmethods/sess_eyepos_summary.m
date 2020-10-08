function S = sess_eyepos_summary(Exp, varargin)
% Summarize eye position / saccade statistics for a session
% S = sess_eyepos_summary(Exp, varargin)
% Input:
%   Exp (struct): the main experiment struct
% Output:
%   S (struct): has fields for each stimulus protocol

ip = inputParser();
ip.addParameter('fid', 1);
ip.addParameter('plot', false);
ip.parse(varargin{:});

%% Organize eye position / saccade info
eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
eyeX = Exp.vpx.smo(:,2);
eyeY = Exp.vpx.smo(:,3);

sacon = Exp.vpx2ephys(Exp.slist(:,1));
sacoff = Exp.vpx2ephys(Exp.slist(:,2));
sacpeak = Exp.slist(:,6);
sacdy = eyeY(Exp.slist(:,5)) - eyeY(Exp.slist(:,4));
sacdx = eyeX(Exp.slist(:,5)) - eyeX(Exp.slist(:,4));
sacamp = hypot(sacdx, sacdy);
peakvel = Exp.vpx.smo(sacpeak,7);



%% Loop over stimulus protocols and summarize
S = struct();

S.global.sacdx = sacdx;
S.global.sacdy = sacdy;
S.global.fixdur = sacon(2:end)-sacoff(1:end-1);

S.exname = strrep(Exp.FileTag, '.mat', '');
binEdges = -20:.5:20;

stimProtocols = {'BackImage', 'Grating', 'Gabors', 'Dots', 'CSD', 'All', 'ITI'};

fid = ip.Results.fid; % in case we want to print to a file

for iStim = 1:numel(stimProtocols)
    
    stim = stimProtocols{iStim};
    fprintf(fid, 'Analyzing [%s] trials\n', stim);
    
    if strcmp(stim, 'ITI')
        trialIdx = 2:numel(Exp.D);
        S.(stim).nTrials = numel(trialIdx)-1; % number of trials
        
        % get trial start and stop
        tstarts = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(trialIdx-1)));
        tstops = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(trialIdx)));
        
    elseif strcmp(stim, 'All')
        trialIdx = 1:numel(Exp.D);
        S.(stim).nTrials = numel(trialIdx); % number of trials
        
        % get trial start and stop
        tstarts = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(trialIdx)));
        tstops = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(trialIdx)));
    else
        trialIdx = io.getValidTrials(Exp, stim);
        S.(stim).nTrials = numel(trialIdx); % number of trials
        
        % get trial start and stop
        tstarts = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(trialIdx)));
        tstops = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(trialIdx)));
    end
    
    if S.(stim).nTrials==0
        continue
    end
    
    
    
    S.(stim).totalDuration = sum(tstops-tstarts);
    
    % analysis of position
    validTimeIdx = getTimeIdx(eyeTime, tstarts, tstops);
    
    % rough counts
    S.(stim).nValidSamples = sum(Exp.vpx.Labels(validTimeIdx) <=2);
    S.(stim).nInvalidSamples = sum(Exp.vpx.Labels(validTimeIdx)>2);
    S.(stim).nTotalSamples = numel(Exp.vpx.Labels(validTimeIdx));
    S.(stim).nFixationSamples = sum(Exp.vpx.Labels(validTimeIdx)==1);
    S.(stim).nSaccadeSamples = sum(Exp.vpx.Labels(validTimeIdx)==2);
    
    % adjust index to include only valid eye samples
    validTimeIdx = find(validTimeIdx & Exp.vpx.Labels <= 2);
    
    
    
    binCenters = binEdges(1:end-1) + mean(diff(binEdges))/2;
    bw = bwlabel(Exp.vpx.Labels(validTimeIdx)==1);
    
    bws = unique(bw);
    nFixations = numel(bws);
    fixDur = zeros(nFixations,1);
    for iFix = 1:nFixations
        fixDur(iFix) = sum(iFix==bw);
    end
    
    remlabel = find(fixDur > 3e3);
    for i = 1:numel(remlabel)
        validTimeIdx(remlabel(i)==bw) = 0;
    end
    
    validTimeIdx(validTimeIdx==0) = [];
    
    cnt = histcounts2(eyeX(validTimeIdx), eyeY(validTimeIdx), binEdges, binEdges)'; % tranpose makes it X by Y
    
    
    if ip.Results.plot
        figure;
        imagesc(binCenters, binCenters, imgaussfilt(cnt, 3)); axis xy
        xlabel('Horizontal Position')
        ylabel('Vertical Position')
        title('Eye position')
    end
    
    
    S.(stim).positionBins = binCenters;
    S.(stim).positionCount = cnt;
    
    
    
    
    % center of mass
    S.(stim).CenterOfMass = [nanmean(eyeX(validTimeIdx)), nanmean(eyeY(validTimeIdx))];
    S.(stim).DistanceBins = 0:.5:20;
    dist = hypot(eyeX(validTimeIdx) - S.(stim).CenterOfMass(1), eyeY(validTimeIdx)-S.(stim).CenterOfMass(2));
    cnts = histcounts(dist, S.(stim).DistanceBins);
    
    distCtr = hypot(eyeX(validTimeIdx), eyeY(validTimeIdx));
    cntsCtr = histcounts(distCtr, S.(stim).DistanceBins);
    
    S.(stim).DistanceBins(end) = [];
    S.(stim).DistanceCountCtrMass = cnts;
    S.(stim).DistanceCountCtrScreen = cntsCtr;
    
    
    % saccade analysis
    validSacIdx = getTimeIdx(sacon, tstarts, tstops);
    
    S.(stim).nSaccades = numel(validSacIdx);
    
    bins = 0:.25:15;
    S.(stim).sacAmpBinsBig = bins(1:end-1);
    S.(stim).sacAmpCntBig = histcounts(sacamp(validSacIdx), bins);
    
    bins = 0:.1:2;
    S.(stim).sacAmpBinsMicro = bins(1:end-1);
    S.(stim).sacAmpCntMicro = histcounts(sacamp(validSacIdx), bins);
    
    binsize = 0.01;
    bins = 0:binsize:1;
    S.(stim).fixationDurationBins = bins(1:end-1);
    fixdur = sacon(2:end)-sacoff(1:end-1);
    S.(stim).fixationDurationCnt = histcounts(fixdur, bins);
    
    thresh = .1;
    S.(stim).fixationDurationMedian = median(fixdur(fixdur > thresh));
    S.(stim).fixationDurationMedianCi = bootci(500, @median, fixdur(fixdur>thresh));
    S.(stim).fixationNum = sum(fixdur>thresh);
    
    if ip.Results.plot
        figure;
        plot(S.(stim).fixationDurationBins, S.(stim).fixationDurationCnt, '-o')
        xlabel('Fixation Duration')
        ylabel('Count')
    end
    
    binsize = 0.002;
    bins = 0:binsize:.1;
    S.(stim).sacDurationBins = bins(1:end-1);
    S.(stim).sacDurationCnt = histcounts(sacoff-sacon, bins);
    
    xax = 0:.1:20;
    yax = 0:10:1000;
    [C, xax, yax] = histcounts2(sacamp(validSacIdx), peakvel(validSacIdx), xax, yax);
    
    good = ~isnan(peakvel) & (validSacIdx); %#ok<*NASGU>
    fun = @(params, x) (params(1)*x).^params(2);
    
    try
        evalc('phat = robustlsqcurvefit(fun, [50 1], sacamp(good), peakvel(good));');
    catch
        phat = nan(1,2);
    end
    
    fprintf(fid, 'Slope: %02.2f, Exponent: %02.2f\n', phat(1), phat(2));
    
    S.(stim).mainSeqXbins = xax;
    S.(stim).mainSeqYbins = yax;
    S.(stim).mainSeqCnt = C';
    S.(stim).mainSeqSlope = phat(1);
    S.(stim).mainSeqExponent = phat(2);
end