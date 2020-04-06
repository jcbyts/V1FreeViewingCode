function [rfs, rect] = quick_sta_rfs(Exp, varargin)
% QUICK STA RFS uses the coarse dots from forage noise to reconstruct the
% spatiotemporal Receptive fields
%
% Input:
%   Exp [struct]: Marmoview experiment struct
%       osp: this function requires the original spikes from Kilosort
%   
% Optional Arguments (as argument pairs):
%   ROI [1 x 4]:        gaze-centered ROI (in pixels)
%   binSize [1 x 1]:    spatial bin size (in pixels)
%   debug [logical]:    pause on each frame to test it (default: false)
%   spikeBinSize:       size of the spike bins (in seconds)
%   latency:            photodiode measured latency of the monitor (in
%                       seconds)
%   nTimeLags           number of time lags (# of frames)
% 
% Output:
%   rfs [N x 1 struct]:  RF analysis for each neuron
%
% Example Call:
% rfs = quick_sta_rfs(Exp, 'ROI', [-200 -200 200 200], 'binSize', .5*Exp.S.pixPerDeg)
% 
% 2019 jly  wrote it

ip = inputParser();
ip.addParameter('ROI', [-500 -500 500 500]);
ip.addParameter('binSize', ceil(Exp.S.pixPerDeg))
ip.addParameter('debug', false)
ip.addParameter('spikeBinSize', 1/Exp.S.frameRate)
ip.addParameter('latency', 0)
ip.addParameter('eyePosExclusion', 400)
ip.addParameter('nTimeLags', ceil(Exp.S.frameRate*.1));
ip.addParameter('verbose', true)
ip.parse(varargin{:});

verbose = ip.Results.verbose;

% --- find valid trials
trialProtocols = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);

if verbose
    disp('Running quick RF map on trials with large dots')
end

ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
if (numel(ephysTrials)/numel(Exp.D)) < 0.6
    warning('quick_sta_rf: Something is wrong. Assuming all trials had ephys')
    ephysTrials = 1:numel(Exp.D);
end

% Forage trials
validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

% dot spatial noise trials
validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));

dotSize = cellfun(@(x) x.P.dotSize, Exp.D(validTrials));
validTrials = validTrials(dotSize==max(dotSize));

numValidTrials = numel(validTrials);
if verbose
    fprintf('Found %d valid trials\n', numValidTrials)
end

if numValidTrials < 5
    rfs = nan;
    rect = nan;
    return
end

% -------------------------------------------------------------------------
% Loop over trials and bin up stimulus after offsetting the eye position

debug = ip.Results.debug;
ROI = ip.Results.ROI;
binSize = ip.Results.binSize;
spikeBinSize = ip.Results.spikeBinSize;
nlags = ip.Results.nTimeLags;

xax = ROI(1):binSize:ROI(3);
yax = ROI(2):binSize:ROI(4);

[xx,yy] = meshgrid(xax, yax);

dims = size(xx);
nDims = prod(dims);

frameTimes = cell2mat(cellfun(@(x) Exp.ptb2Ephys(x.PR.NoiseHistory(:,1)), Exp.D(validTrials), 'uni', 0));

frameTimes = frameTimes + ip.Results.latency;

% convert eye tracker to ephys time
eyeTimes = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    
    
framesPerTrial = [0; cumsum(cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials)))];
numFrames = numel(frameTimes);

stimX = zeros(numFrames, nDims);
validFrames = false(numFrames, 1);

if verbose
    fprintf('Binning stimulus around the eye position\n')
end
for iTrial = 1:numValidTrials
    
    currFrame = framesPerTrial(iTrial);
    if verbose
        fprintf('Processing %d/%d Trials\n', iTrial, numValidTrials);
    end
    
    thisTrial = validTrials(iTrial);
    
    NoiseHistory = Exp.D{thisTrial}.PR.NoiseHistory;
    
    numDots = Exp.D{thisTrial}.PR.noiseNum;
    nFramesTrial = size(NoiseHistory,1);
        
    if debug
        % this is our noise object
        hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
        hNoise.rng.reset(); % reset the random seed to the start of the trial
        hNoise.frameUpdate = 0; % reset the frame counter
        for i = 1:4
            hNoise.afterFrame();
        end
        seedGood = zeros(nFramesTrial,1);
    end

    
    
    
    for iFrame = 1:nFramesTrial
        
        eyeIx = find(eyeTimes >= frameTimes(iFrame + currFrame), 1);
        
        if isempty(eyeIx)
            if verbose==2
                fprintf('No valid eye position on frame %d\n', iFrame + currFrame)
            end
            validFrames(iFrame + currFrame)= false;
            continue
        end
        
        % eye position in pixels
%         eyeX = Exp.D{thisTrial}.eyeSmo(eyeIx,2) * Exp.S.pixPerDeg;
%         eyeY = Exp.D{thisTrial}.eyeSmo(eyeIx,3) * Exp.S.pixPerDeg;
        eyeX = Exp.vpx.smo(eyeIx,2) * Exp.S.pixPerDeg;
        eyeY = Exp.vpx.smo(eyeIx,3) * Exp.S.pixPerDeg;
        
        % exclude eye positions that are off the screen
        eyePosEcc = hypot(eyeX, eyeY);
        if eyePosEcc > ip.Results.eyePosExclusion
            if verbose==2
                fprintf('Eye position outside window on Frame %d %02.2f\n', iFrame + currFrame, eyePosEcc)
            end
            validFrames(iFrame + currFrame)= false;
            continue
        end
        
        % offset for center of the screen
        eyeX = Exp.S.centerPix(1) + eyeX;
        eyeY = Exp.S.centerPix(2) - eyeY;
        
        
        % center ROI on eye position
        xtmp = xx(:) + eyeX;
        ytmp = yy(:) + eyeY;
        
        xdots = NoiseHistory(iFrame,1 + (1:numDots)) + Exp.S.centerPix(1); %/Exp.S.pixPerDeg;
        ydots = -NoiseHistory(iFrame,1 + numDots + (1:numDots)) + Exp.S.centerPix(2); %/Exp.S.pixPerDeg;
        
        I = double(sum(hypot(xdots-xtmp , ydots-ytmp) <= binSize/1.5,2)>0);
        
        if debug
            hNoise.afterFrame();
            
            seedGood(iFrame) = all([hNoise.x hNoise.y] == NoiseHistory(iFrame,2:end));
            I2 = hNoise.getImage();
            
            figure(1); clf
            subplot(1,2,1)
            imagesc(I2); hold on
            plot(xdots, ydots, 'or')
            plot(xtmp, ytmp, 'g.')
            
            subplot(1,2,2)
            imagesc(reshape(I, dims));
            
            pause
        end
        
        stimX(iFrame + currFrame, :) = I(:);
        validFrames(iFrame + currFrame) = true;
    end

end

% -------------------------------------------------------------------------
% --- Bin Spike Times
if verbose
    fprintf('Binning spikes at frame rate\n')
end
% This is the fastest possible way to bin in matlab

Y = binNeuronSpikeTimesFast(Exp.osp, frameTimes, spikeBinSize);
Y = Y(:,Exp.osp.cids);

% % conversion from time to bins
% binfun = @(x) (x==0) + ceil(x / spikeBinSize);
% 
% % bin spike times
% bst = binfun(Exp.osp.st);
% % bin frame times
% bft = binfun(frameTimes);
% % create binned spike times
% Y = sparse(bst, double(Exp.osp.clu), ones(numel(bst), 1));
% % index in with binned frame times
% Y = Y(bft,:);

CN = size(Y,2);

% -------------------------------------------------------------------------
% --- Do STA
fprintf('Computing spike triggered average\n')

ytmp = (Y(validFrames,:));
X = sparse(stimX(validFrames,:));

rfs = repmat(struct('xax', xax, 'yax', yax, ...
    'srf', [], ...
    'trf', [], ...
    'sta', [], ...
    'separability', [], ...
    'lags', (1:nlags)*spikeBinSize), CN, 1);

stas = zeros(prod(dims), nlags, CN);

y = ytmp-mean(ytmp); % subtract mean to account for DC

for ilag = 1:nlags
    lag = ilag;
    fprintf('Lag %d/%d\n', ilag, nlags)
    
    sta = (X(1:end-lag+1,:))'*y(lag:end, :);
    stas(:,ilag,:) = sta ./ sum(X)';
    
end

for kNeuron = 1:CN
    fprintf('Unit %d/%d\n', kNeuron, CN)
%     y = ytmp(:,kNeuron)-mean(ytmp(:,kNeuron)); % subtract mean to account for DC
%     sta = simpleRevcorr(X, y, nlags);
    sta = stas(:,:,kNeuron)';

    [u,s,v] = svd(sta);
    sd = sign(sum(u(:,1)));
    rfs(kNeuron).separability = s(1) / sum(diag(s));
    rfs(kNeuron).sta = sta;
    rfs(kNeuron).trf = u(:,1)*sd;
    rfs(kNeuron).srf = reshape(v(:,1)*sd, dims);
    
end


if nargout > 1
    figure(1); clf
    CN = numel(rfs);
    sx = ceil(sqrt(CN));
    sy = round(sqrt(CN));
    
    for kNeuron = 1:CN
        subplot(sx, sy, kNeuron, 'align')
        imagesc(rfs(kNeuron).xax/Exp.S.pixPerDeg, rfs(kNeuron).yax/Exp.S.pixPerDeg, rfs(kNeuron).srf)
        drawnow
    end
    
    % Find a bounding box around RFs
    clear validrfs
    c = 1;
    sI = 0;
    for kNeuron = 1:CN
        srf = (rfs(kNeuron).srf - min(rfs(kNeuron).srf(:))) / (max(rfs(kNeuron).srf(:)) - min(rfs(kNeuron).srf(:)));
        
        % weird edge case with the first pixel. Easy to exclude
        srf(1) = .1;
        
        regions = regionprops(logical(srf > .5));
        if numel(regions)>0
            sI = sI + srf;
            validrfs(c) = regions(1);
            c = c+1;
        end
    end
    
    figure(2); clf
    imagesc(rfs(1).xax, rfs(1).yax, sI); hold on
    title('Average RF')
    bbox = mode(reshape([validrfs.BoundingBox], 4, []), 2)';
    rect = [interp1(rfs(1).xax, bbox(1)), ...
        interp1(rfs(1).yax, bbox(2)), ...
        interp1(rfs(1).xax, bbox(1)+bbox(3)), ...
        interp1(rfs(1).yax, bbox(2)+bbox(4))];
    
    rect = round(rect * 1.05);
    
    xcoords = rect([1 1 3 3 1]);
    ycoords = rect([2 4 4 2 2]);
    plot(xcoords, ycoords, 'c', 'Linewidth', 2)
    xlabel('Pixels (centered on gaze)')
    ylabel('Pixels (centered on gaze)')
    legend('Bounding box')
end

