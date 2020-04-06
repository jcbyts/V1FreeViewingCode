function [stimX, Robs, opts] = preprocess_spatialmapping_data(Exp, varargin)
% Preprocess spatial mapping data for analysis (and export to Python)
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
%   stim [NT x NDIMS]
%   Robs [NT x NCells]
%   opts [struct]
%       .frameTimes
%       .xpos
%       .ypos
%       .eyeAtFrame
%
% 
% 2020 jly  wrote it

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
validTrials = io.getValidTrials(Exp, 'Dots');
dotSize = cellfun(@(x) x.P.dotSize, Exp.D(validTrials));
validTrials = validTrials(dotSize==max(dotSize));

numValidTrials = numel(validTrials);
if verbose
    fprintf('Found %d valid trials\n', numValidTrials)
end

% -------------------------------------------------------------------------
% Loop over trials and bin up stimulus after offsetting the eye position

debug   = ip.Results.debug;
ROI     = ip.Results.ROI;
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

numDots = cellfun(@(x) size(x.PR.NoiseHistory,2), Exp.D(validTrials));
numDots = (unique(numDots)-1)/2;
assert(numel(numDots)==1, 'preprocess_spatialmapping_data: multiple numbers of dots. this is currently unsupported.')
% Note: to support multiple numbers of dots, initialize by Nans and index in on each trial by the number of dots. doable, but I'm not implementing it now. -- Jake

% save out the x and y position for each frame
xPosition = zeros(numFrames,numDots);
yPosition = zeros(numFrames,numDots);
eyePosAtFrame = zeros(numFrames, 2);

if verbose
    fprintf('Binning stimulus around the eye position\n')
end

% Note: this is written as a slow for-loop to be extra clear what is going on.
% It could be massively optimized, but not going to do that unless it's necessary. This
% function should only be called once per dataset.
for iTrial = 1:numValidTrials
    
    currFrame = framesPerTrial(iTrial);
    if verbose
        fprintf('Processing %d/%d Trials\n', iTrial, numValidTrials);
    end
    
    thisTrial = validTrials(iTrial);
    
    NoiseHistory = Exp.D{thisTrial}.PR.NoiseHistory;
    
    numDots = Exp.D{thisTrial}.PR.noiseNum;
    nFramesTrial = size(NoiseHistory,1);
    
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
        xPosition(iFrame + currFrame,:) = xdots(:);
        yPosition(iFrame + currFrame,:) = ydots(:);
        eyePosAtFrame(iFrame + currFrame,:) = [eyeX eyeY];

    end

end

% -------------------------------------------------------------------------
% --- Bin Spike Times
if verbose
    fprintf('Binning spikes at frame rate\n')
end
% This is the fastest possible way to bin in matlab

Robs = binNeuronSpikeTimesFast(Exp.osp, frameTimes, spikeBinSize);
Robs = Robs(:,Exp.osp.cids);

CN = size(Robs,2);

opts.frameTimes = frameTimes;
opts.xax = xax;
opts.yax = yax;
opts.dims = [numel(yax) numel(xax)];
opts.xPosition = xPosition;
opts.yPosition = yPosition;
opts.eyePosAtFrame = eyePosAtFrame;
opts.validFrames = validFrames;
opts.numDots = numDots;
opts.dotSize = max(dotSize);