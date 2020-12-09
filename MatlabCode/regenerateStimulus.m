function [Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect, varargin)
% REGENERATE STIMULUS reconstructs the stimulus within a specified window
% on the screen. Can be gaze contingent or not
%
% Inputs:
%   Exp [struct]:           marmoview datastruct
%   validTrials [T x 1]:    list of trials to analyze
%   rect [1 x 4]:           rectangle (with respect to center of screen or
%                           eye position)
% 
% Output:
%   Stim [x,y,frames]
%   frameInfo [struct]:
%       'dims'
%       'rect'
%       'frameTimesPtb' frame time in Ptb clock
%       'frameTimesOe'  frame time in OE clock
%       'eyeAtFrame'   [eyeTimeOE eyeX eyeY eyeIx]
%       'probeDistance' probe distance from gaze
%       'probeAtFrame' [X Y id]
%       'seedGood'
% 
% Optional Arguments:
%   spatialBinSize, 1
%   GazeContingent, true
%   ExclusionRadius, 500
%   Latency, 0
%   EyePos, []
%   includeProbe
% [Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect)

ip = inputParser();
ip.addParameter('spatialBinSize', 1)
ip.addParameter('GazeContingent', true)
ip.addParameter('SaveVideo', [])
ip.addParameter('ExclusionRadius', 500)
ip.addParameter('Latency', 0)
ip.addParameter('EyePos', [])
ip.addParameter('includeProbe', true)
ip.addParameter('debug', false)
ip.addParameter('frameIndex', [])
ip.addParameter('usePTBdraw', false)
ip.parse(varargin{:});

spatialBinSize = ip.Results.spatialBinSize;

% get the dimensions of your image sequence
dims = ((rect([4 3]) - rect([2 1]))/spatialBinSize);

% extract the total number of frames
framesPerTrial = cellfun(@(x) sum(~isnan(x.eyeData(6:end,6))),Exp.D(validTrials));
nTotalFrames = sum(framesPerTrial);

Stim = zeros(dims(1), dims(2), nTotalFrames, 'single');

frameInfo = struct('dims', dims, 'rect', rect, ...
    'frameTimesPtb', zeros(nTotalFrames, 1), ...
    'frameTimesOe', zeros(nTotalFrames, 1), ...
    'eyeAtFrame', zeros(nTotalFrames, 4), ... % [eyeTime eyeX eyeY eyeIx]
    'probeDistance', inf(nTotalFrames, 1), ...
    'probeAtFrame', zeros(nTotalFrames, 3), ... % [X Y id]
    'seedGood', true(nTotalFrames, 1));
    
% convert eye tracker to ephys time
eyeTimes = Exp.vpx2ephys(Exp.vpx.smo(:,1));

% get eye position
if ~isempty(ip.Results.EyePos)
    eyePos = ip.Results.EyePos;
    assert(size(eyePos,1)==numel(eyeTimes), 'regenerateStimulus: eyePos input does not match number of eye frames in datafile')
else
    eyePos = Exp.vpx.smo(:,2:3); 
end

eyePos(Exp.vpx.Labels==4,:) = nan; % invalid eye samples are NaN will be skipped

%% loop over trials and regenerate stimuli
nTrials = numel(validTrials);

blocks = 1;

frameCounter = 1;

for iTrial = 1:nTrials
    fprintf('%d/%d trials\n', iTrial, nTrials)
    thisTrial = validTrials(iTrial);

    % extract the frame refresh times from marmoview FrameControl
    frameRefreshes = Exp.D{thisTrial}.eyeData(6:end,6);
    frameRefreshes = frameRefreshes(~isnan(frameRefreshes));

    nFrames = numel(frameRefreshes);
    

    frameRefreshesOe = Exp.ptb2Ephys(frameRefreshes(1:nFrames));
    
    % --- prepare for reconstruction based on stimulus protocol
    switch Exp.D{thisTrial}.PR.name
        case 'ForageProceduralNoise'
            noiseFrames = Exp.D{thisTrial}.PR.NoiseHistory(:,1);
            
            nFrames = min(numel(noiseFrames), nFrames);
            
            % this is our noise object
            if ~isfield(Exp.D{thisTrial}.PR, 'hNoise')
                continue
            end
            
            hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
            hNoise.rng.reset(); % reset the random seed to the start of the trial
            hNoise.frameUpdate = 0; % reset the frame counter
            if isprop(hNoise, 'screenRect')
                hNoise.screenRect = Exp.S.screenRect;
            end
            
            useNoiseObject = true;
            
            % get the probe location on each frame
            probeX = Exp.S.centerPix(1) + round(Exp.D{thisTrial}.PR.ProbeHistory(:,1)*Exp.S.pixPerDeg);
            probeY = Exp.S.centerPix(2) + round(Exp.D{thisTrial}.PR.ProbeHistory(:,2)*Exp.S.pixPerDeg);
            probeId = Exp.D{thisTrial}.PR.ProbeHistory(:,3);
            
            probeX = repnan(probeX, 'previous');
            probeY = repnan(probeY, 'previous');
            
            if ip.Results.includeProbe
                % setup probes
                [Probe, Faces] = protocols.PR_ForageProceduralNoise.regenerateProbes(Exp.D{thisTrial}.P,Exp.S);
                for i = 1:numel(Probe)
                   Probe{i}.screenRect = Exp.S.screenRect; 
                end
            end
            
        case 'BackImage'
            useNoiseObject = false;
            useBackImage = true;
            try
                Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));
            catch
                fprintf(1, 'regenerateStimulus: failed to load image [%s]\n', Exp.D{thisTrial}.PR.imagefile)
                continue
            end
            
            % zero mean
            Im = mean(Im,3)-127;
            Im = imresize(Im, fliplr(Exp.S.screenRect(3:4)));
            
            % no probe
            probeX = nan(nFrames,1);
            probeY = nan(nFrames,1);
            probeId = nan(nFrames,1);
            
            if ip.Results.debug
                figure(999); clf
                ax = subplot(5,5,setdiff(1:25, [5 10 15 20 25]));
                imagesc(Im); hold on
                eyeIx = eyeTimes >= frameRefreshesOe(1) & eyeTimes < frameRefreshesOe(end);
                plot(eyePos(eyeIx,1)*Exp.S.pixPerDeg + Exp.S.centerPix(1), Exp.S.centerPix(2) - eyePos(eyeIx,2)*Exp.S.pixPerDeg, 'r'); axis ij
                colormap gray
                skip = input('skip?');
                if skip
                    continue
                end
            end
            
        case 'FixRsvpStim'
            useNoiseObject = false;
            useBackImage = false;
            
            hObj = stimuli.gaussimages(0, 'bkgd', Exp.S.bgColour, 'gray', false);
            hObj.loadimages('rsvpFixStim.mat');
            hObj.position = [0,0]*Exp.S.pixPerDeg + Exp.S.centerPix;
            hObj.radius = round(Exp.D{thisTrial}.P.faceRadius*Exp.S.pixPerDeg);
            
            noiseFrames = Exp.D{thisTrial}.PR.NoiseHistory(:,1);
            
            nFrames = min(numel(noiseFrames), nFrames);
            probeX = nan(nFrames,1);
            probeY = nan(nFrames,1);
            probeId = nan(nFrames,1);
            
        otherwise
            
            fprintf('regenerateStimulus: [%s] is an unrecognized protocol. Skipping trial %d\n', Exp.D{thisTrial}.PR.name, iTrial)
            continue
    end
    
    
    if isempty(ip.Results.frameIndex)
        frameIndx = 1:nFrames;
    else
        frameIndx = ip.Results.frameIndex(:)';
    end
    
    blockStart = frameCounter;
    
    % --- loop over frames and get noise from that frame
    for iFrame = frameIndx
        
        % find the index into eye position that corresponds to this frame
        eyeIx = find(eyeTimes >= frameRefreshesOe(iFrame) + ip.Results.Latency,1);
        
        % eye position is invalid, skip frame
        if isempty(eyeIx)
            disp('Skipping because of eyeTime')
            blocks = [blocks; frameCounter];
            continue
        end
        
        % eye position in pixels
        eyeX = eyePos(eyeIx,1) * Exp.S.pixPerDeg;
        eyeY = eyePos(eyeIx,2) * Exp.S.pixPerDeg;
        
        % exclude eye positions that are off the screen
        if hypot(eyeX, eyeY) > ip.Results.ExclusionRadius
            blocks = [blocks; frameCounter];
            continue
        end
        
        % offset for center of the screen
        eyeX = Exp.S.centerPix(1) + eyeX;
        eyeY = Exp.S.centerPix(2) - eyeY;
        
        % skip frame if eye position is invalid
        if isnan(eyeX) || isnan(eyeY)
            blocks = [blocks; frameCounter];
            continue
        end
        
        if ip.Results.GazeContingent
            % center on eye position
            tmprect = rect + [eyeX eyeY eyeX eyeY];
        else
            % center on screen
            tmprect = rect + Exp.S.centerPix([1 2 1 2]);
        end
        
        if useNoiseObject
            
            if (frameRefreshes(iFrame)~=noiseFrames(iFrame)) && (iFrame ~=nFrames)
                fprintf('regenerateStimulus: noiseHistory doesn''t equal the frame refresh time on frame %d\n', iFrame)
                blocks = [blocks; frameCounter];
                continue
            end
            
            seedGood = false;
            ctr = 0; % loop counter
            while ~seedGood % try frames until the seeds match
                hNoise.afterFrame(); % regenerate noise stimulus
                switch Exp.D{thisTrial}.PR.noisetype
                    case 1 % FF grating
                        seedGood = all([hNoise.orientation hNoise.cpd] == Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2:end));
                    case 4 % Gabors
                        seedGood = all([hNoise.x(1) hNoise.mypars(2)] == Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2:end));
                    case 5 % dots
                        noiseNum = Exp.D{thisTrial}.PR.noiseNum;
                        seedGood = all([hNoise.x(1:noiseNum) hNoise.y(1:noiseNum)] == Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2:end));
                end
                if ctr > iFrame + 6
                    warning('regenerateStimulus: seed is off')
                    hNoise.rng.reset(); % reset the random seed to the start of the trial
                    hNoise.frameUpdate = 0; % reset the frame counter
                    continue
                end
                ctr = ctr + 1;
            end
            
            % get image from noise object
            I = hNoise.getImage(tmprect, spatialBinSize);
%             figure(1); clf
%             imagesc(I);
%             drawnow
%             keyboard
            
        elseif useBackImage
            imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
            I = imcrop(Im, imrect); % requires the imaging processing toolbox
            I = I(1:spatialBinSize:end,1:spatialBinSize:end);
            
            if ip.Results.debug
                set(gcf, 'currentaxes', ax)
                plot(eyeX, eyeY, 'or')
                plot([imrect(1) imrect(1) + imrect(3)], imrect([2 2]), 'r')
                plot([imrect(1) imrect(1) + imrect(3)], imrect(2)+imrect([4 4]), 'r')
                plot(imrect([1 1]),[imrect(2), imrect(2) + imrect(4)], 'r')
                plot(imrect(1)+imrect([3 3]), [imrect(2), imrect(2) + imrect(4)], 'r')
                subplot(5,5,5)
                imagesc(I)
                drawnow
            end
        
        else % fix rsvp stim
%             posx = Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2);
%             posy = Exp.D{thisTrial}.PR.NoiseHistory(iFrame,3);
            hObj.imagenum = Exp.D{thisTrial}.PR.NoiseHistory(iFrame,4);
            hObj.position = Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2:3);
            I = hObj.getImage(tmprect, spatialBinSize)-Exp.S.bgColour;
            if ip.Results.debug
                figure(1); clf
                imagesc(I);
                pause
            end
            
        end
    
        % save info
        frameInfo.frameTimesPtb(frameCounter) = frameRefreshes(iFrame) + ip.Results.Latency;
        frameInfo.frameTimesOe(frameCounter)  = frameRefreshesOe(iFrame) + ip.Results.Latency;
        frameInfo.eyeAtFrame(frameCounter,:)    = [eyeTimes(eyeIx) eyeX eyeY eyeIx];
        
        % probe distance from center of window
        frameInfo.probeDistance(frameCounter) = hypot(probeX(iFrame)-mean(tmprect([1 3])), probeY(iFrame)-mean(tmprect([2 4])));
        
        % --- handle the probe object
        frameInfo.probeAtFrame(frameCounter,:) = [probeX(iFrame) probeY(iFrame) probeId(iFrame)];
        
        if useNoiseObject && ip.Results.includeProbe % probes can exist on noise objects. for now. Might add them for BackImage soon
            
            probeInWin = (probeX(iFrame) > (tmprect(1)-Probe{1}.radius)) & (probeX(iFrame) < (tmprect(3) + Probe{1}.radius));
            probeInWin = probeInWin & ((probeY(iFrame) > (tmprect(2)-Probe{1}.radius)) & (probeY(iFrame) < (tmprect(4) + Probe{1}.radius)));
        
            if ~isnan(probeId(iFrame)) && probeInWin
            
                
                if probeId(iFrame) > 0 % grating
                    Probe{probeId(iFrame)}.position = [probeX(iFrame) probeY(iFrame)];
                    [pIm, pAlph] = Probe{probeId(iFrame)}.getImage(tmprect, spatialBinSize);
                elseif probeId(iFrame) < 0
                    Faces.imagenum = abs(probeId(iFrame));
                    Faces.position = [probeX(iFrame) probeY(iFrame)];
                    [pIm, pAlph] = Faces.getImage(tmprect, spatialBinSize);
                    pIm = pIm - 127;
                end
                
                % blend I
                try
                    I = pIm + (1-pAlph).*I;
                catch
                    disp('probe not combined')
                end
            end
            
            
        end
        
        
        % check that the seed worked
        if useNoiseObject
            switch Exp.D{thisTrial}.PR.noisetype
                case 4
                    frameInfo.seedGood(frameCounter) = all([hNoise.x(1) hNoise.mypars(2)] == Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2:end));
                case 5
                    noiseNum = Exp.D{thisTrial}.PR.noiseNum;
                    frameInfo.seedGood(frameCounter) = all([hNoise.x(1:noiseNum) hNoise.y(1:noiseNum)] == Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2:end));
            end
        else
            frameInfo.seedGood(frameCounter) = true; % seed doesn't exist. It's an image
        end
        
        try
            Stim(:,:, frameCounter) = I;
        catch
            warning('frame not saved')
            continue
        end
        
        frameCounter = frameCounter + 1;
        
        
    end
    
end

% clean up blocks
% blocks(diff(blocks, [], 2)<50,:) = [];
bd = blocks(diff(blocks)>1);
blocks = [bd(1:end-1)+1 bd(2:end)];
blocks(1) = 1;

Stim(:,:,frameCounter:end) = [];
frameInfo.eyeAtFrame(frameCounter:end,:) = [];
frameInfo.frameTimesOe(frameCounter:end) = [];
frameInfo.frameTimesPtb(frameCounter:end) = [];
frameInfo.probeDistance(frameCounter:end) = [];
frameInfo.seedGood(frameCounter:end) = [];
frameInfo.probeAtFrame(frameCounter:end,:) = [];
frameInfo.blocks = blocks;




