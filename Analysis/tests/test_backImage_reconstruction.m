
%% Load Data
eyesmooth = 3;

% Load dataset
sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'jrclustwf', 'cleanup_spikes', 0);

% Preprocess eye position
eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);

%% % Regenerate
stimulusSet = 'BackImage';

fprintf('Reconstructing [%s] stimuli...\n', stimulusSet)

validTrials = io.getValidTrials(Exp, stimulusSet);


numValidTrials = numel(validTrials);

if numValidTrials == 0
    error('No Valid Trials')
end

% smooth the eye position with an sgolay filter
eyePos = Exp.vpx.smo(:,2:3);


if eyesmooth > 1 
    
    % smoothing window must be odd
    if mod(eyesmooth-1,2)~=0
        smwin = eyesmooth - 1;
    else
        smwin = eyesmooth;
    end
    
    eyePos(:,1) = sgolayfilt(eyePos(:,1), 1, smwin);
    eyePos(:,2) = sgolayfilt(eyePos(:,2), 1, smwin);
    
end

binSize = 1; % pixel
Latency = 8e-3;
fprintf('Regenerating stimuli...\n')
rect = [0 0 50 50];
[Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect, 'spatialBinSize', binSize, 'Latency', Latency, 'eyePos', eyePos, 'includeProbe', true, 'debug', true);