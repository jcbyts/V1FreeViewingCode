
%% Load Data
eyesmooth = 9;

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
% rect = [-50 -60 50 50];
rect = [-20 -60 40 10];
if strcmp(stimulusSet, 'BackImage')
    cloudTrials = validTrials(cellfun(@(x) ~isempty(strfind(x.PR.imagefile, 'cloud')),Exp.D(validTrials)));
    validTrials = [cloudTrials; setdiff(validTrials, cloudTrials)];
    validTrials = sort(validTrials);
end

% validTrials = validTrials(1:40);
[Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect, 'spatialBinSize', binSize, 'Latency', Latency, 'eyePos', eyePos, 'includeProbe', true, 'debug', false);

%%
NT = size(Stim,3);
figure(1); clf

for iFrame = 1:NT
    if iFrame < 200
        imagesc(Stim(:,:,iFrame)); drawnow
    end
    Stim(:,:,iFrame) = Stim(:,:,iFrame) - imgaussfilt(Stim(:,:,iFrame),3);
end

stim = reshape(Stim, prod(frameInfo.dims), NT)';
% 
% [~, inds] = sort(frameInfo.frameTimesOe);
% stim = stim(inds,:);
% frameInfo.eyeAtFrame = frameInfo.eyeAtFrame(inds,:);
% frameInfo.frameTimesOe = frameInfo.frameTimesOe(inds);
%% bin spikes
Robs = binNeuronSpikeTimesFast(Exp.osp, frameInfo.frameTimesOe, 1/Exp.S.frameRate);

Robs(:,sum(Robs)==0)= [];

stim = stim ./ std(stim(:));
%%
labels = Exp.vpx.Labels(frameInfo.eyeAtFrame(:,4));
eyeAtFrame = frameInfo.eyeAtFrame(:,2:3);

% only take central eye positions
ecc = hypot(eyeAtFrame(:,1)-Exp.S.centerPix(1), eyeAtFrame(:,2)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc > 5.1 | labels ~= 1;

NC = size(Robs,2);
nlags = 20;
NT = size(stim,1);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(1); clf
stas = zeros(nlags, size(stim,2), NC);
% Xstim = makeStimRows(stim, nlags);
%%
cc = 8;
rtmp = Robs(:,cc);
rtmp = sum(Robs,2);
rtmp(ix) = 0;
% stim = stim - mean(stim);
f = @(x) (x);
sta = simpleSTC(f(stim(1:end,:)), rtmp(1:end), nlags);
sta = (sta - min(sta(:))) / (max(sta(:)) - min(sta(:)));
figure(1); clf
for ilag = 1:nlags
    subplot(1, nlags, ilag, 'align')
    imagesc(reshape(sta(ilag,:), frameInfo.dims), [0 1])
end

%%

figure,
ilag = 10;
imagesc(reshape(sta(ilag,:), frameInfo.dims), [0 1])

figure, plot(rtmp)
