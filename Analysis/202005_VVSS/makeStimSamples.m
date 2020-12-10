
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);


%% load data

close all
sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'jrclustwf');
% eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);

eyePosOrig = Exp.vpx.smo(:,2:3);

%% find valid trials
stimSet = 'FixRsvpStim';
validTrials = io.getValidTrials(Exp, stimSet);

[~, ind] = sort(cellfun(@(x) numel(x.rewardtimes), Exp.D(validTrials)), 'descend');
[~, ind] = sort(cellfun(@(x) size(x.PR.NoiseHistory,1), Exp.D(validTrials)), 'descend');

thisTrial = validTrials(ind(1));

%% cloud stimuli
stimSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimSet);
if strcmp(stimSet, 'BackImage')
    
    thisTrial = validTrials(cellfun(@(x) ~isempty(strfind(x.PR.imagefile, 'cloud')), Exp.D(validTrials)));
    thisTrial = thisTrial(1);
end

%% gabor noise
stimSet = 'Gabor';
validTrials = io.getValidTrials(Exp, stimSet);

thisTrial = validTrials(1);


%% nat img
stimSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimSet);
if strcmp(stimSet, 'BackImage')
    
    thisTrial = validTrials(~cellfun(@(x) ~isempty(strfind(x.PR.imagefile, 'cloud')), Exp.D(validTrials)));
    thisTrial = thisTrial(15);
end
%%

rect = [-250 -250 250 250];
rect2 = [-20 -10 50 60];

[Stim, frameInfo] = regenerateStimulus(Exp, thisTrial, rect, 'spatialBinSize', 1, ...
    'Latency', 0, 'eyePos', eyePos, 'includeProbe', true, 'GazeContingent', false, 'ExclusionRadius', 600,...
    'frameIndex', 1:355, 'usePTBdraw', true);

%%
frameInfo.eyeAtFrame(:,2:3) = frameInfo.eyeAtFrame(:,2:3)-Exp.S.centerPix;
iFrame = 1;
%%

figure(1); clf
iFrame = iFrame + 1;
if iFrame > size(Stim,3)
    iFrame = 1;
end
imagesc(Stim(:,:,iFrame), [-127 127]); colormap gray
%%
filename = strrep(Exp.FileTag, '.mat', [stimSet '_fixed_noROI.mp4']);

makeStimMovie(Stim, frameInfo, filename, true, []);

% rect2 = [-20 10 40 60];
rect2 = [-20 -10 50 60];

filename = strrep(Exp.FileTag, '.mat', [stimSet '_fixed_ROI.mp4']);

makeStimMovie(Stim, frameInfo, filename, true, rect2);

%% make gaze contingent stim
[Gstim, GframeInfo] = regenerateStimulus(Exp, thisTrial, rect2, 'spatialBinSize', 1, ...
    'Latency', 0, 'eyePos', eyePos, 'includeProbe', true, 'GazeContingent', true, 'ExclusionRadius', 600,...
    'frameIndex', 1:355,'usePTBdraw', true);

figure; clf
imagesc(Gstim(:,:,6)); colormap gray

%%

GframeInfo.eyeAtFrame(:,2:3) = GframeInfo.eyeAtFrame(:,2:3)-Exp.S.centerPix;

filename = strrep(Exp.FileTag, '.mat', [stimSet '_ROI.mp4']);

makeStimMovie(Gstim, GframeInfo, filename, false, []);
%% coarse dot mapping

% [X, Robs, opts] = io.preprocess_spatialmapping_data(Exp, 'validTrials', thisTrial, 'binSize', 10);


%%

figure(1); clf
iFrame = 1;
imagesc(opts.xax, opts.yax, reshape(X(iFrame,:), opts.dims))