

DataFolder = uigetdir();

%% Baic marmoView import. Synchronize with Ephys if it exists
Exp = io.basic_marmoview_import(DataFolder);

%% Import eye position signals
Exp = io.import_eye_position(Exp, DataFolder);

% upsample eye traces to 1kHz
new_timestamps = Exp.vpx.raw(1,1):1e-3:Exp.vpx.raw(end,1);
new_EyeX = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,2), new_timestamps);
new_EyeY = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,3), new_timestamps);
new_Pupil = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,4), new_timestamps);
bad = interp1(Exp.vpx.raw(:,1), double(Exp.vpx.raw(:,2)>31e3), new_timestamps);
Exp.vpx.raw = [new_timestamps(:) new_EyeX(:) new_EyeY(:) new_Pupil(:)];

Exp.vpx.raw(bad>0,2:end) = nan; % nan out bad sample times


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
Exp.vpx2ephys = Exp.ptb2Ephys;

%%
stimSet = 'DriftingGrating';
% stimSet = 'Gabor';
validTrials = io.getValidTrials(Exp, stimSet);

thisTrial = validTrials(2);
%% reconstruct stimulus
rect = [-250 -250 250 250];
rect2 = [-20 -10 50 60];

[Stim, frameInfo] = regenerateStimulus(Exp, thisTrial, rect, 'spatialBinSize', 1, ...
    'Latency', 0, 'includeProbe', false, 'GazeContingent', false, 'ExclusionRadius', inf,...
    'frameIndex', 1:355, 'usePTBdraw', true);

%% FFT of stimulus
pixperdeg = Exp.S.pixPerDeg;
ny = size(Stim,1);
nx = size(Stim,2);
nt = 100;

% vertical = Exp.D{thisTrial}.PR.NoiseHistory(:,2)==90 | Exp.D{thisTrial}.PR.NoiseHistory(:,2)==270;
% 
% iFrame = find(vertical, 1, 'first')+nt;

iFrame = nt + 100;
frNum = find(Exp.D{thisTrial}.PR.NoiseHistory(:,1) - frameInfo.frameTimesPtb(iFrame) > 0, 1);
Exp.D{thisTrial}.PR.NoiseHistory(frNum,[2:3 5:end])

figure(1); clf
subplot(1,2,1)
imagesc((1:ny)/pixperdeg, (1:nx)/pixperdeg, Stim(:,:,iFrame)); colormap gray

subplot(1,2,2)
n = 2^nextpow2(nx);
F = fftshift(fft2(Stim(:,:,iFrame)));



x = -floor(nx/2):floor((nx-1)/2);
x = x/-x(1)*pixperdeg/2;

y = -floor(ny/2):floor((ny-1)/2);
y = y/-y(1)*pixperdeg/2;

imagesc(x, y, abs(F))

figure(2); clf

F = abs(fftshift(fftn(Stim(:,:,iFrame-nt:iFrame))));

t = -floor(nt/2):floor((nt-1)/2);
t = t + .5;
t = t/-t(1)*Exp.S.frameRate/2;

imagesc(t, x,squeeze(sum(F,1))); axis xy
hold on
plot(x, Exp.D{thisTrial}.PR.NoiseHistory(frNum,6)*x, 'r')
colormap gray