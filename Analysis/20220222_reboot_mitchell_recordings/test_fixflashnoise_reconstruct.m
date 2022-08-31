

%% Get file

[fname, fpath] = uigetfile('~/Documents/MATLAB/MarmoV5/Output/*.mat');
Exp = load(fullfile(fpath, fname));


%% Preprocess to use online eye data
Exp.ptb2Ephys = @(x) x;
Exp = io.import_eye_position(Exp, '');

% eliminate double samples (this shouldn't do anything)
[~,ia] =  unique(Exp.vpx.raw(:,1));
Exp.vpx.raw = Exp.vpx.raw(ia,:);

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

%%
validTrials = io.getValidTrials(Exp, 'FixFlashGabor');
validTrials = validTrials(4);

Exp.D{validTrials}.PR.hNoise

%%
rect = [-100 -100 100 100];
[Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect, 'usePTBdraw', true);