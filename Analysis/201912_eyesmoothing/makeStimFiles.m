
%% add paths

user = 'jakework';
addFreeViewingPaths(user);


%% load data
sessId = 12;
[Exp, S] = io.dataFactory(sessId);


%% check that the session looks reasonable
sessionQAfigures(Exp, S)

%% try correcting the calibration using FixCalib protocol
eyePos = io.getCorrectedEyePosFixCalib(Exp);

% smooth eye position with 3rd order sgolay filter, preserves tremor
eyePos(:,1) = sgolayfilt(eyePos(:,1), 3, 9);
eyePos(:,2) = sgolayfilt(eyePos(:,2), 3, 9);

Exp.vpx.smo(:,2:3) = eyePos;

%% regenerate data with the following parameters
eyesmoothing = 9;
t_downsample = 2;
s_downsample = 2;

% Gabor reverse correlation
stim = 'Gabor';
options = {'stimulus', stim, ...
    'testmode', false, ...
    'eyesmooth', eyesmoothing, ... % bins
    't_downsample', t_downsample, ...
    's_downsample', s_downsample, ...
    'includeProbe', true, ...
    'correctEyePos', false};

io.dataGenerate(Exp, S, options{:});

% grating reverse correlation
stim = 'Grating';
options = {'stimulus', stim, ...
    'testmode', false, ...
    'eyesmooth', eyesmoothing, ... % bins
    't_downsample', t_downsample, ...
    's_downsample', s_downsample, ...
    'includeProbe', true, ...
    'correctEyePos', false};

io.dataGenerate(Exp, S, options{:});

% Static Natural images
stim = 'BackImage';
options = {'stimulus', stim, ...
    'testmode', false, ...
    'eyesmooth', eyesmoothing, ... % bins
    't_downsample', t_downsample, ...
    's_downsample', s_downsample, ...
    'includeProbe', true, ...
    'correctEyePos', false};

io.dataGenerate(Exp, S, options{:});

% Static Natural images
stim = 'FixRsvpStim';
options = {'stimulus', stim, ...
    'testmode', false, ...
    'eyesmooth', eyesmoothing, ... % bins
    't_downsample', t_downsample, ...
    's_downsample', s_downsample, ...
    'includeProbe', true, ...
    'correctEyePos', false};

io.dataGenerate(Exp, S, options{:});

