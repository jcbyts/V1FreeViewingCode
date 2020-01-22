
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);


%% load data
sessId = 12;
[Exp, S] = io.dataFactory(sessId);


%% check that the session looks reasonable
% sessionQAfigures(Exp, S)
% eyePos1 = io.getCorrectedEyePosFixCalib(Exp, 'plot', true, 'usePolynomial', true);
% eyePos2 = io.getCorrectedEyePosFixCalib(Exp, 'plot', true);

% %%
% figure(1); clf
% 
% plot(eyePos1(:,1), eyePos2(:,1), '.'); hold on
% xlabel('Old Calibration')
% ylabel('New Calibration')
% plot(xlim, [0 0])
% plot([0 0], ylim)
%% try correcting the calibration using FixCalib protocol
eyePos = io.getCorrectedEyePosFixCalib(Exp, 'plot', true);

% smooth eye position with 3rd order sgolay filter, preserves tremor
eyePos(:,1) = sgolayfilt(eyePos(:,1), 3, 9);
eyePos(:,2) = sgolayfilt(eyePos(:,2), 3, 9);

Exp.vpx.smo(:,2:3) = eyePos;

%% use correction grid
if exist('Data/Marmo20191231_eyecor1.mat', 'file') && sessId ==12
    ec = load('Data/Marmo20191231_eyecor1.mat');
    ppd = Exp.S.pixPerDeg;
    GridCentersX = ec.corr_list(:,1) / ppd;
    GridCentersY = ec.corr_list(:,2) / ppd;
    CorrectionX = GridCentersX + ec.corr_list(:,3) / ppd;
    CorrectionY = GridCentersY + ec.corr_list(:,4) / ppd;
    
    % learn interpolatant function
    Fx = scatteredInterpolant(CorrectionX, CorrectionY, GridCentersX);
    Fy = scatteredInterpolant(CorrectionX, CorrectionY, GridCentersY);
    
    % only fix the central points
    iix = hypot(eyePos(:,1), eyePos(:,2)) < 5*ppd;
    
    eyePos2 = eyePos;
    
    eyePos2(iix, 1) = Fx(eyePos(iix,1), eyePos(iix,2));
    eyePos2(iix, 2) = Fy(eyePos(iix,1), eyePos(iix,2));
    
    Exp.vpx.smo(:,2:3) = eyePos2;
end
%% regenerate data with the following parameters
eyesmoothing = 9;
t_downsample = 2;
s_downsample = 1;

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
stim = 'FixRsvpStim';
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



