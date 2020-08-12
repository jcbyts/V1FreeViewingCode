
%% add paths

user = 'jakework';
addFreeViewingPaths(user);


%% load data
sessId = 12;
[Exp, S] = io.dataFactory(sessId);


%%

% eyePos = io.getCorrectedEyePosFixCalib(Exp, 'plot', false, 'usePolynomial', true);
% eyePos2 = io.getCorrectedEyeposRF(Exp, S, eyePos);


% eyePos3 = io.getCorrectedEyeposRF(Exp, S, eyePos2);
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
eyePos = io.getCorrectedEyePosFixCalib(Exp, 'plot', false, 'usePolynomial', false);
eyeTime = Exp.vpx.smo(:,1);
win = [574480 574600];
eyeTime = eyeTime - eyeTime(win(1)); 
% smooth eye position with 3rd order sgolay filter, preserves tremor
figure(1); clf
plot(eyeTime*1e3, eyePos(:,1)*60, '-o', 'MarkerSize', 2); hold on

eyePos(:,1) = sgolayfilt(eyePos(:,1), 2, 9);
eyePos(:,2) = sgolayfilt(eyePos(:,2), 2, 9);
plot(eyeTime*1e3, eyePos(:,1)*60, '-o', 'MarkerSize', 2);
xlim(eyeTime(win)*1e3)
legend({'Raw', 'Smoothed'})
ylabel('arcmin')
xlabel('sample')
title('Tremor preserved')

Exp.vpx.smo(:,2:3) = eyePos;

%% use correction grid
if exist('Data\Marmo20191231_eyecor1.mat', 'file') && sessId ==12
    fprintf('Correcting using correction grid\n')
    ec = load('Data\Marmo20191231_eyecor1.mat');
    ppd = Exp.S.pixPerDeg;
    GridCentersX = ec.corr_list(:,1) / ppd;
    GridCentersY = ec.corr_list(:,2) / ppd;
    CorrectionX =  ec.corr_list(:,3) / ppd;
    CorrectionY =  ec.corr_list(:,4) / ppd;
    
    X = [GridCentersX(:)'; GridCentersY(:)'];
    opX = rbfcreate(X, CorrectionX','RBFFunction', rbfOpts{:}); 
    opY = rbfcreate(X, CorrectionY','RBFFunction', rbfOpts{:}); 
    gxHat = rbfinterp(X, opX);
    gyHat = rbfinterp(X, opY);
    
    % chefk you got the fit right
    figure(1); clf
    dim = [25 25];
    clim = [-.1 .2];
    subplot(2,2,1)
    imagesc(reshape(CorrectionX, dim), clim)
    title('Measured X')
    subplot(2,2,2)
    imagesc(reshape(gxHat, dim), clim)
    title('fit X')
    subplot(2,2,3)
    imagesc(reshape(CorrectionY, dim), clim)
    title('Measured Y')
    subplot(2,2,4)
    imagesc(reshape(gyHat, dim), clim)
    title('fit Y')
    
    d = max(GridCentersX); % 5.5 degree square to include for correction
    iix = all(eyePos > -d & eyePos < d,2);
    errorX = rbfinterp(eyePos(iix,:)', opX);
    errorY = rbfinterp(eyePos(iix,:)', opY);
    
    eyePos2 = eyePos;
    eyePos2(iix,1) = eyePos2(iix,1) - errorX(:);
    eyePos2(iix,2) = eyePos2(iix,2) - errorY(:);
    
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

%%

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



