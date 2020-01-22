
%% add paths

user = 'jakework';
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
eyePos = io.getCorrectedEyePosFixCalib(Exp, 'plot', false);
eyeTime = Exp.vpx.smo(:,1);
win = [574480 574600];
eyeTime = eyeTime - eyeTime(win(1)); 
% smooth eye position with 3rd order sgolay filter, preserves tremor
figure(1); clf
plot(eyeTime*1e3, eyePos(:,1)*60, '-o', 'MarkerSize', 2); hold on

eyePos(:,1) = sgolayfilt(eyePos(:,1), 2, 9);
eyePos(:,2) = sgolayfilt(eyePos(:,2), 2, 9);
plot(eyeTime*1e3, eyePos(:,1)*60, '-o', 'MarkerSize', 2);
xlim(win)
legend({'Raw', 'Smoothed'})
ylabel('arcmin')
xlabel('sample')
title('Tremor preserved')

Exp.vpx.smo(:,2:3) = eyePos;

%% use correction grid
if exist('Data/Marmo20191231_eyecor1.mat', 'file') && sessId ==12
    ec = load('Data/Marmo20191231_eyecor1.mat');
    ppd = Exp.S.pixPerDeg;
    GridCentersX = ec.corr_list(:,1) / ppd;
    GridCentersY = ec.corr_list(:,2) / ppd;
    CorrectionX =  GridCentersX + ec.corr_list(:,3) / ppd;
    CorrectionY =  GridCentersY + ec.corr_list(:,4) / ppd;
    
    % learn interpolatant function
%     rbfOpts = {'cubic'};
    rbfOpts = {'multiquadric', 'RBFConstant', 1, 'RBFSmooth', 10};
%     rbfOpts = {'multiquadric', 'RBFConstant', 2};
    opX = rbfcreate([CorrectionX(:)'; CorrectionY(:)'], GridCentersX(:)', 'RBFFunction', rbfOpts{:});
    rbfcheck(opX);
    opY = rbfcreate([CorrectionX(:)'; CorrectionY(:)'], GridCentersY(:)', 'RBFFunction', rbfOpts{:});
    rbfcheck(opY);

    % only fix the central points
    iix = hypot(eyePos(:,1), eyePos(:,2)) < 5*ppd;
    
    eyePos2 = eyePos;
    eyePos2(iix,1) = rbfinterp(eyePos(iix,:)', opX)';
    eyePos2(iix,2) = rbfinterp(eyePos(iix,:)', opY)';    
    
    Exp.vpx.smo(:,2:3) = eyePos2;
%     %%
%     figure(6); clf, plot(GridCentersX, GridCentersY, '.')
%     hold on
%     plot(CorrectionX, CorrectionY, 'o')
%     plot(Fx(CorrectionX, CorrectionY), Fy(CorrectionX, CorrectionY), 'o')

% %%     
% %     rbfOpts = {'cubic','RBFSmooth', 500};
%     rbfOpts = {'multiquadric', 'RBFConstant', 1, 'RBFSmooth', 10};
% %     rbfOpts = {'multiquadric', 'RBFConstant', 2000, 'RBFSmooth', 5000};
%     opX = rbfcreate([CorrectionX(:)'; CorrectionY(:)'], GridCentersX(:)', 'RBFFunction', rbfOpts{:});
%     rbfcheck(opX);
%     opY = rbfcreate([CorrectionX(:)'; CorrectionY(:)'], GridCentersY(:)', 'RBFFunction', rbfOpts{:});
%     rbfcheck(opY);
%     gxHat = rbfinterp([CorrectionX(:)'; CorrectionY(:)'], opX);
%     figure(7); clf
%     plot3(CorrectionX, CorrectionY, GridCentersX, '.'); hold on;
%     plot3(CorrectionX, CorrectionY, gxHat', 'o')
%     
%     %%
%     rbfOpts = {'cubic','RBFSmooth', 10000};
%     rbfOpts = {'multiquadric', 'RBFConstant', 1, 'RBFSmooth', 10};
%     X = [GridCentersX(:)'; GridCentersY(:)'];
%     op = rbfcreate(X, CorrectionX(:)'-GridCentersX(:)','RBFFunction', rbfOpts{:}); 
%     rbfcheck(op)
%     gxHat = rbfinterp(X, op);
%     
%     figure(7); clf
%     plot3(GridCentersX, GridCentersY, CorrectionX-GridCentersX, '.'); hold on;
%     plot3(GridCentersX, GridCentersY, gxHat', 'o')
%     
%     figure(8); clf
%     subplot(1,2,1)
%     imagesc(reshape(gxHat, [25 25]), [-.1 .2])
%     subplot(1,2,2)
%     imagesc(reshape(CorrectionX(:)'-GridCentersX(:)', [25 25]),[-.1 .2])
%     %%
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



