%% Open Marmoview screen with parameters from Forage protocol
currDir = pwd;


sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'jrclustwf');

stimSet = 'Grating';
validTrials = io.getValidTrials(Exp, stimSet);

thisTrial = validTrials(1);

%%
S = Exp.S;
S.screenNumber = max(Screen('Screens'));

S.DummyScreen = true;
S.DataPixx = false;
S.DummyEye = true;
% S.arrington = false;
% S.eyelink
cd(fileparts(which('MarmoV5')))

sca % clear any open windows

P = Exp.D{thisTrial}.P;

% Open Window
S = Exp.S;
S.screenNumber = max(Screen('Screens'));
S.screenRect = S.screenRect + [1e3 0 1e3 0];
S.DummyScreen = true;
S.DataPixx = false;
S.DummyEye = true;

featureLevel = 0; % use the 0-255 range in psychtoolbox (1 = 0-1)
PsychDefaultSetup(featureLevel);

% disable ptb welcome screen
Screen('Preference','VisualDebuglevel',3);
% close any open windows
Screen('CloseAll');
% setup the image processing pipeline for ptb
PsychImaging('PrepareConfiguration');

PsychImaging('AddTask', 'General', 'UseRetinaResolution');
PsychImaging('AddTask','General','FloatingPoint32BitIfPossible', 'disableDithering',1);

% Applies a simple power-law gamma correction
PsychImaging('AddTask','FinalFormatting','DisplayColorCorrection','SimpleGamma');

% create the ptb window...
[A.window, A.screenRect] = PsychImaging('OpenWindow',0,S.bgColour,S.screenRect);


A.frameRate = FrameRate(A.window);

% set alpha blending/antialiasing etc.
Screen(A.window,'BlendFunction',GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    

winPtr = A.window;

%% outside frame loop, initialize protocol
PR = protocols.(['PR_' Exp.D{thisTrial}.PR.name])(winPtr);
PR.generate_trialsList(S,P);
PR.initFunc(S, P);

% set random seed 
rng(Exp.D{thisTrial}.P.rng_before_trial);

P = PR.next_trial(S,Exp.D{thisTrial}.P);

[FP,TS] = PR.prep_run_trial();

%
% iFrame = 1;

% x = (eyepos(1)-o.c(1)) / (o.dx*o.pixPerDeg);
% y = (eyepos(2)-o.c(2)) / (o.dy*o.pixPerDeg);

% eye position x,y needs to be in pixels

% drop = state_and_screen_update(o,currentTime,x,y) 

hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
% properties(hNoise)
hNoise.winPtr = winPtr;
hNoise.updateTextures()

%% setp 1

FCount = 6;   % flip counter, why at 5 though? 
% first frame on 6
state = Exp.D{thisTrial}.eyeData(FCount,5);
eyepos = Exp.D{thisTrial}.eyeData(FCount,2:3);
currentTime = Exp.D{thisTrial}.eyeData(FCount,1);
x = (eyepos(1) - Exp.D{thisTrial}.c(1)) / (Exp.D{thisTrial}.dx*Exp.S.pixPerDeg);
y = (eyepos(2) - Exp.D{thisTrial}.c(2)) / (Exp.D{thisTrial}.dy*Exp.S.pixPerDeg);

PR.state_and_screen_update(currentTime,x,y);

Screen('Flip', winPtr, 0);

% updateNoise(o,xx,yy,currentTime)

%% Update 
% hNoise = stimuli.gabornoise(winPtr);

hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
% properties(hNoise)
hNoise.winPtr = winPtr;
hNoise.updateTextures()
% hNoise.rng.reset(); % reset the random seed to the start of the trial
% hNoise.frameUpdate = 0; 

%%
hNoise.afterFrame()
hNoise.beforeFrame()

Screen('Flip', winPtr, 0);

%%
rect = 400+[100 100 400 400];
im1 = Screen('GetImage', winPtr, rect);
figure(2); clf
imagesc(im1)

%% show fft
ppd = Exp.S.pixPerDeg; % sampling rate (space)

figure(2); clf

I = mean(im1,3);
I = I - mean(I(:));
dims = size(I);

subplot(1,2,1)
x = (0:dims(2))/ppd;
x = x - x(end)/2;
imagesc(x, x, I, [-127 127]); hold on
plot(x, -tand(hNoise.orientation)*x, 'r')
title('Spatial Domain')


xax =  -floor((ny-1)/2):floor((ny-1)/2); %-ceil(dims(1)/2:dims/2;
xax = -xax./xax(1);
xax = xax * ppd/2;

fim = abs(fftshift(fft2(I)));
subplot(1,2,2)
imagesc(xax, xax, fim); hold on


[kY, kX] = pol2cart(hNoise.orientation/180*pi, hNoise.cpd);
plot(kX, kY, 'or')
plot([0 kX], [0 kY], 'r')
plot(xlim, [0 0], 'r')
plot([0 0], ylim, 'r')

title('Fourier Domain')


%%
sca
cd(currDir)