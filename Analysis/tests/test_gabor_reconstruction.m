%% Open Marmoview screen with parameters from Forage protocol
currDir = pwd;


sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'jrclustwf');

stimSet = 'Gabor';
validTrials = io.getValidTrials(Exp, stimSet);

thisTrial = validTrials(1);

%%
S = Exp.S;
S.screenNumber = max(Screen('Screens'));
S.screenRect = S.screenRect + [1e3 0 1e3 0];
S.DummyScreen = true;
S.DataPixx = false;
S.DummyEye = true;
% S.arrington = false;
% S.eyelink
cd(fileparts(which('MarmoV5')))

sca % clear any open windows
% [S,P] = Forage7_GarboriumNoise();

P = Exp.D{thisTrial}.P;

A = marmoview.openScreen(S, struct());
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

%%

Im = hNoise.getImage(rect);
figure(1); clf
imagesc(Im, [-127 127])
colormap gray

%%
sca