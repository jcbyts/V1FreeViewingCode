
currDir = pwd;

%% Open Marmoview screen with parameters from Forage protocol
sca % clear any open windows

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

%% Update 
P.snoisediam = inf; % diameter of noise
P.range = 127;

% create two versions of the same grating
gratpro = stimuli.grating_procedural(winPtr);
grat = stimuli.grating(winPtr);

% set paramters
cpd = 5;
ori = 90;
phi = 180;
gauss = true;

%%
P.snoisediam = 5;
% match the two gratings
grat.screenRect = S.screenRect;
gratpro.screenRect = S.screenRect;
gratpro.radius = round((P.snoisediam/2)*S.pixPerDeg);
grat.radius = round((P.snoisediam/2)*S.pixPerDeg);


gratpro.phase = phi; % black
grat.phase = phi;  % black
grat.orientation = ori; % cpd will be zero => all one color
gratpro.orientation = ori; % cpd will be zero => all one color
grat.cpd = cpd; % when cpd is zero, you get a Gauss
gratpro.cpd = cpd; % when cpd is zero, you get a Gauss

grat.range = P.range;
grat.square = false; % true;  % if you want circle
grat.gauss = gauss;
grat.bkgd = P.bkgd;
grat.transparent = 0.5;
grat.pixperdeg = S.pixPerDeg;


gratpro.range = P.range;
gratpro.square = false; % true;  % if you want circle
gratpro.gauss = gauss;
gratpro.bkgd = P.bkgd;
gratpro.transparent = .5;
gratpro.pixperdeg = S.pixPerDeg;


grat.updateTextures();
gratpro.updateTextures();

grat.position = [700 360];
gratpro.position = [700 460];
%%
grat.drawGrating()
gratpro.drawGrating();
% 
rect = CenterRectOnPointd([0 0 gratpro.pixperdeg gratpro.pixperdeg]*P.snoisediam, gratpro.position(1), gratpro.position(2));
% rect([2 4]) = rect([2 4]) + gratpro.pixperdeg*3;
% Screen('FillRect', winPtr, 0, rect);
% 
Screen('Flip', winPtr, 0);
I1 = Screen('GetImage', winPtr, rect);

%% check resolution

figure(2); clf
dims = size(I1);
x = (1:dims(2))/Exp.S.pixPerDeg;
y = (1:dims(1))/Exp.S.pixPerDeg;
imagesc(x, y, I1)

%% test gratpro fast
t0 = GetSecs;
while GetSecs < t0 +5
    if rand < .25
        Screen('FillRect', winPtr, 127);
        gratpro.orientation = rand*360;
        
    end
    
    gratpro.drawGrating();
    
    %
    rect = CenterRectOnPointd([0 0 gratpro.pixperdeg gratpro.pixperdeg]*P.snoisediam, gratpro.position(1), gratpro.position(2));
    rect([2 4]) = rect([2 4]) + gratpro.pixperdeg*3;
    Screen('FillRect', winPtr, 0, rect);
    
    Screen('Flip', winPtr, 0, 2);
    Screen('FillRect', winPtr, 127);
%     Screen('Flip', winPtr, 127);
end

%% test grat pro reconstruction
gratpro.position = [500 504];
gratpro.phase = 180;
gratpro.gauss = false;
gratpro.orientation = gratpro.orientation + 1;
gratpro.drawGrating();
Screen('Flip', winPtr);
rect = [0 0 1270 720];
% rect = CenterRectOnPointd([0 0 gratpro.pixperdeg gratpro.pixperdeg]*P.snoisediam, gratpro.position(1), gratpro.position(2));
% rect = round(rect);
%
% gratpro.phase = 186+3;
I = gratpro.getImage(rect, 1);

I1 = Screen('GetImage', winPtr, rect);
I1 = mean(I1, 3);
figure(1); clf;
subplot(1,3,1)
imagesc(I)
subplot(1,3,2)
imagesc(I1)
subplot(1,3,3)
imagesc(I1 - I, [-10 10])




%% close textures
grat.CloseUp
gratpro.CloseUp

%% close screen if done
sca

