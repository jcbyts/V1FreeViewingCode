%% Open Marmoview screen with parameters from Forage protocol
sca % clear any open windows

[S,P] = FixRsvp();
A = marmoview.openScreen(S, struct());

winPtr = A.window;
%% Update 
Faces = stimuli.gaussimages(winPtr,'bkgd',S.bgColour,'gray',false);   % color images
Faces.loadimages('./SupportData/rsvpFixStim.mat');
Faces.position = [0,0]*S.pixPerDeg + S.centerPix;
Faces.radius = round(P.faceRadius*S.pixPerDeg);
Faces.imagenum = 1;
%%
Faces.imagenum = Faces.imagenum + 1;
Faces.beforeFrame()

Screen('Flip', winPtr, 0);

%%
Faces2 = stimuli.gaussimages(0,'bkgd',S.bgColour,'gray',false);   % color images
Faces2.loadimages('./SupportData/rsvpFixStim.mat');
Faces2.position = [0,0]*S.pixPerDeg + S.centerPix;
Faces2.radius = round(P.faceRadius*S.pixPerDeg);
Faces2.imagenum = Faces.imagenum;
Im = Faces2.getImage(A.screenRect);
figure(1); clf
imagesc(Im, [0 255])
colormap gray

%%
sca