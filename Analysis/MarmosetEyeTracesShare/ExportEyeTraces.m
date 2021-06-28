
%%
for i = 23:57
    Exp = io.dataFactory(i);
    evalc("validTrials = io.getValidTrials(Exp, 'BackImage');");
    fprintf('Session %d) %d Trials\n', i, numel(validTrials))
end


%%

Exp = io.dataFactory(57);
io.checkCalibration(Exp)

%%
eyePos = io.getEyeCalibrationFromRaw(Exp);

%%

Exp.vpx.smo(:,2) = sgolayfilt(Exp.vpx.smo(:,2), 1, 19);
Exp.vpx.smo(:,3) = sgolayfilt(Exp.vpx.smo(:,3), 1, 19);
% figure(1); clf
% plot(Exp.vpx.smo(:,2))

tt = Exp.vpx2ephys(Exp.vpx.smo(:,1));
eyePos = Exp.vpx.smo(:,2:3);
eyeLabels = Exp.vpx.Labels;

validTrials = io.getValidTrials(Exp, 'BackImage');
numel(validTrials)

%%

D = struct();

% iTrial = iTrial + 1;
% if iTrial > numel(validTrials)
%     iTrial = 1;
% end
for iTrial = 1:numel(validTrials)
    
thisTrial = validTrials(iTrial);

imdir = fileparts(which('MarmoV5'));
imfile = fullfile(imdir, Exp.D{thisTrial}.PR.imagefile);
im = imread(imfile);

im = imresize(im, fliplr(Exp.D{thisTrial}.PR.destRect(3:4)));

iix = tt > Exp.D{thisTrial}.START_EPHYS & tt < Exp.D{thisTrial}.END_EPHYS;
figure(1); clf
imagesc(im)
hold on

x = eyePos(iix,1) * Exp.S.pixPerDeg + Exp.S.centerPix(1);
y = -eyePos(iix,2) * Exp.S.pixPerDeg + Exp.S.centerPix(2);
x(eyeLabels(iix)==4) = nan;
y(eyeLabels(iix)==4) = nan;

plot(x,y, 'r')

[~, imgname, ext] = fileparts(Exp.D{thisTrial}.PR.imagefile);
D(iTrial).imgfile = [imgname ext];
D(iTrial).time = tt(iix) - tt(find(iix,1));
D(iTrial).eyePos = [x y];
D(iTrial).eyeLabels = eyeLabels(iix);
D(iTrial).pixPerDeg = Exp.S.pixPerDeg;
D(iTrial).screenPix = Exp.S.screenRect(3:4);

end

save('Analysis/MarmosetEyeTracesShare/exampleEyeTraces.mat', '-v7.3', 'D')
