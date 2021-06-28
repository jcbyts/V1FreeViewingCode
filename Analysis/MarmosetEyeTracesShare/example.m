
load Analysis/MarmosetEyeTracesShare/exampleEyeTraces.mat

iTrial = 1;
%%
iTrial = iTrial + 1;
imgDir = fileparts(which('example.m'));
figure(1); clf
im = imread(fullfile(imgDir, D(iTrial).imgfile));
im = imresize(im, fliplr(D(iTrial).screenPix));
imagesc(im)
hold on
plot(D(iTrial).eyePos(:,1), D(iTrial).eyePos(:,2), 'r', 'Linewidth', 2)

figure(2); clf
ctr = D(iTrial).screenPix/2;
eyePosDeg = (D(iTrial).eyePos-ctr)/D(iTrial).pixPerDeg;
plot(D(iTrial).time, eyePosDeg(:,1)); hold on
plot(D(iTrial).time, eyePosDeg(:,2));
legend({'Horizontal', 'Vertical'})
xlabel('Time (seconds)')
ylabel('Degrees')
