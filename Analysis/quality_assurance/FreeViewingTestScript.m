
%% addpaths
testmode = true; % only use 10 trials while regenerating stimuli

user = 'jakelaptop';
% user = 'jakework';
dataPath = addFreeViewingPaths(user);

%% load data


sessionId = 6;
Exp = dataFactory(sessionId, dataPath);

%% find specific trial lists
trialProtocols = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);
protocols = unique(trialProtocols);
numProtocols = numel(protocols);
fprintf('Session was run with %d protocols\n', numProtocols);
for i = 1:numProtocols
    
    nTrials = sum(strcmp(trialProtocols, protocols{i}));
    fprintf('%d) %d trials of %s\n', i, nTrials, protocols{i});
end

%% Quick spatial RF map

rect = [-200 -200 200 200];
binSize = .5*Exp.S.pixPerDeg; % we just want to find the rect

Latency = 8.3e-3;
RF = quick_sta_rfs(Exp, 'ROI', rect, 'binSize', binSize, 'latency', Latency);

%%
% plot spatial RFs
figure(1); clf
CN = numel(RF);
sx = ceil(sqrt(CN));
sy = round(sqrt(CN));

for kNeuron = 1:CN
    subplot(sx, sy, kNeuron, 'align')
    imagesc(RF(kNeuron).xax/Exp.S.pixPerDeg, RF(kNeuron).yax/Exp.S.pixPerDeg, RF(kNeuron).srf)
    drawnow
end

% Find a bounding box around RFs
clear validrfs
c = 1;
sI = 0;
for kNeuron = 1:CN
    srf = (RF(kNeuron).srf - min(RF(kNeuron).srf(:))) / (max(RF(kNeuron).srf(:)) - min(RF(kNeuron).srf(:)));
    
    % weird edge case with the first pixel. Easy to exclude
    srf(1) = .1; 
    
    regions = regionprops(logical(srf > .5));
    if numel(regions)==1
        sI = sI + srf;
        validrfs(c) = regions;
        c = c+1;
    end
end

figure(2); clf
imagesc(sI); hold on
title('Average RF')
bbox = mode(reshape([validrfs.BoundingBox], 4, []), 2)';
xcoords = bbox([1 1 1 1 1]) + [0 0 bbox([3 3]) 0];
ycoords = bbox([2 2 2 2 2]) + [0 bbox([4 4]) 0 0];
plot(xcoords, ycoords, 'c', 'Linewidth', 2)    
legend('Bounding box')
rect = [interp1(RF(1).xax, bbox(1)), ...
    interp1(RF(1).yax, bbox(2)), ...
    interp1(RF(1).xax, bbox(1)+bbox(3)), ...
    interp1(RF(1).yax, bbox(2)+bbox(4))];

rect = round(rect * 1.05);

%% Select trials to analyze

% --- find the trials that we want to analyze
ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
if (numel(ephysTrials)/numel(Exp.D)) < 0.6
    disp('Something is wrong. Assuming all trials had ephys')
    ephysTrials = 1:numel(Exp.D);
end

% Forage trials
validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

% dot spatial noise trials
validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));

dotSize = cellfun(@(x) x.P.dotSize, Exp.D(validTrials));
validTrials = validTrials(dotSize==min(dotSize));

numValidTrials = numel(validTrials);
fprintf('Found %d valid trials\n', numValidTrials)

%% Reconstruct stimuli

binSize = 1; % pixel
[Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect, 'spatialBinSize', binSize, 'Latency', Latency);


%% bin spike times
spikeBinSize = 1/Exp.S.frameRate;
frameTimes = frameInfo.frameTimesOe;

Y = binNeuronSpikeTimesFast(Exp.osp, frameTimes, spikeBinSize);
Y = Y(:,Exp.osp.cids); % index into units

% reshape the stimulus
X = reshape(Stim, prod(frameInfo.dims), numel(frameTimes))';

goodIx = frameInfo.probeDistance > 50;
fprintf('%d frames excluded because the probe was in the window\n', sum(~goodIx))
goodIx = goodIx & frameInfo.seedGood;
fprintf('%d frames excluded because the seed failed\n', sum(~frameInfo.seedGood))

% find saccades and exclude them

goodVelTimes = cell2mat(cellfun(@(x) x.eyeSmo(x.eyeSmo(:,7)<20,1) + x.STARTCLOCKTIME, Exp.D(validTrials), 'uni', 0));
goodVelTimes = Exp.ptb2Ephys(goodVelTimes);
bvt = binfun(goodVelTimes);
goodVel = (sparse(bvt, ones(numel(bvt), 1), ones(numel(bvt), 1)));
goodVel = full(goodVel(bft)) > 0;

goodIx = goodIx & goodVel;

hasSaccades = cellfun(@(x) ~isempty(x.slist), Exp.D(validTrials));
saccadeOnsets = cell2mat(cellfun(@(x) x.slist(:,1) + x.STARTCLOCKTIME, Exp.D(validTrials(hasSaccades)), 'uni', 0));
saccadeOnsets = Exp.ptb2Ephys(saccadeOnsets);
saccadeBinned = binfun(saccadeOnsets);
saccadeBinned = unique(bsxfun(@plus, saccadeBinned, -10:20));
saccadeBinned(saccadeBinned < 1) = [];
saccadeBinned(saccadeBinned > numel(goodIx)) = [];
fprintf('%d frames excluded because during saccade\n', numel(saccadeBinned))

% goodIx(saccadeBinned) = false;

%%
fprintf('%d valid time bins\n', sum(goodIx))
X = double(X(goodIx,:));
Y = Y(goodIx,:);

y = Y - mean(Y);

kNeuron = 0;

nNeurons = size(Y,2);
%%
%[59, 60, 63, 67]
xax = rect(1):binSize:(rect(3)-binSize);
yax = rect(2):binSize:(rect(4)-binSize);
xax = xax / Exp.S.pixPerDeg;
yax = yax / Exp.S.pixPerDeg;

lags = 1:2:20;

kNeuron = kNeuron + 1;

nlags = numel(lags);
figure(1); clf
set(gcf, 'Color', 'w')
stas = nan(prod(frameInfo.dims), nlags);
for i = 1:nlags
    lag = lags(i);
    sta = abs(X(1:end-lag+1,:))'*y(lag:end, kNeuron);
    stas(:,i) = sta;
end

stas = (stas - min(stas(:)) ) ./ (max(stas(:))-min(stas(:)));
for i = 1:nlags
    subplot(1,nlags, i)
    lag = lags(i);
    imagesc(xax, yax, reshape(stas(:,i), frameInfo.dims), [0 1])
    colormap(parula)
    title(sprintf('Neuron %d lag %d', kNeuron, lag))
end


figure(2); clf
set(gcf, 'Color', 'w')
stas = nan(prod(frameInfo.dims), nlags);
for i = 1:nlags
    lag = lags(i);
    sta = (X(1:end-lag+1,:))'*y(lag:end, kNeuron);
    stas(:,i) = sta;
end

stas = (stas - min(stas(:)) ) ./ (max(stas(:))-min(stas(:)));
for i = 1:nlags
    subplot(1,nlags, i)
    lag = lags(i);
    imagesc(xax, yax, reshape(stas(:,i), frameInfo.dims), [0 1])
    colormap(parula)
    title(sprintf('Neuron %d lag %d', kNeuron, lag))
end
%% Now analyze the GABOR NOISE trials


% Forage trials
validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

validTrials = validTrials(cellfun(@(x) x.PR.noisetype==4, Exp.D(validTrials)));

% validTrials = intersect(find(strcmp(trialProtocols, 'BackImage')), ephysTrials);
numValidTrials = numel(validTrials);
fprintf('Found %d valid trials\n', numValidTrials)

%% REGENERATE STIMULUS

if testmode
    fprintf('**** TEST MODE ****\nOnly using 10 trials\n')
    validTrials = randsample(validTrials, 10);
end

binSize = 1; % pixel
fprintf('Regenerating stimuli...\n')
[Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect, 'spatialBinSize', binSize, 'Latency', Latency);

% get sync function to convert from vpx to ephys time
vpx2Ephys = synchtime.sync_vpx_to_ephys_clock(Exp);
%%
ix = frameInfo.eyeAtFrame(:,1)~=0;
t1 = Exp.ptb2Ephys(frameInfo.eyeAtFrame(ix,1));
x1 = frameInfo.eyeAtFrame(ix,2);
figure(1); clf


t2 = vpx2Ephys(Exp.vpx.raw(:,1));

%**** eye transform
pixPerDeg = Exp.S.pixPerDeg;
cx = cellfun(@(x) x.c(1), Exp.D(validTrials));
cy = cellfun(@(x) x.c(2), Exp.D(validTrials));
dx = cellfun(@(x) x.dx, Exp.D(validTrials));
dy = cellfun(@(x) x.dy, Exp.D(validTrials));
      
cx = mode(cx);
cy = mode(cy);
dx = mode(dx);
dy = mode(dy);

vxx = Exp.vpx.raw(:,2);
vyy = Exp.vpx.raw(:,3);
        
vxx = (vxx - cx)/(dx);
vyy = 1 - vyy;
vyy = (vyy - cy)/(dy);

vxx = vxx + Exp.S.centerPix(1);
vyy = vyy + Exp.S.centerPix(2);
        
plot(t2, vxx, '-o', 'MarkerSize', 2); hold on
plot(t1, x1, '.');

%%
figure(1); clf

plot(diff(Exp.vpx.raw(:,1)), '.')

% plot(Exp.ptb2Ephys(Exp.vpx.raw(:,1)), Exp.vpx.raw(:,2))
thisTrial = 10;
%% bin spike times

figure(1); clf;
thisTrial = thisTrial + 1;
t = Exp.D{thisTrial}.eyeSmo(:,1);
x = Exp.D{thisTrial}.eyeSmo(:,2)*60;
plot(t, x, 'r'); hold on
% xs = sgolayfilt(x, 1, 9);
xs = imgaussfilt(x, 10);
plot(t, xs, 'k')

sacoff = Exp.D{thisTrial}.slist(1:end-1,2);
sacon = Exp.D{thisTrial}.slist(2:end,1);

n = numel(sacoff);
fixations = cell(n,1);
for i = 1:n
   fixations{i} = find(t > sacoff(i) & t < sacon(i));
end

figure(2); clf
ix = fixations{4};

subplot(1,2,1)
plot(t, x); hold on
plot(t(ix), x(ix), 'r')

ix = cell2mat(fixations);
s = xs(ix)-x(ix);
s(abs(s) > 5) = [];
figure(1); clf
plot(s, '-o', 'MarkerSize', 2);
s(isnan(s)) = [];

figure(1); clf
[Pxx, bins] = pwelch(s, [], [], [], 540);

plot(bins, imgaussfilt(10*log10(Pxx), 2))
%%
spikeBinSize = 1/Exp.S.frameRate;
frameTimes = frameInfo.frameTimesOe;

X = reshape(Stim, prod(frameInfo.dims), numel(frameTimes))';
Y = binNeuronSpikeTimesFast(Exp.osp, frameTimes, spikeBinSize);
Y = Y(:,Exp.osp.cids); % index into good units

%% find good trials
goodIx = (frameInfo.probeDistance > 50) | isnan(frameInfo.probeDistance);
% fprintf('%d frames excluded because the probe was in the window\n', sum(~goodIx))
% goodIx = goodIx & frameInfo.seedGood;
% fprintf('%d frames excluded because the seed failed\n', sum(~frameInfo.seedGood))

% find saccades and exclude them

goodVelTimes = cell2mat(cellfun(@(x) x.eyeSmo(x.eyeSmo(:,7)<20,1) + x.STARTCLOCKTIME, Exp.D(validTrials), 'uni', 0));
goodVelTimes = Exp.ptb2Ephys(goodVelTimes);
bvt = binfun(goodVelTimes);
goodVel = (sparse(bvt, ones(numel(bvt), 1), ones(numel(bvt), 1)));
goodVel = full(goodVel(bft)) > 0;

goodIx = goodIx & goodVel;

fprintf('%d valid time bins\n', sum(goodIx))
X = double(X(goodIx,:));
X = zscore(X);
Y = Y(goodIx,:);

y = Y - mean(Y);

kNeuron = 0;

nNeurons = size(Y,2);
%%


%%
%[59, 60, 63, 67]
xax = rect(1):binSize:(rect(3)-binSize);
yax = rect(2):binSize:(rect(4)-binSize);
xax = xax / Exp.S.pixPerDeg;
yax = yax / Exp.S.pixPerDeg;

lags = 1:2:20;

kNeuron = kNeuron + 1;

nlags = numel(lags);
figure(1); clf
set(gcf, 'Color', 'w')
stas = nan(prod(frameInfo.dims), nlags);
for i = 1:nlags
    lag = lags(i);
    sta = abs(X(1:end-lag+1,:))'*y(lag:end, kNeuron);
    stas(:,i) = sta;
end

stas = (stas - min(stas(:)) ) ./ (max(stas(:))-min(stas(:)));
for i = 1:nlags
    subplot(1,nlags, i)
    lag = lags(i);
    imagesc(xax, yax, reshape(stas(:,i), frameInfo.dims), [0 1])
    colormap(parula)
    title(sprintf('Neuron %d lag %d', kNeuron, lag))
end


figure(2); clf
set(gcf, 'Color', 'w')
stas = nan(prod(frameInfo.dims), nlags);
for i = 1:nlags
    lag = lags(i);
    sta = (X(1:end-lag+1,:))'*y(lag:end, kNeuron);
    stas(:,i) = sta;
end

stas = (stas - min(stas(:)) ) ./ (max(stas(:))-min(stas(:)));
for i = 1:nlags
    subplot(1,nlags, i)
    lag = lags(i);
    imagesc(xax, yax, reshape(stas(:,i), frameInfo.dims), [0 1])
    colormap(parula)
    title(sprintf('Neuron %d lag %d', kNeuron, lag))
end

%% BACK IMAGE ANALYSIS

backImageTrials = find(strcmp(trialProtocols, 'BackImage'));

validTrials = backImageTrials;

iTrial = 2;
thisTrial = validTrials(iTrial);

figure(1); clf

stimRect = Exp.D{thisTrial}.PR.destRect - [Exp.S.centerPix Exp.S.centerPix];


marmoViewPath = fileparts(which('MarmoV5'));
imageFile = fullfile(marmoViewPath, Exp.D{thisTrial}.PR.imagefile);
im = imread(imageFile);
imWidth = Exp.D{thisTrial}.PR.destRect(3) - Exp.D{thisTrial}.PR.destRect(1);
imHeight = Exp.D{thisTrial}.PR.destRect(4) - Exp.D{thisTrial}.PR.destRect(2);
im = imresize(im, [imHeight imWidth]);

rect = [-imWidth -imHeight imWidth imHeight]/2 + [Exp.S.centerPix Exp.S.centerPix];
assert(all(rect==Exp.D{thisTrial}.PR.destRect), 'One of my assumptions was wrong about the stimulus size');

imxax = ((1:imWidth)-imWidth/2) / Exp.S.pixPerDeg;
imyax = ((1:imHeight)-imHeight/2) / Exp.S.pixPerDeg;

imagesc(imxax, imyax, im);

hold on
plot(Exp.D{thisTrial}.eyeSmo(:,2), Exp.D{thisTrial}.eyeSmo(:,3), '.')



%%

ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
ephysTrials = 1:numel(Exp.D);
validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);


validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));

validTrials = validTrials(cellfun(@(x) x.P.numDots==100, Exp.D(validTrials)));
% validTrials = validTrials(cellfun(@(x) x.P.numDots==1000, Exp.D(validTrials)));


numValidTrials = numel(validTrials);
fprintf('Found %d valid trials\n', numValidTrials)


framesPerTrial = cellfun(@(x) size(x.PR.NoiseHistory, 1), Exp.D(validTrials));
nFrames = sum(framesPerTrial);

rect = [-20 0 50 100];

% rect = CenterRectOnPointd(rect, Exp.S.centerPix(1), Exp.S.centerPix(1));
spatialBinSize = 2;
dims = (rect([4 3]) - rect([2 1]))/spatialBinSize;

% X = spalloc(nFrames, prod(dims), 1500*nFrames);
X = zeros(nFrames, prod(dims));
CN = numel(Exp.sp);
Y = nan(nFrames, CN); % counting spikes for the photodiode

binSize = 1./Exp.S.frameRate;

%% loop over trials and regenerate stimuli
nTrials = numel(validTrials);
frameCounter = 1;
nTrials = 10;

for iTrial = 1:nTrials
    fprintf('%d/%d trials\n', iTrial, nTrials)
    thisTrial = validTrials(iTrial);

%     frameRefreshes = Exp.D{thisTrial}.eyeData(6:end,6);
    frameRefreshes = Exp.D{thisTrial}.PR.NoiseHistory(:,1);
    nFrames = numel(frameRefreshes);

    nFrames = min(nFrames, size(Exp.D{thisTrial}.PR.NoiseHistory,1));
    frameRefreshes = Exp.ptb2Ephys(frameRefreshes(1:nFrames));
    
    
    % get eye calibration
    c = Exp.D{thisTrial}.c;
    dx = Exp.D{thisTrial}.dx;
    dy = Exp.D{thisTrial}.dy;
    
%     eyeX = (eyeX - c(1))/(dx*Exp.S.pixPerDeg);
%     eyeY = (eyeY - c(2))/(dx*Exp.S.pixPerDeg);
    % this is our noise object
    hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
    hNoise.rng.reset(); % reset the random seed to the start of the trial
    hNoise.frameUpdate = 0; % reset the frame counter

    eyeTimes = Exp.ptb2Ephys(Exp.D{thisTrial}.eyeSmo(:,1) + Exp.D{thisTrial}.STARTCLOCKTIME);
    % loop over frames and get noise from that frame
    for iFrame = 1:nFrames
%         disp(iFrame)
        hNoise.afterFrame();
        
        eyeIx = find(eyeTimes >= frameRefreshes(iFrame),1);
        eyeX = Exp.D{thisTrial}.eyeSmo(eyeIx,2)*Exp.S.pixPerDeg;
        eyeY = Exp.D{thisTrial}.eyeSmo(eyeIx,3)*Exp.S.pixPerDeg;
        
        if hypot(eyeX, eyeY) > 500
            continue
        end
%         eyeX = (eyeX - c(1))/dx;
        
%         x = (eyeX - c(1)) / dx;
%         y = (eyeY - c(2)) / dy;
        eyeX = Exp.S.centerPix(1)+eyeX;
        eyeY = Exp.S.centerPix(2)-eyeY;
        
        if isnan(eyeX) || isnan(eyeY)
            continue
        end
        
%         tmprect = CenterRectOnPointd(rect, eyeX, eyeY);
        tmprect = rect + [eyeX eyeY eyeX eyeY];
        I = hNoise.getImage(tmprect, spatialBinSize);
    
        X(frameCounter,:) = I(:);
        
        for kUnit = 1:CN
            ix = Exp.sp{kUnit}.st >= frameRefreshes(iFrame) & Exp.sp{kUnit}.st < frameRefreshes(iFrame) + binSize;
            Y(frameCounter, kUnit) = sum(ix);
        end
        frameCounter = frameCounter + 1;
    end
    
end

%%
X = X(1:frameCounter-1,:);
Y = Y(1:frameCounter-1,:);
y = Y - mean(Y);

%%
xax = rect(1):spatialBinSize:(rect(3)-spatialBinSize);
yax = rect(2):spatialBinSize:(rect(4)-spatialBinSize);
xax = xax / Exp.S.pixPerDeg;
yax = yax / Exp.S.pixPerDeg;

lag = 2; 
sta = abs(X(1:end-lag+1,:))'*sum(y(lag:end, :),2);

figure(1); clf
imagesc(xax, yax, reshape(sta, dims))

%% plot neurons by lag
figure(1); clf
nlags = 10;
nNeurons = 20;
sx = nNeurons; %ceil(sqrt(nlags));
sy = nlags; %round(sqrt(nlags));

ax = pdsa.tight_subplot(sx, sy, 0.01, 0.01);
for i = 1:nlags
    lag = i;
    
    for kUnit = 1:nNeurons
        set(gcf, 'currentaxes', ax((kUnit-1)*nlags + i))
    
        sta = abs(X(1:end-lag+1,:))'*y(lag:end, kUnit);
        imagesc(xax, yax, reshape(sta, dims));
        axis off
        drawnow
    end
end



%%




%% Run again at a finer resolution (time to make this a function, huh)

% ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
ephysTrials = 1:numel(Exp.D);
validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);


validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));
validTrials = validTrials(cellfun(@(x) x.P.numDots==1000, Exp.D(validTrials)));


numValidTrials = numel(validTrials);
fprintf('Found %d valid trials\n', numValidTrials)


%% regenerate stimulus
framesPerTrial = cellfun(@(x) size(x.PR.NoiseHistory, 1), Exp.D(validTrials));
nFrames = sum(framesPerTrial);


% window
% rect = ceil([0 0 1 2]*Exp.S.pixPerDeg);
rect = [-20 0 50 100];

spatialBinSize = 1;
dims = ((rect([4 3]) - rect([2 1]))/spatialBinSize);

X = zeros(nFrames, prod(dims));
CN = numel(Exp.sp);
Y = nan(nFrames, CN); % counting spikes

binSize = 1./Exp.S.frameRate;

%% loop over trials and regenerate stimuli
nTrials = numel(validTrials);
frameCounter = 1;
nTrials = 50;

for iTrial = 1:nTrials
    fprintf('%d/%d trials\n', iTrial, nTrials)
    thisTrial = validTrials(iTrial);

    frameRefreshes = Exp.D{thisTrial}.PR.NoiseHistory(:,1);
    nFrames = numel(frameRefreshes);

    nFrames = min(nFrames, size(Exp.D{thisTrial}.PR.NoiseHistory,1));
    frameRefreshes = Exp.ptb2Ephys(frameRefreshes(1:nFrames));
    
    
    % get eye calibration
    c = Exp.D{thisTrial}.c;
    dx = Exp.D{thisTrial}.dx;
    dy = Exp.D{thisTrial}.dy;
    
    % this is our noise object
    hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
    hNoise.rng.reset(); % reset the random seed to the start of the trial
    hNoise.frameUpdate = 0; % reset the frame counter

    eyeTimes = Exp.ptb2Ephys(Exp.D{thisTrial}.eyeSmo(:,1) + Exp.D{thisTrial}.STARTCLOCKTIME);
    
    % loop over frames and get noise from that frame
    for iFrame = 1:nFrames
%         disp(iFrame)
        hNoise.afterFrame();
        
        eyeIx = find(eyeTimes >= frameRefreshes(iFrame),1);
        eyeX = Exp.D{thisTrial}.eyeSmo(eyeIx,2)*Exp.S.pixPerDeg;
        eyeY = Exp.D{thisTrial}.eyeSmo(eyeIx,3)*Exp.S.pixPerDeg;
        
        if hypot(eyeX, eyeY) > 500
            continue
        end

        eyeX = Exp.S.centerPix(1)+eyeX;
        eyeY = Exp.S.centerPix(2)-eyeY;
        
        if isnan(eyeX) || isnan(eyeY)
            continue
        end
        
%         tmprect = CenterRectOnPointd(rect, eyeX, eyeY);
        tmprect = rect + [eyeX eyeY eyeX eyeY];
        I = hNoise.getImage(tmprect, spatialBinSize);
    
        X(frameCounter,:) = I(:);
        
        for kUnit = 1:CN
            ix = Exp.sp{kUnit}.st >= frameRefreshes(iFrame) & Exp.sp{kUnit}.st < frameRefreshes(iFrame) + binSize;
            Y(frameCounter, kUnit) = sum(ix);
        end
        frameCounter = frameCounter + 1;
    end
    
end


%% truncate data
X = X(1:frameCounter-1,:);
Y = Y(1:frameCounter-1,:);

y = Y - mean(Y);

%%

% build stimulus axes
xax = rect(1):spatialBinSize:(rect(3)-spatialBinSize);
yax = rect(2):spatialBinSize:(rect(4)-spatialBinSize);
xax = xax / Exp.S.pixPerDeg;
yax = yax / Exp.S.pixPerDeg;

lag = lag + 1; 
sta = abs(X(1:end-lag+1,:))'*sum(y(lag:end, :),2);

figure(2); clf
imagesc(xax, yax, reshape(sta, dims))
title(lag*4)

%%
figure(3); clf
lags = [1:2:20];
nlags = numel(lags);
neuronIx = [3 21 22 24 54 66 68 74 75 86 87];
% neuronIx = 80:CN;
nNeurons = numel(neuronIx);
sx = nNeurons; %ceil(sqrt(nlags));
sy = nlags; %round(sqrt(nlags));

ax = pdsa.tight_subplot(sx, sy, 0.01, 0.01);
for kUnit = 1:nNeurons
    rf = zeros(dims(1), dims(2), nlags);
    for i = 1:nlags
        lag = lags(i);
        sta = abs(X(1:end-lag+1,:))'*y(lag:end, neuronIx(kUnit));
        rf(:,:,i) = reshape(sta, dims);
    end
        
    rf = (rf - min(rf(:))) ./ (max(rf(:))-min(rf(:)));
    
    for i = 1:nlags
        
        set(gcf, 'currentaxes', ax((kUnit-1)*nlags + i))
        
        rI = imgaussfilt(reshape(rf(:,:,i), dims), .2);
        imagesc(xax, yax, rI, [0 1]);
        if kUnit==1
            title(lags(i)*4)
        end
        xlim([-.5 1.5])
        ylim([0 2])
        axis off
        drawnow
    end
end


%%

validTrials = find(strcmp(trialProtocols, 'ForageProceduralNoise'));

noiseTypes = cellfun(@(x) x.PR.noisetype, Exp.D(validTrials));

validTrials = validTrials(noiseTypes==4);
numTrials = numel(validTrials);

fprintf('Found %d Gabor Trials\n', numTrials)


iTrial = 10;
thisTrial = validTrials(iTrial);


Exp.D{thisTrial}.PR
%%
hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
hNoise.rng.reset(); % reset the random seed to the start of the trial
hNoise.frameUpdate = 0; % reset the frame counter

%%
tic
hNoise.afterFrame();

rect = CenterRectOnPointd([0 0 100 100], Exp.S.centerPix(1), Exp.S.centerPix(2));

I = hNoise.getImage(rect);
toc



%%
figure(1); clf
imagesc(I)