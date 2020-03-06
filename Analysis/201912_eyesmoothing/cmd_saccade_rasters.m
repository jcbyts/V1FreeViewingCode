
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

switch user
    case 'jakework'
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\NIMclass
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\sNIMclass
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\L1General'))    
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\minFunc_2012'))  
    case 'jakelaptop'
        addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
        addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
        addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))    
        addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))  
end

%% load data
sessId = 12;
[Exp, S] = io.dataFactory(sessId);


%% get valid trials

stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

%% bin spikes and eye pos
binsize = 1e-3;
win = [-.1 .5];



eyeTime = Exp.vpx2ephys(Exp.vpx.raw(:,1));
remove = find(diff(eyeTime)==0);

eyeX = sgolayfilt(Exp.vpx.smo(:,2), 3, 9);
eyeX(isnan(eyeX)) = 0;
eyeY = sgolayfilt(Exp.vpx.smo(:,3), 3, 9);
eyeY(isnan(eyeY)) = 0;

eyeTime(remove) = [];
eyeX(remove) = [];
eyeY(remove) = [];

% resample at higher temporal resolution with fixed sample times
nuTime = eyeTime(1):binsize:eyeTime(end);

nuEyeX = interp1(eyeTime, eyeX, nuTime, 'pchip');
nuEyeY = interp1(eyeTime, eyeY, nuTime, 'pchip');

figure(1); clf
plot(nuTime, nuEyeX, '.'); hold on
plot(eyeTime, eyeX, '.')

%%

% find start and stop of valid trials
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

% get all detected saccades
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));

% exclude saccade times that don't have ephys
bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];

% only saccades that occured within a BackImage trial
valid = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);

% sort by fixation duration
fixdur = sacon - fixon;
[~, ind] = sort(fixdur);

% bin spike times at binsize
lags = win(1):binsize:win(2);
nlags = numel(lags);
n = numel(fixon);
NC = numel(S.cids);
spks = zeros(n,NC,nlags);

for i = 1:nlags
    y = binNeuronSpikeTimesFast(Exp.osp,fixon+lags(i), binsize);
    spks(:,:,i) = full(y(:,S.cids));
end

k = 1;

% %% 
% tic
% binTimes =  min(Exp.osp.st):binsize:max(Exp.osp.st);
% valid = getTimeIdx(fixon, tstart, tstop);
% binTimes = binTimes(valid);
% Y = binNeuronSpikeTimesFast(Exp.osp, binTimes, binsize);
% toc
%%
xpos = zeros(n,nlags);
ypos = zeros(n,nlags);
for i = 1:n
    xpos(i,:) = interp1(eyeTime, eyeX, fixon(i)+lags, 'pchip');
    ypos(i,:) = interp1(eyeTime, eyeY, fixon(i)+lags, 'pchip');
end
%%

% k = mod(k + 1, NC); k = max(k, 1);


figure(1); clf
set(gcf, 'Color', 'w')
cc = mod(cc + 1, NC); cc = max(cc, 1);
% for cc = 1:NC
%     subplot(6, ceil(NC/6), cc, 'align')
    [i, j] = find(spks(ind,cc,:));
    plot.raster(lags(j), i, 10); hold on
    plot([0 0], ylim, 'r')
    plot(fixdur(ind), 1:numel(ind), 'g')
    xlim(win)
    ylim([1 numel(ind)])
    title(sprintf('Unit: %d', cc))
    axis off
%     ylim(ind(end)-[400 0])
% end

%% sort eye velocity by tremor peak
xvel = diff(xpos, [], 2);
yvel = diff(ypos, [], 2);

figure(2); clf
vel = hypot(xvel, yvel);
% vel = xvel;
% range to analyze over
iix = lags > 0 & lags < .2;
goodTrial = zeros(n,1);
for iTrial = 1:n
    goodTrial(iTrial) = all(vel(iTrial,iix & lags < fixdur(iTrial))<.1);
end

goodTrial = goodTrial & fixdur > .2;
useVel = false;
if useVel
    v = vel(goodTrial,iix);
else
%     v = hypot(xpos(goodTrial,iix), ypos(goodTrial,iix));
    v = xpos(goodTrial,iix); 
end

% v = bsxfun(@minus, v, mean(v,2));
for i = 1:size(v,1)
    v(i,:) = detrend(v(i,:));
end

vf = v - filter(ones(5,1)/5, 1, v')';
%% check that you see a peak at 80 over all
figure(1); clf
pwelch(reshape(v', [], 1), [], [], [], 1/binsize)
hold on
[Pxx, xax] = pwelch(reshape(vf', [], 1), [], [], [], 1/binsize);
plot(xax, log10(Pxx)*10, 'r')
plot(80*[1 1], ylim, 'g')

%% make a gabor with 80 cycles/sec
freq = 80*(2*useVel);
nd = 40; % 40ms wide
nl = size(v,2);

figure(1); clf
filt = cos(linspace(-nd/2, nd/2, nd)*freq)'; %.*hanning(nd);
filt = filt / sum(filt);
nv = size(v,1);

shift = nan(nv,1);
for i = 1:nv
    xc = xcorr(vf(1,:), vf(i,:));
    ii2 = (1:10)+nl;
    
    [~, id] = max(xc(ii2));
    shift(i) = ii2(id)-nl;
end

%%

figure(1); clf
i = mod(i + 1, size(v,1)); i = max(i,1);
for j = 1:2:15
plot(imboxfilt(v(i,:),j)*60); hold on
end
% plot(xcorr(vf(1,:), vf(i,:))); hold on
% plot(xlim, 0.01*[1 1])

%%
vs = nan(size(v));
sps = spks(goodTrial,:,iix);
for i = 1:nv
% plot(v(i,:)); hold on
    vs(i,:) = circshift(v(i,:), shift(i));
    for cc = 1:NC
        sps(i,cc,:) = circshift(sps(i,cc,:), shift(i));
    end
end

%%


figure(1); clf
subplot(2,1,1)
imagesc(vs, [-.1 .1])
subplot(2,1,2)
plot(mean(vs))

%%
M = squeeze(mean(sps,1));
M0 = squeeze(mean(spks(goodTrial,:,iix),1));
figure(1); clf
cc = mod(cc + 1, NC); cc = max(cc, 1);
plot(M0(cc,:)/1e-3); hold on
plot(M(cc,:)/1e-3)
title(cc)
%%
%%


figure(2); clf
set(gcf, 'Color', 'w')
cc = mod(cc + 1, NC); cc = max(cc, 1);
%     subplot(6, ceil(NC/6), cc, 'align')
[i, j] = find(squeeze(sps(:,cc,:)));
plot.raster((j), i, 10); hold on
% imagesc(squeeze(sps(:,cc,:)))
title(sprintf('Unit: %d', cc))
axis off
%     ylim(ind(end)-[400 0])




%%
t = t + 1;
figure(2); clf
plot(v(t,:))
% spectrogram(v(t,:)-mean(v(t,:)), kaiser(128,18), 1, 100, 1e3)
% 
% xlim([10 250])


% imagesc(vel(ind,iix))

%% 
%%

subplot(1,2,1)

imagesc(lags(2:end),1:size(xpos,1), xvel, [-.05 .05])
axis xy
subplot(1,2,2)
imagesc(diff(ypos(ind,:), [],2), [-.05 .05])
axis xy


%%
k = mod(k + 1, NC); k = max(k, 1);
figure(2); clf
set(gcf, 'Color', 'w')
% plot(lags, squeeze(var(spks(fixdur > .25,k,:)))'/binsize); hold on
thresh = .25;
mFR = squeeze(mean(spks(fixdur > thresh,:,:)))'/binsize;
mFR = filter(ones(5,1)/5, 1, mFR);

plot(lags, mFR); hold on
plot(thresh*[1 1], ylim, 'k--')
xlabel('Time from fixation onset')
ylabel('Spike Rate')

% plot(squeeze(mean(spks(fixdur < .2,:,:),1))');
figure(3); clf
set(gcf, 'Color', 'w')
% plot(lags, squeeze(var(spks(fixdur > .25,k,:)))'/binsize); hold on
thresh = .25;
mFR = squeeze(mean(spks(fixdur > thresh,:,:)))'/binsize;
mFR = filter(ones(5,1)/5, 1, mFR);
% mTot = full(mean(Y(:,S.cids)))/binsize;
mTot = mean(mFR(lags < .250 & lags > .1,:));
mTot(mTot < 1) = nan;
plot(lags, mFR./mTot); hold on
plot(thresh*[1 1], ylim, 'k--')
plot(xlim, [1 1], 'k--')
xlabel('Time from fixation onset')
ylabel('Relative Rate')

figure(1); clf
plot(mean(mFR(lags > .15,:)), mean(mFR(lags < .1,:)), '.'); hold on
plot(xlim, xlim, 'k')

%%

labels = Exp.vpx.Labels;
good = labels == 1 | labels ==2;
[bw, num] = bwlabel(good);
ar = nan(num, 1);
for i = 1:num
    ar(i) = sum(bw==i);
end

%%
   
[~, ind] = sort(ar, 'descend');

longest = bw == ind(1);


t = Exp.vpx.smo(:,1);
x = sgolayfilt(Exp.vpx.smo(:,2), 2, 7);
y = Exp.vpx.smo(:,3);

figure(1); clf
plot(t(longest), x(longest), 'k')

%%
clf
xx = linspace(0, 10, 1000);
yy = sin(10*xx);
cy = cos(10*xx);
dt = mean(diff(xx));
dy = [0 diff(yy)/dt];
plot(yy);
hold on
plot(dy)
plot(cy)
%%
clf
plot(fftshift(abs(fft(yy)))); hold on
plot(fftshift(abs(fft(dy)))); hold on
