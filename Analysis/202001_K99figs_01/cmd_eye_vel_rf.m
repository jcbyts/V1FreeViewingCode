
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
sessId = 57;
[Exp, S] = io.dataFactoryGratingSubspace(sessId);

%% get valid trials
% ForageStaticLines
stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

%%

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));

win = [-.1 .5];
bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];

valid = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);

fixdur = sacon - fixon;
[~, ind] = sort(fixdur);


binsize = 1e-3;


lags = win(1):binsize:win(2);
nlags = numel(lags);
n = numel(fixon);
NC = numel(S.cids);
spks = zeros(n,NC,nlags);
yvel = zeros(n,nlags);
xvel = zeros(n,nlags);

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1)); % time
remove = find(diff(eyeTime)==0); % bad samples

% filter eye position with 3rd order savitzy-golay filter
eyeXvel = Exp.vpx.smo(:,5); % smooth (preserving tremor)
eyeXvel(isnan(eyeXvel)) = 0;
eyeYvel = Exp.vpx.smo(:,6);
eyeYvel(isnan(eyeYvel)) = 0;

% remove bad samples
eyeTime(remove) = [];
eyeXvel(remove) = [];
eyeYvel(remove) = [];

eyePosInterpolationMethod = 'linear'; %'pchip'

for i = 1:nlags-1
    y = binNeuronSpikeTimesFast(Exp.osp,fixon+lags(i), binsize);
    spks(:,:,i) = full(y(:,S.cids));
    xvel(:,i) = interp1(eyeTime, eyeXvel, fixon+lags(i), eyePosInterpolationMethod);
    yvel(:,i) = interp1(eyeTime, eyeYvel, fixon+lags(i), eyePosInterpolationMethod);
end
k = 1;
disp('Done')




%% plot saccade rasters

figure(1); clf
set(gcf, 'Color', 'w')
for cc = 1:NC
    subplot(6, ceil(NC/6), cc, 'align')
    [i, j] = find(spks(ind,cc,:));
    plot.raster(lags(j), i, 10); hold on
    plot([0 0], ylim, 'r')
    plot(fixdur(ind), 1:numel(ind), 'g')
    xlim(win)
    ylim([1 numel(ind)])
    title(sprintf('Unit: %d', cc))
    axis off
end

%% plot eye velocity 
figure(2); clf
imagesc(lags, 1:n, hypot(xvel(ind,:), yvel(ind,:)))
xlabel('lags')
ylabel('fixation #')

%%
nth = 8;
nrho = 5;
bs = 360/nth;
th0 = 0:bs:(360-bs);
rh0 = .5*2.^(0:nrho-1);

[th, rh] = meshgrid(th0,rh0);

xax = linspace(0, 360, 1000)';
figure(10); clf
circdiff = @(th1, th2) angle(exp(1i*(th1-th2)/180*pi))/pi*180;

subplot(121)
plot(xax, max(1-abs(circdiff(xax,th0))/bs, 0))
subplot(122)
xax = linspace(1e-3, max(rh0), 1000);
plot(xax, max(1-abs(log2(xax(:))-log2(rh0(:)')), 0))

%%
[thvel, rhvel] = cart2pol(xvel(:), yvel(:));
thvel = thvel/pi*180;

thbas = max(1-abs(circdiff(thvel(:),th(:)'))/bs, 0);
rhbas = max(1-abs(log2(rhvel(:))-log2(rh(:)')), 0);

X = thbas .* rhbas;
Y = reshape(permute(spks, [1 3 2]), [], NC);

%%
cc = cc + 1;
sta = simpleRevcorr(X, Y(:,cc), 100);

figure(1); clf
imagesc(sta./sum(X))



%%

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
xlabel('Sustained')
ylabel('Transient')
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

