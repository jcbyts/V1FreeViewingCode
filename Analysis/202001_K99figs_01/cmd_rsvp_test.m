% We have a RSVP fixation point that flashes processed natural images in
% the central 

%% Load dataset

[Exp, S] = io.dataFactory(12);
out = load('Data/gmodsac0_output.mat');

%% get valid trials
validTrials = io.getValidTrials(Exp, 'FixRsvpStim');


%% bin spikes and eye pos
binsize = 2e-3; % 1 ms bins for rasters
win = [-.1 2]; % -100ms to 2sec after fixation onset
% resample the eye position at the rate of the time-resolution of the
% ephys. When upsampling, use linear or spline (pchip) interpolation
eyePosInterpolationMethod = 'linear'; %'pchip'


% --- get eye position
eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1)); % time
remove = find(diff(eyeTime)==0); % bad samples

% filter eye position with 3rd order savitzy-golay filter
eyeX = sgolayfilt(Exp.vpx.smo(:,2), 3, 9); % smooth (preserving tremor)
eyeX(isnan(eyeX)) = 0;
eyeY = sgolayfilt(Exp.vpx.smo(:,3), 3, 9);
eyeY(isnan(eyeY)) = 0;

% remove bad samples
eyeTime(remove) = [];
eyeX(remove) = [];
eyeY(remove) = [];

% --- get spike times

% trial length
n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials));

% time fixation start
tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1,1), Exp.D(validTrials)));
tend = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(end,1), Exp.D(validTrials)));

% trials < 100 frames remove
bad = n < 100;
tstart(bad) = [];
tend(bad) = [];
n(bad) = [];

% sort trials by fixation duration
[~, ind] = sort(n, 'descend');

% bin spike times at specific lags
lags = win(1):binsize:win(2);
nlags = numel(lags);
nt = numel(tstart);
NC = numel(S.cids);
spks = zeros(nt,NC,nlags);
xpos = zeros(nt,nlags);
ypos = zeros(nt,nlags);
NC = 58;
Robs = zeros(nt,NC,nlags);
Ypred = zeros(nt,NC,nlags);

ft = out.frameTimes;
bad = ft==0;
R = out.Robs;
Rhat = out.predrate;
R(bad,:) = [];
Rhat(bad,:) = [];
ft(bad) = [];
% Do the binning here
for i = 1:nt
    y = binNeuronSpikeTimesFast(Exp.osp,tstart(i)+lags, binsize);
    spks(i,:,:) = full(y(:,S.cids))';
    % resample eye position at the time resolution of the spike trains
    xpos(i,:) = interp1(eyeTime, eyeX, tstart(i)+lags, eyePosInterpolationMethod);
    ypos(i,:) = interp1(eyeTime, eyeY, tstart(i)+lags, eyePosInterpolationMethod);
    
	Robs(i,:,:) = interp1(ft, R, tstart(i)+lags, eyePosInterpolationMethod)';
    Ypred(i,:,:) = interp1(ft, Rhat, tstart(i)+lags, eyePosInterpolationMethod)';
    
end

% initialize iterator for plotting cells
cc = 19;
%% remove data after saccade
bdur = ceil(n(:)/Exp.S.frameRate/binsize);
for i = 1:nt
    
    Robs(i,:,bdur(i):end) = nan;
    Ypred(i,:,bdur(i):end) = nan;
end

%%
cc = mod(cc + 1, NC); cc = max(cc, 1);
figure(10); clf
set(gcf, 'Color', 'w')

subplot(4,1,1:3)
[i, j] = find(spks(ind,cc,:));
plot.raster(lags(j), i, 1); hold on
plot([0 0], ylim, 'r')
plot(n(ind)/Exp.S.frameRate, 1:numel(ind), 'g')
xlim(lags([1 end]))
ylabel('Trial')

title(cc)
subplot(4,1,4)
m = squeeze(mean(spks(:,cc,:),1))/binsize;
plot(lags, m, 'k')
xlim(lags([1 end]))
xlabel('Time from fixation onset')
ylabel('Firing Rate')

figure(1); clf
subplot(1,2,1)
I0 = squeeze(Robs(ind,cc,:));
I0(12,:) = [];
h = imagesc(I0);
title('Data')

subplot(1,2,2)
I1 = squeeze(Ypred(ind,cc,:));
I1(12,:) = [];
imagesc(I1,[0 .3])
colormap parula
title('Model')

figure(2); clf
plot(nanmean(I0)); hold on
plot(nanmean(I1)); hold on
title(cc)
%%

cc = mod(cc + 1, NC); cc = max(cc, 1);
% cc = 20;

figure(1); clf
set(gcf, 'Color', 'w')

subplot(4,1,1:3)
[i, j] = find(spks(ind,cc,:));
plot.raster(lags(j), i, 1); hold on
plot([0 0], ylim, 'r')
plot(n(ind)/Exp.S.frameRate, 1:numel(ind), 'g')
xlim(lags([1 end]))
ylabel('Trial')

title(cc)
subplot(4,1,4)
m = squeeze(mean(spks(:,cc,:),1))/binsize;
plot(lags, m, 'k')
xlim(lags([1 end]))
xlabel('Time from fixation onset')
ylabel('Firing Rate')

%% plot eye position aligned to fixation onset

% image of x and y position to see the pattern
figure(2); clf
subplot(1,2,1)
imagesc(lags, 1:nt, xpos(ind,:), [-.5 .5]); axis xy
xlabel('Time from fixation onset')
ylabel('Fixation #')
title('X position')

subplot(1,2,2)
imagesc(ypos(ind,:), [-.5 .5]); axis xy
xlabel('Time from fixation onset')
ylabel('Fixation #')
title('Y position')

% plot the individual traces from each trial
figure(3); clf
subplot(2,1,1)
for i = 1:nt
    iix = lags < (n(i)/Exp.S.frameRate);
    plot(lags(iix), xpos(i,iix)); hold on
end
plot(win, .5*[1 1], 'k--')
plot(win, -.5*[1 1], 'k--')
xlim(win)
ylim([-1 1])
xlabel('Time from fixation onset')
ylabel('X position (d.v.a)')
title('Marmoset "Fixations"')
subplot(2,1,2)
for i = 1:nt
    iix = lags < (n(i)/Exp.S.frameRate);
    plot(lags(iix), ypos(i,iix)); hold on
end
plot(win, .5*[1 1], 'k--')
plot(win, -.5*[1 1], 'k--')

xlim(win)
ylim([-1 1])
xlabel('Time from fixation onset')
ylabel('Y position (d.v.a)')

%% plot eye velocity raster
% looks like there are TONS of microsaccades, but they happen at similar
% times. Plot eye velocity
clf
spd = hypot(diff(xpos, [], 2), diff(ypos, [], 2));
imagesc(lags, 1:nt, spd(ind,:), [0 .15]); hold on
plot(n(ind)/Exp.S.frameRate, 1:numel(ind), 'g')
axis xy
xlabel('Time from fixation onset')
ylabel('Fixation #')
title('Eye speed')

%% plot example eye trace
% plot the individual traces from each trial
figure(3); clf
% subplot(2,1,1)
iTrial = 15; %mod(iTrial + 1,nt); iTrial = max(iTrial, 1);
iix = lags > 0 & lags < (n(iTrial)/Exp.S.frameRate);
% smoothfun = @(x) sgolayfilt(x, 2, 13);
% smoothfun = @(x) imboxfilt(x, 9);
smoothfun = @(x) imgaussfilt(x, 3);
cmap = lines;
plot(lags(iix), 5+60*smoothfun(xpos(iTrial,iix)), 'Color',cmap(1,:), 'Linewidth', 1); hold on
plot(lags(iix), 50*smoothfun(ypos(iTrial,iix)), 'Color',cmap(5,:), 'Linewidth', 1);

plot(win, 30*[1 1], 'k--')
plot(win, -30*[1 1], 'k--')
xlim([0 1.6])
ylim([-20 15])
xlabel('Time from fixation onset')
ylabel('X position (arcmin)')
% title(iTrial)
plot.fixfigure(gcf, 12, [2.5 2])
set(gca, 'YTick', -20:10:20)
legend({'X position', 'Y position'}, 'Box', 'off')
saveas(gcf, 'Figures/example/eyeTrace.pdf')


%% plot stimulus
% stim = load('Data/L20191231_FixRsvpStim.mat');
stim = load('Data/L20191231_BackImage.mat');
% stim = load('Data/L20191231_Gabor.mat');
%%
iix = 4910:5195; %back image
% iix = 9700:9900;
stim.valdata = stim.valdata(iix);
stim.labels = stim.labels(iix);
stim.probeDist = stim.probeDist(iix);
stim.frameTimes = stim.frameTimes(iix);
stim.eyeAtFrame = stim.eyeAtFrame(iix,:);
stim.stim = stim.stim(iix,:);
stim.Robs = stim.Robs(iix,:);

%%
dim = stim.NX*[1 1];
NT = numel(stim.frameTimes);
X = reshape(stim.stim, [NT, dim]);
xax = (1:dim(1))*1.5;
yax = xax;
[xx,tt,yy] = meshgrid(xax, stim.frameTimes, yax);

%%


%%
% [xx,yy] = meshgrid(xax, yax);
% I = reshape(a, [dims nlags]);
% I = permute(I, [3 2 1]);
% iTrial = mod(iTrial + 1, numel(n)); iTrial = max(iTrial,1);
% times = tstart(iTrial)+(0.1:.008:1);
% iix = find(stim.frameTimes > times(1) & stim.frameTimes < times(end));
% iix = iix(:)';
iix = 1:ceil(numel(stim.frameTimes)/3);
figure(2); clf
set(gcf, 'Color', 'w')

subplot(3,1,2)
h = slice(xx,tt,yy,X,[],stim.frameTimes(iix(1:2:end)),[]); hold on


for i = iix(1:2:end)
plot3(xax, stim.frameTimes(i)*ones(size(xax)), yax(end)*ones(size(xax)), 'k')
plot3(xax, stim.frameTimes(i)*ones(size(xax)), yax(1)*ones(size(xax)), 'k')
plot3(yax(1)*ones(size(xax)), stim.frameTimes(i)*ones(size(xax)), yax, 'k')
plot3(yax(end)*ones(size(xax)), stim.frameTimes(i)*ones(size(xax)), yax, 'k')
end

set(gca, 'CLim', 200*[-1 1])
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
%     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end

view(79,11)
colormap gray
xlim([-20 60])
view(86,33)
axis off

subplot(3,1,1)


% smoothfun = @(x) imgaussfilt(x, 1);
% plot3(0*stim.eyeAtFrame(:,2), stim.frameTimes, stim.eyeAtFrame(:,1)); hold on
% plot3(0*stim.eyeAtFrame(:,2), stim.frameTimes, stim.eyeAtFrame(:,2))
% % plot(lags(iix), 5+60*smoothfun(xpos(iTrial,iix)), 'Color',cmap(1,:), 'Linewidth', 1); hold on
% view(79,11)
% view(86,33)
% plot(lags(iix), 50*smoothfun(ypos(iTrial,iix)), 'Color',cmap(5,:), 'Linewidth', 1);
t0 = stim.frameTimes(iix(1));
plot(stim.frameTimes(iix)- t0, detrend(stim.eyeAtFrame(iix,1), 'constant'), 'o-', 'MarkerSize', 2); hold on
plot(stim.frameTimes(iix)-t0, detrend(stim.eyeAtFrame(iix,2), 'constant'), 'o-', 'MarkerSize', 2); hold on
axis tight
set(gca, 'Box', 'off')

subplot(3,1,3)
% plot3(0*stim.frameTimes(iix),stim.frameTimes(iix)-t0, bsxfun(@plus, stim.Robs(iix,:), 1:size(stim.Robs,2)), 'k') 
plot(stim.frameTimes(iix)-t0, bsxfun(@plus, stim.Robs(iix,:), 1:size(stim.Robs,2)), 'k') 
set(gca, 'Box', 'off')
% view(86,33)
axis tight

%%
plot.fixfigure(gcf, 10, [6 5], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', 'sampleStim.png'))

%%



