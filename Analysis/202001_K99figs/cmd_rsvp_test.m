% We have a RSVP fixation point that flashes processed natural images in
% the central 

%% Load dataset

[Exp, S] = io.dataFactoryGratingSubspace(56, 'spike_sorting', 'jrclustwf', 'cleanup_spikes', 0);



%% get valid trials
validTrials = io.getValidTrials(Exp, 'FixRsvpStim');

%% bin spikes and eye pos
binsize = 1e-3; % 1 ms bins for rasters
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

bad = n < 100;

validTrials(bad) = [];

tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1), Exp.D(validTrials)));
tend = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(end,1), Exp.D(validTrials)));
n(bad) = [];

% sort trials by fixation duration
[~, ind] = sort(n, 'descend');

% bin spike times at specific lags
lags = win(1):binsize:win(2);
nlags = numel(lags);
nt = numel(tstart);
S.cids = Exp.osp.cids;
NC = numel(S.cids);

spks = zeros(nt,NC,nlags);
xpos = zeros(nt,nlags);
ypos = zeros(nt,nlags);

% Do the binning here
for i = 1:nt
    y = binNeuronSpikeTimesFast(Exp.osp,tstart(i)+lags, binsize);
    spks(i,:,:) = full(y(:,S.cids))';
    % resample eye position at the time resolution of the spike trains
    xpos(i,:) = interp1(eyeTime, eyeX, tstart(i)+lags, eyePosInterpolationMethod);
    ypos(i,:) = interp1(eyeTime, eyeY, tstart(i)+lags, eyePosInterpolationMethod);
end

fprintf('Done\n')
% initialize iterator for plotting cells
cc = 19;
%%

NC = numel(Exp.osp.cids);
cc = mod(cc + 1, NC); cc = max(cc, 1);
% cc = 37;

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

%%

figure(4); clf
i = 1;
nt = size(xpos,1);
for j = 1:nt
    h(j) = plot(xpos(j,i:i+5), ypos(j,i:i+5), '-'); hold on
end
xlim([-1 1])
ylim([-1 1])
for i = 1:size(xpos,2)-5
    for j = 1:nt
       h(j).XData = xpos(j,i:i+5);
       h(j).YData = ypos(j,i:i+5);
    end
        
    pause(0.02)
   
end

