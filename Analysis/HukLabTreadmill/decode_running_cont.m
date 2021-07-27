function stat = decode_running_cont(D, sessionId, varargin)
% Decode running continuous
% Some simple running analyses
% stat = decode_running_cont(D, sessionId)



nboot = 100;
runThresh = 3;

sessionId = sessionId + 1; % pick a session id
if sessionId > max(D.sessNumSpikes)
    sessionId = 1;
end

% index into  times from that session
win = [min(D.spikeTimes(D.sessNumSpikes==sessionId)) max(D.spikeTimes(D.sessNumSpikes==sessionId))];

% find treadmill times that correspond to this session 
treadIx = D.treadTime > win(1) & D.treadTime < win(end);
treadTime = D.treadTime(treadIx);
treadSpeed = D.treadSpeed(treadIx);

% get spike rate
spikeIds = unique(D.spikeIds(D.sessNumSpikes==sessionId));
NC = numel(spikeIds);
spikeRate = zeros(numel(treadTime), NC);

bs = diff(treadTime);
for cc = 1:NC
    spikeRate(:,cc) = [0 histcounts(D.spikeTimes(D.spikeIds==spikeIds(cc)), treadTime)./bs'];
end

% spikeRate = spikeRate ./ std(spikeRate); % normalize for visualization


dt = median(diff(treadTime));

% find running epochs
isrunning = treadSpeed > runThresh;
onsets = find(diff(isrunning) == 1);
offsets = find(diff(isrunning) == -1);

if onsets(1) > offsets(1)
    onsets = [1; onsets];
end

if offsets(end) < onsets(end)
    offsets = [offsets; numel(treadSpeed)];
end

assert(numel(onsets)==numel(offsets), "onset offset mismatch")

% plot raw data
figure(1); clf
subplot(2,1,1)
imagesc(treadTime, 1:NC, spikeRate')
xlim(treadTime([1 end]))

subplot(2,1,2)
plot(treadTime, treadSpeed); hold on
for i = 1:numel(onsets)
    fill(treadTime([onsets(i) onsets(i) offsets(i) offsets(i)]), [ylim fliplr(ylim)], 'r', 'FaceColor', 'r', 'FaceAlpha', .5, 'EdgeColor', 'none')
end
xlim(treadTime([1 end]))


runInds = find(isrunning);
statInds = find(~isrunning);
nn = min(numel(runInds), numel(statInds));


muR = zeros(NC, nboot);
muS = zeros(NC, nboot);

for iboot = 1:nboot
    
    rinds = randsample(runInds, nn, true);
    sinds = randsample(statInds, nn, true);
    
    muR(:,iboot) = mean(spikeRate(rinds,:));
    muS(:,iboot) = mean(spikeRate(sinds,:));
end

rateR = prctile(muR, [2.5 50 97.5], 2);
rateS = prctile(muS, [2.5 50 97.5], 2);
rateDiff = prctile(muR-muS, [2.5 50 97.5], 2);
%%

figure(1); clf
subplot(1,2,1)
for cc = 1:NC
    h = plot(rateS(cc,[2 2]), rateR(cc, [1 3])); hold on
    plot(rateS(cc,[1 3]), rateR(cc, [2 2]), 'Color', h.Color);
    plot(rateS(cc,2), rateR(cc, 2), 'o', 'Color', h.Color, 'MarkerFaceColor', h.Color);
end
plot(xlim, xlim, 'k')
xlabel('Stationary')
ylabel('Running')

subplot(1,2,2)
for cc = 1:NC
    h = plot(cc*[1 1], rateDiff(cc,[1 3])); hold on
    plot(cc, rateDiff(cc,2), 'o', 'Color', h.Color)
end




%%
    
    
    
rateR = (spikeRate'*isrunning) ./ sum(isrunning);
rateS = (spikeRate'*~isrunning) ./ sum(~isrunning);


figure(2); clf
plot(rateS, rateR, 'o'); hold on
plot(xlim, xlim, 'k')
xlabel('Stationary Rate')
ylabel('Running Rate')

figure(3); clf
ix = (offsets - onsets) > 50;
win = [-200 200];
lags = win(1):win(2);
nb = 10;
R = filter(ones(nb, 1)/nb, 1, spikeRate);
an = eventTriggeredAverage(R, onsets(ix), win);
an = an - mean(an);
subplot(2,1,1)
imagesc(lags*dt, 1:NC, an')

subplot(2,1,2)
plot(lags*dt, mean(an,2))
xlabel('Lags (sec)')


%%

% %%
% nb = 10;
% R = filter(ones(nb,1)/nb, 1, spikeRate);


figure(1); clf
subplot(2,1,1)
R = spikeRate;
x = mean(R,2);
y = treadSpeed/max(treadSpeed);
n = numel(y);
plot(x); hold on
plot(y)

subplot(2,1,2)
dt = median(diff(treadTime));
ceil(.5/dt)
nlags = 200;

xcnull = zeros(nboot, nlags*2 + 1);
for iboot = 1:nboot
    R = spikeRate((randi(n, [n 1])),:);
    x = mean(R,2);
    xcnull(iboot,:) = xcorr(x, y, nlags, 'Coeff');
end

R = spikeRate;
x = mean(R,2);
[xc, lags] = xcorr(x, y, nlags, 'Coeff');
lags = lags*dt;
xci = prctile(xcnull, [2.5 97.5]);
plot(lags, xci, 'k--'); hold on
plot(lags, xc)

%%
figure(2); clf
l = 10;
plot(x(l:end)/std(x), y(1:end-l+1)/std(y), '.')
