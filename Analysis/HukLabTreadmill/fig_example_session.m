% Plot a short segment from an example session

%% setup paths
fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');

flist = dir(fullfile(fdir, '*.mat'));
figdir = 'Figures/HuklabTreadmill/manuscript/';
%% figure

id = find(strcmp('brie_20210917_grat.mat', {flist.name}));

D = load(fullfile(flist(id).folder, flist(id).name));

t0 = 665;

figure(1); clf
cids = unique(D.spikeIds);

NC = numel(cids);
mapId = zeros(max(cids),1);
mapId(cids) = 1:NC;


ax = axes('Position', [.2 .1 .7 .8]);
h = plot.raster(D.spikeTimes, mapId(D.spikeIds)*.5, .5, 'k', 'Linewidth', .05);

xd = t0 + [-1 15];
xlim(xd)

hold on
plot(D.frameTimes, sind(D.framePhase).*5.*D.frameContrast + NC*.7 -2, 'k')

scale = 2;
x = D.eyePos(:,1)/scale;
y = D.eyePos(:,2)/scale;
offset = -10;
plot(D.eyeTime, offset + x, 'Color', 'k')
plot(D.eyeTime, offset + y, 'Color', .5*[1 1 1])
x(D.eyeLabels==1) = nan;
y(D.eyeLabels==1) = nan;
plot(D.eyeTime, offset + x, 'Color', 'r')
plot(D.eyeTime, offset + y, 'Color', 'r')
plot(t0*[1 1], offset + [0 10]/scale, 'k', 'Linewidth', 2 )
% pupil
offset = -30;
p =  D.eyePos(:,3);
p = (p - nanmean(p))/ nanstd(p);
plot(D.eyeTime, offset + 5*p, 'k')

% running
offset = -50;

treadSpeed = D.treadSpeed;
treadSpeed(isnan(treadSpeed)) = 0;
treadSpeed = lowpass(treadSpeed, .5);
plot(D.treadTime, offset + treadSpeed/2, 'k')

plot(t0*[1 1], offset + [0 10]/2, 'k', 'Linewidth', 2)

plot(t0 + [0 5], -60*[1 1], 'k', 'Linewidth', 5)
axis off

%% save figure
plot.fixfigure(gcf, 12, [8 8])
title(strrep(flist(id).name, '_', ' '))
saveas(gcf, fullfile(figdir, sprintf('session_%d.pdf', id)) )


