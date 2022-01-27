
%%
fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');

flist = dir(fdir);
figdir = 'Figures/HuklabTreadmill/sfn/';
%%

% id = find(strcmp('brie_20210917_grat.mat', {flist.name}));
% 

%%
id = 38;
D = load(fullfile(fdir, flist(id).name));


figure(1); clf
cids = unique(D.spikeIds);


NC = numel(cids);
mapId = zeros(max(cids),1);
mapId(cids) = 1:NC;


ax = axes('Position', [.2 .1 .7 .8]);
plot.raster(D.spikeTimes, mapId(D.spikeIds)*.5, .5)
t0 = 482;
xlim(t0 + [-1 10])

hold on
plot(D.frameTimes, sind(D.framePhase).*5.*D.frameContrast + NC -2, 'k')

x = D.eyePos(:,1)/2;
y = D.eyePos(:,2)/2;
offset = -10;
plot(D.eyeTime, offset + x, 'Color', 'k')
plot(D.eyeTime, offset + y, 'Color', .5*[1 1 1])
x(D.eyeLabels==1) = nan;
y(D.eyeLabels==1) = nan;
plot(D.eyeTime, offset + x, 'Color', 'r')
plot(D.eyeTime, offset + y, 'Color', 'r')

% pupil
offset = -30;
p =  D.eyePos(:,3);
p = (p - nanmean(p))/ nanstd(p);
plot(D.eyeTime, offset + 5*p, 'k')

% running
offset = -50;
plot(D.treadTime, offset + D.treadSpeed/3, 'k')

plot(t0 + [0 5], -60*[1 1], 'k', 'Linewidth', 5)
axis off

plot.fixfigure(gcf, 12, [8 8])
saveas(gcf, fullfile(figdir, sprintf('session_%d.pdf', id)) )


