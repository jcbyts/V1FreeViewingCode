
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

%% load data

sessId = 'L20191205';
[Exp, S] = io.dataFactory(sessId);

%% Example 1: regenerate stimulus for gabors
options = {'stimulus', 'Gabor', ...
    'testmode', 20, ...
    'eyesmooth', 3, ... % bins
    't_downsample', 2, ...
    's_downsample', 2, ...
    'includeProbe', false};
  

fname = io.dataGenerate(Exp, S, options{:});

%% --- see that it worked
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
load(fullfile(dataDir, fname))
whos

% plot saccade triggered average
[an, ~, widx] = eventTriggeredAverage(mean(Robs,2), slist(:,1), [-1 1]*ceil(.2/dt));
figure(1); clf
plot(widx * dt, an / dt)
xlabel('Time from saccade onset')
ylabel('Firing Rate')

figname = strrep(fname, '.mat', '');
fontSize = 10;
Dimensions = [5 3]; % inches

plot.fixfigure(gcf, fontSize, Dimensions)
saveas(gcf, fullfile('Figures', 'example', [figname '_saccadeAverage.pdf']))

% STA on a subset of units
figure(2); clf
ix = valdata == 1 & labels == 1 & probeDist > 50;
for k = 1:min(25, size(Robs,2))
    sta = simpleSTC((stim(ix,:)), Robs(ix,k), ceil(.1/dt) );

    subplot(5, 5, k, 'align')
    imagesc(sta)
    set(gca, 'XTickLabel', '', 'YTickLabel', '')
end

plot.fixfigure(gcf, 10, [10 10], 'OffsetAxes', false)

hx = plot.suplabel('space (x by y)', 'x'); set(hx, 'FontSize', 14)
hy = plot.suplabel('time (runs up)', 'y');  set(hy, 'FontSize', 14)
ht = plot.suplabel('STA in on raw pixels (Gabors)', 't');  set(ht, 'FontSize', 14)

saveas(gcf, fullfile('Figures', 'example', [figname '_staRaw.pdf']))

% STA with squared pixel values
figure(2); clf
ix = valdata == 1 & labels == 1 & probeDist > 50;
for k = 1:min(25, size(Robs,2))
    sta = simpleSTC((stim(ix,:)).^2, Robs(ix,k), ceil(.1/dt) );

    subplot(5, 5, k, 'align')
    imagesc(sta)
    set(gca, 'XTickLabel', '', 'YTickLabel', '')
    
end

plot.fixfigure(gcf, 10, [10 10], 'OffsetAxes', false)

hx = plot.suplabel('space (x by y)', 'x'); set(hx, 'FontSize', 14)
hy = plot.suplabel('time (runs up)', 'y');  set(hy, 'FontSize', 14)
ht = plot.suplabel('STA in on squared pixels (Gabors)', 't');  set(ht, 'FontSize', 14)

saveas(gcf, fullfile('Figures', 'example', [figname '_staQuad.pdf']))


%% Example 2: regenerate stimulus for Gratings (Hartley, but not really)
options = {'stimulus', 'Grating', ...
    'testmode', 5, ...
    'eyesmooth', 3, ... % bins
    'fft', true, ... % convert to fourier domain
    't_downsample', 2, ...
    's_downsample', 2, ...
    'includeProbe', false};
  

fname = io.dataGenerate(Exp, S, options{:});

%% --- see that it worked
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
load(fullfile(dataDir, fname))
whos


figname = strrep(fname, '.mat', '');
fontSize = 10;
Dimensions = [5 3]; % inches


% STA on a subset of units
figure(2); clf
ix = valdata == 1 & labels == 1 & probeDist > 50;
for k = 1:min(25, size(Robs,2))
    sta = simpleRevcorr(stim(ix,:), Robs(ix,k)-mean(Robs(ix,k)), ceil(.1/dt) );

    subplot(5, 5, k, 'align')
    imagesc(sta)
end
colormap gray


plot.fixfigure(gcf, 10, [10 10], 'OffsetAxes', false)

hx = plot.suplabel('space (x by y)', 'x'); set(hx, 'FontSize', 14)
hy = plot.suplabel('time (runs up)', 'y');  set(hy, 'FontSize', 14)
ht = plot.suplabel('STA in fourier domain (Gratings)', 't');  set(ht, 'FontSize', 14)

saveas(gcf, fullfile('Figures', 'example', [figname '_staRaw.pdf']))