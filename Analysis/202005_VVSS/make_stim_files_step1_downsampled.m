%%
RoiS = struct();
RoiS.('logan_20200304') = [-20 -60 50 10];
RoiS.('logan_20200306') = [-20 -60 50 10];
RoiS.('logan_20191231') = [-20 -60 50 10];
%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);


%% load data

close all
sessId = 57;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'jrclustwf', 'cleanup_spikes', 0);

eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);




%% get visually driven units

spkS = io.get_visual_units(Exp, 'plotit', true);

%% plot spatial RFs to try to select a ROI
unit_mask = 0;
NC = numel(spkS);
hasrf = find(~isnan(arrayfun(@(x) x.x0, spkS)));
figure(2); clf
set(gcf, 'Color', 'w')
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

xax = spkS(1).xax/Exp.S.pixPerDeg;
yax = spkS(1).yax/Exp.S.pixPerDeg;
cc = lines(NC);
for cc = 1:NC %hasrf(:)'
    subplot(sx,sy,cc,'align')
    rf = abs(spkS(cc).unit_mask)/max(abs(spkS(cc).unit_mask(:)));
    
    I = spkS(cc).srf;
    I = I - mean(I);
    I = I / max(abs(I(:)));
    if isnan(sum(I(:)))
        I = zeros(size(I));
    else
        unit_mask = unit_mask + rf;
    end
    imagesc(xax, yax, I, [-1 1]); hold on
    colormap parula
    axis xy
    plot([0 0], ylim, 'w')
    plot(xlim,[0 0], 'w')
%     xlim([0 40])
%     ylim([-40 0])
%     [~, h] = contour(xax, yax, rf, [.75:.05:1], 'Color', cmap(cc,:)); hold on
end

figure(1); clf
ppd = Exp.S.pixPerDeg;
imagesc(xax*ppd, yax*ppd,unit_mask); axis xy
[i,j] = find(unit_mask>NC);
[min(xax(j)) max(xax(j))]


%% regenerate data with the following parameters

close all

S.rect = RoiS.(strrep(Exp.FileTag, '.mat', ''));
% pixels run down (where do we want to enforce this?)
S.rect([2 4]) = sort(-S.rect([2 4]));
fname = {};

eyesmoothing = 19;
t_downsample = 2;
s_downsample = 2;

% Gabor reverse correlation
stimset = 'Gabor';
options = {'stimulus', stimset, ...
    'debug', false, ...
    'testmode', false, ...
    'eyesmooth', eyesmoothing, ... % bins
    't_downsample', t_downsample, ...
    's_downsample', s_downsample, ...
    'includeProbe', true, ...
    'correctEyePos', false, ...
    'nonlinearEyeCorrection', false, ...
    'usePTBdraw', true, ...
    'overwrite', true};

%%
options{find(strcmp(options, 'usePTBdraw')) + 1} = true;
fname{1} = io.dataGenerate(Exp, S, options{:});

% options{find(strcmp(options, 'usePTBdraw')) + 1} = false;
% options{find(strcmp(options, 'overwrite')) + 1} = true;
% fname{2} = io.dataGenerate(Exp, S, options{:});

%% grating reverse correlation
stimset = 'Grating';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

fname{2} = io.dataGenerate(Exp, S, options{:});

%%

% RSVP images
stimset = 'FixRsvpStim';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

fname{3} = io.dataGenerate(Exp, S, options{:});

%% Static Natural images
stimset = 'BackImage';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

fname{4} = io.dataGenerate(Exp, S, options{:});


%% test that it worked
id = 1;
load(fullfile('Data', fname{id}))
iFrame = 1;

%% show sample frame
NY = size(stim,2)/NX;

iFrame = iFrame + 1;
I = reshape(stim(iFrame,:), [NX,NY]);
figure(id); clf
imagesc(I);% axis xy
colorbar

%% get STAs to check that you have the right rect

stim = zscore(stim);

% only take central eye positions
ecc = hypot(eyeAtFrame(:,1)-Exp.S.centerPix(1), eyeAtFrame(:,2)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc > 5.1 | labels ~= 1;

NC = size(Robs,2);
nlags = 10;
NT = size(stim,1);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(1); clf
stas = zeros(nlags, size(stim,2), NC);

for cc = 1:NC
    rtmp = Robs(:,cc);
    rtmp(ix) = 0; % remove spikes we don't want to analyze
    sta = simpleSTC(stim, rtmp, nlags);
    subplot(sx, sy, cc, 'align')
    plot(sta)
    stas(:,:,cc) = sta;
    drawnow
end

%%
cc = 0;

%% plot one by one
figure(2); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end

sta = stas(:,:,cc);
NY = size(stim,2)/NX;
sta = (sta - min(sta(:))) ./ (max(sta(:)) - min(sta(:)));
x = xax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
y = yax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% xax = (1:NX)/Exp.S.pixPerDeg*60;
% yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(1,nlags, ilag, 'align')
   imagesc(x, y, reshape(sta(ilag,:), [NX NY])', [0 1])
end
colormap gray
title(cc)

%% copy to server
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
for iFile = 1:numel(fname)
    command = [command fullfile(data_dir, fname{iFile}) ' '];
end

command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname{:})
