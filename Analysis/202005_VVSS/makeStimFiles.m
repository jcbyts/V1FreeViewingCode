
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);


%% load data

close all
sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'kilo', 'cleanup_spikes', 1);
% eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);

eyePosOrig = Exp.vpx.smo(:,2:3);
%% check if the eye correction exists on the server
server_string = 'jcbyts@sigurros';
output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
importname = strrep(Exp.FileTag, '.mat', '_eyetraces.mat');
command = ['scp ' server_string ':' output_dir importname ' ' data_dir];

try 
    system(command)
    tmp = load(fullfile(data_dir, importname));
    ft0 = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    [ft, ia] = unique(tmp.frameTime);

    eyeXold = Exp.vpx.smo(:,2);
    eyeXnew = interp1(ft, tmp.eyeShift(ia,1), ft0, 'nearest')/Exp.S.pixPerDeg + eyeXold;
    eyeYold = Exp.vpx.smo(:,3);
    eyeYnew = interp1(ft, tmp.eyeShift(ia,2), ft0, 'nearest')/Exp.S.pixPerDeg + eyeYold;

    Exp.vpx.smo(:,2) = eyeXnew;
    Exp.vpx.smo(:,3) = eyeYnew;
catch 
    fprintf('could not find %s on server\n', importname)
end
%%
figure; plot(eyeXnew); hold on; plot(eyePosOrig(:,1))

%% get visually driven units

spkS = io.get_visual_units(Exp);

%% plot spatial RFs to try to select a ROI
unit_mask = 0;
NC = numel(spkS);
hasrf = find(~isnan(arrayfun(@(x) x.x0, spkS)));
figure(2); clf
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

xax = spkS(1).xax;
yax = spkS(1).yax;
cc = lines(NC);
for cc = hasrf(:)'
    subplot(sx,sy,cc,'align')
    rf = abs(spkS(cc).unit_mask)/max(abs(spkS(cc).unit_mask(:)));
    unit_mask = unit_mask + rf;
    imagesc(xax, yax, spkS(cc).unit_mask)
    xlim([0 40])
    ylim([-40 0])
%     [~, h] = contour(xax, yax, rf, [.75:.05:1], 'Color', cmap(cc,:)); hold on
end

figure(1); clf

imagesc(xax, yax,unit_mask); axis xy
[i,j] = find(unit_mask>NC);
[min(xax(j)) max(xax(j))]


%% regenerate data with the following parameters

close all

S.rect = [-20 -60 40 10];
% pixels run down (where do we want to enforce this?)
S.rect([2 4]) = sort(-S.rect([2 4]));
fname = {};

eyesmoothing = 9;
t_downsample = 2;
s_downsample = 2;

% Gabor reverse correlation
stimset = 'Gabor';
options = {'stimulus', stimset, ...
    'testmode', false, ...
    'eyesmooth', eyesmoothing, ... % bins
    't_downsample', t_downsample, ...
    's_downsample', s_downsample, ...
    'includeProbe', true, ...
    'correctEyePos', false, ...
    'nonlinearEyeCorrection', false, ...
    'overwrite', true};

fname{1} = io.dataGenerate(Exp, S, options{:});

%% grating reverse correlation
stimset = 'Grating';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

fname{2} = io.dataGenerate(Exp, S, options{:});

%%

% RSVP images
stimset = 'FixRsvpStim';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

fname{3} = io.dataGenerate(Exp, S, options{:});

% Static Natural images
stimset = 'BackImage';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

fname{4} = io.dataGenerate(Exp, S, options{:});


%% test that it worked

load(fullfile('Data', fname{1}))
iFrame = 1;

%% show sample frame
NY = size(stim,2)/NX;

iFrame = iFrame + 1;
I = reshape(stim(iFrame,:), [NX,NY]);
figure(1); clf
imagesc(I)
% 
% 
%  imagesc( (I - imgaussfilt(I,2)).^2)


%% get STAs to check that you have the right rect

stim = zscore(stim);

% only take central eye positions
ecc = hypot(eyeAtFrame(:,1)-Exp.S.centerPix(1), eyeAtFrame(:,2)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc > 3.1 | labels ~= 1;

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
xax = (1:NX)/Exp.S.pixPerDeg*60;
yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(1,nlags, ilag, 'align')
   imagesc(xax, yax, reshape(sta(ilag,:), [NX NY]), [0 1])
end
title(cc)

%% copy to server
server_string = 'jcbyts@sigurros';
output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
for iFile = 4%1:numel(fname)
    command = [command fullfile(data_dir, fname{iFile}) ' '];
end

command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname{:})
