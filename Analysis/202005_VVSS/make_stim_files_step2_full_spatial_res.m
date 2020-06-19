
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);


%% load data

close all
sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'kilo', 'cleanup_spikes', 1);

eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);

Exp.vpx.smo(:,2) = sgolayfilt(Exp.vpx.smo(:,2), 1, 9);
Exp.vpx.smo(:,3) = sgolayfilt(Exp.vpx.smo(:,3), 1, 9);

eyePosCorrSmo = Exp.vpx.smo(:,2:3);

%%
data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
importname = strrep(Exp.FileTag, '.mat', '_eyetraces.mat');
server_string = 'jcbyts@sigurros';
output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

fname = fullfile(data_dir, importname);
if ~exist(fname, 'file')
    command = ['scp ' server_string ':' output_dir importname ' ' data_dir];
    system(command)
end
    
tmp = load(fname); % load the eye correction file

ft0 = Exp.vpx2ephys(Exp.vpx.smo(:,1)); % original eye-tracker time (in ephys seconds)
[ft, ia] = unique(tmp.frameTime); % don't double-count filestarts
    
% find the shift amout
[~, ~, ind] = histcounts(ft, ft0); % index frame times into eyetracker time

eaf = tmp.eyeAtFrame(ia,:)/Exp.S.pixPerDeg; % convert to d.v.a
eaf(:,2) = -eaf(:,2); % flip y (from pixels to d.v.a)
rem = ind==0; % remove non-matched frame times
eaf(rem,:) = [];
ind(rem) = [];

% find shift by subtracting the current eye estimate from the corrected eye
% at frame
xshift = eaf(:,1) - Exp.vpx.smo(ind,2);
yshift = eaf(:,2) - Exp.vpx.smo(ind,3);

% don't shift between gaps in time
gaps = find(diff(ft) > .1);
xshift(gaps) = nan;
yshift(gaps) = nan;

% don't recenter the ROI
xshift = xshift - nanmean(xshift);
yshift = yshift - nanmean(xshift); % i think this is reasonable. don't recenter things
    
% shift eye pos
eyeXold = Exp.vpx.smo(:,2);
eyeYold = Exp.vpx.smo(:,3);

% deprecated: we no longer use saved shift values (this is not robust.
% recalculate shift amount instead)
eyeXnew2 = interp1(ft, tmp.eyeShift(ia,1), ft0, 'previous')/Exp.S.pixPerDeg + eyeXold;
eyeYnew2 = interp1(ft, tmp.eyeShift(ia,2), ft0, 'previous')/Exp.S.pixPerDeg + eyeYold;

ft(rem) = [];

eyeXnew = interp1(ft, xshift, ft0, 'previous') + eyeXold;
eyeYnew = interp1(ft, yshift, ft0, 'previous') + eyeYold;

ix = ~(isnan(eyeXnew) | isnan(eyeYnew));

% correct absolute offset ( we don't want to shift the center of where we
% think the monkey is looking --> there's probably a better way to do this)
yoffset = mean(eyeYnew(ix))-nanmean(eyePosCorrSmo(ix,2));
xoffset = mean(eyeXnew(ix))-nanmean(eyePosCorrSmo(ix,1));

Exp.vpx.smo(ix,2) = eyeXnew(ix) - xoffset;
Exp.vpx.smo(ix,3) = eyeYnew(ix) - yoffset;

%%

figure(1); clf
plot(ft0, eyePosOrig(:,1)); hold on
plot(ft, eaf(:,1), '.')
plot(ft0, eyeXnew)

figure(2); clf
plot(ft0, eyePosOrig(:,2)); hold on
plot(ft, eaf(:,2), '.')
plot(ft0, eyeYnew)

%% confirm monkey is looking centrally
validTrials = io.getValidTrials(Exp, 'FixRsvpStim');
C = 0;
binEdges = -2:.05:2;

for iTrial = 1:numel(validTrials)
    tstart = Exp.ptb2Ephys(Exp.D{validTrials(iTrial)}.STARTCLOCKTIME);
    tstop =  Exp.ptb2Ephys(Exp.D{validTrials(iTrial)}.ENDCLOCKTIME);

    tt = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    ix = tt > tstart & tt < tstop;
    ex = Exp.vpx.smo(ix,2);
    ey = Exp.vpx.smo(ix,3);
    tt = tt(ix) - tstart;

    C_ = histcounts2(ex, ey, binEdges, binEdges);
    C = C + C_;
end

figure(1); clf
imagesc(binEdges, binEdges, C')


% figure(1); clf
% plot(tt, ex)
% plot(Exp.D{validTrials(iTrial)}.PR.NoiseHistory(:,3))
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
cmap = lines(NC);
for cc = 1:NC %hasrf(:)'
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

% S.rect = [-20 -60 40 10];
% S.rect = [-10 -60 35 -20];
S.rect = [-10 -40 35 0];
% pixels run down (where do we want to enforce this?)
S.rect([2 4]) = sort(-S.rect([2 4]));
fname = {};

eyesmoothing = 1;
t_downsample = 2;
s_downsample = 1;

% Gabor reverse correlation
stimset = 'Gabor';
options = {'stimulus', stimset, ...
    'testmode', false, ...
    'eyesmooth', eyesmoothing, ... % bins
    't_downsample', t_downsample, ...
    's_downsample', s_downsample, ...
    'binSize', 1, ...
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
stim = stim / std(stim(:));

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

% stim = zscore(stim);


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
    sta = simpleSTC(stim-mean(stim,2), rtmp, nlags);
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
for iFile = 1:numel(fname)
    command = [command fullfile(data_dir, fname{iFile}) ' '];
end

command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname{:})
