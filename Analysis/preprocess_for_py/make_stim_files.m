%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);

%% ROIs for each session

RoiS = struct();
RoiS.('logan_20200304') = [-20 -60 50 10];
RoiS.('logan_20200306') = [-20 -60 50 10];
RoiS.('logan_20191231') = [-20 -60 50 10];

%% load data

close all
sessId = 'logan_20200304';
spike_sorting = 'kilowf';
[Exp, S] = io.dataFactory(sessId, 'spike_sorting', spike_sorting, 'cleanup_spikes', 0);

eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);


%% get visually driven units
[spkS, W] = io.get_visual_units(Exp, 'plotit', true);

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

eyesmoothing = 19;

%% Gabor reverse correlation
stimset = 'Gabor';
options = {'stimulus', stimset, ...
    'debug', false, ...
    'testmode', true, ...
    'eyesmooth', eyesmoothing, ... % bins
    'includeProbe', true, ...
    'correctEyePos', false, ...
    'nonlinearEyeCorrection', false, ...
    'usePTBdraw', true, ...
    'overwrite', false};

options{find(strcmp(options, 'testmode')) + 1} = true;
fname = io.dataGenerateHdf5(Exp, S, options{:});
options{find(strcmp(options, 'testmode')) + 1} = false;
fname = io.dataGenerateHdf5(Exp, S, options{:});

%% grating reverse correlation
stimset = 'Grating';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

options{find(strcmp(options, 'testmode')) + 1} = true;
io.dataGenerateHdf5(Exp, S, options{:});
options{find(strcmp(options, 'testmode')) + 1} = false;
io.dataGenerateHdf5(Exp, S, options{:});

%%

% RSVP images
stimset = 'FixRsvpStim';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

options{find(strcmp(options, 'testmode')) + 1} = true;
io.dataGenerateHdf5(Exp, S, options{:});
options{find(strcmp(options, 'testmode')) + 1} = false;
h5name = io.dataGenerateHdf5(Exp, S, options{:});

%% Static Natural images
stimset = 'BackImage';
options{find(strcmp(options, 'stimulus')) + 1} = stimset; % change the stimulus set

options{find(strcmp(options, 'testmode')) + 1} = true;
io.dataGenerateHdf5(Exp, S, options{:});
options{find(strcmp(options, 'testmode')) + 1} = false;
% options{find(strcmp(options, 'overwrite')) + 1} = false;
io.dataGenerateHdf5(Exp, S, options{:});

%% Add Neuron meta data
h5create(fname,'/Neurons/cgs', size(Exp.osp.cgs))
h5write(fname, '/Neurons/cgs', Exp.osp.cgs)
% TODO: include CSD depths / burstiness index / different spike sorting

%% add spike sorting

goodspikes = ismember(Exp.osp.clu, Exp.osp.cids);

h5create(fname, ['/Neurons/' spike_sorting '/cids'], size(Exp.osp.cids))
h5write(fname, ['/Neurons/' spike_sorting '/cids'], Exp.osp.cids)

h5create(fname, ['/Neurons/' spike_sorting '/times'], size(Exp.osp.st(goodspikes)))
h5write(fname, ['/Neurons/' spike_sorting '/times'], Exp.osp.st(goodspikes))

h5create(fname, ['/Neurons/' spike_sorting '/cluster'], size(Exp.osp.clu(goodspikes)))
h5write(fname, ['/Neurons/' spike_sorting '/cluster'], Exp.osp.clu(goodspikes))

h5create(fname, ['/Neurons/' spike_sorting '/cgs'], size(Exp.osp.cgs))
h5write(fname, ['/Neurons/' spike_sorting '/cgs'], Exp.osp.cgs)

h5create(fname, ['/Neurons/' spike_sorting '/peakval'], size([W.peakval]))
h5write(fname, ['/Neurons/' spike_sorting '/peakval'], [W.peakval])

h5create(fname, ['/Neurons/' spike_sorting '/troughval'], size([W.troughval]))
h5write(fname, ['/Neurons/' spike_sorting '/troughval'], [W.troughval])

exci = reshape([W.ExtremityCiRatio], 2, []);
h5create(fname, ['/Neurons/' spike_sorting '/ciratio'], size(exci))
h5write(fname, ['/Neurons/' spike_sorting '/ciratio'], exci)

%% test that it worked
id = 1;
stim = 'Gabor';
set = 'Train';

sz = h5readatt(fname, ['/' stim '/' set '/Stim'], 'size');

iFrame = 1;

%% show sample frame
iFrame = iFrame + 1;
I = h5read(fname, ['/' stim '/' set '/Stim'], [iFrame, 1,1], [1 sz(1:2)']);
I = squeeze(I);
% I = h5read(fname{1}, ['/' stim '/' set '/Stim'], [1,1,iFrame], [sz(1:2)' 1]);
figure(id); clf
subplot(1,2,1)
imagesc(I)
subplot(1,2,2)
imagesc(I(1:2:end,1:2:end));% axis xy
colorbar
colormap gray


%% get STAs to check that you have the right rect

Stim = h5read(fname, ['/' stim '/' set '/Stim']);
Robs = h5read(fname, ['/' stim '/' set '/Robs']);
ftoe = h5read(fname, ['/' stim '/' set '/frameTimesOe']);
frate = h5readatt(fname, ['/' stim '/' set '/Stim'], 'frate');
st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe, 1/frate);

% Robs = 
eyeAtFrame = h5read(fname, ['/' stim '/' set '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' set '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);
%%

Stim = reshape(Stim, size(Stim, 1), NX*NY);

% reshape(Stim, 
Stim = zscore(single(Stim));

%% forward correlation
NC = size(Robs,2);
nlags = 20;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;

stas = zeros(nlags, nstim, NC);
for idim = 1:nstim
    fprintf('%d/%d\n', idim, nstim)
    Xstim = conv2(Stim(:,idim), eye(nlags), 'full');
    Xstim = Xstim(1:end-nlags+1,:);
    stas(:, idim, :) = Xstim(ix,:)'*Rdelta(ix,:);
end

% %%
% % only take central eye positions
% ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% ix = ecc > 5.1 | labels ~= 1;
% 
% NC = size(Robs,2);
% nlags = 10;
% NT = size(Stim,1);
% sx = ceil(sqrt(NC));
% sy = round(sqrt(NC));
% 
% figure(1); clf
% stas = zeros(nlags, size(Stim,2), NC);
% 
% for cc = 1:NC
%     if sum(Robs(:,cc))==0
%         continue
%     end
%     rtmp = Robs(:,cc);
%     rtmp(ix) = 0; % remove spikes we don't want to analyze
%     rtmp = rtmp - mean(rtmp);
%     sta = simpleRevcorr(Stim, rtmp, nlags);
%     subplot(sx, sy, cc, 'align')
%     plot(sta)
%     stas(:,:,cc) = sta;
%     drawnow
% end
% 
% %%
% % clearvars -except Exp Stim Robs
%%
cc = 0;

%% plot one by one
figure(2); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end

sta = stas(:,:,cc);
% NY = size(Stim,2)/NX;
sta = (sta - min(sta(:))) ./ (max(sta(:)) - min(sta(:)));
% x = xax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% y = yax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% xax = (1:NX)/Exp.S.pixPerDeg*60;
% yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(1,nlags, ilag, 'align')
   imagesc(reshape(sta(ilag,:), [NY NX])', [0 1])
end
colormap parula
title(cc)

%% manual update

h5writeatt(fname, '/BackImage/Train/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/BackImage/Train/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/BackImage/Train/Stim', 'viewdist', Exp.S.screenDistance)

h5writeatt(fname, '/BackImage/Test/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/BackImage/Test/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/BackImage/Test/Stim', 'viewdist', Exp.S.screenDistance)

h5writeatt(fname, '/FixRsvpStim/Train/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/FixRsvpStim/Train/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/FixRsvpStim/Train/Stim', 'viewdist', Exp.S.screenDistance)

h5writeatt(fname, '/FixRsvpStim/Test/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/FixRsvpStim/Test/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/FixRsvpStim/Test/Stim', 'viewdist', Exp.S.screenDistance)

h5writeatt(fname, '/Gabor/Train/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/Gabor/Train/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/Gabor/Train/Stim', 'viewdist', Exp.S.screenDistance)

h5writeatt(fname, '/Gabor/Test/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/Gabor/Test/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/Gabor/Test/Stim', 'viewdist', Exp.S.screenDistance)

h5writeatt(fname, '/Grating/Train/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/Grating/Train/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/Grating/Train/Stim', 'viewdist', Exp.S.screenDistance)

h5writeatt(fname, '/Grating/Test/Stim', 'frate', Exp.S.frameRate)
h5writeatt(fname, '/Grating/Test/Stim', 'center', Exp.S.centerPix)
h5writeatt(fname, '/Grating/Test/Stim', 'viewdist', Exp.S.screenDistance)


%% copy to server
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command fname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)
