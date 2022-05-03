
%%


% to hack my code, you need to set the SERVER Directory
setpref('FREEVIEWING', 'SERVER_DATA_DIR', '~/Dropbox/MarmoLabWebsite/PSA/DDPI_Before_Processed/')
setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', '~/Dropbox/MarmoLabWebsite/PSA/DDPI_Before_Processed/')


%%
S = struct();
S.processedFileName = 'allen_20220216';
S.rawFilePath = 'Allen_160222_V1_flip64/';
S.spikeSorting = 'kilo';

Exp = io.importFreeViewing(S);

fname = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), S.rawFilePath, [S.processedFileName '.mat']);
save(fname, '-v7.3', '-struct', 'Exp')

%%

trialIdx = io.getValidTrials(Exp, 'Gabor');

iTrial = 1;
Exp.D{trialIdx(iTrial)}.PR

%% get RFs

dotTrials = io.getValidTrials(Exp, 'Dots');
if ~isempty(dotTrials)
    
    BIGROI = [-1 -.5 1 .5]*10;
%     BIGROI = [-5 -5 5 5];
%     eyePos = C.refine_calibration();
    eyePos = Exp.vpx.smo(:,2:3);
%     eyePos(:,1) = -eyePos(:,1);
%     eyePos(:,2) = -eyePos(:,2);
    binSize = .3;
    Frate = 60;
    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
        'eyePosExclusion', 2e3, ...
        'eyePos', eyePos, 'frate', Frate, 'fastBinning', true);
    
    % use indices while fixating
    ecc = hypot(opts.eyePosAtFrame(:,1), opts.eyePosAtFrame(:,2))/Exp.S.pixPerDeg;
    ix = opts.eyeLabel==1 & ecc < 5.2;
    
end


spike_rate = mean(RobsSpace)*Frate;

figure(1); clf
plot.raster(Exp.osp.st, Exp.osp.clu, 1); hold on
ylabel('Unit ID')
xlabel('Time (seconds)')
cmap = lines;
stims = {'Dots', 'BackImage', 'Gabor'};

for istim = 1:numel(stims)
    validTrials = io.getValidTrials(Exp, stims{istim});
    for iTrial = validTrials(:)'
        t0 = Exp.ptb2Ephys(Exp.D{iTrial}.STARTCLOCKTIME);
        t1 = Exp.ptb2Ephys(Exp.D{iTrial}.ENDCLOCKTIME);
        yd = ylim;
        h(istim) = fill([t0 t0 t1 t1], [yd(1) yd(2) yd(2) yd(1)], 'k', 'FaceColor', cmap(istim,:), 'FaceAlpha', .5, 'EdgeColor', 'none');
    end
end

legend(h, stims)

%% threshold units by spike rate
figure(2); clf; set(gcf, 'Color', 'w')
subplot(3,1,1:2)
h = [];
h(1) = stem(spike_rate, '-ok', 'MarkerFaceColor', 'k', 'MarkerSize', 4);

ylabel('Firing Rate During Stimulus')
hold on
goodunits = find(spike_rate > .5);
h(2) = plot(goodunits, spike_rate(goodunits), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 4);
h(3) = plot(xlim, [1 1], 'r-');
legend(h, {'All units', 'Good Units', '1 Spike/Sec'}, 'Location', 'Best')

subplot(3,1,3)
stem(spike_rate, '-ok', 'MarkerFaceColor', 'k', 'MarkerSize', 4)
ylim([0, 1])
xlabel('Unit #')
ylabel('Firing rate of bad units')


fprintf('%d / %d fire enough spikes to analyze\n', numel(goodunits), size(RobsSpace,2))
drawnow

%% do forward correlation

win = [-1 10];
cc = 2;
stas = forwardCorrelation(Xstim, mean(RobsSpace,2), win);
% stas = forwardCorrelation(Xstim, RobsSpace(:,cc), win);
stas = stas / std(stas(:)) - mean(stas(:));
wm = [min(stas(:)) max(stas(:))];
nlags = size(stas,1);
figure(1); clf
for ilag = 1:nlags
    subplot(2, ceil(nlags/2), ilag)
    imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, reshape(stas(ilag, :), opts.dims), wm)
    title(sprintf('lag: %02.2f', ilag*16))
    axis xy
end



%% find ROI
ilag = 4;
S.rect = [-60 -20 10 50];
% S.rect = 
figure(1); clf
ppd = Exp.S.pixPerDeg;
imagesc(opts.xax, opts.yax, reshape(stas(ilag, :), opts.dims))
axis xy

hold on
plot(S.rect([1 3]), S.rect([2 2]), 'r', 'Linewidth', 2)
plot(S.rect([1 3]), S.rect([4 4]), 'r', 'Linewidth', 2)
plot(S.rect([1 1]), S.rect([2 4]), 'r', 'Linewidth', 2)
plot(S.rect([3 3]), S.rect([2 4]), 'r', 'Linewidth', 2)
title('Average Spatial RF & ROI')
xlabel('Azimuth (pixels)')
ylabel('Elevation (pixels)')

%% Do high-res reconstruction using PTB (has to replay the whole experiment)
Exp.FileTag = S.processedFileName;
% pixels run down so enforce this here
S.rect([2 4]) = sort(-S.rect([2 4]));
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}, 'overwrite', false);
%% Copy to server
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command fname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)


%% test that it worked
id = 1;
stim = 'Gabor';
tset = 'Train';

sz = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'size');

iFrame = 1;

%% show sample frame
iFrame = iFrame + 1;
I = h5read(fname, ['/' stim '/' tset '/Stim'], [iFrame, 1,1], [1 sz(1:2)']);
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
spike_sorting = 'kilo';
Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
% Robs = h5read(fname, ['/' stim '/' set '/Robs']);
ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);

frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');
st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);

% Robs = 
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
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
    Xstim = conv2(Stim(:,idim).^2, eye(nlags), 'full');
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
% sta = (sta - min(sta(:))) ./ (max(sta(:)) - min(sta(:)));
sta = (sta - mean(sta(:))) ./ std(sta(:));
% x = xax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% y = yax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% xax = (1:NX)/Exp.S.pixPerDeg*60;
% yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(1,nlags, ilag, 'align')
   imagesc(reshape(sta(ilag,:), [NY NX])', [-1 1]*4)
end

% colormap(plot.viridis)
colormap(gray)
title(cc)


%%
figure(2); clf
cc = cc + 1;
% NC = numel(W);
if cc > NC
    cc = 1;
end
sta = stas(:,:,cc);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[~, bestlag] = max(std(sta,[],2));
figure(1); clf

subplot(2,2,2)
imagesc(reshape(sta(bestlag,:), [NX NY])')

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,4)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')
