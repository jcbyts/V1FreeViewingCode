%% run make_stim_files_step1_downsampled first
%% get STAs
data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
importname = 'logan_20200304_Gabor_STAs.mat';
server_string = 'jcbyts@sigurros';
output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

fname = fullfile(data_dir, importname);
% if ~exist(fname, 'file')
command = ['scp ' server_string ':' output_dir importname ' ' data_dir];
system(command)
% end
    
tmp = load(fname); % load the eye correction file

%%
[stim, robs, gopts] = io.preprocess_grating_subspace_data(Exp);
robs = robs(:,Exp.osp.cids);

% X = makeStimRows(stim{1}, nlags);
nstim = size(stim{1},2);
NC = size(robs,2);

stg = zeros(nlags+1, nstim, NC);
stge = zeros(nlags+1, nstim, NC);

for i = 1:nstim
    ev = find(stim{1}(:,i));
    [an, sd] = eventTriggeredAverage(robs*gopts.fs_stim, ev, [0 nlags]);
    stg(:,i,:) = an;
    stge(:,i,:) = sd / sqrt(numel(ev));
end


%%
% logan_20200304_cell38_model
cc = 0
NY = 30; %size(stim,2)/NX;
%% plot them
figure(1); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end
cc = 38;
% cc = 57;
sta = tmp.stas(:,:,cc)';
sta = (sta - min(sta(:)))/(max(sta(:)) - min(sta(:)));
nlags = size(sta,1);
x = xax(1:2:end)/Exp.S.pixPerDeg*60;
y = yax(1:2:end)/Exp.S.pixPerDeg*60;

tpower = std(sta,[],2);
[~,bestlag] = max(tpower);

for ilag = 1:nlags
   subplot(1,nlags, ilag, 'align')
%    subplot(1,nlags, ilag)
   imagesc(x, y, reshape(sta(ilag,:), [NX NY]), [0 1])
   if ilag > 1
   axis off
   end
   title(ilag*8.33 + 4.167)
end
colormap(viridis)
plot.fixfigure(gcf, 12, [24 2])
saveas(gcf, 'Figures/sta.pdf')
%%
sta = tmp.stas(:,:,cc)';
sta = (sta - mean(sta(:))) / std(sta(:));
figure(2); clf
subplot(121)
I =reshape(sta(bestlag,:), [NX, NY]);
I = I - mean(I(:));
imagesc(x,y,I)
peak = find( (abs(I) == max(abs(I(:)))));
xlabel('Soace (arcminutes)')
ylabel('Space (arcminutes)')
subplot(122)
plot((1:nlags)*8.33 + 4.167, sta(:,peak), 'k', 'Linewidth', 2)
xlabel('Lag (ms)')
colormap(viridis)
plot.fixfigure(gcf, 12, [5 2], 'OffsetAxes', false)
saveas(gcf, 'Figures/staPeak.pdf')

%%
stga = squeeze(stg(:,:,cc));
stgae = squeeze(stge(:,:,cc));
figure(3); clf
subplot(131)
imagesc(stga)

subplot(132)

% tpower = std(stga,[],2);
tpower = max(stga, [],2);
[~, bestlag] = max(tpower);
rf = reshape(stga(bestlag,:), gopts.dim);
rfe= reshape(stgae(bestlag,:), gopts.dim);
[i,j] = find(rf == max(rf(:)));


errorbar(gopts.oris, rf(:,j), rfe(:,j))

subplot(1,3,3)
errorbar(gopts.cpds, rf(i,:), rfe(i,:))

figure(4); clf
subplot(1,4,1)
plot.plotWaveforms(Wf(cc));
title(cc)
subplot(1,4,3:4)
plot.errorbarFill(Wf(cc).lags, Wf(cc).isi, Wf(cc).isiE, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', 'none', 'FaceAlpha', .5); hold on
plot(Wf(cc).lags, Wf(cc).isi, '-o', 'MarkerSize', 2, 'Color', cmap(1,:))
xlim([0 50])
% set(gca, 'xscale', 'log')
title(cc)
xlabel('Time from spike (ms)')
ylabel('Excess Firing Rate (sp/s)')


plot.fixfigure(gcf, 12, [6 4])
saveas(gcf, 'Figures/waveform.pdf')
%%


figure(1); clf
st = Exp.osp.st;
clu = Exp.osp.clu;
t0 = 1600;
ix = st > t0 & st < t0+10;
plot.raster(st(ix)-t0, clu(ix), 1)

%%

cc = cc+1;
if cc > NC
    cc = 1;
end

%%

rfcenter = nan(NC, 2);
figure(10); clf
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
hasRF = [1:5 9:12 16 20 22:25 27 28 31 32 33 35:39 41:45 47 49 52:59 65];
% hasRF = intersect(find([Wf.x] > 100), hasRF);
for cc = hasRF(:)' %1:NC
    sta = tmp.stas(:,:,cc)';
    tpower = std(sta,[],2);
    [~,bestlag] = max(tpower);
    
    
    I = reshape(sta(bestlag,:), [NX NY]);
    
    subplot(sx,sy,cc,'align')
    imagesc(x,y,I)
%     imagesc(I);
    hold on
    
    I = I - mean(I(:));
    I = max(I, 0);
    
    [i,j] = radialcenter( I);
       
    sz = size(I);
    
    x0 = interp1(1:sz(2), x, i);
    y0 = interp1(1:sz(1), y, j);
%     plot(i, j, 'or', 'MarkerSize', 10, 'MarkerFaceColor', 'r')
%     plot(x0, y0, 'or', 'MarkerSize', 5); %, 'MarkerFaceColor', 'r')

    rfcenter(cc,:) = [x0 y0];
    title(cc)
    axis off
    

end
%%
spkS = io.get_visual_units(Exp);
Wf = io.get_waveform_stats(Exp.osp);

%%

figure(1); clf
ix = [Wf.x] < 100;% & [Wf.uQ] > 10;
cmap = tab10;
plot(rfcenter(ix,1), rfcenter(ix,2), 'o', 'MarkerSize', 2, 'Color', cmap(1,:), 'MarkerFaceColor', cmap(1,:)); hold on
% plot.plotellipse(mean(rfcenter(ix,:)), cov(rfcenter(ix,:)), 1, 'Color', cmap(1,:))
ix = [Wf.x] > 100;% & [Wf.uQ] > 10;
plot(rfcenter(ix,1), rfcenter(ix,2), 'o', 'MarkerSize', 2, 'Color', cmap(2,:), 'MarkerFaceColor', cmap(2,:)); hold on
% plot.plotellipse(mean(rfcenter(ix,:)), cov(rfcenter(ix,:)), 1, 'Color', cmap(2,:))

d = hypot(rfcenter(:,1), rfcenter(:,2));
xlim(x([1 end]))
ylim(y([1 end]))
axis ij

th = linspace(0, 2*pi, 100);
plot(30*cos(th), 30*sin(th), 'Color', .5*[1 1 1]);
plot(xlim, [0 0], 'Color', .5*[1 1 1]);
plot([0 0], ylim, 'Color', .5*[1 1 1]);

plot.fixfigure(gcf, 12, [4 4])
saveas(gcf, 'Figures/rflocation.pdf')

%%
figure
num_lags = size(tmp.staMod,2);
vmn = min(tmp.staMod(:));
vmx = max(tmp.staMod(:));
for ilag = 1:num_lags
    subplot(2,num_lags, ilag, 'align')
    imagesc(reshape(tmp.staMod(:,ilag), [NX,NY]), [vmn vmx])
    axis off
    subplot(2,num_lags, num_lags + ilag, 'align')
    imagesc(reshape(tmp.stas(:,ilag,39), [NX,NY]), [vmn vmx/1.2])
    axis off
end
colormap(viridis)

plot.fixfigure(gcf, 12, [24 4])
saveas(gcf, 'Figures/staMod.pdf')
%%

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

[ft, ia] = unique(tmp.frameTime(:));
% Do the binning here
for i = 1:nt
    y = binNeuronSpikeTimesFast(Exp.osp,tstart(i)+lags, binsize);
    spks(i,:,:) = full(y(:,S.cids))';
    % resample eye position at the time resolution of the spike trains
    xpos(i,:) = interp1(ft, tmp.Rhat(ia), tstart(i)+lags, eyePosInterpolationMethod);
    ypos(i,:) = interp1(ft, tmp.R(ia), tstart(i)+lags, eyePosInterpolationMethod);
end

fprintf('Done\n')
% initialize iterator for plotting cells
cc = 19;
%%

NC = numel(Exp.osp.cids);
% cc = mod(cc + 1, NC); cc = max(cc, 1);
cc = 38;

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

figure(2); clf


R  = ypos(ind,:);
Rhat = xpos(ind,:);

subplot(121)
imagesc(imgaussfilt(R,[.1 5])); axis xy
subplot(122)
imagesc(Rhat); axis xy

figure(3); clf
plot(mean(imgaussfilt(R,[.1 5]))); hold on
plot(mean(Rhat))

%%


[stim, robs, opts] = io.preprocess_grating_subspace_data(Exp);

%%
[ft, ia] = unique(tmp.frameTime(:));
Rhat = interp1(ft, tmp.Rhat(ia), opts.frameTime);
R = interp1(ft, tmp.R(ia), opts.frameTime);

nstim = size(stim{1},2);
nlags = 14;

stg = zeros(nlags+1, nstim);
stge = zeros(nlags+1, nstim);

for i = 1:nstim
    ev = find(stim{1}(:,i));
    [an, sd] = eventTriggeredAverage(R*gopts.fs_stim, ev, [0 nlags]);
    stg(:,i) = an;
    stge(:,i) = sd / sqrt(numel(ev));
end

stgH = zeros(nlags+1, nstim);
stgeH = zeros(nlags+1, nstim);
lm = fitlm(Rhat, R);


for i = 1:nstim
    ev = find(stim{1}(:,i));
    
    [an, sd] = eventTriggeredAverage(lm.predict(Rhat)*gopts.fs_stim, ev, [0 nlags]);
%     [an, sd] = eventTriggeredAverage((Rhat)*gopts.fs_stim, ev, [0 nlags]);
    stgH(:,i) = an;
    stgeH(:,i) = sd / sqrt(numel(ev));
end

%%



%%
figure(3);clf


subplot(121)

% tpower = std(stga,[],2);
tpower = max(stg, [],2);

[~, bestlag] = max(tpower);
rf = reshape(stg(bestlag,:), gopts.dim);
rfH = reshape(stgH(bestlag,:), gopts.dim);

rfe= reshape(stge(bestlag,:), gopts.dim);
rfeH= reshape(stgeH(bestlag,:), gopts.dim);
[i,j] = find(rf == max(rf(:)));

nfun = @(x) x/max(x(:));
errorbar(gopts.oris, rf(:,j), rfe(:,j)); hold on
errorbar(gopts.oris, rfH(:,j), rfeH(:,j))
title('Orientation Tuning')
legend('Data', 'Model')
subplot(1,2,2)
errorbar(gopts.cpds, rf(i,:), rfe(i,:)); hold on
errorbar(gopts.cpds, rfH(i,:), rfeH(i,:))
title('Spatial Frequency Tuning')

%%
sta = simpleRevcorr(stim{1}, R-mean(R), 10);
staH = simpleRevcorr(stim{1}, Rhat-mean(Rhat), 10);


win = [0 10];
% [an,sd] = eventTriggeredAverage(R, find(stim{1}(:,1)), win);
figure(1); clf
subplot(121)
imagesc(sta)
I = reshape(sta(1,:), opts.dim);
imagesc(I)

subplot(122)
imagesc(staH)

I2 = reshape(staH(1,:), opts.dim);
imagesc(I2)

% plot(I2(:,2))

%% get model
%% get STAs
data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
importname = 'logan_20200304_cell38_model2.mat';
server_string = 'jcbyts@sigurros';
output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

fname = fullfile(data_dir, importname);
% if ~exist(fname, 'file')
command = ['scp ' server_string ':' output_dir importname ' ' data_dir];
system(command)
% end
    
tmp2 = load(fname); % load the eye correction file

%%
tmp2 = load('~/Downloads/HRcnim37.mat')

%%
figure(1); clf
nsubs = size(tmp2.ws01,2);
nlags = size(tmp2.ws00,1);
m = size(tmp2.ws00,2);
d = sqrt(size(tmp2.ws01,1)/m);
for isub = 1:nsubs
    I = reshape(tmp2.ws01(:,isub), [m d*d]);
    w2 = tmp2.ws00*I;
    tpower = std(w2,[],2);
    tpower = max(w2,[],2);
    [~, bestlag] = max(tpower);
    
    srf = reshape(w2(bestlag,:), [d, d]);
    vmn = min(srf(:));
    vmx = max(srf(:));
    
    vmx = max(abs(vmn), abs(vmx));
    vmn = -vmx;
    subplot(nsubs, 3, (isub-1)*3 + 1)
    
    h = imagesc(srf, [vmn vmx]);
%     h = imagesc(srf);
    subplot(nsubs,3,(isub-1)*3 + 2)
    
    id = find(srf==max(srf(:)));
    plot(w2(:,id), 'k', 'Linewidth', 2); hold on
    id = find(srf==min(srf(:)));
    plot(xlim, [0 0], '--', 'Color', .5*[1 1 1], 'Linewidth', 2)
%     plot(w2(:,id), 'r')
    
    subplot(nsubs,3,(isub-1)*3 + 3)
    if isub<nsubs
        plot(-10:10, max(-10:10,0), 'k', 'Linewidth', 2)
    else
        plot(-10:10, -min(-10:10,0), 'k', 'Linewidth', 2)
    end
        
    
%     for ilag = 1:nlags
%         subplot(nsubs, nlags, (isub-1)*nlags + ilag, 'align')
%         h = imagesc(reshape(w2(ilag,:), [12, 12]), [vmn vmx]);
% %         h.AlphaData = abs(h.CData)/max(h.CData(:));
%     end
end
colormap(gray)
plot.fixfigure(gcf, 12, [6 6])
saveas(gcf, 'Figures/subunits.pdf')
%%
n = size(tmp2.ws02,1);
m = n / nsubs;
o = sqrt(m);
figure(2); clf
w = reshape(tmp2.ws02, [nsubs, m]);
for isub = 1:nsubs
    subplot(1,nsubs,isub)
    
    h = imagesc(reshape(w(isub,:), [o,o])');
    h.AlphaData = h.CData/max(h.CData(:));
    
    cmap = (gray);
    if isub < 3
        cmap(:,1:2) = 0;
    else
        cmap(:,2:3) = 0;
    end
    
    colormap(gca, cmap)
    
    xlim([-9 o+9])
    ylim([-9 o+9])
end

plot.fixfigure(gcf, 12, [6 2], 'OffsetAxes', false)
saveas(gcf, 'Figures/pooling.pdf')

% cmap = (gray);
% cmap(:,2:3) = 0;
% colormap(cmap)

%%