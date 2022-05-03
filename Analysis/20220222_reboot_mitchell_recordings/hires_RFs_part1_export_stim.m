%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
addpath Analysis/HukLabTreadmill/

%%
close all
sessId = 46;
[Exp, S] = io.dataFactoryTreadmill(sessId);
Exp.osp.cgs = ones(size(Exp.osp.cids))*2;

io.checkCalibration(Exp);
%%
% BIGROI = [-10 -8 10 8];
BIGROI = [-30 -15 30 15];

% eyePos = eyepos;
eyePos = Exp.vpx.smo(:,2:3);
% eyePos(:,1) = -eyePos(:,1);
% eyePos(:,2) = -eyePos(:,2);

stat = spat_rf_helper(Exp, 'ROI', BIGROI, ...
    'win', [0 12],...
    'binSize', .5, 'plot', true, 'debug', false, 'spikesmooth', 1, 'eyePos', eyePos);

%% GET SPATIAL RFS

dotTrials = io.getValidTrials(Exp, 'Dots');
if ~isempty(dotTrials)
    
    BIGROI = [-1 -.5 1 .5]*30;
    
    % eyePos = eyepos;
    eyePos = Exp.vpx.smo(:,2:3);
    % eyePos(:,1) = -eyePos(:,1);
    % eyePos(:,2) = -eyePos(:,2);
    binSize = .5;
    Frate = 60;
    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
        'eyePos', eyePos, 'frate', Frate);
    
    % use indices while fixating
    ecc = hypot(opts.eyePosAtFrame(:,1), opts.eyePosAtFrame(:,2))/Exp.S.pixPerDeg;
    ix = opts.eyeLabel==1 & ecc < 5.2;
    
end


spike_rate = mean(RobsSpace)*Frate;


goodunits = find(spike_rate > 3);

fprintf('%d / %d fire enough spikes to analyze\n', numel(goodunits), size(RobsSpace,2))

% Analyze population
R = RobsSpace(:,goodunits);
win=[-1 10];
stas = forwardCorrelation(Xstim, R, win, find(ix));
NC = size(R,2);

%% Get spatial and temporal RFs
RFspop = zeros([opts.dims NC]);
RFtpop = zeros([size(stas,1), NC]);

for cc = 1:NC

    I = squeeze(stas(:,:,cc));
    I = (I - mean(I(:)) )/ std(I(:));

    [bestlag,j] = find(I==max(I(:)));
    I = I(min(max(bestlag(1)+[-1 0 1], 1), nlags),:);
    I = mean(I);
    I = imgaussfilt(reshape(I, opts.dims), 1);
    RFspop(:,:,cc) = I;
    
    id = find(I(:)==max(I(:)));
    
    RFtpop(:,cc) = stas(:,id,cc);
        
    
end


%% plotting
sx = 15;
sy = ceil(NC/sx);
nperfig = sx*sy;
nfigs = ceil(NC / nperfig);
fprintf('%d figs\n', nfigs)
nlags = size(stas,1);
if nfigs == 1
    figure(1); clf; set(gcf, 'Color', 'w')
end

RFspop = zeros([opts.dims NC]);

for cc = 1:NC
    if nfigs > 1
        fig = ceil(cc/nperfig);
        figure(fig); set(gcf, 'Color', 'w')
    end

    si = mod(cc, nperfig);  
    if si==0
        si = nperfig;
    end
    subplot(sy,sx, si)

    I = squeeze(stas(:,:,cc));
    I = (I - mean(I(:)) )/ std(I(:));

    [bestlag,j] = find(I==max(I(:)));
    I = I(min(max(bestlag(1)+[-1 0 1], 1), nlags),:);
    I = mean(I);
    I = imgaussfilt(reshape(I, opts.dims), 1);
    RFspop(:,:,cc) = I;
%     I = std(I);
    xax = opts.xax/Exp.S.pixPerDeg;
    yax = opts.yax/Exp.S.pixPerDeg;
    imagesc(xax, yax, I, [-2, 2])
    colormap(plot.coolwarm)
    title(Exp.osp.cids(goodunits(cc)))
end


%%

I = reshape(RFspop, prod(opts.dims), NC);

nFac = 6;
rng(1234)
[w, h] = nnmf(I, nFac);

figure(1); clf
xax = opts.xax/Exp.S.pixPerDeg;
yax = opts.yax/Exp.S.pixPerDeg;

for i = 1:nFac
    subplot(ceil(nFac/2), 2, i)
    imagesc(xax, yax, reshape(w(:,i), opts.dims))
    title(i)
end

factors = [1 2 3];
%%
figure(2); clf
plot3(h(factors(1),:), h(factors(2),:), h(factors(3),:), 'o')

rf_units = find(any(h(factors,:)>0.05,1));
N = numel(rf_units);
fprintf('%d/%d units have RFs\n', N, NC)

sx = round(sqrt(N));
sy = ceil(sqrt(N));

[gridx, gridy] = meshgrid(-25:5:25);

figure(2); clf


%%

figure(3); clf
ax = plot.tight_subplot(sx, sy, 0.01, 0.01);
for cc = 1:N
    I = RFspop(:,:,rf_units(cc));
    set(gcf, 'currentaxes', ax(cc))
    imagesc(xax, yax, I, [-2 2]); hold on
    plot(gridx, gridy, 'w'); 
    plot(gridy, gridx, 'w')
    xlim(xax([1 end]))
    ylim(yax([1 end]))
    
end

colormap(plot.coolwarm)
%%

[xx,yy] = meshgrid(xax, yax);
pow = 20;



factors = [1 3];
figure(1); clf
i = 3;
I = reshape(w(:,i), opts.dims);
I = I ./ max(I(:));
I(I < 0.1) = 0;

Iw = I.^pow/sum(I(:).^pow);
x = xx(:)'*Iw(:);
y = yy(:)'*Iw(:);

subplot(1,2,1)
imagesc(xax, yax, I)
hold on
plot(x, y, 'ok')

% initial parameter guess
par0 = [x y 1];
    
% gaussian function
gfun = @(params, X) exp(-.5 * sum(((X-[params(1) params(2)])*pinv([params(3) params(4); params(4) params(3)]'*[params(3) params(4); params(4) params(3)])).*(X-[params(1) params(2)]),2));
    
    
subplot(1,2,2)

xy = get_rf_contour(xx, yy, I, 'thresh', .05, 'plot', true);

ROI = [min(xy(:,1)) min(xy(:,2)) max(xy(:,1)) max(xy(:,2))];

plot(ROI([1 3]), ROI([2 2]), 'r', 'Linewidth', 2)
plot(ROI([1 3]), ROI([4 4]), 'r', 'Linewidth', 2)
plot(ROI([1 1]), ROI([2 4]), 'r', 'Linewidth', 2)
plot(ROI([3 3]), ROI([2 4]), 'r', 'Linewidth', 2)

title('Average Spatial RF & ROI')
xlabel('Azimuth (pixels)')
ylabel('Elevation (pixels)')

ROI = ceil(Exp.S.pixPerDeg*ROI);


%% regenerate data with the following parameters
close all
% pixels run down so enforce this here
S.rect = ROI;
S.rect([2 4]) = sort(-S.rect([2 4]));
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots'}, 'overwrite', false);

flist{1} = fname;

%% copy to server
for i = 1:numel(flist)
    fname = flist{i};
    server_string = 'jake@bancanus'; %'jcbyts@sigurros';
    output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';
    
    data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
    command = 'scp ';
    command = [command fname ' '];
    command = [command server_string ':' output_dir];
    
    system(command)
    
    fprintf('%s\n', fname)
end
%% test that it worked
id = 1;
stim = 'Dots';
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

Stim = h5read(fname, ['/' stim '/' tset '/Stim']);

ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);

frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');
% st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
% clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
% cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
% sp.st = st;
% sp.clu = clu;
% sp.cids = cids;
% Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);
Robs = h5read(fname, ['/' stim '/' tset '/Robs']);

% Robs = 
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);

Stim = reshape(Stim, size(Stim, 1), NX*NY);

% reshape(Stim, 
Stim = zscore(single(Stim)).^2;

%% forward correlation
NC = size(Robs,2);
nlags = 10;
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


%%

goodunits = find(mean(Robs)*frate > 1);
NC = numel(goodunits);

plotit = true;
rfs = struct();
rfs.srf = zeros(NX, NY, NC);
rfs.ctrs = nan(NC,2);
rfs.area = nan(NC,1);
rfs.ecc = nan(NC,1);
rfs.wfx = nan(NC,1);
rfs.com = nan(NC,2);
rfs.contours = cell(NC,1);

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

xax = (ROI(1):ROI(3))/Exp.S.pixPerDeg;
yax = -(ROI(2):ROI(4))/Exp.S.pixPerDeg;
xax = xax(2:end);
yax = yax(2:end);


for cc = 1:NC
    sta = stas(:,:,goodunits(cc));
    nlags = size(sta,1);
    
    sta = reshape(sta, [nlags, NX NY]);
    sta = (sta - mean(sta(:))) / std(sta(:));
    
    sflat = reshape(sta, nlags, []);
    if any(isnan(sflat(:)))
        continue
    end
    
    [bestlag, ~] = find(max(abs(sflat(:)))==abs(sflat));
    
    extrema = max(max(sta(:)), max(abs(sta(:))));
    
    post = squeeze(sta(bestlag,:,:));
    
    rfs.srf(:,:,cc) = post;

    dim = size(post);
    
    [xi, yi] = meshgrid(xax, yax);
    
    sda = abs(post);
    sda = imgaussfilt(sda, 2);
    
    [xc, yc] = radialcenter(sda.^4);
    xc = interp1(1:dim(2), xax, xc);
    yc = interp1(1:dim(1), yax, yc);
    
    rfs.com(cc,1) = xc;
    rfs.com(cc,2) = yc;
    
    [xy, rfs.area(cc), rfs.ctrs(cc,:)] = get_rf_contour(xi, yi, sda, 'plot', true, 'thresh', 3.5);
    rfs.ecc(cc) = hypot(rfs.ctrs(cc,1), rfs.ctrs(cc,2));
    
    rfs.contours{cc} = xy;
end


rfs.xax = xax;
rfs.yax = yax;

%% 

figure(2); clf
set(gcf, 'Color', 'w')
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
cmap = hsv(NC);
cmap = (cmap + .5*ones(size(cmap)))/2;
for cc = 1:NC
    subplot(sx, sy, cc)
    try
    imagesc(rfs.xax, rfs.yax, rfs.srf(:,:,cc)); hold on
    plot(rfs.contours{cc}(:,1), rfs.contours{cc}(:,2), 'Color', 'r', 'Linewidth', 1); hold on
    axis xy
    plot(xlim, [0 0], 'y')
    plot([0 0], ylim, 'y')
    end
end
colormap gray
% colormap(plot.viridis)

%%
figure(1); clf
set(gcf, 'Color', 'w')
cmap = hsv(NC);
cmap = (cmap + .5*ones(size(cmap)))/2;
for cc = 1:NC   
    plot(rfs.contours{cc}(:,1), rfs.contours{cc}(:,2), 'Color', cmap(cc,:), 'Linewidth', 2); hold on
end

xlim([-1 1]*2)
ylim([-1 1]*2)
plot.polar_grid(0:45:360, 0:1:2)

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
   imagesc(reshape(sta(ilag,:), [NX NY])', [-1 1]*4)
end

% colormap(plot.viridis)
colormap(gray)
title(cc)
%%


%%
figure(2); clf
cc = cc + 1;
NC = numel(W);
if cc > NC
    cc = 1;
end
sta = stas(:,:,W(cc).cid);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[~, bestlag] = max(std(sta,[],2));
figure(1); clf

subplot(2,2,1)
plot(W(cc).wavelags, W(cc).ctrChWaveform, 'k'); hold on
plot(W(cc).wavelags, W(cc).ctrChWaveformCiHi, 'k--')
plot(W(cc).wavelags, W(cc).ctrChWaveformCiLo, 'k--')
title(cc)
axis tight
ax = subplot(2,2,3);
plot(W(cc).lags, W(cc).isi)
title(W(cc).isiV)
% ax.XScale ='log';

subplot(2,2,2)
imagesc(reshape(sta(bestlag,:), [NX NY])')

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,4)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')


%% explore waveforms
cids = [0,2,3,8,9,10,11,15,18,19,21,22,23,24,26,27,29,30,31,32,34,35,36,37,38,40,41,42,43,45,48,52,56,57,58,60,63,64]+1;
figure(10); clf
plot.plotWaveforms(W, 1, 'cids', cids)

%%
sid = regexp(sessId, '_', 'split');

fname2 = ['Figures/2021_pytorchmodeling/rfs_' sid{2} '_' spike_sorting '.mat'];
if exist(fname2, 'file')
    tmp = load(fname2);
end

% tmp.shiftx = flipud(tmp.shiftx);
% tmp.shifty = flipud(tmp.shifty);

tmp.shiftx = tmp.shiftx / 60 * 5.2;
tmp.shifty = tmp.shifty / 60 * 5.2;
%%

figure(2); clf
subplot(1,2,1)
contourf(tmp.xspace, tmp.yspace, tmp.shiftx, 'w'); colorbar
subplot(1,2,2)
contourf(tmp.xspace, tmp.yspace, tmp.shifty, 'w'); colorbar

%%
% 
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
eyeX = (eyeAtFrame(:,2) - Exp.S.centerPix(1)) / Exp.S.pixPerDeg;
eyeY = (eyeAtFrame(:,3) - Exp.S.centerPix(2)) / Exp.S.pixPerDeg;

figure(1); clf
subplot(2,1,1)
plot(eyeX); hold on
plot(eyeY)

shiftX = interp2(tmp.xspace, tmp.yspace, tmp.shiftx, eyeX, eyeY, 'linear', 0);
shiftY = interp2(tmp.xspace, tmp.yspace, tmp.shifty, eyeX, eyeY, 'linear', 0);

subplot(2,1,2)
plot(shiftX); hold on
plot(shiftY)

%%
Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
StimS = zeros(size(Stim), 'like', Stim);
%%
dims = size(Stim);
NT = dims(1);
dims(1) = [];

% xax = floor(-dims(2)/2:dims(2)/2)/Exp.S.pixPerDeg*60;
% yax = floor(-dims(1)/2:dims(1)/2)/Exp.S.pixPerDeg*60;
xax = linspace(-1,1,dims(2));
yax = linspace(-1,1,dims(1));
[xgrid,ygrid] = meshgrid(xax(1:dims(2)), yax(1:dims(1)));

disp("shift correcting stimulus...")
for iFrame = 1:NT
    xsample = xgrid+shiftX(iFrame);
    ysample = ygrid+shiftY(iFrame);
    I = interp2(xgrid, ygrid, single(squeeze(Stim(iFrame,:,:))), xsample, ysample, 'linear', 0);
    StimS(iFrame,:,:) = I;
end
disp("Done")

% I0 = single(squeeze(Stim(iFrame,:,:)));
% figure(1); clf
% subplot(1,3,1)
% imagesc(I0)
% subplot(1,3,2)
% imagesc(I)
% subplot(1,3,3)
% imagesc(I0-I)

%%
NC = size(Robs,2);
nlags = 20;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;
ix = sum(conv2(double(ix), eye(nlags), 'full'),2) == nlags; % all timelags good

stas = zeros(nlags, dims(1), dims(2), NC);
for ii = 1:dims(1)
    fprintf('%d/%d\n', ii, dims(1))
    for jj = 1:dims(2)
        Xstim = conv2(StimS(:,ii,jj), eye(nlags), 'full');
        Xstim = Xstim(1:end-nlags+1,:);
        stas(:, ii,jj, :) = Xstim(ix,:)'*Rdelta(ix,:);
    end
end

%%
% clf
% plot.plotWaveforms(W)

cmap = [[ones(128,1) repmat(linspace(0,1,128)', 1, 2)]; [flipud(repmat(linspace(0,1,128)', 1, 2)) ones(128,1)]];
xax = linspace(-1,1,NX)*30;
yax = linspace(-1,1,NY)*30;

figure(1); clf
for cc = 1:NC
%     sta = stas(:,:,:,W(cc).cid);
    sta = tmp.stas_post(:,:,:,cc);
    nlags = size(sta,1);
    sta = reshape(sta, nlags, []);
    sta = (sta - mean(sta(:))) ./ std(sta(:));
    [bestlag, ~] = find(max(abs(sta(:)))==abs(sta));
    extrema = max(max(sta(:)), max(abs(sta(:))));
    I = reshape(sta(bestlag,:), [NX NY])';
    ind = 10:60;
    I = I(ind, ind);
%     abs(I).^10
    imagesc(xax(ind) + W(cc).x, yax(ind) + W(cc).depth, I, .5*[-1 1]*extrema); hold on
    xlim([-50 250])
    ylim([-50 1.1*max([W.depth])])
    drawnow
    
end

colormap(cmap)

%%
figure(2); clf
cc = cc + 1;
NC = numel(W);
if cc > NC
    cc = 1;
end

sta = stas(:,:,:,W(cc).cid);


% sta = tmp.stas_post(:,:,:,cc);
% nlags = size(tmp.stas_pre,1);

nlags = size(sta,1);
sta = reshape(sta, nlags, []);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[bestlag, ~] = find(max(abs(sta(:)))==abs(sta));
% [~, bestlag] = max(max(sta,[],2));
figure(1); clf

subplot(2,2,1)
plot(W(cc).wavelags, W(cc).ctrChWaveform, 'k'); hold on
plot(W(cc).wavelags, W(cc).ctrChWaveformCiHi, 'k--')
plot(W(cc).wavelags, W(cc).ctrChWaveformCiLo, 'k--')
title(cc)
axis tight
ax = subplot(2,2,3);
plot(W(cc).lags, W(cc).isi)
title(W(cc).isiV)
% ax.XScale ='log';

extrema = max(max(sta(:)), max(abs(sta(:))));
subplot(2,2,2)
imagesc(reshape(sta(bestlag,:), [NX NY])', [-1 1]*extrema)
% colormap(plot.viridis)
colormap parula
% axis xy

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,4)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')


figure(3); clf
for ilag = 1:nlags
    subplot(1,nlags,ilag)
    imagesc(reshape(sta(ilag,:), [NX NY])', [-1 1]*extrema)
end