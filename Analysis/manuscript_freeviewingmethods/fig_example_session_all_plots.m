

sesslist = io.dataFactoryGratingSubspace;
sesslist = sesslist(1:57); % exclude monash sessions

%% Load analyses
% Spatial RFs
sfname = fullfile('Data', 'spatialrfs.mat');
load(sfname)

% Grating RFs
gfname = fullfile('Data', 'gratrf.mat');
load(gfname)

% FFT RF
fftname = fullfile('Data', 'fftrf.mat');
load(fftname)

%% Explore

iEx = 45;
fprintf('Loading session [%s]\n', sesslist{iEx})

%% Refit space?
Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
stat = spat_rf_helper(Exp, 'plot', true, 'stat', Srf{iEx}, 'debug', true, 'boxfilt', 5);
Srf{iEx} = stat;

%% refit grating?
Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Sgt{iEx}.sorter);
stat = grat_rf_helper(Exp, 'plot', true, 'stat', Sgt{iEx}, 'debug', false, 'boxfilt', 1, 'sftuning', 'loggauss', 'upsample', 2);
Sgt{iEx} = stat;

%% refit fftrf?
Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Sgt{iEx}.sorter);
rf_post = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, 'debug', false, 'plot', true, 'usestim', 'post', 'alignto', 'fixon');
fftrf{iEx}.rfs_post = rf_post;


%% step through units

cc = cc + 1;
% 
if cc > numel(Srf{iEx}.rffit)
    cc = 1;
end

fields = {'rfs_post', 'rfs_pre'};
field = fields{1};
cmap = lines;


figure(100); clf
subplot(3,2,1) % spatial RF
imagesc(Srf{iEx}.xax, Srf{iEx}.yax, Srf{iEx}.spatrf(:,:,cc)); axis xy
hold on
plot(fftrf{iEx}.(field)(cc).rfLocation(1), fftrf{iEx}.(field)(cc).rfLocation(2), 'or')

subplot(3,2,2) % grating fit
imagesc(fftrf{iEx}.(field)(cc).rf.kx, fftrf{iEx}.(field)(cc).rf.ky, fftrf{iEx}.(field)(cc).rf.Ifit')
title(cc)

for  f = 1:2
    field = fields{f};
    subplot(3,2,3+(f-1)*2) % X proj
    bar(fftrf{iEx}.(field)(cc).xproj.bins, fftrf{iEx}.(field)(cc).xproj.cnt, 'FaceColor', .5*[1 1 1]); hold on
    lev = fftrf{iEx}.(field)(cc).xproj.levels(1);
    iix = fftrf{iEx}.(field)(cc).xproj.bins <= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(5,:));
    lev = fftrf{iEx}.(field)(cc).xproj.levels(2);
    iix = fftrf{iEx}.(field)(cc).xproj.bins >= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(1,:));
    
    subplot(3,2,4+(f-1)*2) % PSTH
    mrate = fftrf{iEx}.(field)(cc).rateHi;
    srate = fftrf{iEx}.(field)(cc).stdHi / sqrt(fftrf{iEx}.(field)(cc).nHi);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:)); hold on
    mrate = fftrf{iEx}.(field)(cc).rateLow;
    srate = fftrf{iEx}.(field)(cc).stdLow / sqrt(fftrf{iEx}.(field)(cc).nLow);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:));
    xlim([-.05 .4])
end

plot.suplabel(strrep(sesslist{iEx}, '_', ' '), 't');


frf = fftrf{iEx}.rfs_post(cc).frf;
nsteps = numel(fftrf{iEx}.rfs_post(cc).frfsteps);
figure(10); clf
clim = [min(frf(:)) max(frf(:))];
for i = 1:nsteps
    subplot(1,nsteps+1, i)
    imagesc(fftrf{iEx}.(field)(cc).rf.kx, fftrf{iEx}.(field)(cc).rf.ky, frf(:,:,i), clim)
end


figure(iEx); clf
subplot(321, 'align')
Imap = Srf{iEx}.spatrf(:,:,cc);

xax = Srf{iEx}.xax;
yax = Srf{iEx}.yax;

imagesc(xax, yax, Imap)
colorbar
colormap(plot.viridis)
axis xy
hold on
xlabel('Azimuth (d.v.a)')
ylabel('Elevation (d.v.a)')


% ROI
mu = Srf{iEx}.rffit(cc).mu;
C = Srf{iEx}.rffit(cc).C;

% significance (NEED TO REDO)
if isempty(mu) % no fit was run because RF never crossed threshold
    sig = 0;
else
    ms = (Srf{iEx}.rffit(cc).mushift/Srf{iEx}.rffit(cc).ecc);
    sz = (Srf{iEx}.maxV(cc)./Srf{iEx}.rffit(cc).ecc);
    sig = ms < .25 & sz > 5;
end

zrf = Sgt{iEx}.rf(:,:,cc)*Sgt{iEx}.fs_stim / Sgt{iEx}.sdbase(cc);
z = reshape(zrf(Sgt{iEx}.timeax>=0,:), [], 1);
zthresh = 6;
sigg = sum(z > zthresh) > (1-normcdf(zthresh));

sigs = sig;

fprintf('%d) spat: %d, grat: %d\n', cc, sigs, sigg)


if sigs
    
    offs = trace(C)*10;
    xd = [-1 1]*offs + mu(1);
    xd = min(xd, max(xax)); xd = max(xd, min(xax));
    yd = [-1 1]*offs + mu(2);
    yd = min(yd, max(yax)); yd = max(yd, min(yax));
    
    plot(xd([1 1]), yd, 'r')
    plot(xd([2 2]), yd, 'r')
    plot(xd, yd([1 1]), 'r')
    plot(xd, yd([2 2]), 'r')
    
    plot.plotellipse(mu, C, 1, 'g', 'Linewidth', 2);
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
end

%     xlim(xd)
%     ylim(yd)

if sigg && ~isempty(Sgt{iEx}.rffit(cc).srf)
    subplot(3,2,2) % grating RF and fit
    contourf(Sgt{iEx}.xax, Sgt{iEx}.yax, Sgt{iEx}.rffit(cc).srf, 10, 'Linestyle', 'none'); hold on
    contour(Sgt{iEx}.xax, Sgt{iEx}.yax, Sgt{iEx}.rffit(cc).srfHat/max(Sgt{iEx}.rffit(cc).srfHat(:)), [1 .5], 'r', 'Linewidth', 2)
    axis xy
    xlabel('Orientation (deg)')
    ylabel('Spatial Frequency')
    title(Sgt{iEx}.rffit(cc).oriPref)
end

% TIME (SPATIAL MAPPING)
subplot(3,2,3)
if ~isnan(Srf{iEx}.spmx(cc))
    plot(Srf{iEx}.timeax, Srf{iEx}.rf(:,Srf{iEx}.spmx(cc),cc)*Srf{iEx}.fs_stim, 'b-o', 'Linewidth', 2); hold on
    plot(Srf{iEx}.timeax, Srf{iEx}.rf(:,Srf{iEx}.spmn(cc),cc)*Srf{iEx}.fs_stim, 'r-o', 'Linewidth', 2);
    plot(Srf{iEx}.peaklagt(cc)*[1 1], ylim, 'k', 'Linewidth', 2)
    plot(Sgt{iEx}.peaklagt(cc)*[1 1], ylim, 'k--', 'Linewidth', 2)
    
    xlabel('Time Lag (ms)')
    ylabel('Firing Rate')
    
    subplot(3,2,4)
    [~, spmx] = max(reshape(Sgt{iEx}.rffit(cc).srf', [], 1));
    [~, spmn] = min(reshape(Sgt{iEx}.rffit(cc).srf', [], 1));
    try
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmx,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmx,cc)*Sgt{iEx}.fs_stim, 'b', 'FaceColor', 'b', 'EdgeColor', 'b'); hold on
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmn,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmn,cc)*Sgt{iEx}.fs_stim, 'r', 'FaceColor', 'r', 'EdgeColor', 'r');
    
    plot(Srf{iEx}.peaklagt(cc)*[1 1], ylim, 'k', 'Linewidth', 2)
    plot(Sgt{iEx}.peaklagt(cc)*[1 1], ylim, 'k--', 'Linewidth', 2)
    xlabel('Time Lag (ms)')
    axis tight
    end
end

if sigg
% PLOTTING FIT
[xx,yy] = meshgrid(Sgt{iEx}.rffit(cc).oriPref/180*pi, 0:.1:15);
X = [xx(:) yy(:)];





% plot data RF tuning
par = Sgt{iEx}.rffit(cc).pHat;
if isempty(par)
sigg = false;
end

lag = Sgt{iEx}.peaklag(cc);
if lag == 0
    sigg = false;
end

fs = Sgt{iEx}.fs_stim;
srf = reshape(Sgt{iEx}.rf(lag,:,cc)*fs, Sgt{iEx}.dim);
srfeb = reshape(Sgt{iEx}.rfsd(lag,:,cc)*fs, Sgt{iEx}.dim);

if Sgt{iEx}.ishartley
    
    [kx,ky] = meshgrid(Sgt{iEx}.xax, Sgt{iEx}.yax);
    sf = hypot(kx(:), ky(:));
    sfs = min(sf):max(sf);
    
    ori0 = Sgt{iEx}.rffit(cc).oriPref/180*pi;
    
    [Xq, Yq] = pol2cart(ori0*ones(size(sfs)), sfs);
    
    r = interp2(Sgt{iEx}.xax, Sgt{iEx}.yax, srf, Xq, Yq);
    
    subplot(3,2,6)
    plot(sfs, r, '-o'); hold on
    xlabel('Spatial Frequency (cpd)')
    
    oris = 0:(pi/10):pi;
    sf0 = Sgt{iEx}.rffit(cc).sfPref;
    [Xq, Yq] = pol2cart(oris, sf0*ones(size(oris)));
    
    r = interp2(Sgt{iEx}.xax, Sgt{iEx}.yax, srf, Xq, Yq, 'linear');
    
    subplot(3,2,5)
    plot(oris/pi*180, r, '-o'); hold on
    xlabel('Orientation (deg)')
    ylabel('Firing Rate')

else
    %%
    [i,j] = find(srf == max(srf(:)));
    ori0 = Sgt{iEx}.xax(j)/180*pi;
    sf0 = Sgt{iEx}.yax(i);
    
    subplot(3,2,5)
    errorbar(Sgt{iEx}.xax, srf(i,:), srfeb(i,:), 'o-', 'Linewidth', 2); hold on
    xlim([0 180])
    xlabel('Orientation (deg)')
    ylabel('Firing Rate')
    
    subplot(3,2,6)
    errorbar(Sgt{iEx}.yax, srf(:,j), srfeb(:,j), 'o-', 'Linewidth', 2); hold on
    xlabel('Spatial Frequency (cpd)')
    
end

% if sigg % Only plot fits if it's "significant"
    orientation = 0:.1:pi;
    spatfreq = 0:.1:20;
    
    % plot tuning curves
    orientationTuning = prf.parametric_rf(par, [orientation(:) ones(numel(orientation), 1)*sf0], strcmp(Sgt{iEx}.sftuning, 'loggauss'));
    spatialFrequencyTuning = prf.parametric_rf(par, [ones(numel(spatfreq), 1)*ori0 spatfreq(:)], strcmp(Sgt{iEx}.sftuning, 'loggauss'));
    
    subplot(3,2,5)
    plot(orientation/pi*180, orientationTuning, 'r', 'Linewidth', 2)
    
    
    subplot(3,2,6)
    plot(spatfreq, spatialFrequencyTuning, 'r', 'Linewidth', 2)
    plot(par(3)*[1 1], ylim, 'b')
    
    
    % plot bandwidth
    if strcmp(Sgt{iEx}.sftuning, 'loggauss')
        a = sqrt(-log(.5) * par(4)^2 * 2);
        b1 = (par(3) + 1) * exp(-a) - 1;
        b2 = (par(3) + 1) * exp(a) - 1;
    else
        a = acos(.5);
        logbase = log(par(4));
        b1 = par(3)*exp(-a*logbase);
        b2 = par(3)*exp(a*logbase);
    end
    
    plot(b2*[1 1], ylim, 'b')
    plot(b1*[1 1], ylim, 'k')
    plot( Sgt{iEx}.rffit(cc).sfPref*[1 1], ylim)
end

plot.suplabel(sprintf('%s: %d', strrep(sesslist{iEx}, '_', ' '), cc), 't');
plot.fixfigure(gcf, 10, [4 8], 'offsetAxes', false)