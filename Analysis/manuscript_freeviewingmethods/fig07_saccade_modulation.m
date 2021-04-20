% This script produces the 

%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

% addpath Analysis/202001_K99figs_01  
addpath Analysis/manuscript_freeviewingmethods/
figDir = 'Figures/manuscript_freeviewing/fig07';
%% Load analyses from prior steps

sesslist = io.dataFactory;
sesslist = sesslist(1:57); % exclude monash sessions

% Spatial RFs
sfname = fullfile('Data', 'spatialrfsreg.mat');
load(sfname)

% Grating RFs
fittype = 'basis';
gfname = fullfile('Data', sprintf('gratrf_%s.mat', fittype));
load(gfname)


%% get fftrf modulation
% Analyze FFTRF
fftrf = cell(numel(sesslist),1);

fftname = fullfile('Data', sprintf('fftrf_%s.mat', fittype));
if exist(fftname, 'file')==2
    disp('Loading FFT RFs')
    load(fftname)
else
    for iEx = 1:numel(sesslist)
        if isempty(fftrf{iEx})
                if isempty(Srf{iEx}) || isempty(Sgt{iEx})
                    continue
                end
                
                if ~isfield(Srf{iEx}, 'fine')
                    continue
                end
                
                if sum(Srf{iEx}.fine.sig & Sgt{iEx}.sig) == 0 % no significant RFs
                    continue
                end

                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
                
%                 evalc('rf_pre = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, ''plot'', false, ''usestim'', ''pre'', ''alignto'', ''fixon'');');
                evalc('rf_post = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, ''plot'', true, ''usestim'', ''post'', ''alignto'', ''fixon'');');
                
                fftrf{iEx} = struct();
%                 fftrf{iEx}.rfs_pre = rf_pre;
                fftrf{iEx}.rfs_post = rf_post;
                fftrf{iEx}.sorter = Srf{iEx}.sorter;
        end
    end
    
    
    save(fftname, '-v7.3', 'fftrf')
    
end

%% get fixrate modulation by stimulus

fixrat = cell(numel(sesslist),1);
fixname = fullfile('Data', 'fixrate.mat');
if exist(fixname, 'file')==2
    disp('Loading Fixation Rates')
    load(fixname)
else
    
    for iEx = 1:numel(sesslist)
        if isempty(fixrat{iEx})
            try
                
                if isempty(Srf{iEx}) || isempty(Sgt{iEx})
                    continue
                end
                
%                 try % use JRCLUST sorts if they exist
%                     sorter = 'jrclustwf';
%                     Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
%                 catch % otherwise, use Kilosort
                sorter = 'kilowf';
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
%                 end
                
                evalc("stat = fixrate_by_stim(Exp, 'stimSets', {'Grating', 'BackImage'}, 'plot', false);");
                
                fixrat{iEx} = stat;
                fixrat{iEx}.sorter = sorter;
            
            catch me
                disp('ERROR ERROR')
                disp(me.message)
            end
        end
    end
    
    
    save(fixname, '-v7.3', 'fixrat')
end

%% get waveform stats
Waveforms = cell(numel(sesslist), 1);
wname = fullfile('Data', 'waveforms.mat');
if exist(wname, 'file')==2
    disp('Loading Waveforms')
    load(wname)
else
    
    for iEx = 1:numel(sesslist)
        if ~isempty(Srf{iEx})
            Exp = io.dataFactory(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
            Waveforms{iEx} = io.get_waveform_stats(Exp.osp);
        end
    end
    save(wname, '-v7.3', 'Waveforms')
    
end
%% plot example Unit
fields = {'rfs_post'}; %, 'rfs_pre'
field = fields{1};
cmap = lines;

iEx = 56;
if ~exist('cc', 'var'), cc = 1; end
cc = cc + 1;

if cc > numel(fftrf{iEx}.(field)) 
    cc = 1;
end

srf = Srf{iEx}.fine;
% 13, 17
% cc = 17; 
figure(1); clf
subplot(3,2,1) % spatial RF
imagesc(srf.xax, srf.yax, srf.srf(:,:,cc)); axis xy
hold on
plot(fftrf{iEx}.(field)(cc).rfLocation(1), fftrf{iEx}.(field)(cc).rfLocation(2), 'or')
xlabel('Azimuth')
ylabel('Elevation')

subplot(3,2,2) % grating fit
imagesc(fftrf{iEx}.(field)(cc).rf.kx, fftrf{iEx}.(field)(cc).rf.ky, fftrf{iEx}.(field)(cc).rf.Ifit)
title(cc)
xlabel('Frequency x')
ylabel('Frequency y')

for  f = 1
    field = fields{f};
    subplot(3,2,3+(f-1)*2) % X proj
    bar(fftrf{iEx}.(field)(cc).xproj.bins, fftrf{iEx}.(field)(cc).xproj.cnt, 'FaceColor', .5*[1 1 1]); hold on
    lev = fftrf{iEx}.(field)(cc).xproj.levels(1);
    iix = fftrf{iEx}.(field)(cc).xproj.bins <= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(5,:));
    lev = fftrf{iEx}.(field)(cc).xproj.levels(2);
    iix = fftrf{iEx}.(field)(cc).xproj.bins >= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(1,:));
    xlabel('Generator Signal')
    ylabel('Count')
    
    subplot(3,2,4+(f-1)*2) % PSTH
    mrate = fftrf{iEx}.(field)(cc).rateHi;
    srate = fftrf{iEx}.(field)(cc).stdHi / sqrt(fftrf{iEx}.(field)(cc).nHi);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:)); hold on
    mrate = fftrf{iEx}.(field)(cc).rateLow;
    srate = fftrf{iEx}.(field)(cc).stdLow / sqrt(fftrf{iEx}.(field)(cc).nLow);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:));
    xlim([-.05 .4])
    title(strrep(field, '_', ' '))
    xlabel('Time from fixation')
    ylabel('Firing Rate')
end

plot.suplabel(sprintf('%s Unit %d', strrep(sesslist{iEx}, '_', ' '), cc), 't');

%% Loop over and get relevant summary statistics

ar = []; % sqrt area (computed from gaussian fit)
ecc = []; % eccentricity
maxV = []; % volume of RF blob

sfPref = [];  % spatial frequency preference
sfBw = [];    % spatial frequency bandwidth (FWHM)
oriPref = []; % orientation preference
oriBw  = [];  % orientation bandwidth (FWHM)

sigg = []; % boolean: grating RF is significant
sigs = []; % boolean: spatial RF is significant

r2 = [];   % r-squared from gaussian fit to RF
gtr2 = []; % r-squared of parametric fit to frequecy RF

ctr = []; % counter for tracking cell number
cgs = []; % cluster quality

mshift = []; % how much did the mean shift during fitting (measure of whether we got stuck in local minimum)

% FFT modulations
mrateHi = [];
mrateLow = [];
fftcorr = [];
fftpBY = [];
fftpBH = [];
fftpval = [];

field = 'rfs_post';

% fixrate modulations
nipeakt = []; % time of peak for nat images
nipeakv = []; % value at peak (in relative rate) for nat images
nitrot = []; % trough time
nitrov = []; % trough value
nimrate = []; % mean rate
grpeakt = []; % time of peak for gratings
grpeakv = []; % value at peak (in relative rate) for gratings
grtrot = []; % trough time
grtrov = []; % trough value
grmrate = []; % mean rate

rflag = []; % peak lag of the temporal RF

wf = [];

zthresh = 8;
for ex = 1:numel(Srf)
    
    if isempty(fftrf{ex})
        continue
    end
    srf = Srf{ex};
    if isfield(srf, 'fine')
        srf = srf.fine;
    end
    
    if ~isfield(srf, 'rffit') || ~isfield(Sgt{ex}, 'rffit') || (numel(Sgt{ex}.rffit) ~= numel(srf.rffit))
        continue
    end
    
    NC = numel(srf.rffit);
    for cc = 1:NC
        if ~isfield(srf.rffit(cc), 'mu')
            continue
        end
         
         if isempty(Sgt{ex}.rffit(cc).r2) || isempty(srf.rffit(cc).r2)
             continue
         end
         
%          maxV = [maxV; srf.maxV(cc)];
         
         % Tuning preferences
         oriPref = [oriPref; Sgt{ex}.rffit(cc).oriPref];
         oriBw = [oriBw; Sgt{ex}.rffit(cc).oriBandwidth];
         sfPref = [sfPref; Sgt{ex}.rffit(cc).sfPref];
         sfBw = [sfBw; Sgt{ex}.rffit(cc).sfBandwidth];
         gtr2 = [gtr2; Sgt{ex}.rffit(cc).r2];
             
         % significance
         sigg = [sigg; Sgt{ex}.sig(cc)];
         sigs = [sigs; srf.sig(cc)];
        
         r2 = [r2; srf.rffit(cc).r2]; % store r-squared
         ar = [ar; srf.rffit(cc).ar];
         ecc = [ecc; srf.rffit(cc).ecc];
        
         rflag = [rflag; Sgt{ex}.peaklagt(cc)];
         % unit quality metric
         cgs = [cgs; srf.cgs(cc)];
        
         mshift = [mshift; srf.rffit(cc).mushift]; %#ok<*AGROW>
         
         % FFT stuff
         mrateHi = [mrateHi; fftrf{ex}.(field)(cc).rateHi];
         mrateLow = [mrateLow; fftrf{ex}.(field)(cc).rateLow];
         fftcorr = [fftcorr; fftrf{ex}.(field)(cc).corrrho];
         
        fftpBY = [fftpBY; benjaminiYekutieli(fftrf{ex}.(field)(cc).corrp, 0.05)];
        fftpBH = [fftpBH; benjaminiHochbergFDR(fftrf{ex}.(field)(cc).corrp, 0.05)];
        fftpval = [fftpval; fftrf{ex}.(field)(cc).corrp];
        
         % fixrate
         nipeakt = [nipeakt; fixrat{ex}.BackImage.peakloc(cc)];
         nipeakv = [nipeakv; fixrat{ex}.BackImage.peak(cc)];
         nitrot = [nitrot; fixrat{ex}.BackImage.troughloc(cc)];
         nitrov = [nitrov; fixrat{ex}.BackImage.trough(cc)];
         nimrate = [nimrate; fixrat{ex}.BackImage.meanRate(cc,:)]; % mean rate
         grpeakt = [grpeakt; fixrat{ex}.Grating.peakloc(cc)];
         grpeakv = [grpeakv; fixrat{ex}.Grating.peak(cc)];
         grtrot = [grtrot; fixrat{ex}.Grating.troughloc(cc)];
         grtrov = [grtrov; fixrat{ex}.Grating.trough(cc)];
         grmrate = [grmrate; fixrat{ex}.Grating.meanRate(cc,:)];
        
         wf = [wf; Waveforms{ex}(cc)];
         % Counter
         ctr = [ctr; [numel(r2) numel(gtr2) size(mrateHi,1)]];
         
         if ctr(end,1) ~= ctr(end,3)
             keyboard
         end
    end
end

% wrap orientation
oriPref(oriPref < 0) = 180 + oriPref(oriPref < 0);
oriPref(oriPref > 180) = oriPref(oriPref > 180) - 180;

fprintf('%d (Spatial) and %d (Grating) of %d Units Total are significant\n', sum(sigs), sum(sigg), numel(sigs))

cg = arrayfun(@(x) x.cg, wf);
ecrl = arrayfun(@(x) x.ExtremityCiRatio(1), wf);
ecru = arrayfun(@(x) x.ExtremityCiRatio(2), wf);
wfamp = arrayfun(@(x) x.peakval - x.troughval, wf);


%% LOAD SESSION FOR EXAMPLES IN FIGURE
iEx = 45;
fprintf('Loading session [%s]\n', sesslist{iEx})
Exp = io.dataFactory(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
% run fftrf analysis with additional meta data
[rf_post, plotmeta] = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, 'debug', false, 'plot', false, 'usestim', 'post', 'alignto', 'fixon');
% run fixrate analysis with additional meta data
[~, pmeta] = fixrate_by_stim(Exp, 'stimSets', {'Grating', 'BackImage'}, 'plot', false);

%% MAKE FIGURE 07
         
fig = figure(3); clf
fig.Position = [100 100 600 400];
fig.Color = 'w';
m = 3;
n = 4;
layout1 = tiledlayout(m,n);
layout1.TileSpacing = 'compact';
layout1.Padding = 'compact';

% cc = 17;
% cc = cc + 1;
% if cc > numel(rf_post)
%     cc = 1;
% end
% disp(cc)
cc = 40;
% cc = 22; 35
% cc = 38;
% cc = 23; 
% -- RASTER
ax = nexttile([2 2]);

lags = fixrat{iEx}.BackImage.lags;
stims = {'Grating', 'BackImage'};
clrs = cmap([4 1],:);

for istim = 1:2
    stim = stims{istim};
    
    fixdur = pmeta.(stim).fixdur;
    
    [~, ind] = sort(fixdur);

    ind = ind(fixdur(ind) >= 0.25 & fixdur(ind) <= 0.5);
    
    if istim==2
        nn = numel(ind);
        ind = ind(1:floor(nn/num):nn);
    end
    
    [i,j] = find(squeeze(pmeta.(stim).spks(ind,cc,:)));


    
    ix = find(i >= min(ind) & i <= max(ind));
    i = i - min(i(ix));

    if istim==1
        num = numel(unique(i(ix)));
        off = 0;
    else
        off = num;
    end
    
    plot.raster(lags(j(ix)), i(ix)+off, 2, 'Color', clrs(istim,:)); axis tight
    hold on
    cmap = lines;
    plot.raster(fixdur(ind), (1:numel(ind))+off, 2, 'Color', 'k');
end


ylabel('Flashed Gratings               Natural Images')
ax.YTick = [];
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;
ax.XTick = -.1:.1:.5;
xlim([-.1 .5])
xlabel('Time From Fixation (s)')

% -- MEAN FIRING RATE BY STIMULUS
ax = nexttile(2*n+1, [1 2]);

% BackImage
m = fixrat{iEx}.BackImage.meanRate(cc,:);
s = 2*fixrat{iEx}.BackImage.sdRate(cc,:) / sqrt(fixrat{iEx}.BackImage.numFix);
plot.errorbarFill(lags, m, s, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', 'none', 'FaceAlpha', .5); hold on
h = plot(lags, m, 'Color', cmap(1,:));

% Grating
lags = fixrat{iEx}.Grating.lags;
m = fixrat{iEx}.Grating.meanRate(cc,:);
s = 2*fixrat{iEx}.Grating.sdRate(cc,:) / sqrt(fixrat{iEx}.Grating.numFix);
plot.errorbarFill(lags, m, s, 'k', 'FaceColor', cmap(4,:), 'EdgeColor', 'none', 'FaceAlpha', .5); hold on
h(2) = plot(lags, m, 'Color', cmap(4,:));


xlim(lags([1 end]))
xlabel('Time From Fixation (s)')
ylabel('Firing Rate (sp s^{-1})')
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;
ax.XTick = -.1:.1:.5;
% ax.YTick
% legend(h, {'Natural Image', 'Flashed Grating'}, 'Location', 'Best')
xlim([-.1 .5])
plot.offsetAxes(ax, true, 8)

% -- POPULATION FIRING RATE
ix = sigs & sigg;
ix = ix & wfamp > 40; %ecrl > 1 & ecru > 2;

ax = nexttile(3, [2 2]);
mnorm = median([nimrate(ix,:) grmrate(ix,:)],2);
relrateni = nimrate(ix,:) ./ mnorm;
relrategr = grmrate(ix,:) ./ mnorm;
lags = linspace(-.1, .5, 601);
cmap = lines;
plot.errorbarFill(lags, mean(relrateni), 2*std(relrateni)/sqrt(sum(ix)), 'k', 'FaceColor', cmap(1,:), 'EdgeColor', 'None', 'FaceAlpha', .8); hold on
plot(lags, mean(relrateni), 'Color', cmap(1,:));
plot.errorbarFill(lags, mean(relrategr), 2*std(relrategr)/sqrt(sum(ix)), 'k', 'FaceColor', cmap(4,:), 'EdgeColor', 'None', 'FaceAlpha', .8);
plot(lags, mean(relrategr), 'Color', cmap(4,:));
xlim([-.05 .25])
text(.1, 2.5, sprintf('n = %d', sum(ix)))
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;
ax.XTick = -0.05:0.05:.25;
xlabel('Time from Fixation (s)')
ylabel('Relative Rate')

% -- SUMMARY STATISTICS
ax = nexttile(2*n+3, [1 1]);
plot(nipeakt(ix), grpeakt(ix), 'ok', 'MarkerFaceColor', 'w', ...
    'MarkerSize', 1); hold on
xlim([0 .2])
ylim([0 .2])
plot(xlim, xlim, 'k')
title('Peak Lag', 'FontWeight', 'Normal')
xlabel('Natural Images (s)')
ylabel('Flashed Gratings (s)')
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;

ax = nexttile(2*n+4, [1 1]);
plot(nipeakv(ix), grpeakv(ix), 'ok', 'MarkerFaceColor', 'w', ...
    'MarkerSize', 1); hold on

plot(xlim, xlim, 'k')
xlabel('Natural Image')
ylabel('Flashed Gratings')
title('Peak Rate', 'Fontweight', 'normal')
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;

plot.fixfigure(fig, 8, [5 4], 'offsetAxes', false)
% saveas(gcf, fullfile(figDir, sprintf('stim_rate_%s_%d.pdf', sesslist{iEx}, cc)));


fig = figure(4); clf
fig.Position = [100 100 600 400];
set(fig, 'Color', 'w')
% -- FIXATIONS ON NATURAL IMAGE
% 
ax = axes('Position', [.05 .65 .25 .25]); %'Units', 'Pixels');
% ax.Position = [40 280 150 100];
C = plotmeta.clustMeta(plotmeta.clusts(cc));
ii = 2168; %randi(size(C.fixMeta,1));
thisTrial = C.fixMeta(ii,1);
fixix = find(C.fixMeta(:,1)==thisTrial);
fixix = fixix(1:4);
eyeX = C.fixMeta(fixix,2);
eyeY = C.fixMeta(fixix,3);

% LOAD IMAGE (THIS WILL ONLY WORK ON MARMOV5 SESSIONS -- see fixrate_by_fftrf for how to load other sessions)
Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));

ppd = Exp.S.pixPerDeg;
Im = mean(Im,3)-127;
Im = imresize(Im, fliplr(Exp.S.screenRect(3:4)));
imagesc(Im); colormap gray
hold on
et = C.fixMeta(fixix,4);
xy = Exp.vpx.smo(et(1):et(end), 2:3)*Exp.S.pixPerDeg.*[1 -1] + Exp.S.centerPix;
% plot(xy(:,1), xy(:,2), 'c')
plot(eyeX - C.rfCenter(1)*ppd, eyeY + C.rfCenter(2)*ppd, 'c');
plot(eyeX - C.rfCenter(1)*ppd, eyeY + C.rfCenter(2)*ppd, 'oc', 'MarkerFaceColor', 'none', 'MarkerSize', 5)
ax.YTick = [];
ax.XTick = [];
tmprect = C.rect + [eyeX(end) eyeY(end) eyeX(end) eyeY(end)];
imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
plot([imrect(1) imrect(1) + imrect(3)], imrect([2 2]), 'r', 'Linewidth', 1)
plot([imrect(1) imrect(1) + imrect(3)], imrect(2)+imrect([4 4]), 'r', 'Linewidth', 1)
plot(imrect([1 1]),[imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 1)
plot(imrect(1)+imrect([3 3]), [imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 1)

plot([imrect(1) .8*max(xlim)], [imrect(2) min(ylim)], '--r', 'Linewidth', 1)
plot([imrect(1) .8*max(xlim)], [imrect(2)+imrect(4) .6*max(ylim)], '--r', 'Linewidth', 1)

% -- SPATIAL WINDOW 
ax = axes('Position', [.22 .75 .15 .15]);
% ax.Position = [100 280 150 100];
I = squeeze(C.fixIms(:,:,fixix(end),2));
imagesc(I); hold on
plot(xlim, [1 1], 'r', 'Linewidth', 1)
plot(xlim, max(ylim)*[1 1], 'r', 'Linewidth', 1)
plot(min(xlim)*[1 1], ylim, 'r', 'Linewidth', 1)
plot(max(xlim)*[1 1], ylim, 'r', 'Linewidth', 1)
axis off
% ax.YTick = []; ax.XTick = [];
axis square
title("Image Patch")

aa = annotation('textarrow', [.35 .39], [.83 .83]);
aa.HeadWidth = 5;
aa.HeadLength = 5;
aa.HeadStyle = 'vback1';

% -- FFT IN WINDOW
ax = axes('Position', [.41 .75 .1 .15]);
I = abs(squeeze(C.fftIms(:,:,fixix(end),2)));
imagesc(fftrf{iEx}.rfs_post(cc).rf.kx, fftrf{iEx}.rfs_post(cc).rf.ky, I);
% ax.YTick = [-8 0 8]; ax.XTick = [];
axis square
title("Frequency")
xlim([-10 10])
ylim([-10 10])
xlabel('cyc/deg')

aa = annotation('textarrow', [.52 .55], [.83 .83]);
aa.HeadWidth = 5;
aa.HeadLength = 5;
aa.HeadStyle = 'vback1';

% -- GRATING RF
ax = axes('Position', [.56 .75 .1 .15]);
imagesc(fftrf{iEx}.rfs_post(cc).rf.kx, fftrf{iEx}.rfs_post(cc).rf.ky, fftrf{iEx}.rfs_post(cc).rf.Ifit)
xlim([-8 8])
ylim([-8 8])
ax.YTick = []; ax.XTick = [];
title("Frequency RF")

aa = annotation('textarrow', [.67 .70], [.83 .83]);
aa.HeadWidth = 5;
aa.HeadLength = 5;
aa.HeadStyle = 'vback1';

% -- Generator Signal
ax = axes('Position', [.72 .68 .25 .25]) ;
[x,y] = stairs(fftrf{iEx}.rfs_post(cc).xproj.bins, fftrf{iEx}.rfs_post(cc).xproj.cnt);
mn = min(x);
mx = max(x);
my = max(y);
fill([x(1) x' x(end)], [0 y' 0], 'k', 'FaceColor', .5*[1 1 1], 'EdgeColor', .5*[1 1 1]); hold on
% Null
iix = find(fftrf{iEx}.rfs_post(cc).xproj.bins <= fftrf{iEx}.rfs_post(cc).xproj.levels(1));
if numel(iix) == 1, iix = [iix iix + 1]; end
[x,y] = stairs(fftrf{iEx}.rfs_post(cc).xproj.bins(iix), fftrf{iEx}.rfs_post(cc).xproj.cnt(iix));
fill([x(1) x' x(end)], [0 y' 0], 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:))
% Preferred
iix = fftrf{iEx}.rfs_post(cc).xproj.bins >= fftrf{iEx}.rfs_post(cc).xproj.levels(2);
[x,y] = stairs(fftrf{iEx}.rfs_post(cc).xproj.bins(iix), fftrf{iEx}.rfs_post(cc).xproj.cnt(iix));
fill([x(1) x' x(end)], [0 y' 0], 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:))
ax.YTick = []; ax.XTick = [0];
ax.TickDir = "out";
ax.TickLength = ax.TickLength*4;
ax.Box = "off";
xlabel('Generator Signal')
text(mn-.1*(mx-mn), my*.8, '"Null"', 'Color', cmap(5,:))
text(mx*.6, my*.4, '"Preferred"', 'Color', cmap(1,:))
xlim([mn-.1*(mx-mn) max(xlim)])


% --- FFTRF RATE (example unit)
ax = axes('Position', [.08 .3 .25 .25]);
lags = fftrf{iEx}.rfs_post(cc).lags;
m1 = fftrf{iEx}.rfs_post(cc).rateHi;
m2 = fftrf{iEx}.rfs_post(cc).rateLow;
s1 = fftrf{iEx}.rfs_post(cc).stdHi/sqrt(fftrf{iEx}.rfs_post(cc).nHi);
s2 = fftrf{iEx}.rfs_post(cc).stdLow/sqrt(fftrf{iEx}.rfs_post(cc).nLow);
plot.errorbarFill(lags, m1, s1, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:)); hold on
plot.errorbarFill(lags, m2, s2, 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:));
xlim([-.05 .3])
xlabel('Time (s)')
ylabel('Firing Rate')
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*4;
plot.offsetAxes(ax, true, 4)

% --- FFTRF RATE (POPULATION AVERAGE)
ax = axes('Position', [.42 .3 .25 .25]);
lags = fftrf{1}.rfs_post(1).lags;
iix = lags>-.05 & lags<0; 

mnorm = max(nanmean(mrateHi(:,iix),2), 1);

nrateHi = mrateHi ./ mnorm;
nrateLow = mrateLow ./ mnorm;

cmap = lines;
plot.errorbarFill(lags, nanmean(nrateHi(ix,:)), nanstd(nrateHi(ix,:))/sqrt(sum(ix)), 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .8); hold on
plot(lags, nanmean(nrateHi(ix,:)), 'Color', cmap(1,:));
plot.errorbarFill(lags, nanmean(nrateLow(ix,:)), nanstd(nrateLow(ix,:))/sqrt(sum(ix)), 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:), 'FaceAlpha', .8);
plot(lags, nanmean(nrateLow(ix,:)), 'Color', cmap(5,:));
xlim([-.05 .3])
plot(xlim, [1 1], 'k--')

text(0.2, 4, sprintf("n=%d",sum(ix)))
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*4;
plot.offsetAxes(ax, true, 4)
xlabel('Time (s)')
ylabel('Relative Rate')

% --- Fraction of cells with significant modulation
ax = axes('Position', [.75 .3 .22 .25]);
% plot(lags, mean(fftpval < 0.05)); hold on
% plot(lags, mean(fftpBH(ix,:))); hold on
x = lags;
y = mean(fftpBY(ix,:));
[x,y] = stairs(lags, mean(fftpBY(ix,:)));
fill([x(1) x(:)' x(end)],[0 y(:)' 0], 'k', 'FaceColor', .5*[1 1 1], 'EdgeColor', 'none'); hold on
plot(x, y, 'k')
% plot(lags, mean(fftpval < 0.05/600));
xlim([-.05 .3])
xlabel('Time (s)')
ylabel('Fraction of cells')
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*4;
plot.offsetAxes(ax, true, 4)

plot.fixfigure(fig, 8, [5 4], 'offsetAxes', false)
% set(gcf, 'renderer', 'painters')
fname = fullfile(figDir, sprintf('fft_rate_%s_%d.pdf', sesslist{iEx}, cc));
% saveas(gcf, fname);
print(gcf,fname,'-dpdf','-painters');


%%
figure(1); clf

psig = fftpBY(ix,:);
rho = fftcorr(ix,:);
rho(~psig) = nan;
plot.errorbarFill(lags, nanmean(rho), nanstd(rho)./sqrt(sum(psig)), 'k--', 'FaceColor', 'w'); hold on
plot(lags, nanmean(rho), 'k')
xlabel('Time (s)')
ylabel('Correlation Coefficient')
xlim([-.05 .3])
%%
cc = cc + 1;
if cc > numel(rf_post)
    cc = 1;
end
figure(1); clf
subplot(1,2,1);
plot(rf_post(cc).lags, rf_post(cc).rateHi); hold on
plot(rf_post(cc).lags, rf_post(cc).rateLow);
xlim([-0.05 .3]);

subplot(1,2,2)
plot(rf_post(cc).lags, rf_post(cc).corrrho); hold on
sigix = benjaminiHochbergFDR(rf_post(cc).corrp, 0.05);
plot(rf_post(cc).lags(sigix), rf_post(cc).corrrho(sigix), '.');
sigix = benjaminiYekutieli(rf_post(cc).corrp, 0.05);
plot(rf_post(cc).lags(sigix), rf_post(cc).corrrho(sigix), 'o');
xlim([-0.05 .3]);





%%
earlyix = lags > 0.01 & lags < 0.1;
emodh = sum(nrateHi(ix, earlyix),2);
emodl = sum(nrateLow(ix, earlyix),2);
lateix = lags > 0.1 & lags < 0.25;
lmodh = sum(nrateHi(ix, lateix),2);
lmodl = sum(nrateLow(ix, lateix),2);

figure(4); clf
plot(lmodl, lmodh, 'o'); hold on
plot(emodl, emodh, 'o'); hold on
plot(xlim, xlim, 'k')

%%
cc = randsample(find(ix), 1);
figure(4); clf; %plot(nrateHi(cc,:), nrateLow(cc,:)); hold on


subplot(1,2,1)
[cnt, xedges, yedges] = histcounts2(nrateHi(ix,earlyix), nrateLow(ix,earlyix), 0:.05:5, 0:.05:5);
imagesc(xedges, yedges, cnt); hold on
plot(xlim, xlim, 'r')
plot([0 1], [1 1], 'r')
plot([1 1], [0 1], 'r')
axis xy

subplot(1,2,2)
[cnt, xedges, yedges] = histcounts2(nrateHi(ix,lateix), nrateLow(ix,lateix), 0:.05:5, 0:.05:5);
imagesc(xedges, yedges, cnt); hold on
plot(xlim, xlim, 'r')
plot([0 1], [1 1], 'r')
plot([1 1], [0 1], 'r')
axis xy
colormap(plot.viridis)
%% Population summary

lags = fftrf{1}.rfs_post(1).lags;
iix = lags>-.05 & lags<0; 

mnorm = mean(mrateHi(:,iix),2);

nrateHi = mrateHi ./ mnorm;
nrateLow = mrateLow ./ mnorm;

% index
ix = sigs & sigg;
ix = ix & wfamp > 40; %ecrl > 1 & ecru > 2;

figure(1); clf
imagesc(lags, find(ix), (nrateHi(ix,:) - nrateLow(ix,:)))

figure(2); clf
cmap = lines;
plot.errorbarFill(lags, nanmean(nrateHi(ix,:)), nanstd(nrateHi(ix,:))/sqrt(sum(ix)), 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .8); hold on
plot(lags, nanmean(nrateHi(ix,:)), 'Color', cmap(1,:));
plot.errorbarFill(lags, nanmean(nrateLow(ix,:)), nanstd(nrateLow(ix,:))/sqrt(sum(ix)), 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:), 'FaceAlpha', .8);
plot(lags, nanmean(nrateLow(ix,:)), 'Color', cmap(5,:));
xlim([-.05 .3])

text(0.2, 4, sprintf("n=%d",sum(ix)))

figure(3); clf
X = [rflag(ix), nipeakt(ix)*1e3];
mu = mean(X);
[u,s] = svd(cov(X));
plot(X(:,1), X(:,2), 'o')
hold on
m = u(2)./u(1);
b = mu(2) - m * mu(1);
xd = xlim;
yd = ylim;
xd(1) = min(xd(1), yd(1));
xd(2) = max(xd(2), yd(2));
plot(xd, b + m*xd, 'r')
xlabel("Receptive Field Lag (ms)")
ylabel("Fixation Transient (ms)")
plot(xd, xd, 'k')
xlim(xd)
ylim(xd)


X = [rflag(ix), nipeakt(ix)*1e3];
[rho, pval] = corr(X(:,1), X(:,2));
if pval < 0.001
    fprintf("RF lag and Fixation transient are significantly correlated:\n")
else
    fprintf("RF lag and Fixation transient are NOT significantly correlated:\n")
end
fprintf("Rho: %02.3f, pval: %d\n", rho, pval)


%% plot

% ix = ix & nipeakt > -0.05 & nipeakt < 0.1 & nitrot > -0.05 & nitrot < 0.1;
% ix = ix & grpeakt > -0.05 & grpeakt < 0.1 & grtrot > -0.05 & grtrot < 0.1;
% 

sum(ix)


figure(1); clf
relrateni = nimrate(ix,:) ./ median(nimrate(ix,:),2) - 1;
relrategr = grmrate(ix,:) ./ median(grmrate(ix,:),2) - 1;
lags = linspace(-.1, .5, 601);
cmap = lines;
plot.errorbarFill(lags, mean(relrateni), 2*std(relrateni)/sqrt(sum(ix)), 'k', 'FaceColor', cmap(1,:), 'EdgeColor', 'None', 'FaceAlpha', .8); hold on
plot.errorbarFill(lags, mean(relrategr), 2*std(relrategr)/sqrt(sum(ix)), 'k', 'FaceColor', cmap(4,:), 'EdgeColor', 'None', 'FaceAlpha', .8);
xlim([-.1 .25])


figure(2); clf
plot(nipeakt(ix), nipeakv(ix), 'o', 'Color', 'w', 'MarkerFaceColor', cmap(1,:)); hold on
plot(grpeakt(ix), grpeakv(ix), 'o', 'Color', 'w', 'MarkerFaceColor', cmap(4,:));
plot(nitrot(ix), nitrov(ix), 'o', 'Color', cmap(1,:), 'MarkerFaceColor', 'w'); hold on
plot(grtrot(ix), grtrov(ix), 'o', 'Color', cmap(4,:), 'MarkerFaceColor', 'w');


X = [nipeakt(ix), nipeakv(ix)];
mu = mean(X);
plot.plotellipse(mean(X), cov(X), 1, '-', 'Color', cmap(1,:), 'Linewidth', 2); hold on
X = [grpeakt(ix), grpeakv(ix)];
disp(mu-mean(X))
plot.plotellipse(mean(X), cov(X), 1, '-', 'Color', cmap(4,:), 'Linewidth', 2);


figure(4); clf
plot(nipeakt(ix), grpeakt(ix), 'o'); hold on
xlim([0 .2])
ylim([0 .2])
plot(xlim, xlim, 'k')
xlabel('Natural Image (s)')
ylabel('Flashed Gratings (s)')
title('Peak Lag')

%%