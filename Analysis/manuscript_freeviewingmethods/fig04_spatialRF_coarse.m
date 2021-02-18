%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
figDir = 'Figures/manuscript_freeviewing/fig04';

%% Loop over examples, get Srf

BIGROI = [-14 -10 14 10];
binSize = 0.3; % d.v.a.

clear Srf

sesslist = io.dataFactoryGratingSubspace;
sesslist = sesslist(1:57); % exclude monash sessions

sfname = fullfile('Data', 'spatialrfs.mat');
Srf = cell(numel(sesslist),1);

if exist(sfname, 'file')==2
    disp('Loading Spatial RFs')
    load(sfname)
else
    
    for iEx = 1:numel(sesslist)
        if isempty(Srf{iEx})
            try
                try % use JRCLUST sorts if they exist
                    sorter = 'jrclustwf';
                    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                catch % otherwise, use Kilosort
                    sorter = 'kilowf';
                    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                end
                
                evalc('tmp = spat_rf_helper(Exp, ''plot'', true, ''ROI'', BIGROI, ''binSize'', binSize, ''spikesmooth'', 0);');
                drawnow
                tmp.sorter = sorter;
                Srf{iEx} = tmp;
            catch me
                disp(me.message)
                disp('ERROR ERROR')
            end
        end
    end
    save(sfname, '-v7.3', 'Srf')
end

%% Grating RFs
Sgt = cell(numel(sesslist),1);
fittype = 'loggauss';
gfname = fullfile('Data', sprintf('gratrf_%s.mat', fittype));
if exist(gfname, 'file')==2
    disp('Loading Grating RFs')
    load(gfname)
else
    
    for iEx = 1:numel(sesslist)
        if isempty(Sgt{iEx})
            
            try
                
                try % use JRCLUST sorts if they exist
                    sorter = 'jrclustwf';
                    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                catch % otherwise, use Kilosort
                    sorter = 'kilowf';
                    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                end
                
                evalc("tmp = grat_rf_helper(Exp, 'plot', false, 'upsample', 2);");
                
                tmp.sorter = sorter;
                Sgt{iEx} = tmp;
                
            catch me
                disp('ERROR ERROR')
                disp(me.message)
            end
        end
    end
    
    save(gfname, '-v7.3', 'Sgt')
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


%% Example units
D = struct();
% details for example sessions

D.('ellie_20190107').flipx = 1;
D.('ellie_20170731').flipx = 0;
D.('logan_20200304').flipx = 0;

D.('ellie_20190107').spike_sorting = 'kilowf';
D.('ellie_20170731').spike_sorting = 'jrclustwf';
D.('logan_20200304').spike_sorting = 'jrclustwf';

D.('ellie_20190107').example_unit = 11;
D.('ellie_20170731').example_unit = 5;
D.('logan_20200304').example_unit = 9;


%% plot units
exnames = fieldnames(D);
% single units
% D.('ellie_20190107').example_unit = D.('ellie_20190107').example_unit + 1; %1;% 8, 10
% D.('ellie_20170731').example_unit = D.('ellie_20170731').example_unit + 1; % 2, 11, 30
% D.('logan_20200304').example_unit = D.('logan_20200304').example_unit + 1; %9; % 3, 16, 25

ex = 2;

iEx = find(strcmp(sesslist, exnames{ex}));
cc = D.(sesslist{iEx}).example_unit;
if cc > numel(Srf{iEx}.rffit)
    cc = 1;
    D.(sesslist{iEx}).example_unit = 1;
end


figure(iEx); clf
t = tiledlayout(3,1);
t.TileSpacing = 'Compact';

ax = nexttile;
% subplot(311, 'align') % SPATIAL RF
Imap = Srf{iEx}.spatrf(:,:,cc);

xax = Srf{iEx}.xax;
yax = Srf{iEx}.yax;

imagesc(xax, yax, Imap)
hc = colorbar;
colormap(plot.viridis);
axis xy
hold on
xlabel('Azimuth (d.v.a)')
ylabel('Elevation (d.v.a)')

% ROI
mu = Srf{iEx}.rffit(cc).mu;
C = Srf{iEx}.rffit(cc).C;

sigs = Srf{iEx}.sig(cc);
sigg = Sgt{iEx}.sig(cc);

fprintf('%d) spat: %d, grat: %d\n', cc, sigs, sigg)


if sigs % plot RF fit
    
    offs = trace(C)*2;
    xd = [-1 1]*offs + mu(1);
    xd = min(xd, max(xax)); xd = max(xd, min(xax));
    yd = [-1 1]*offs + mu(2);
    yd = min(yd, max(yax)); yd = max(yd, min(yax));
%     
%     plot(xd([1 1]), yd, 'r')
%     plot(xd([2 2]), yd, 'r')
%     plot(xd, yd([1 1]), 'r')
%     plot(xd, yd([2 2]), 'r')
    
    plot.plotellipse(mu, C, 2, 'r', 'Linewidth', 2);
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
end

if sigg && ~isempty(Sgt{iEx}.rffit(cc).srf)
    nexttile
%     ax = subplot(3,1,2); % grating RF and fit
    srf = Sgt{iEx}.rffit(cc).srf;
    srf = [rot90(srf,2) srf(:,2:end)];
    
    srfHat = Sgt{iEx}.rffit(cc).srfHat/max(Sgt{iEx}.rffit(cc).srfHat(:));
    srfHat = [rot90(srfHat,2) srfHat(:,2:end)];
    
    xax = [-fliplr(Sgt{iEx}.xax(:)') Sgt{iEx}.xax(2:end)'];
    contourf(xax, -Sgt{iEx}.yax, srf, 10, 'Linestyle', 'none'); hold on
    contour(xax, -Sgt{iEx}.yax, srfHat, [1 .5], 'r', 'Linewidth', 2)
    axis xy
    xlabel('Frequency (cyc/deg)')
    ylabel('Frequency (cyc/deg)')
%     title(Sgt{iEx}.rffit(cc).oriPref)
    hc = colorbar;
    
    nexttile
    [~, spmx] = max(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));
    [~, spmn] = min(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));

    cmap = lines;
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmx,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmx,cc)*Sgt{iEx}.fs_stim, 'b', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .8); hold on
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmn,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmn,cc)*Sgt{iEx}.fs_stim, 'r', 'FaceColor', 'r', 'EdgeColor', 'r', 'FaceAlpha', .8);
    
    ylabel('\Delta Firing Rate (sp s^{-1})')
    xlabel('Time Lag (ms)')
    axis tight

end

% plot.suplabel(sprintf('%s: %d', strrep(sesslist{iEx}, '_', ' '), cc), 't');
plot.fixfigure(gcf, 10, [4 8], 'offsetAxes', false)

% plot tuning

figure(100+iEx); clf    
% TIME (SPATIAL MAPPING)
subplot(3,2,3)
if ~isnan(Srf{iEx}.spmx(cc))
    plot(Srf{iEx}.timeax, Srf{iEx}.rf(:,Srf{iEx}.spmx(cc),cc)*Srf{iEx}.fs_stim, 'b-o', 'Linewidth', 2); hold on
    plot(Srf{iEx}.timeax, Srf{iEx}.rf(:,Srf{iEx}.spmn(cc),cc)*Srf{iEx}.fs_stim, 'r-o', 'Linewidth', 2);
    plot(Srf{iEx}.peaklagt(cc)*[1 1], ylim, 'k', 'Linewidth', 2)
    plot(Sgt{iEx}.peaklagt(cc)*[1 1], ylim, 'k--', 'Linewidth', 2)
    
    xlabel('Time Lag (ms)')
    ylabel('Firing Rate')
    
    
end

if sigg
% PLOTTING FIT
[xx,yy] = meshgrid(Sgt{iEx}.rffit(cc).oriPref/180*pi, 0:.1:15);
X = [xx(:) yy(:)];





% plot data RF tuning
par = Sgt{iEx}.rffit(cc).pHat;

lag = Sgt{iEx}.peaklag(cc);

fs = Sgt{iEx}.fs_stim;
srf = reshape(Sgt{iEx}.rf(lag,:,cc)*fs, Sgt{iEx}.dim);
srfeb = reshape(Sgt{iEx}.rfsd(lag,:,cc)*fs, Sgt{iEx}.dim);

if Sgt{iEx}.ishartley
    
    [kx,ky] = meshgrid(Sgt{iEx}.xax, Sgt{iEx}.yax);
    sf = hypot(kx(:), ky(:));
    sfs = min(sf):max(sf);
    
    ori0 = Sgt{iEx}.rffit(cc).oriPref/180*pi;
    
    [Yq, Xq] = pol2cart(ori0*ones(size(sfs)), sfs);
    
    r = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srf, Xq, Yq);
    eb = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srfeb, Xq, Yq);
    
    subplot(3,2,6)
    h = errorbar(sfs, r, eb, 'ok'); hold on
    h.CapSize = 0;
    h.MarkerSize = 2;
    h.MarkerFaceColor = 'k';
    h.LineWidth = 1.5;
    xlabel('Spatial Frequency (cpd)')
    xlim([0 8])
    
    oris = 0:(pi/10):pi;
    sf0 = Sgt{iEx}.rffit(cc).sfPref;
    [Yq, Xq] = pol2cart(oris, sf0*ones(size(oris)));
    
    r = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srf, Xq, Yq, 'linear');
    eb = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srfeb, Xq, Yq);
    
    subplot(3,2,5)
    h = errorbar(oris/pi*180, r, eb, 'ok'); hold on
    h.CapSize = 0;
    h.MarkerSize = 2;
    h.MarkerFaceColor = 'k';
    h.LineWidth = 1.5;
    xlabel('Orientation (deg)')
    ylabel('Firing Rate')
    xlim([0 180])
    
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
%     plot(par(3)*[1 1], ylim, 'b')
    
    
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
    
%     plot(b2*[1 1], ylim, 'b')
%     plot(b1*[1 1], ylim, 'k')
%     plot( Sgt{iEx}.rffit(cc).sfPref*[1 1], ylim)
end





%% refit guassian on spatial map
for iEx = 26:numel(Srf)
    if isempty(Srf{iEx})
        continue
    end
    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
    stat = spat_rf_helper(Exp, 'plot', false, 'stat', Srf{iEx}, 'debug', false);
    Srf{iEx} = stat;
end

save(sfname, '-v7.3', 'Srf')

%% refit grating parametric
for iEx = 1:28%numel(Sgt)
    if isempty(Sgt{iEx})
        continue
    end
    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Sgt{iEx}.sorter);
    stat = grat_rf_helper(Exp, 'plot', false, 'stat', Sgt{iEx}, 'debug', false, 'boxfilt', 1, 'sftuning', fittype);
    Sgt{iEx} = stat;
end

save(gfname, '-v7.3', 'Sgt')

%% Loop over SRF struct and get relevant statistics
w = [];  % crude area computation
r2 = []; % r-squared from gaussian fit to RF
ar = []; % sqrt area (computed from gaussian fit)
ar2 = []; % double check
ecc = []; % eccentricity
maxV = [];

sfPref = []; % spatial frequency preference
sfBw = [];
oriPref = [];
oriBw  = [];
sigg = [];
sigs = [];

gtr2 = []; % r-squared of parametric fit to frequecy RF

ctr = [];
mus = [];
Cs = [];
cgs = [];
mshift = [];

sess = {};

wf = [];

zthresh = 8;
for ex = 1:numel(Srf)
    
    if ~isfield(Srf{ex}, 'rffit') || ~isfield(Sgt{ex}, 'rffit') || (numel(Sgt{ex}.rffit) ~= numel(Srf{ex}.rffit))
        continue
    end
    
    NC = numel(Srf{ex}.rffit);
    for cc = 1:NC
        if ~isfield(Srf{ex}.rffit(cc), 'mu')
            continue
        end
        
         mu = Srf{ex}.rffit(cc).mu;
         C = Srf{ex}.rffit(cc).C;
         maxV = [maxV; Srf{ex}.maxV(cc)];
         
         if isempty(Sgt{ex}.rffit(cc).r2) || isempty(Srf{ex}.rffit(cc).r2)
             wf = [wf; Waveforms{ex}(cc)];
             oriPref = [oriPref; nan];
             oriBw = [oriBw; nan];
             sfPref = [sfPref; nan];
             sfBw = [sfBw; nan];
             gtr2 = [gtr2; nan];
             
             r2 = [r2; nan]; % store r-squared
             ar = [ar; nan];
             ecc = [ecc; nan];
             sigs = [sigs; false];
             sigg = [sigg; false];
             mus = [mus; nan(1,2)];
             Cs = [Cs; nan(1,4)];
             cgs = [cgs; nan];
             mshift = [mshift; nan];
             sess = [sess; sesslist{ex}];
             continue
         end
         
         mus = [mus; mu];
         Cs = [Cs; C(:)'];
%          plot.plotellipse(mu, C, 1); hold on
         
         oriPref = [oriPref; Sgt{ex}.rffit(cc).oriPref];
         oriBw = [oriBw; Sgt{ex}.rffit(cc).oriBandwidth];
         sfPref = [sfPref; Sgt{ex}.rffit(cc).sfPref];
         sfBw = [sfBw; Sgt{ex}.rffit(cc).sfBandwidth];
         gtr2 = [gtr2; Sgt{ex}.rffit(cc).r2];
             
         sigs = [sigs; Srf{ex}.sig(cc)];
         sigg = [sigg; Sgt{ex}.sig(cc)];
        
         r2 = [r2; Srf{ex}.rffit(cc).r2]; % store r-squared
         ar = [ar; Srf{ex}.rffit(cc).ar];
         ecc = [ecc; Srf{ex}.rffit(cc).ecc];
        
         ctr = [ctr; [numel(r2) numel(gtr2)]];
         
         cgs = [cgs; Srf{ex}.cgs(cc)];
         mshift = [mshift; Srf{ex}.rffit(cc).mushift];
         
         wf = [wf; Waveforms{ex}(cc)];
         sess = [sess; sesslist{ex}];
         
         if ctr(end,1) ~= ctr(end,2)
             keyboard
         end
    end
end

% wrap
% wrap orientation
oriPref(oriPref < 0) = 180 + oriPref(oriPref < 0);
oriPref(oriPref > 180) = oriPref(oriPref > 180) - 180;

fprintf('%d (Spatial) and %d (Grating) of %d Units Total are significant\n', sum(sigs), sum(sigg), numel(sigs))

ecrl = arrayfun(@(x) x.ExtremityCiRatio(1), wf);
ecru = arrayfun(@(x) x.ExtremityCiRatio(2), wf);
wfamp = arrayfun(@(x) x.peakval - x.troughval, wf);

%% 
figure(1); clf
ix = sigs & sigg; %mshift./ecc < .25 & (maxV./ecc)>5;
su = ecrl > 1 & ecru > 2;
ix = ix & su; % wfamp > 40;
plot(ecc(ix), sfPref(ix), '.'); hold on

set(gca, 'xscale', 'log', 'yscale', 'log')
xlabel('Eccentricity (d.v.a)')
ylabel('Spatial Frequency Preference ')
ylim([0.1 22])
xlim([0 15])


eccx = .5:.1:20;

% Receptive field size defined as sqrt(RF Area)
rosa_fit = exp( -0.764 + 0.495 * log(eccx) + 0.050 * log(eccx) .^2); % rosa 1997 marmoset
cebus = exp( (-.56 + .25*log(eccx) + 0.06*log(eccx).^2 ) ).^2; %Gattas 8
 
figure(2); clf
cmap = lines;
x = ecc(ix);
scaleFactor = sqrt(-log(.5))*2; % scaling to convert from gaussian SD to FWHM
y = ar(ix) * scaleFactor;

plot(x, y, 'wo', 'MarkerFaceColor', cmap(1,:), 'MarkerSize', 5)

b0 = [0.764, 0.495 ,0.050]; % initialize with Rosa fit
fun = @(p,x) exp( -p(1) + p(2)*log(x) + p(3)*log(x).^2);
evalc('bhat = lsqcurvefit(fun, b0, x, y);');

xlim([0 15])
ylim([0 15])

hold on
plot(eccx, fun(bhat,eccx))
plot(eccx, rosa_fit)

set(gca, 'xscale', 'log', 'yscale', 'log')
xlabel('Eccentricity (d.v.a)')
ylabel('RF size (d.v.a)')
set(gcf, 'Color', 'w')
legend({'Data', 'Fit', 'Rosa 1997'}, 'Box', 'off')
set(gca, 'XTick', [.1 1 10], 'YTick', [.1 1 10])
xt = get(gca, 'XTick');
set(gca, 'XTickLabel', xt)
yt = get(gca, 'YTick');
set(gca, 'YTickLabel', yt)

text(1, 5, sprintf('n=%d', sum(ix)))

plot.fixfigure(gcf, 7, [1.75 2.5], 'FontName', 'Arial', ...
    'LineWidth',1, 'OffsetAxes', false);

saveas(gcf, fullfile(figDir, 'fig04_ecc_vs_RFsize.pdf'))


%%
mus(mus(:,1) < -5,1) = - mus(mus(:,1) < -5,1);

figure(10); clf
for ii = find(ix)'
    plot.plotellipse(mus(ii,:), reshape(Cs(ii,:), [2 2]), 1); hold on
end

xlim([-14 14])
ylim([-10 10])

%%

figure(1); clf
plot(ar(ix) * scaleFactor, sfBw(ix),'o')
ylim([0 10])

%%
figure(1); clf
gix = sigg==1 & oriBw < 90;% & su; % & sfBw./sfPref > .55 & oriBw > 32 & oriBw < 88;
% gix = ix;
subplot(1,2,1)

h = histogram(wrapTo180(oriPref(gix)), 'binEdges', 0:10:180); hold on
text(160, .7*max(h.Values), sprintf('n=%d', sum(gix)))
xlabel('Orientation Preference ')
ylabel('Count')
plot(90*[1 1], ylim, 'r')

subplot(1,2,2)
plot(wrapTo180(oriPref(gix)), oriBw(gix), 'o'); hold on
obw = oriBw(gix);
be = -20:20:200;
be2 = be(2:end);
be = be(1:end-1);
obw = arrayfun(@(x,y) mean(obw(oriPref(gix)>x & oriPref(gix)<y)), be,be2);
plot((be + be2)/2, obw)

figure(2); clf
cmap = lines;
plot(sfPref(gix), sfBw(gix), 'ow', 'MarkerFaceColor', cmap(1,:), 'MarkerSize', 4 ); hold on
x = 0:15;
ylim([0 20])
xlabel('Preferred Stim. (cpd)')
ylabel('Bandwidth (cpd)')
[B,STATS] = robustfit(sfPref(gix), sfBw(gix));
plot(x, x*B(2) + B(1), 'k', 'Linewidth', 2)




%%