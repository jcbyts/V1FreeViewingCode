%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
figDir = 'Figures/manuscript_freeviewing/fig04';

%% Loop over examples, get Srf


binSize = 0.3; % d.v.a.
sorter = 'kilowf'; % id tag for using kilosort spike-sorting

clear Srf

sesslist = io.dataFactory;
sesslist = sesslist(1:57);

sfname = fullfile('Data', 'spatialrfs.mat');
Srf = cell(numel(sesslist),1);

%%
if exist(sfname, 'file')==2
    disp('Loading Spatial RFs')
    load(sfname)
else
    %%
    for iEx = 1:numel(sesslist)
%         if isempty(Srf{iEx})
            try % not all sessions 
                switch sesslist{iEx}(1)
                    case 'l' % foveal sessions
                        BIGROI = [-4 -4 4 4];
                    otherwise
                        BIGROI = [-14 -10 14 10];     
                end
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                
                tmp = spat_rf_helper(Exp, 'plot', false, 'debug', false, ...
                    'ROI', BIGROI, 'binSize', binSize, ...
                    'spikesmooth', 1, 'boxfilt', 3);
%                 evalc("tmp = spat_rf_helper(Exp, 'plot', true, 'debug', true, 'ROI', BIGROI, 'binSize', binSize, 'spikesmooth', 0);");
%                 drawnow
                tmp.sorter = sorter;
                Srf{iEx} = tmp;
            catch me
                disp(me.message)
                disp('ERROR ERROR')
            end
%         end
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
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                
                tmp = grat_rf_helper(Exp, 'plot', false, 'stat', Sgt{iEx}, 'debug', false, 'boxfilt', 1, 'sftuning', fittype, 'upsample', 2);
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
D.('ellie_20170728').flipx = 0;
D.('logan_20200304').flipx = 0;

D.('ellie_20190107').spike_sorting = 'kilowf';
D.('ellie_20170731').spike_sorting = 'kilowf';
D.('ellie_20170728').spike_sorting = 'kilowf';
D.('logan_20200304').spike_sorting = 'kilowf';

D.('ellie_20190107').example_unit = 11;
D.('ellie_20170731').example_unit = 5;
D.('logan_20200304').example_unit = 9;
D.('ellie_20170728').example_unit = 3;


%% plot units
exnames = fieldnames(D);
% single units
% D.('ellie_20190107').example_unit = D.('ellie_20190107').example_unit + 1; %1;% 8, 10
% D.('ellie_20170731').example_unit = D.('ellie_20170731').example_unit + 1; % 2, 11, 30
% D.('logan_20200304').example_unit = D.('logan_20200304').example_unit + 1; %9; % 3, 16, 25

ex = 3;

iEx = find(strcmp(sesslist, exnames{ex}));
cc = D.(sesslist{iEx}).example_unit;
if cc > numel(Srf{iEx}.rffit)
    cc = 1;
    D.(sesslist{iEx}).example_unit = 1;
end

figure(100); clf
subplot(1,2,1)
plot(Waveforms{iEx}(cc).wavelags, Waveforms{iEx}(cc).ctrChWaveform, 'k'); hold on
plot(Waveforms{iEx}(cc).wavelags, Waveforms{iEx}(cc).ctrChWaveformCiHi, 'k--');
plot(Waveforms{iEx}(cc).wavelags, Waveforms{iEx}(cc).ctrChWaveformCiLo, 'k--');
title(Waveforms{iEx}(cc).ExtremityCiRatio)

subplot(1,2,2)
plot(Waveforms{iEx}(cc).lags, Waveforms{iEx}(cc).isi, 'k')
title(Waveforms{iEx}(cc).isiRate)

figure(iEx); clf
t = tiledlayout(3,1);
t.TileSpacing = 'Compact';

ax = nexttile;
% subplot(311, 'align') % SPATIAL RF
Imap = Srf{iEx}.spatrf(:,:,cc);

xax = Srf{iEx}.xax;
yax = Srf{iEx}.yax;

imagesc(xax, yax, Imap); hold on
hc = colorbar;
colormap(plot.viridis);
plot(0,0,'+w')
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
    
    plot.plotellipse(mu, C, 2, 'r', 'Linewidth', 1);
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
    imagesc([-5 5], [-5 5], ones(2)*min(srf(:))); hold on
    
    contourf(xax, -Sgt{iEx}.yax, srf, 10, 'Linestyle', 'none'); hold on
    contour(xax, -Sgt{iEx}.yax, srfHat, [1 .5], 'r', 'Linewidth', 1)
    
    axis xy
    xlabel('Frequency (cyc/deg)')
    ylabel('Frequency (cyc/deg)')
    xlim([-5 5])
    ylim([-5 5])
%     title(Sgt{iEx}.rffit(cc).oriPref)
    hc = colorbar;
    
    nexttile
    [~, spmx] = max(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));
    [~, spmn] = min(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));

    cmap = [0 0 0; .5 .5 .5];
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmx,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmx,cc)*Sgt{iEx}.fs_stim, 'b', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .8); hold on
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmn,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmn,cc)*Sgt{iEx}.fs_stim, 'r', 'FaceColor', cmap(2,:), 'EdgeColor', cmap(2,:), 'FaceAlpha', .8);
    
    ylabel('\Delta Firing Rate (sp s^{-1})')
    xlabel('Time Lag (ms)')
    axis tight

end

% plot.suplabel(sprintf('%s: %d', strrep(sesslist{iEx}, '_', ' '), cc), 't');
% plot.fixfigure(gcf, 10, [4 8], 'offsetAxes', false)


plot.fixfigure(gcf, 8, [2.5 6], 'offsetAxes', false)
saveas(gcf, fullfile(figDir, sprintf('example_%s_%d.pdf', sesslist{iEx}, cc)))


%% refit guassian on spatial map
for iEx = 21
    fprintf('running session [%s]\n', sesslist{iEx})
    if isempty(Srf{iEx})
        continue
    end
    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
%     Srf{iEx}
    stat = spat_rf_helper(Exp, 'plot', true, 'debug', true,...
        'ROI', BIGROI, 'binSize', binSize, 'spikesmooth', 0);
%     stat = spat_rf_helper(Exp, 'plot', false, 'stat', [], 'debug', true);
    Srf{iEx} = stat;
end

% save(sfname, '-v7.3', 'Srf')

%% refit grating parametric
for iEx = 1:numel(Sgt)
    if isempty(Sgt{iEx})
        continue
    end
    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Sgt{iEx}.sorter);
    stat = grat_rf_helper(Exp, 'plot', true, 'stat', Sgt{iEx}, 'debug', false, 'boxfilt', 1, 'sftuning', fittype, 'upsample', 2);
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

exnum = [];

zthresh = 8;
for ex = 1:numel(Srf)
    
    if ~isfield(Srf{ex}, 'rffit') || ~isfield(Sgt{ex}, 'rffit') || (numel(Sgt{ex}.rffit) ~= numel(Srf{ex}.rffit))
        continue
    end
    
    if sum(Srf{ex}.sig)<2 && sum(Sgt{ex}.sig)<2
        continue
    end
        
    NC = numel(Srf{ex}.rffit);
    for cc = 1:NC
        if ~isfield(Srf{ex}.rffit(cc), 'mu')
            fprintf('Skipping because no fit %s\n', sesslist{ex})
            continue
        end
        
        
         mu = Srf{ex}.rffit(cc).mu;
         C = Srf{ex}.rffit(cc).C;
         maxV = [maxV; Srf{ex}.maxV(cc)];
         
         if isempty(Sgt{ex}.rffit(cc).r2) || isempty(Srf{ex}.rffit(cc).r2)
             fprintf('Skipping because bad fit [%s] %d\n', sesslist{ex}, cc)
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
             exnum = [exnum; ex];
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
         
         exnum = [exnum; ex];
         
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
ix = ix & wfamp > 40;
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
scaleFactor = 1; %sqrt(-log(.5))*2; % scaling to convert from gaussian SD to FWHM
y = ar(ix) * scaleFactor;

plot(x, y, 'wo', 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 2); hold on
% plot(ecc(ix&su), ar(ix&su), 'o')
b0 = [0.764, 0.495 ,0.050]; % initialize with Rosa fit
fun = @(p,x) exp( -p(1) + p(2)*log(x) + p(3)*log(x).^2);
evalc('bhat = lsqcurvefit(fun, b0, x, y);');

xlim([0 15])
ylim([0 15])

hold on
% plot(eccx, fun(bhat,eccx))
plot(eccx, rosa_fit)

set(gca, 'xscale', 'log', 'yscale', 'log')
xlabel('Eccentricity (d.v.a)')
ylabel('RF size (d.v.a)')
set(gcf, 'Color', 'w')
legend({'Data', 'Rosa 1997'}, 'Box', 'off')
set(gca, 'XTick', [.1 1 10], 'YTick', [.1 1 10])
xt = get(gca, 'XTick');
set(gca, 'XTickLabel', xt)
yt = get(gca, 'YTick');
set(gca, 'YTickLabel', yt)

text(1, 5, sprintf('n=%d', sum(ix)))

plot.fixfigure(gcf, 7, [2 2], 'FontName', 'Arial', ...
    'LineWidth',.5, 'OffsetAxes', false);

saveas(gcf, fullfile(figDir, 'fig04_ecc_vs_RFsize.pdf'))

% figure
% for ex = unique(exnum(:))'
%     ii = ex==exnum(ix);
%     plot(x(ii), y(ii), '.'); hold on; %'wo', 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 2)
% end

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
gix = sigg==1 & oriBw < 180;% & su; % & sfBw./sfPref > .55 & oriBw > 32 & oriBw < 88;
% gix = ix;
subplot(1,2,1)

oriPref = real(oriPref);
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
plot(sfPref(gix), sfBw(gix), 'ow', 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 2 ); hold on
x = 0:15;
xlim([0 10])
ylim([0 10])
xlabel('Preferred Stim. (cpd)')
ylabel('Bandwidth (cpd)')

gixx = gix & sfBw < 20 & sfPref > 0 & sfPref < 10;
X = [sfPref(gixx), sfBw(gixx)];
[u,s] = svd(cov(X));
mu = mean(X);



[B,STATS] = robustfit(sfPref(gixx), sfBw(gixx));
plot(x, x*B(2) + B(1), 'k', 'Linewidth', 2)
plot(xlim, xlim, 'k--')

% plot(mu(1), mu(2), 'or')
m = u(2,1)./u(1,1);
b = mu(2) - mu(1)*m;
plot(xlim, xlim*m + b, 'r')

plot.fixfigure(gcf, 8, [2 2], 'OffsetAxes', false);
plot.offsetAxes(gca, true, 4)

saveas(gcf, fullfile(figDir, 'fig04_sf_vs_bw.pdf'))

%%
figure(10); clf
cmap = lines(max(exnum));
for ex = unique(exnum(:))'
    ii = ex==exnum(:) & gix;
    plot(sfPref(ii), sfBw(ii), '.', 'Color', cmap(ex,:)); hold on; %'wo', 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 2)
end
xlim([0 10])
ylim([0 10])

%%
clf
plot(exnum(gix), sfPref(gix), '.')
ylim([0 10])