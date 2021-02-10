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
D.('ellie_20170731').example_unit = 29;
D.('logan_20200304').example_unit = 9;

%% plot units
exnames = fieldnames(D);
zthresh = 6;

% single units
D.('ellie_20190107').example_unit = D.('ellie_20190107').example_unit + 1; %1;% 8, 10
D.('ellie_20170731').example_unit = D.('ellie_20170731').example_unit + 1; % 2, 11, 30
D.('logan_20200304').example_unit = D.('logan_20200304').example_unit + 1; %9; % 3, 16, 25

ex = 1;

iEx = find(strcmp(sesslist, exnames{ex}));
cc = D.(sesslist{iEx}).example_unit;
if cc > numel(Srf{iEx}.rffit)
    cc = 1;
    D.(sesslist{iEx}).example_unit = 1;
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
for iEx = 1:numel(Sgt)
    if isempty(Sgt{iEx})
        continue
    end
    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', Sgt{iEx}.sorter);
    stat = grat_rf_helper(Exp, 'plot', true, 'stat', Sgt{iEx}, 'debug', false, 'boxfilt', 1, 'sftuning', fittype);
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
         if ctr(end,1) ~= ctr(end,2)
             keyboard
         end
    end
end

% wrap
oriPref(oriPref < 0) = 180 + oriPref(oriPref < 0);
oriPref(oriPref > 180) = oriPref(oriPref > 180) - 180;

[sum(sigs) sum(sigg)]
%%
NC = numel(r2);

mnx = (mus(:,1) - 1);
mxx = (mus(:,1) + 1);
mny = (mus(:,2) - 1);
mxy = (mus(:,2) + 1);

figure(1); clf

outbounds = mnx < BIGROI(1) | ...
    mxx > BIGROI(3) | ...
    mny < BIGROI(2) | ...
    mxy > BIGROI(4);


plot(mus(outbounds, 1), mus(outbounds, 2), 'o'); hold on
plot(mus(~outbounds, 1), mus(~outbounds, 2), 'o')

plot(BIGROI([1 3]), BIGROI([2 2]), 'k')
plot(BIGROI([1 1]), BIGROI([2 4]), 'k')
plot(BIGROI([1 3]), BIGROI([4 4]), 'k')
plot(BIGROI([3 3]), BIGROI([2 4]), 'k')



%% 
figure(1); clf
% ix = sigs > 0.05 & sigs < 1;
% ix = ix & (r2 > 0.025); % & (maxV./ecc)>10; % & ~outbounds;
% ix = cgs == 2 | cgs == 1;
ix = sigs & sigg; %mshift./ecc < .25 & (maxV./ecc)>5;

plot(ecc(ix), sfPref(ix), '.'); hold on
% plot(ecc(ix & cgs==2), sfPref(ix & cgs==2), 'o')
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
evalc('bhat = lsqcurvefit(fun, b0, x, y, [0 0 0]);');

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
plot(BIGROI([1 3]), BIGROI([2 2]), 'k')
plot(BIGROI([1 1]), BIGROI([2 4]), 'k')
plot(BIGROI([1 3]), BIGROI([4 4]), 'k')
plot(BIGROI([3 3]), BIGROI([2 4]), 'k')
xlim([-14 14])
ylim([-10 10])
%%
figure(1); clf
gix = sigg==1 & oriBw < 85; % & sfBw./sfPref > .55 & oriBw > 32 & oriBw < 88;
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