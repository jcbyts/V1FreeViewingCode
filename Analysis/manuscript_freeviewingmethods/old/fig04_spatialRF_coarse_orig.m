%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

figDir = 'Figures/manuscript_freeviewing/fig04';

switch user
    case 'jakework'
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\NIMclass
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\sNIMclass
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\L1General'))
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\minFunc_2012'))
        addpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\lowrankrgc')
    case 'jakelaptop'
        addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
        addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
        addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))
        addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))
end

%% Panel A-C: How gaze-contingent retinotopic mapping works

% pick a session, where we can reconstruct full stimulus examples
Exp = io.dataFactoryGratingSubspace(56);

% index into Dot trials
validTrials = io.getValidTrials(Exp, 'Dots');

iTrial = validTrials(1);

frames = [500 700:10:800];

% full screen image
[Stim, frameInfo] = regenerateStimulus(Exp, iTrial, Exp.S.screenRect - [Exp.S.centerPix Exp.S.centerPix], 'spatialBinSize', 1, ...
    'includeProbe', true, 'frameIndex', frames, 'GazeContingent', false);

% gaze-contingent ROI
ppd = Exp.S.pixPerDeg;
ROI = [-14 -10 14 10];
[StimGC, ~] = regenerateStimulus(Exp, iTrial, round(ROI*ppd), 'spatialBinSize', 1, ...
    'includeProbe', true, 'frameIndex', frames, 'GazeContingent', true);

% most sessions showed only white dots, which is what's primarily reported in the
% text. Some sessions showed both black and white dots. Regardless, when we
% do retinotopic mapping, we ignore sign (assuming some level of complex
% cell
Stim = abs(Stim);
StimGC = abs(StimGC);

%% output

for i = 1:size(Stim,3)
    figure(1); clf
    imagesc(Stim(:,:,i), [-127 127]); colormap gray
    axis equal
    hold on
    plot(frameInfo.eyeAtFrame(i,2), frameInfo.eyeAtFrame(i,3), 'or')
    axis off
    
    plot.fixfigure(gcf, 10, [5 4])
    saveas(gcf, fullfile(figDir, sprintf('fig04_exampleframe%d.pdf', i)))
    
    drawnow
end
hold on
tstart = Exp.ptb2Ephys(Exp.D{iTrial}.STARTCLOCKTIME);
tstop = Exp.ptb2Ephys(Exp.D{iTrial}.ENDCLOCKTIME);

tt = Exp.vpx2ephys(Exp.vpx.smo(:,1));
ix = tt > tstart & tt < tstop;
ppd = Exp.S.pixPerDeg;
ctr = Exp.S.centerPix;
x = Exp.vpx.smo(ix,2)*ppd + ctr(1);
y = ctr(2) - Exp.vpx.smo(ix,3)*ppd;
x(Exp.vpx.Labels(ix) > 2) = nan;
plot(x, y, '-c', 'MarkerSize', 2)
plot.fixfigure(gcf, 10, [5 4])
saveas(gcf, fullfile(figDir, 'fig04_example_eyepos.pdf'))


%% example coarse grid
xax = 0:ppd:size(StimGC,2);
n = numel(xax);
dim = size(StimGC);
dim(3) = [];
SE = {};
xax = [round(xax+1), dim(1)];

for i = 1:size(StimGC,3)
    figure(1); clf
    imagesc(StimGC(:,:,i)); hold on
    for j = 1:n
        for k = 1:n
            plot(xax([j j]), ylim, 'r')
            plot(xlim, xax([k k]), 'r')
        end
    end
    
    xd = xlim;
    xlim(xd + [-10 10]);
    yd = ylim;
    ylim(yd + [-10 10]);
    axis off
    
    plot.fixfigure(gcf, 10, [5 4])
    saveas(gcf, fullfile(figDir, sprintf('fig04_ROI_coarse_grid_%d.pdf', i)))
    
    % average in coarse grid
    [xx,yy] = meshgrid(xax);
    I = zeros(size(xx));
    Iraw = StimGC(:,:,i);
    for ii = 1:n
        for jj = 1:n
            rows = xax(ii):xax(ii+1);
            cols = xax(jj):xax(jj+1);
            m = min(numel(rows), numel(cols));
            ind = sub2ind(dim, cols(1:m), rows(1:m));
            I(jj,ii) = mean(Iraw(ind));
        end
    end
    
    SE{i} = I;
    
    figure(2); clf
    imagesc(I, [-127 127]/4)
    colormap gray
    axis off
    plot.fixfigure(gcf, 10, [5 4])
    saveas(gcf, fullfile(figDir, sprintf('fig04_ROI_coarse_binned_%d.png', i)))
    
end

%% Plot the stimulus ensemble
nframes = numel(frames);
sz = size(SE{1});
I = reshape(cell2mat(SE), [sz nframes]);
for i = 1:nframes
    I(1,:,i) = -200;
    I(sz(1)-1,:,i) = -200;
    I(:,1,i) = -200;
    I(:,sz(1)-1,i) = -200;
    
end
I = permute(I, [3 2 1]);


xax = 1:sz(2);
yax = 1:sz(1);

cmap = gray;
[xx,tt,yy] = meshgrid(xax, (1:nframes)*8, yax);

figure(2); clf
set(gcf, 'Color', 'w')
h = slice(xx,tt, yy, I, [], (1:10)*8,[]);

set(gca, 'CLim', [-9 9])
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
    %     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end
view(79,11)
colormap(cmap)
axis off

hold on
% plot3(10+[10 16], [1 1]*39, -[50 50], 'r', 'Linewidth', 2)
% plot3(10+[10 16], [1 1]*30.5, -[50 50], 'r', 'Linewidth', 2)
% text(40, 32, -50, '0.1 d.v.a', 'Color', 'r')

plot.fixfigure(gcf, 8, [6 3])
saveas(gcf, fullfile(figDir, 'fig04_binned_course_ensemble.png'))


%% Loop over examples, get Srf

D = struct();
% details for example sessions

D.('ellie_20190107').flipx = 1;
D.('ellie_20170731').flipx = 0;
D.('logan_20200304').flipx = 0;

D.('ellie_20190107').spike_sorting = 'kilowf';
D.('ellie_20170731').spike_sorting = 'jrclustwf';
D.('logan_20200304').spike_sorting = 'jrclustwf';

D.('ellie_20190107').example_unit = 11; %1;
D.('ellie_20170731').example_unit = 29;
D.('logan_20200304').example_unit = 9; %9;


BIGROI = [-14 -10 14 10];

% exnames = fieldnames(D);
clear Srf

sesslist = io.dataFactoryGratingSubspace;
sesslist = sesslist(1:57); % exclude monash sessions
Srf = repmat(struct('coarse', [], 'fine', [], 'sorter', []), numel(sesslist), 1);

if exist('Srf.mat', 'file')==2
    disp('Loading')
    load Srf.mat
else
    
    for iEx = 1:numel(sesslist)
        
        try
            try % use JRCLUST sorts if they exist
                sorter = 'jrclustwf';
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
            catch % otherwise, use Kilosort
                sorter = 'kilowf';
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
            end
            
            evalc('tmp = spatial_RF_single_session(Exp, ''plot'', true, ''ROI'', BIGROI, ''numspace'', 20);');
            tmp.sorter = sorter;
            Srf(iEx) = tmp;
        catch
            disp('ERROR ERROR')
        end
        
    end
    save('Srf.mat', '-v7.3', 'Srf')
end

%%
% tmp = spatial_RF_single_session(Exp, 'plot', true, 'ROI', BIGROI, 'numspace', 20);
%% 
Sgt = [];
if exist('Sgt.mat', 'file')==2
    disp('Loading Grating RFs')
    load Sgt.mat
else
    
    for iEx = 1:numel(sesslist)
        try
            
            try % use JRCLUST sorts if they exist
                sorter = 'jrclustwf';
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
            catch % otherwise, use Kilosort
                sorter = 'kilowf';
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
            end
            
            evalc('rfdeets = grating_RF_single_session(Exp, ''plot'', false);');
            for cc = 1:numel(rfdeets)
                rfdeets(cc).sorter = sorter;
            end
            Sgt(iEx).rfs = rfdeets;
            
        catch
            disp('ERROR ERROR')
        end
    end
    
    save('Sgt.mat', '-v7.3', 'Sgt')
end


%% get fftrf

fftrf = [];
if exist('Fftrf.mat', 'file')==2
    disp('Loading FFT RFs')
    load Fftrf.mat
else
    
    for iEx = 1:numel(sesslist)
        try
            
            try % use JRCLUST sorts if they exist
                sorter = 'jrclustwf';
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
            catch % otherwise, use Kilosort
                sorter = 'kilowf';
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
            end
            evalc('rfdeets = fixrate_by_fftrf(Exp, Srf(iEx), Sgt(iEx).rfs, ''plot'', true);');
            for cc = 1:numel(rfdeets)
                rfdeets(cc).sorter = sorter;
            end
            fftrf(iEx).rfs = rfdeets;
            
        catch
            disp('ERROR ERROR')
        end
    end
    
    save('Fftrf.mat', '-v7.3', 'fftrf')
end
%% Example units

exnames = fieldnames(D);

% single units
D.('ellie_20190107').example_unit = 1;% 8, 10
D.('ellie_20170731').example_unit = D.('ellie_20170731').example_unit + 1; % 2, 11, 30
D.('logan_20200304').example_unit = 9; % 3, 16, 25

ex = 2;

iEx = find(strcmp(sesslist, exnames{ex}));

cc = D.(sesslist{iEx}).example_unit;
if cc > numel(Srf(iEx).fine)
    cc = 1;
end
cc

ppd = Srf(iEx).coarse.details(cc).xax(end)/BIGROI(3);
figure(ex); clf
subplot(311, 'align')
Imap = Srf(iEx).coarse.details(cc).srf;
Imap = Imap / max(Imap(:));

xax = Srf(iEx).coarse.details(cc).xax/ppd;
yax = Srf(iEx).coarse.details(cc).yax/ppd;

if D.(sesslist{iEx}).flipx
    xax = -xax;
end

imagesc(xax, yax, Imap)
colormap(plot.viridis)
axis xy
hold on

if ~isnan(Srf(iEx).fine(cc).ROI)
    ROI = Srf(iEx).fine(cc).ROI;
    if D.(sesslist{iEx}).flipx
        ROI([1 3]) = sort(-ROI([1 3]));
    end
    
    plot(ROI([1 3]), ROI([2 2]), 'r', 'Linewidth', 1)
    plot(ROI([1 3]), ROI([4 4]), 'r', 'Linewidth', 1)
    plot(ROI([1 1]), ROI([2 4]), 'r', 'Linewidth', 1)
    plot(ROI([3 3]), ROI([2 4]), 'r', 'Linewidth', 1)
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
    
    subplot(312, 'align')
    Imap = Srf(iEx).fine(cc).srf;
    Imap = Imap / max(Imap(:));
    xax = Srf(iEx).fine(cc).xax;
    yax = Srf(iEx).fine(cc).yax;
    mu = Srf(iEx).fine(cc).gfit.mu;
    C = Srf(iEx).fine(cc).gfit.cov;
    if D.(sesslist{iEx}).flipx
        xax = -xax;
        mu(1) = -mu(1);
        C([2 3]) = -C([2 3]);
    end
    
    imagesc(xax, yax, Imap)
    
    hold on
    
    plot.plotellipse(mu, C, 2, 'r', 'Linewidth', 2);
    axis xy
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
    
    subplot(313, 'align')
    
    [~, peak_id] = max(Srf(iEx).fine(cc).srf(:));
    [~, min_id] = min(Srf(iEx).fine(cc).srf(:));
    sta = Srf(iEx).fine(cc).sta / max(Srf(iEx).fine(cc).sta(:));
    tk = sta(:,peak_id);
    tmin = sta(:,min_id);
    
%     [u,s,v] = svd(Srf(iEx).fine(cc).sta);
%     tk = u(:,1);
%     tk = tk*sign(sum(tk));
%     
    tk = flipud(tk);
    tmin = flipud(tmin);
    plot((1:numel(tk))*(1e3/120), tk, '-ok', 'MarkerFaceColor', 'k', 'MarkerSize', 2); hold on
    plot((1:numel(tk))*(1e3/120), tmin, '-o', 'Color', .5*[1 1 1], 'MarkerFaceColor', .5*[1 1 1], 'MarkerSize', 2);
    xlabel('Lags (ms)')
    
    plot.fixfigure(gcf, 7, [1 3], 'OffsetAxes', false, 'FontName', 'Arial');
    
    colormap(plot.viridis)
    exname = sesslist{iEx};
    saveas(gcf, fullfile(figDir, sprintf('fig04_SRF_%s_%d.pdf', exname, cc)));
end


%% grating RF

iEx = 1;
cc = 1;
I = Sgt(iEx).rfs(cc).srf;

subplot(1,2,1)
imagesc(I)

subplot(1,2,2)
imagesc([I; rot90(I, 2)])

% I = zeros(size(I));

% I = [rot90(I, 2) I]; % mirror symmetric
I = [I; rot90(I, 2)];
yax = [-flipud(Sgt(iEx).rfs(cc).cpds); Sgt(iEx).rfs(cc).cpds];
xax = Sgt(iEx).rfs(cc).oris;

figure(1); clf
[xx,yy] = meshgrid(xax, yax);
[ori, cpd] = cart2pol(xx,yy);

iix = ori==0 | (ori-pi).^2 < .001;
iix = iix & (cpd - 4.0).^2 < 0.001;
% I(iix) = 1;


% contourf(cos(ori - pi/2)); colormap gray
subplot(121)
imagesc(xax, yax, I, [-1 1]);
%%
contourf(xx, yy, I, 'Linestyle', 'none')

params = [Sgt(iEx).rfs(cc).rffit.oriBandwidth/180*pi, ...
    Sgt(iEx).rfs(cc).rffit.oriPref/180*pi, ...
    Sgt(iEx).rfs(cc).rffit.sfPref,...
    Sgt(iEx).rfs(cc).rffit.sfBandwidth,...
    Sgt(iEx).rfs(cc).rffit.amp, ...
    Sgt(iEx).rfs(cc).rffit.base];
Ifit = prf.parametric_rf(params, [ori(:), cpd(:)]);

% hold on
subplot(122)
contourf(xx,yy,reshape(Ifit, size(I)), 'r', 'Linestyle', 'none')
% scatteredInterpolant(xx(:), yy(:), 100*ones(size(xx(:))), I(:) + min(I(:)))



%%
N = 150;
maxr = 20;
theta = linspace(0, 2*pi+.1, N+1);
r = linspace(0, maxr, N+1);
theta = theta(1:end-1);  %throw away 2*pi since it duplicates 0
r = r(2:end);            %throw away 0 since that's at the center
[THETA, R] = ndgrid(theta, r);
Z = sin(THETA*10).*R;
[X, Y] = pol2cart(THETA,R);
surf(X, Y, Z, 'Linestyle', 'none');
view(0,90)
%% Loop over SRF struct and get relevant statistics
w = []; % crude area computation
r2 = []; % r-squared from gaussian fit to RF
ar = []; % sqrt area (computed from gaussian fit)
ar2 = []; % double check
ecc = []; % eccentricity
amp = []; % gaussian fit amplitude
sfPref = []; % spatial frequency preference
gtr2 = []; % r-squared of parametric fit to frequecy RF
rfMad = [];

for ex = 1:numel(Srf)
    if numel(Sgt(ex).rfs) ~= numel(Srf(ex).fine)
        continue
    end
    
    for cc = 1:numel(Srf(ex).fine)
        
        if isfield(Srf(ex).fine(cc).gfit, 'r2') % fit was successful
            
            r2 = [r2; Srf(ex).fine(cc).gfit.r2]; % store r-squared
            amp = [amp; Srf(ex).fine(cc).gfit.amp]; % store amplitude
            
            mu = Srf(ex).fine(cc).gfit.mu;
            C = Srf(ex).fine(cc).gfit.cov; % covariance matrix
            
            % convert multivariate gaussian to ellipse
            trm1 = (C(1) + C(4))/2;
            trm2 = sqrt( ((C(1) - C(4))/2)^2 + C(2)^2);
            
            % half widths?
            l1 =  trm1 + trm2;
            l2 = trm1 - trm2;
            
            [U,S] = svd(C);
            
            
            % convert to sqrt of area to match Rosa et al., 1997
            ar = [ar; sqrt(2 * l1 * l2)];
            ar2 = [ar2; sqrt(prod(2*abs(diag(U*(S)))))];
            ecc = [ecc; hypot(mu(1), mu(2))];
            
            try
                sfPref = [sfPref; Sgt(ex).rfs(cc).rffit.sfPref];
                gtr2 = [gtr2; Sgt(ex).rfs(cc).rffit.r2];
                
                I = Sgt(ex).rfs(cc).sta;
                d = abs(I - median(I(:))); % deviations
                mad = median(d(:)); % median absolute deviation
                rfMad = [rfMad; mean(d(:) > thresh*mad)];
            catch
                sfPref = [sfPref; nan];
                gtr2 = [gtr2; nan];
                rfMad = [rfMad; nan];
            end
        else
            r2 = [r2; nan];
            ar = [ar; nan];
            ar2 = [ar2; nan];
            ecc = [ecc; nan];
            amp = [amp; nan];
            sfPref = [sfPref; nan];
            gtr2 = [gtr2; nan];
        end

    I = (Srf(ex).fine(cc).srf - Srf(ex).fine(cc).gfit.base) / (Srf(ex).fine(cc).gfit.amp - Srf(ex).fine(cc).gfit.base);
    bw = bwlabel(I>0.8);
    sz = sort(sum(bw(:)==unique(bw(:))'), 'descend');
    if numel(sz)==1
        width = nan;
    else
        width = sqrt(sz(2))*Srf(ex).fine(cc).binSize;
        
        figure(1); clf 
        subplot(2,2,1) 
        imagesc(Srf(ex).fine(cc).xax, Srf(ex).fine(cc).yax,I>.5);
        hold on
        mu = Srf(ex).fine(cc).gfit.mu;
        C = Srf(ex).fine(cc).gfit.cov;
        plot.plotellipse(mu, C, 1, 'r');

        subplot(2,2,2)
        [xx,yy] = meshgrid(Srf(ex).fine(cc).xax, Srf(ex).fine(cc).yax);
        I2 = mvnpdf([xx(:) yy(:)], mu, C);
        I2 = (I2 - min(I2(:))) / (max(I2(:))-min(I2(:)));
        plot(I(:)); hold on
        plot(I2(:)); 
        title(Srf(ex).fine(cc).gfit.r2)
        
        subplot(2,2,3)
        I = Sgt(ex).rfs(cc).srf;
        if any(Sgt(ex).rfs(cc).oris>45) % it's orientation / sf, not hartley
            xax = Sgt(ex).rfs(cc).cpds;
            isHartley = false;
        else
            isHartley = true;
            xax = [flipud(-Sgt(ex).rfs(cc).cpds); Sgt(ex).rfs(cc).cpds];
        end
        
        if isHartley
            imagesc(Sgt(ex).rfs(cc).oris, xax, [rot90(I,2); I]); axis xy
        else
            imagesc(Sgt(ex).rfs(cc).oris, Sgt(ex).rfs(cc).cpds, I); axis xy
        end
        

        hold on
        
        [ori,cpd] = meshgrid(Sgt(ex).rfs(cc).oris, xax);
        if isHartley
            [ori, cpd] = cart2pol(ori,cpd);
        end

        params = [Sgt(ex).rfs(cc).rffit.oriBandwidth/180*pi, ...
            Sgt(ex).rfs(cc).rffit.oriPref/180*pi, ...
            Sgt(ex).rfs(cc).rffit.sfPref,...
            Sgt(ex).rfs(cc).rffit.sfBandwidth,...
            Sgt(ex).rfs(cc).rffit.amp, ...
            Sgt(ex).rfs(cc).rffit.base];
        Ifit = prf.parametric_rf(params, [ori(:), cpd(:)]);
        
        subplot(2,2,4)
        imagesc(Sgt(ex).rfs(cc).oris, xax, reshape(Ifit, size(ori))); axis xy

        %         [kx, ky] = pol2cart(Sgt(ex).rfs(cc).rffit.oriPref/180*pi);
        
        pause
    end
    w = [w; width];
    end
end

figure(1); clf
plot(ar, ar2, '.'); hold on
plot(xlim, xlim, 'k')


%% 
figure(1); clf
ix = (r2 > .5 & gtr2>.5);
plot(ecc(ix), sfPref(ix), '.')
% set(gca, 'xscale', 'log', 'yscale', 'log')
xlabel('Eccentricity (d.v.a)')
ylabel('Spatial Frequency Preference ')
ylim([0.1 10])
xlim([0 15])

% clf
% plot(sfPref(ix), ar(ix), '.')
%%
eccx = .1:.1:20;
rosa_fit = exp( -0.764 + 0.495 * log(eccx) + 0.050 * log(eccx) .^2);

% gaussian must have amplitude that isn't absurd and rsquared above .5 to interpret the parameters
thresh = .5; % only include reasonable fits
ix = amp > .85 & amp < 1.25 & r2 > thresh & gtr2 > thresh; 

figure(1); clf
cmap = lines;
plot(ecc(ix), ar(ix), 'wo', 'MarkerFaceColor', cmap(1,:), 'MarkerSize', 2)

x = ecc(ix);
y = ar(ix);

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

plot.fixfigure(gcf, 7, [1.75 2.5], 'FontName', 'Arial', ...
    'LineWidth',1, 'OffsetAxes', false);

saveas(gcf, fullfile(figDir, 'fig04_ecc_vs_RFsize.pdf'))

%%




%% plot individual unit

iEx = 2;
cc = cc + 1;


NC = numel(Srf(iEx).fine);
if cc > NC
    cc = 1;
end
nframes = size(Srf(iEx).fine(cc).sta, 1);

ppd = Srf(iEx).coarse.details(cc).xax(end)/BIGROI(3);
figure(1); clf
subplot(311, 'align')
Imap = Srf(iEx).coarse.details(cc).srf;
Imap = Imap / max(Imap(:));
imagesc(Srf(iEx).coarse.details(cc).xax/ppd, Srf(iEx).coarse.details(cc).yax/ppd, Imap)
colormap(plot.viridis)
axis xy
hold on
if ~isnan(Srf(iEx).fine(cc).ROI)
    plot(Srf(iEx).fine(cc).ROI([1 3]), Srf(iEx).fine(cc).ROI([2 2]), 'r')
    plot(Srf(iEx).fine(cc).ROI([1 3]), Srf(iEx).fine(cc).ROI([4 4]), 'r')
    plot(Srf(iEx).fine(cc).ROI([1 1]), Srf(iEx).fine(cc).ROI([2 4]), 'r')
    plot(Srf(iEx).fine(cc).ROI([3 3]), Srf(iEx).fine(cc).ROI([2 4]), 'r')
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
    
    subplot(312, 'align')
    Imap = Srf(iEx).fine(cc).srf;
    Imap = Imap / max(Imap(:));
    imagesc(Srf(iEx).fine(cc).xax, Srf(iEx).fine(cc).yax, Imap)
    hold on
    plot.plotellipse(Srf(iEx).fine(cc).gfit.mu, Srf(iEx).fine(cc).gfit.cov, 2, 'r', 'Linewidth', 2);
    axis xy
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
    subplot(313, 'align')
    [u,s,v] = svd(Srf(iEx).fine(cc).sta);
    tk = u(:,1);
    tk = tk*sign(sum(tk));
    plot((1:numel(tk))*(1e3/120), tk, 'k');
    
    plot.fixfigure(gcf, 7, [1 3], 'OffsetAxes', false, 'FontName', 'Arial');
    colormap(plot.viridis)
    exname = strrep(Exp.FileTag, '.mat','');
    saveas(gcf, fullfile(figDir, sprintf('fig04_SRF_%s_%d.pdf', exname, cc)));
end


% figure(2); clf
% I = Srf(iEx).fine(cc).sta;
% I = (I - min(I(:))) / (max(I(:)) - min(I(:)));
% sz = size(Srf(iEx).fine(cc).srf);
% 
% I = reshape(I, [nframes sz]);
% I = permute(I, [3 2 1]);
% 
% for i = 1:nframes
%     I(1,:,i) = -1;
%     I(sz(1)-1,:,i) = -1;
%     I(:,1,i) = -1;
%     I(:,sz(1)-1,i) = -1;
%     
% end
% I = permute(I, [3 1 2]);
% 
% cmap = gray;
% 
% xax = 1:sz(2);
% yax = 1:sz(1);
% 
% [xx,tt,yy] = meshgrid(xax, (1:nframes)*8, yax);
% 
% figure(2); clf
% set(gcf, 'Color', 'w')
% h = slice(xx,tt, yy, I, [], (1:nframes)*8,[]);
% 
% set(gca, 'CLim', [-1 1])
% for i = 1:numel(h)
%     h(i).EdgeColor = 'none';
% end
% view(79,11)
% colormap(cmap)
% axis off
% 
% 
% title(cc)
% 
% plot.fixfigure(gcf, 8, [6 3]);
% exname = strrep(Exp.FileTag, '.mat','');
% saveas(gcf, fullfile(figDir, sprintf('fig04_StaRF_%s_%d.pdf', exname, cc)));

%% plot individual unit. Regularized poisson regression done in python. See fig04_example_units_GLM.py

flip = [-1 1 1];
cmap = lines;
figure(1); clf
figure(2); clf
figure(3); clf

useSVD = false;


for iEx = 1:3
    
    Exp = io.dataFactoryGratingSubspace(exnames{iEx}, 'spike_sorting', D.(exnames{iEx}).spike_sorting);
    
    figure(1)
    plot(flip(iEx)*D.(exnames{iEx}).ROI([1 3]), D.(exnames{iEx}).ROI([2 2]), 'Color', cmap(iEx,:)); hold on
    plot(flip(iEx)*D.(exnames{iEx}).ROI([1 3]), D.(exnames{iEx}).ROI([4 4]), 'Color', cmap(iEx,:))
    plot(flip(iEx)*D.(exnames{iEx}).ROI([1 1]), D.(exnames{iEx}).ROI([2 4]), 'Color', cmap(iEx,:))
    plot(flip(iEx)*D.(exnames{iEx}).ROI([3 3]), D.(exnames{iEx}).ROI([2 4]), 'Color', cmap(iEx,:))
    
    
    figure(2)
    
    fdir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/FVmanuscript';
    fname = sprintf('Fig04_glm_%s_%d.mat', exnames{iEx}, D.(exnames{iEx}).example_unit - 1);
    datadir = '/Users/jcbyts/Dropbox/Projects/FreeViewing/Data/';
    
    try
        M = load(fullfile(datadir, fname));
    catch
        server_command = 'scp jake@bancanus:';
        
        command = [server_command fullfile(fdir, fname) ' ' datadir];
        
        system(command)
        M = load(fullfile(datadir, fname));
    end
    
    
    
    
    if useSVD
        I = reshape(M.ws00, [M.num_lags, M.NX*M.NY]);
        [u,s,v]= svd(I);
        title(s(1)/sum(s(:)))
        
        sg = sign(sum(u(:,1)));
        
        subplot(1,3,iEx)
        imagesc(M.xax/Exp.S.pixPerDeg, M.yax/Exp.S.pixPerDeg, reshape(sg*v(:,1), [M.NX, M.NY]) ); axis xy
        colormap gray
        
        figure(3)
        subplot(1,3,iEx)
        plot(sg*u(:,1))
        
    else
        I = reshape(M.ws00, [M.num_lags, M.NX, M.NY]);
        I = (I - min(I(:))) / (max(I(:)) - min(I(:)));
        
        [~, peak_lag] = max(std(reshape(M.ws00, [M.num_lags, M.NX*M.NY]),[],2));
        
        subplot(1,3,iEx)
        imagesc(M.xax/Exp.S.pixPerDeg, M.yax/Exp.S.pixPerDeg, squeeze(I(peak_lag,:,:))); axis xy
        colormap(plot.viridis)
        
        
        I = reshape(M.ws00, [M.num_lags, M.NX*M.NY]);
        [~, peak_space] = max(I(peak_lag,:));
        [~, trough_space] = min(I(peak_lag,:));
        
        figure(3)
        subplot(1,3,iEx)
        plot((1:size(I,1))*8,I(:,peak_space), 'k'); hold on
        plot((1:size(I,1))*8,I(:,trough_space), 'b')
        hold on
        plot(xlim, [0 0], 'Color', .5*[1 1 1])
        plot(8*peak_lag*[1 1], ylim, 'Color', .5*[1 1 1])
    end
    
end

plot.fixfigure(1, 8, [4 4])
saveas(1, fullfile(figDir, 'fig04_RF_ROIs.pdf'))

plot.fixfigure(2, 8, [8 4])
saveas(2, fullfile(figDir, 'fig04_RF_spatialRFs.pdf'))

plot.fixfigure(3, 8, [8 4])
saveas(3, fullfile(figDir, 'fig04_RF_temporalRFs.pdf'))


%% Poisson regression in Matlab
ppd = Exp.S.pixPerDeg;
ROI = D.(exnames{iEx}).ROI;
binSize = D.(exnames{iEx}).binSize;

% discretize stimulus
[Stim, Robs, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', ROI*ppd, 'binSize', ppd*binSize);

% embed time
nlags = 15;
Xstim = makeStimRows(Stim, nlags);

% find valid fixations
eyeTimes = Exp.vpx2ephys(Exp.vpx.smo(:,1));

cnt = find(histcounts(opts.frameTimes(opts.validFrames>0), eyeTimes)); % map frame times to eye samples

% find if samples are fixations and map backwards to find valid frames
valid = find(histcounts(eyeTimes(cnt(Exp.vpx.Labels(cnt)==1)), opts.frameTimes(opts.validFrames>0)));

NT = numel(valid);
fprintf('%d valid (fixation) samples\n', NT)

Robs = Robs(:,D.(exnames{iEx}).example_unit);

sta = [Xstim(valid,:) ones(NT,1)]'*Robs(valid);
sta = sta' ./ [mean(Xstim(valid,:)) 1];

%%
spar = NIM.create_stim_params([nlags opts.dims]); %, 'tent_spacing', 1);

test_inds = ceil(NT*2/5):ceil(NT*3/5);
train_inds = setdiff(1:NT,test_inds);

% initialize model
LN0 = NIM( spar, {'lin'}, 1, ...
    'xtargets', 1, ...
    'd2t', 10e1, ...
    'l1', 10e-3, ...
    'd2x', 10e1);

I = reshape(sta(1:end-1), nlags, []);
I = I - mean(I);

plot(I)
LN0.subunits(1).filtK = I(:) / norm(I);

LN = LN0.fit_filters(Robs(valid), {Xstim(valid,:)}, train_inds);

LN.display_model()
%%


% LN1 = LN.reg_path( Robs(valid), Xstim(valid,:), train_inds, test_inds, 'lambdaID', 'd2t' );
% LN2 = LN1.reg_path( Robs(valid), Xstim(valid,:), train_inds, test_inds, 'lambdaID', 'd2x' );
LN3 = LN.reg_path( Robs(valid), Xstim(valid,:), train_inds, test_inds, 'lambdaID', 'l1' );

%%

LN3.display_model()
%%
cc = 13
LN.subunits(1).reg_lambdas.d2x = 2e3;
LN.subunits(1).reg_lambdas.d2t = 10e1;
LN.subunits(1).reg_lambdas.l1 = 20;
LN = LN.fit_filters(Robs(:,cc), {f(X)}, find(ix));

% % compute STA (in units of delta spike rate)
% Rvalid = RobsSpace(valid,:) - mean(RobsSpace(valid,:));
% sta = (Xstim(valid,:)'*Rvalid) ./sum(Xstim(valid,:))';