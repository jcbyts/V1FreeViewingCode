
addpath Analysis/manuscript_freeviewingmethods/

%% Load session
sessid = 24;

try
    sorter = 'jrclustwf';
    Exp = io.dataFactoryGratingSubspace(sessid, 'spike_sorting', sorter);
catch % otherwise, use Kilosort
    sorter = 'kilowf';
    Exp = io.dataFactoryGratingSubspace(sessid, 'spike_sorting', sorter);
end
           
%% Old way of doing things (slow), not great

BIGROI = [-14 -10 14 10];
spatrf = spatial_RF_single_session(Exp, 'plot', true, 'ROI', BIGROI, 'numspace', 20);
gratrf = grating_RF_single_session(Exp, 'plot', false);
% evalc('rfdeets = fixrate_by_fftrf(Exp, Srf(iEx), Sgt(iEx).rfs, ''plot'', true);');

%% plot output
NC = numel(gratrf);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(10); clf
for cc = 1:NC
    subplot(sx,sy,cc)
    plot(spatrf.coarse.details(cc).sta)
end


figure(11); clf
for cc = 1:NC
    subplot(sx,sy,cc)
    plot(gratrf(cc).sta)
end


%% start: big window
BIGROI = [-14 -10 14 10];
stat = spat_rf_helper(Exp, 'sdthresh', 8, 'ROI', BIGROI, 'binSize', .3);

%% plot
NC = size(stat.rf,3);
thresh = 8;
zthresh = (1 - normcdf(thresh));

cc = cc + 1
if cc > NC
    cc = 1;
end


figure(2); clf


subplot(1,2,1)
plot(stat.rf(:,:,cc)/stat.sdbase(cc))
rf = stat.rf(:,:,cc);
zrf = (rf - mean(rf(:)))/stat.sdbase(cc);
sig = abs(zrf) > thresh;
nsig = mean(mean(sig)) > zthresh;
title(sprintf("sig = %d", nsig))

subplot(1,2,2)
spower = reshape(mean(rf.^2), stat.dim);
[bestlag,~] = find(max(rf(:))==rf);

I = reshape(zrf(bestlag,:), stat.dim);
I = sqrt(I.^2);

I = imgaussfilt(I,.5);
imagesc(stat.xax, stat.yax, I); hold on
title(cc)
axis xy


% fit gaussian
[y0, x0] = find(I==max(I(:)));
x0 = stat.xax(x0);
y0 = stat.yax(y0);

% [x0,y0] = radialcenter(I); % initialize center with radial symmetric center
% x0 = interp1(1:size(I,2), stat.xax, x0);
% y0 = interp1(1:size(I,1), stat.yax, y0);
% 
% if isempty(x0)
%     x0 = nan;
%     y0 = nan;
% end

plot(x0, y0, 'o')

mnI = min(I(:));
mxI = max(I(:));
par0 = [x0 y0 5 0 2 mxI];
    

gfun = @(params, X) params(5) + (params(6) - params(5)) * exp(-.5 * sum(((X-[params(1) params(2)])*pinv([params(3) params(4); params(4) params(3)]'*[params(3) params(4); params(4) params(3)])).*(X-[params(1) params(2)]),2));

[xx,yy] = meshgrid(stat.xax,  stat.yax);
X = [xx(:) yy(:)];


% least-squares
lb = [min(xx(:)) min(yy(:)) 0 -1000 0 5];
ub = [max(xx(:)) max(yy(:)) 10 1000 5 mxI];

try
    evalc('phat = lsqcurvefit(gfun, par0, X, I(:), lb, ub);');
catch
    phat = par0;
end
mu = phat(1:2);
C = [phat(3) phat(4); phat(4) phat(3)]'*[phat(3) phat(4); phat(4) phat(3)];

plot.plotellipse(mu, C, 1, 'Linewidth', 2);
drawnow

% get r2
Ihat = gfun(phat, X);
r2 = rsquared(I(:), Ihat(:));

% convert multivariate gaussian to ellipse
trm1 = (C(1) + C(4))/2;
trm2 = sqrt( ((C(1) - C(4))/2)^2 + C(2)^2);
            
% half widths
l1 =  trm1 + trm2;
l2 = trm1 - trm2;
            
% convert to sqrt of area to match Rosa et al., 1997
ar = sqrt(2 * l1 * l2);
ecc = hypot(mu(1), mu(2));
fprintf('ecc: %02.2f, area: %02.2f, r^2:%02.2f\n', ecc, ar, r2)

%% cluster
% cluster until the each group has a median distance less than 2 d.v.a.
numClusts = 1; % initial number of clusters
condition = false;
thresh = 2;
collapse = false;

rfLocations = stat.rfLocations;

ix = ~any(isnan(rfLocations),2);

X = rfLocations(ix,:);

distfun = @(Xi,Xj) sum(sqrt(bsxfun(@minus,Xi,Xj).^2),2); % .* sqrt(Xj.^2) ,2);% .* hypot(Xi,Xj);
Z = linkage(X, 'ward', 'euclidean');
figure(1); clf

while ~condition

    c = cluster(Z,'Maxclust',numClusts);
    mdist = 0;
    for cc = 1:numClusts
        ii = cc==c;
        meddist = mean(mean( hypot(X(ii,1)-X(ii,1)', X(ii,2)-X(ii,2)'),2)) / Exp.S.pixPerDeg;
        mdist = max(mdist, meddist);
    end
    
    if mdist < thresh
        condition = true;
        fprintf('distance=%02.2f\tFinished with %d clusters\n', mdist, numClusts)
    else
        numClusts = numClusts + 1;
        fprintf('distance=%02.2f\tswitching to %d clusters\n', mdist, numClusts)
    end
        
end

if collapse
    % now collapse all clusters with only one unit in it (they're probably
    % noise)
    uclusts = unique(c);
    goodclusts = uclusts(sum(c==uclusts')>1);
    
    newc = ones(size(c));
    for ic = 1:numel(goodclusts)
        newc(goodclusts(ic)==c) = ic + 1;
    end
else
    newc = c;
end

clusts = zeros(size(rfLocations,1), 1);
clusts(ix) = newc;

numClusts = max(clusts);
rois = zeros(numClusts,4);
cmap = lines;
figure(2); clf
for c = 1:numClusts
    ii = clusts==c;
    roix = [min(rfLocations(ii,1)) max(rfLocations(ii,1))];
    roiy = [min(rfLocations(ii,2)) max(rfLocations(ii,2))];
    % expand by 20%
    roix = roix + ([-1 1] .* (.2*abs(roix)));
    roiy = roiy + ([-1 1] .* (.2*abs(roiy)));
    
    % ROI must be at least this big
    scaleecc = max(3*Exp.S.pixPerDeg,1*hypot(mean(rfLocations(ii,1)), mean(rfLocations(ii,2))));
    
    xoff = scaleecc - diff(roix);
    yoff = scaleecc - diff(roiy);
    
    if xoff > 0
        roix = roix + [-1 1].*xoff;
    end
    
    if yoff > 0
        roiy = roiy + [-1 1].*yoff;
    end
    
    rois(c,:) = [roix(1) roiy(1) roix(2) roiy(2)]/Exp.S.pixPerDeg;
    plot(rfLocations(ii,1), rfLocations(ii,2), 'o', 'Color', cmap(c,:)); hold on
    plot(roix, roiy([1 1]), 'Color', cmap(c,:))
    plot(roix, roiy([2 2]), 'Color', cmap(c,:))
    plot(roix([1 1]), roiy, 'Color', cmap(c,:))
    plot(roix([2 2]), roiy, 'Color', cmap(c,:))
end
xlabel('Position (pixels)')
ylabel('Position (pixels)')
title('RF clusters')


%% loop over clusters and recompute
statclust = repmat(stat, numClusts,1);

numBins = 15;

for cc = 1:numClusts
    fprintf('Running cluster %d/%d\n', cc, numClusts)
    ROI = rois(cc,:);
    binSize = max((ROI(3)-ROI(1))/numBins, .25);
    statclust(cc) = spat_rf_helper(Exp, 'ROI', ROI, ...
        'binSize', binSize, 'sdthresh', 5);
end

%%
for cc = 1:numClusts
    fprintf('Running cluster %d/%d\n', cc, numClusts)
    ROI = rois(cc,:);
    stat2 = spat_rf_helper_regress(Exp, 'ROI', ROI, ...
        'binSize', (ROI(3)-ROI(1))/numBins, 'sdthresh', 3);
end

%%
cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(1); clf
subplot(1,2,1)
imagesc(stat1.rf(:,:,cc))
subplot(1,2,2)
imagesc(stat2.rf(:,:,cc))
axis xy

%%
figure(2); clf
plot(stat1.rfLocations(:,1), stat1.rfLocations(:,2), 'o'); hold on
plot(stat2.rfLocations(:,1), stat2.rfLocations(:,2), 'o');



%% New way: forward correlation

% build default options
ip = inputParser();
ip.addParameter('ROI', [-14 -10 14 10])
ip.addParameter('binSize', 1)
ip.addParameter('numlags', 15)
ip.addParameter('numspace', 20)
ip.addParameter('plot', true)
ip.parse()

%% build stimulus matrix for spatial mapping
eyePos = Exp.vpx.smo(:,2:3);

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', ip.Results.ROI*Exp.S.pixPerDeg, 'binSize', ip.Results.binSize*Exp.S.pixPerDeg, 'eyePos', eyePos);

%% do forward correlation
d = size(Xstim, 2);
NC = size(RobsSpace,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

win = [-5 15];
num_lags = diff(win)+1;
mrf = zeros(num_lags, d, NC);

% smooth spike trains
Rdelta = imgaussfilt(RobsSpace-mean(RobsSpace), [1 0.01]);

% loop over stimulus dimensions
for k = 1:d
    fprintf('%d/%d\n', k, d)

    ind = find(Xstim(:,k) > 1);
    s = fliplr(makeStimRows(Xstim(:,k), num_lags));
    s = circshift(s, win(1));
    s = s';
    an = s*Rdelta;
    
    mrf(:,k,:) = an;
end

%% plot / compute cell-by cell quantities
figure(2); clf

[xx,yy]=meshgrid(opts.xax, opts.yax);

rfLocations = nan(NC,2);

fs_stim = round(1/median(diff(opts.frameTimes)));
xax = 1e3*(win(1):win(2))/fs_stim;

for cc = 1:NC
    
    subplot(sx,sy,cc)
    I = mrf(:,:,cc); % individual neuron STA
    
    s = std(reshape(I(xax<0, :), [], 1)); %standard deviation of STA before stim onset (noise estimate)
    t = abs(I./s) > 4; % does the unit have significant excursions?
    
    if sum(t(:)) < 2
        axis off
        continue
    end
    
    % get center xy
    I(~t) = 0;
    [i,j] = find(abs(I)==max(abs(I(:))));
    
    Ispace = reshape(I(i,:), opts.dims);
    
    imagesc(opts.xax, opts.yax, Ispace); hold on
    
    pow = 10;
    Ispace = Ispace.^pow ./ sum(Ispace(:).^pow);
    
    % get softmax center
    x0 = xx(:)'*Ispace(:);
    y0 = yy(:)'*Ispace(:);
    
    plot(x0, y0, 'or')
    
    rfLocations(cc,:) = [x0 y0];
    
    title(cc)

end

%%
figure(2); clf
plot(rfLocations(:,1), rfLocations(:,2), 'o')
xlim(opts.xax([1 end]))
ylim(opts.yax([1 end]))

% distance matrix
cmap = lines;
d = hypot(rfLocations(:,1)-rfLocations(:,1)', rfLocations(:,2)-rfLocations(:,2)')/Exp.S.pixPerDeg;
histogram(nanmedian(d,2), 100)

if any(d(:) > 2) % if any centers are more than 2 degrees away, try to group them
    
    ix = ~any(isnan(rfLocations),2);
    n = 3;
    AIC = nan(n,1);
    BIC = nan(n,1);

    gmdists = cell(n,1);
    for c = 1:n
        try
            evalc('gmdists{c} = fitgmdist(rfLocations(ix,:), c, ''CovarianceType'', ''diagonal'', ''RegularizationValue'', .1);');
            gmdist = gmdists{c};

            AIC(c) = gmdist.AIC;
            BIC(c) = gmdist.BIC;
            
%             contourf(xx,yy,reshape(gmdist.pdf([xx(:) yy(:)]), size(xx)))
        end
        
    end
    
    [~, id] = min(BIC);
    id = max(id, 2); % if we entered this condition, at least two groups
    gmdist = gmdists{id};
    
    % cluster and show RF clusters
    clusts = cluster(gmdist, rfLocations);
    mostCommon = mode(clusts);
    clusts(isnan(clusts)) = mostCommon;
    
else
    clusts = ones(size(rfLocations,1),1);
end

figure(1); clf
for c = 1:id
    ii = clusts==c;
    roix = [min(rfLocations(ii,1)) max(rfLocations(ii,1))];
    roiy = [min(rfLocations(ii,2)) max(rfLocations(ii,2))];
    % expand by 10%
    roix = roix + ([-1 1] .* (.1*abs(roix)));
    roiy = roiy + ([-1 1] .* (.1*abs(roiy)));
    
    plot(rfLocations(ii,1), rfLocations(ii,2), 'o', 'Color', cmap(c,:)); hold on
    plot(roix, roiy([1 1]), 'Color', cmap(c,:))
    plot(roix, roiy([2 2]), 'Color', cmap(c,:))
    plot(roix([1 1]), roiy, 'Color', cmap(c,:))
    plot(roix([2 2]), roiy, 'Color', cmap(c,:))
end
    
%%

%%
% figure(2); clf
% imagesc(d)

%% Now try for gratings

forceHartley = true;
[Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', true);
Robs = Robs(:,Exp.osp.cids);

if numel(unique(grating.oris)) > 20 % stimulus was not run as a hartley basis
    forceHartley = false;
    [Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', false);
    Robs = Robs(:,Exp.osp.cids);
end

%%

d = size(Stim{1}, 2);
win = [-5 25];
num_lags = diff(win)+1;
mrf = zeros(num_lags, d, NC);
srf = zeros(num_lags, d, NC);

Rdelta = Robs-mean(Robs);

for k = 1:d
    ind = find(diff(Stim{1}(:,k))>0);
    [an, sd] = eventTriggeredAverage(Rdelta, ind, win);
    sd = sd./sqrt(numel(ind));
    mrf(:,k,:) = an;
    srf(:,k,:) = sd;
end


figure(1); clf

for cc = 1:NC
    subplot(sx, sy, cc)
    I = mrf(:,:,cc)*grating.fs_stim;
    xax = 1e3*(win(1):win(2))/grating.fs_stim;
    s = std(reshape(I(xax<0, :), [], 1));
    t = abs(I./s) > 4;
    plot( xax , I, 'k'); hold on
    I(~t) = nan;
    plot(xax, I, 'r.')
%     imagesc(I .* t)
    
    axis tight
    title(cc)
end