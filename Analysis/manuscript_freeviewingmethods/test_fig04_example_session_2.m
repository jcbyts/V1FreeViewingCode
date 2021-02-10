
addpath Analysis/manuscript_freeviewingmethods/

%% Load session
sessid = 5;

try
    sorter = 'jrclustwf';
    Exp = io.dataFactoryGratingSubspace(sessid, 'spike_sorting', sorter);
catch % otherwise, use Kilosort
    sorter = 'kilowf';
    Exp = io.dataFactoryGratingSubspace(sessid, 'spike_sorting', sorter);
end
           

%% spatial
BIGROI = [-14 -10 14 10];
stat = spat_rf_helper(Exp, 'sdthresh', 4, 'ROI', BIGROI, 'binSize', .3, 'spikesmooth', 1);


%%
figure(1); clf
NC = size(stat.rf, 3);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC))*2;

for cc = 1:NC
    subplot(sx, sy, (cc-1)*2 + 1)
    plot.errorbarFill(stat.timeax, stat.rf(:,stat.spmx(cc),cc)*stat.fs_stim, stat.rfsd(:,stat.spmx(cc),cc)*stat.fs_stim, 'b', 'EdgeColor', 'b'); hold on
    plot.errorbarFill(stat.timeax, stat.rf(:,stat.spmn(cc),cc)*stat.fs_stim, stat.rfsd(:,stat.spmn(cc),cc)*stat.fs_stim, 'r', 'EdgeColor', 'r');
    
    subplot(sx, sy, (cc-1)*2 + 2)
    imagesc(stat.xax, stat.yax, stat.spatrf(:,:,cc)); hold on
    plot.plotellipse(stat.rffit(cc).mu, stat.rffit(cc).C, 1, 'r');
    axis xy
end
% stat

%%


cc = 0;
%% plot
NC = size(stat.rf,3);
thresh = 6;
zthresh = (1 - normcdf(thresh));

for cc = 1:NC
    % cc = cc + 1;
    % if cc > NC
    %     cc = 1;
    % end
    
    %
    figure(cc); clf
    
    
    subplot(1,3,1)
    
    rf = stat.rf(:,:,cc);
    zrf = (rf)/stat.sdbase(cc);
    
    % for ilag = 1:size(zrf,1)
    %     zrf(ilag,:) = reshape(imgaussfilt(reshape(zrf(ilag,:), stat.dim), 1), 1, []);
    % end
    
    plot(stat.timeax, zrf); hold on
    
    thresh = max(max(abs(zrf(stat.timeax<=0,:))))*1.15;
    sig = abs(zrf) > thresh;
    plot(xlim, [1 1]*thresh, 'r')
    plot(stat.timeax(stat.peaklag(cc))*[1 1], ylim, 'b')
    % plot(xlim, [1 1]*thresh*1.3636, 'r--')
    
    plot([0 0], ylim, 'k')
    nsig = sum(sig(:)); %mean(mean(sig)) > zthresh;
    title(sprintf("sig = %d", nsig))
    
    subplot(1,3,2)
    
    [bestlag,~] = find(max(rf(:))==rf);
    [~, spmx] = max(rf(bestlag,:));
    [~, spmn] = min(rf(bestlag,:));
    plot(stat.timeax, rf(:,spmx)*stat.fs_stim, 'b-o'); hold on
    plot(stat.timeax, rf(:,spmn)*stat.fs_stim, 'r-o');
    
    subplot(1,3,3)
    
    I = reshape(zrf(bestlag,:), stat.dim);
    spatrf = reshape(rf(bestlag,:), stat.dim);
    
    I = sqrt(I.^2);
%     
%     I = imgaussfilt(I,1);
    imagesc(stat.xax, stat.yax, spatrf); hold on
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
    par0 = [x0 y0 5 0 4];
    
    
    gfun = @(params, X) params(5) + (mxI - params(5)) * exp(-.5 * sum(((X-[params(1) params(2)])*pinv([params(3) params(4); params(4) params(3)]'*[params(3) params(4); params(4) params(3)])).*(X-[params(1) params(2)]),2));
    
    [xx,yy] = meshgrid(stat.xax,  stat.yax);
    X = [xx(:) yy(:)];
    
    % least-squares
    lb = [min(xx(:)) min(yy(:)) 0 -1000 mnI 5];
    ub = [max(xx(:)) max(yy(:)) 10 1000 thresh mxI];
    
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
    fprintf('%d) ecc: %02.2f, area: %02.2f, r^2:%02.2f\n', cc, ecc, ar, r2)
end

%% Now try for gratings

grat = grat_rf_helper(Exp, 'fitminmax', true, 'sdthresh', 6);

%%
cc = cc + 1;
if cc > NC
    cc = 1;
end

grat.rffit(cc)
figure(1); clf
set(gcf, 'Color', 'w')
subplot(1,3,1)
srf = squeeze(grat.rf(grat.peaklag(cc),:,cc));
imx = find(srf==max(srf));
imn = find(srf==min(srf));

cmap = lines;
plot.errorbarFill(grat.timeax, grat.rf(:, imx, cc)*grat.fs_stim, grat.rfsd(:, imx, cc)*grat.fs_stim, 'b', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:)); hold on
plot.errorbarFill(grat.timeax, grat.rf(:, imn, cc)*grat.fs_stim, grat.rfsd(:, imn, cc)*grat.fs_stim, 'b', 'FaceColor', cmap(2,:), 'EdgeColor', cmap(2,:)); hold on
axis tight
plot(xlim, [0 0], 'k--')
xlabel('Time lag (ms)')
ylabel('\Delta Firing Rate')
title(sprintf('Unit: %d', cc))

subplot(1,3,2)
imagesc(grat.xax, grat.yax, grat.rffit(cc).srf);
axis xy
xlabel('sf_x')
ylabel('sf_y')
title('Measured RF')

subplot(1,3,3)
imagesc(grat.xax, grat.yax, grat.rffit(cc).srfHat);
axis xy
xlabel('sf_x')
ylabel('sf_y')
title('Parametric Fit')

% prf.parametric_rf(


%%
forceHartley = true;
[Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', true);
Robs = Robs(:,Exp.osp.cids);

if numel(unique(grating.oris)) > 20 % stimulus was not run as a hartley basis
    forceHartley = false;
    [Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', false);
    Robs = Robs(:,Exp.osp.cids);
end

%%
NC = size(Robs,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

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
    t = abs(I./s) > thresh;
    plot( xax , I, 'k'); hold on
    I(~t) = nan;
    plot(xax, I, 'r.')
%     imagesc(I .* t)
    
    axis tight
    title(cc)
end

%%
cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(2); clf
subplot(1,3,1)
rf = mrf(:,:,cc);
sd = srf(:,:,cc);
[bestlag, ~] = find(rf==max(rf(:)));

errorbar(rf(bestlag,:), sd(bestlag,:))

subplot(1,3,2)
frf = reshape(rf(bestlag,:), grating.dim)' * grating.fs_stim;

imagesc(grating.oris, grating.cpds, frf); axis xy

subplot(1,3,3)

% --- fit parametric RF
mxFR = max(frf(:));
mnFR = min(frf(:));

[y0,x0] = find(frf==mxFR, 1);

x0 = grating.oris(x0);
y0 = grating.cpds(y0);

if forceHartley
    [theta0, cpd0] = cart2pol(x0, y0);
else
    theta0 = x0/180*pi;
    cpd0 = y0;
end


mxI = max(frf(:));
mnI = min(frf(:));

% least-squares for parametric rf
fun = @(params) (prf.parametric_rf([params mxI mnI], X) - frf(:)).^2;

if forceHartley % space is hartley
    [xx,yy] = meshgrid(grating.oris, grating.cpds);
    [X(:,1), X(:,2)] = cart2pol(xx(:), yy(:));
else
    [xx,yy] = meshgrid(grating.oris/180*pi, grating.cpds);
    X = [xx(:) yy(:)];
end

par0 = [2 theta0 cpd0 1]; % initial parameters


% least-squares
try
    %  parameters are:
    %   1. Orientation Kappa
    %   2. Orientation Preference
    %   3. Spatial Frequency Preference
    %   4. Spatial Frequency Sigma
    %   5. Gain
    %   6. Offset
    lb = [.01 -pi 0.1 .2];
    ub = [10 2*pi 20 1];
    
    evalc('phat = lsqnonlin(fun, par0, lb, ub)');
    %         evalc('phat = lsqcurvefit(fun, par0, X, I(:), lb, ub);');
catch
    phat = nan(size(par0));
end

phat = [phat mxI mnI];

Ihat = reshape(prf.parametric_rf(phat, X), size(xx));
Ihat0 = reshape(prf.parametric_rf([par0 phat(5:6)], X), size(xx));

imagesc(grating.oris, grating.cpds,Ihat)
axis xy
colorbar

rsquared(frf(:), Ihat(:))