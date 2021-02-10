function [stat, opts] = spat_rf_helper(Exp, varargin)
% [stat, opts] = spat_rf_helper(Exp, varargin)
% Calculate spatiotemporal RF using forward correlation (including
% pre-stimulus lags)
% This is like a basic spike-triggered average but it calculates the
% modulation in forward time and using units of spike rate. Therefore, the
% interpretation is what the change in spike rate would be for a flash at
% each location. This the impulse reponse function of the neuron over
% space.
%
% Inputs:
%   Exp <struct> Main experiment structure
% Optional:
%   'ROI',       [-14 -10 14 10]
%   'binSize',   1
%   'win',       [-5 15]
%   'numspace',  20
%   'plot',      true
%   'sdthresh',  4
% Output:
%   stat <struct>
%   opts <struct>

% build default options
ip = inputParser();
ip.addParameter('ROI', [-14 -10 14 10])
ip.addParameter('binSize', 1)
ip.addParameter('win', [-5 15])
ip.addParameter('numspace', 20)
ip.addParameter('plot', true)
ip.addParameter('sdthresh', 4)
ip.addParameter('boxfilt', 5)
ip.addParameter('spikesmooth', 0)
ip.addParameter('stat', [])
ip.addParameter('debug', false)
ip.parse(varargin{:})

%% build stimulus matrix for spatial mapping

if ~isempty(ip.Results.stat)
    % existing STA was passed in. re-do neuron stats, but don't recompute
    % forward correlation
    stat = ip.Results.stat;
    NC = numel(stat.thresh);
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    nlags = numel(stat.timeax);
    
else
    eyePos = Exp.vpx.smo(:,2:3);
    
    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', ip.Results.ROI*Exp.S.pixPerDeg, 'binSize', ip.Results.binSize*Exp.S.pixPerDeg, 'eyePos', eyePos);
    
    %% do forward correlation
    d = size(Xstim, 2);
    NC = size(RobsSpace,2);
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    
    win = ip.Results.win;
    num_lags = diff(win)+1;
    mrf = zeros(num_lags, d, NC);
    srf = zeros(num_lags, d, NC);
    
    % smooth spike trains
    if ip.Results.spikesmooth > 0
        Rdelta = imgaussfilt(RobsSpace-mean(RobsSpace), [ip.Results.spikesmooth 0.001]); % Gaussian smoothing along the time dimension
    else
        Rdelta = RobsSpace-mean(RobsSpace);
    end
    
    % loop over stimulus dimensions
    disp('Running forward correlation...')
    for k = 1:d
        %     fprintf('%d/%d\n', k, d)
        s = fliplr(makeStimRows(Xstim(:,k), num_lags)); % flip for forward time-embedded stimulus
        s = circshift(s, win(1)); % shift back by the pre-stimulus lags
        for cc = 1:NC
            x = s.*Rdelta(:,cc);
            mrf(:,k,cc) = sum(x)./sum(s);
            srf(:,k,cc) = std(x)./sum(s);
        end
        
    end
    disp('Done')
    
    stat.rf = mrf;
    stat.rfsd = srf;
    
    fs_stim = round(1/median(diff(opts.frameTimes)));
    tax = 1e3*(win(1):win(2))/fs_stim;
    
    stat.timeax = tax;
    stat.xax = opts.xax/Exp.S.pixPerDeg;
    stat.yax = opts.yax/Exp.S.pixPerDeg;
    stat.fs_stim = fs_stim;
    stat.dim = opts.dims;
    stat.ppd = Exp.S.pixPerDeg;
    stat.roi = ip.Results.ROI;
    stat.spatrf = zeros([opts.dims NC]);
    stat.peaklag = nan(NC, 1);
    stat.peaklagt = nan(NC, 1);
    
    stat.maxV = nan(NC, 1);
    stat.thresh = nan(NC, 1);
    
    stat.spmx = nan(NC,1);
    stat.spmn = nan(NC,1);
    stat.cgs = Exp.osp.cgs(:);
    
end

for cc = 1:NC
    stat.rffit(cc).warning = 1;
end
%% compute cell-by cell quantities / plot (if true)

if ip.Results.plot
    fig = figure; clf
end

rfLocations = nan(NC,2);

% build input for Gaussian fit
[xx,yy] = meshgrid(stat.xax,  stat.yax);
X = [xx(:) yy(:)];

for cc = 1:NC
    
    if ip.Results.plot
        figure(fig)
        ax = subplot(sx,sy,cc);
    end
    
    rfflat = stat.rf(:,:,cc)*stat.fs_stim; % individual neuron STA
    
    % smooth RF / calculate Median Absolute Deviations (MAD)
    rf3 = reshape(rfflat, [numel(stat.timeax), stat.dim]); % 3D spatiotemporal tensor
    % clip edges
    rf3([1 end],:,:) = 0;
    rf3(:,[1 end],:) = 0;
    rf3(:,:,[1 end]) = 0;
    
    rf = imboxfilt3(rf3, ip.Results.boxfilt*[1 1 1]); % average in space and time
    
    adiff = abs(rf - median(rf(:))); % positive difference
    mad = median(adiff(:)); % median absolute deviation
    
    adiff = (rf - median(rf(:))) ./ mad; % normalized (units of MAD)
    
    nlags = size(rf,1);
    
    % compute threshold using pre-stimulus lags
    thresh = max(max(adiff(stat.timeax<=0,:)))*1.25;
    
    % threshold and find the largest connected component
    bw = bwlabeln(adiff > thresh);
    s = regionprops3(bw);
    if isempty(s)
        % no connected components found (nothing crossed threshold)
        continue
    end
    
    [maxV, bigg] = max(s.Volume); % largest connected component is the RF
    ytx = s.Centroid(bigg,:); % location (in space / time)
    peaklagt = interp1(1:nlags, stat.timeax, ytx(2)); % peak lag using centroid
    
    nlags = size(rf3,1);
    peaklagf = max(floor(ytx(2))-1, find(stat.timeax==0));
    peaklagc = min(ceil(ytx(2))+1, nlags);
    
    % spatial RF
    srf = squeeze(mean(rf3(peaklagf:peaklagc,:,:), 1));
    stat.spatrf(:,:,cc) = srf;
    
    % fit gaussian    
    I = abs(srf-mean(srf(:))); % change of variable name (why?)
    
    % initialize mean
    x0 = interp1(1:numel(stat.xax), stat.xax, ytx(3));
    y0 = interp1(1:numel(stat.yax), stat.yax, ytx(1));
    
    % range
    mnI = min(I(:));
    mxI = max(I(:));
    
    % initial parameter guess
    par0 = [x0 y0 max(2,hypot(x0, y0)*.5) 0 mean(I(:))];
    
    % gaussian function
    gfun = @(params, X) params(5) + (mxI - params(5)) * exp(-.5 * sum(((X-[params(1) params(2)])*pinv([params(3) params(4); params(4) params(3)]'*[params(3) params(4); params(4) params(3)])).*(X-[params(1) params(2)]),2));
    
    % bounds
    lb = [min(xx(:)) min(yy(:)) 0 -1 mnI 5]; %#ok<*NASGU>
    ub = [max(xx(:)) max(yy(:)) 10 1 thresh mxI];
    
    % least-squares
    try
%         evalc('[phat,RESNORM,RESIDUAL,EXITFLAG] = lsqcurvefit(gfun, par0, X, I(:), lb, ub);');
        options = statset('RobustWgtFun', 'bisquare', 'Tune', 10, 'MaxIter', 1000);
        cstr = evalc("[phat,R,~,COVB,~,minfo] = nlinfit(X, I(:), gfun, par0, options);");
        if contains(cstr, 'Warning:')
            warningFlag = true;
        else
            warningFlag = false;
        end
        CI = nlparci(phat, R, 'covar', COVB);
    catch
        phat = nan(size(par0));
        CI = nan(numel(phat), 2);
        warningFlag = true;
    end
    % convert paramters
    mu = phat(1:2);
    C = [phat(3) phat(4); phat(4) phat(3)]'*[phat(3) phat(4); phat(4) phat(3)];
    
    mushift = hypot(mu(1)-x0, mu(2)-y0);
    rfflat = reshape(rf, nlags, []);
    peaklag = round(ytx(2));
    [~, spmx] = max(rfflat(peaklag,:));
    [~, spmn] = min(rfflat(peaklag,:));
    
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
    
    if ip.Results.plot
        set(fig, 'currentaxes', ax)
        imagesc(stat.xax, stat.yax, srf); hold on
        colormap(plot.viridis)
        axis xy
        hold on
        xlabel('Azimuth (d.v.a)')
        ylabel('Elevation (d.v.a)')

        plot.plotellipse(mu, C, 1, 'r');

        title(cc)
    end
    
    if ip.Results.debug
        
        figure(1); clf
        subplot(1,2,1)
        plot(stat.timeax, max(reshape(adiff, numel(stat.timeax), []),[],2)); hold on
        plot(xlim, thresh*[1 1], 'k')
        plot(peaklagt*[1 1], ylim, 'k')
        
        subplot(1,2,2)
        imagesc(stat.xax, stat.yax, srf); hold on
        plot(x0, y0, '+r')
        plot.plotellipse(mu, C, 1, 'r');
        keyboard
       
    end
    
    rfLocations(cc,:) = mu;
    
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
    stat.rffit(cc).amp = mxI;
    stat.rffit(cc).C = C;
    stat.rffit(cc).mu = mu;
    stat.rffit(cc).r2 = r2;
    stat.rffit(cc).ar = ar;
    stat.rffit(cc).ecc = ecc;
    stat.rffit(cc).beta = phat;
    stat.rffit(cc).betaCi = CI;
    stat.rffit(cc).warning = warningFlag;
    stat.rffit(cc).mushift = mushift;
    
    stat.maxV(cc) = maxV;
    stat.peaklagt(cc) = peaklagt;
    stat.peaklag(cc) = peaklag;
    stat.spmx(cc) = spmx;
    stat.spmn(cc) = spmn;
    stat.mads(cc) = mad;
    stat.thresh(cc) = thresh;
    
end

stat.rfLocations = rfLocations;

% estimate significance
stat.sig = false(NC,1);
for cc = 1:NC
    if isfield(stat.rffit(cc), 'mu') && ~isempty(stat.rffit(cc).mu)
        ms = (stat.rffit(cc).mushift/stat.rffit(cc).ecc);
        sz = (stat.maxV(cc)./stat.rffit(cc).ecc);
        stat.sig(cc) = ms < .25 & sz > 5;
    end
end