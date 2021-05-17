function stat = grat_rf_helper(Exp, varargin)
% Get Grating Receptive Field using forward correlation
% stat = grat_rf_helper(Exp, varargin)
% Inputs:
%   Exp <struct>    main experiment struct
%   
% Optional:
%   'win'         [-5 25]
%   'plot'        true
%   'fitminmax'   false
%   'stat'        <struct>      use existing forward correlation and refit
%                               gaussian
%   'debug'       <logical>     enter debugger to pause/evaluate each unit
%   'boxfilt'     <odd integer> if > 1, filter RF with spatio temporal boxcar
%                               before thresholding
%   'sftuning'    <string>      which parameterization of the spatial
%                               frequency tuning curve to use
%                               ('raisedcosine', 'loggauss')
%   'upsample'    <integer>     upsample spatial RF before fitting (acts as
%                               regularization (default: 1 = no upsampling)
%   'vm01'        <logical>     normalize the von mises between 0 and 1
%                               (default: true -- will automatically relax 
%                               if real bandwidth is too wide)
%   

ip = inputParser();
ip.addParameter('win', [-5 25])
ip.addParameter('plot', true)
ip.addParameter('fitminmax', false)
ip.addParameter('stat', [])
ip.addParameter('debug', false)
ip.addParameter('boxfilt', 1, @(x) mod(x,2)~=0);
ip.addParameter('sftuning', 'loggauss', @(x) ismember(x, {'loggauss', 'raisedcosine'}))
ip.addParameter('upsample', 1)
ip.addParameter('vm01', true)
ip.parse(varargin{:});

switch ip.Results.sftuning
    case 'loggauss'
        log_gauss = 1;
    case 'raisedcosine'
        log_gauss=0;
end



% --- load stimulus
% Two stimulus sets were used in the dataset. Hartley basis, or Gratings
% defined by orientation and spatial frequency
forceHartley = true; % 
[Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', true);
Robs = Robs(:,Exp.osp.cids);

if numel(unique(grating.cpds)) > 20 % stimulus was not run as a hartley basis
    forceHartley = false;
    [Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', false);
    Robs = Robs(:,Exp.osp.cids);
end

NC = size(Robs,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

if ~isempty(ip.Results.stat)
    stat = ip.Results.stat;
%     win = stat.timeax([1 end])/1e3*grating.fs_stim;
else
    % Do forward correlation
    win = ip.Results.win;
    
    d = size(Stim{1}, 2);
    
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
    
    
    stat.rf = mrf;
    stat.rfsd = srf;
    stat.timeax = 1e3*(win(1):win(2))/grating.fs_stim;
    stat.dim = grating.dim;
    stat.xax = grating.oris;
    stat.yax = grating.cpds;
    stat.fs_stim = grating.fs_stim;
    
    stat.peaklag = zeros(NC,1);
    stat.sdbase = zeros(NC,1);
    stat.r2 = zeros(NC,1);
    
end

if ip.Results.plot
    figure(1); clf
end

% initialize variables
stat.peaklagt = nan(NC,1);
stat.thresh = nan(NC,1);
stat.maxV = zeros(NC,1);
stat.rffit = repmat(struct('pHat', [], ...
    'oriPref', [], ...
    'oriBandwidth', [], ...
    'sfPref', [], ...
    'sfBandwidth', [], ...
    'base', [], ...
    'amp', [], ...
    'r2', [], ...
    'srfHat', [], ...
    'cid', []), NC, 1);

stat.ishartley = forceHartley;
stat.sftuning = ip.Results.sftuning;

% Main Loop over cells:
% 1) Threshold RF, find centroid
% 2) Fit parametric model initialized on centroid
for cc = 1:NC
    fprintf('Fitting unit %d/%d...\t', cc, NC)
    
    % initialize the von-mises to be normalized. If the bandwidth is too
    % wide, then relax this constraint
    vm01 = ip.Results.vm01; % is the von-mises bound between 0 and 1 (messes up bandwidth calculation)
    
    % time x space (flattened) receptive field in units of spikes/sec
    rfflat = stat.rf(:,:,cc)*grating.fs_stim;
    
    
    % time x space x space RF
    rf3 = reshape(rfflat, [numel(stat.timeax), stat.dim]); % 3D spatiotemporal tensor
    
    % spatiotemporal average before threshold?
    if ip.Results.boxfilt > 1
        rf = imboxfilt3(rf3, ip.Results.boxfilt*[1 1 1]); % average in space and time
    else
        rf = rf3;
    end
    
    % Calculate median absolute deviations
    adiff = abs(rf - nanmedian(rf(:))); % absolute difference
    mad = nanmedian(adiff(:)); % median absolute deviation
    
    adiff = adiff ./ mad; % normalized (units of MAD)
    
    nlags = size(rf,1);
    
    % compute threshold using pre-stimulus lags
    adiffpre = adiff(stat.timeax<=0,:);
    thresh = max(adiffpre(:)); % largest value pre-stimulus
    thresh = max(thresh, .25*max(adiff(:)));
    
    stat.sdbase(cc) = mad;
    stat.thresh(cc) = thresh;
    
    if ip.Results.plot && ~ip.Results.debug
        subplot(sx, sy, cc)
        plot( stat.timeax , reshape(adiff, nlags, [])); hold on
        plot( xlim, thresh*[1 1], 'k', 'Linewidth', 2)
        axis tight
        title(cc)
    end
    
    % threshold and find the largest connected component
    bw = bwlabeln(adiff > thresh);
    s = regionprops3(bw);
    
    if isempty(s)
        fprintf('No RF\n')
        % no connected components found (nothing crossed threshold)
        continue
    end
    
    [maxV, bigg] = max(s.Volume); % largest connected component is the RF
    
    rfmask = abs(rf3) .* (bw == bigg);
    rfmask = rfmask / sum(rfmask(:));
    
    % if debug on, plot every lag of the forward correlation
    if ip.Results.debug
        figure(10); clf
        plot( stat.timeax , reshape(adiff, nlags, [])); hold on
        plot( xlim, thresh*[1 1], 'k', 'Linewidth', 2)
        axis tight
        title(cc)
        
        figure(11); clf
        nx = ceil(sqrt(numel(stat.timeax)));
        ny = round(sqrt(numel(stat.timeax)));
        for ilag = 1:numel(stat.timeax)
            subplot(nx, ny, ilag)
            imagesc(squeeze(rfmask(ilag,:,:)), [min(rfmask(:)) max(rfmask(:))]); hold on
%             imagesc(squeeze(rf(ilag,:,:)), [min(rf(:)) max(rf(:))]); hold on
%             evalc("contour(squeeze(rfmask(ilag,:,:)), 5, 'r')");
        end
        keyboard
    end
    
    stat.maxV(cc) = maxV;
    
    if maxV < ip.Results.boxfilt
        fprintf('No RF\n')
        continue
    end
    
    
    % get center of mass for the mask
    sz = size(rfmask);
    [tt,yy,xx] = ndgrid(1:sz(1), 1:sz(2), 1:sz(3));
    ytx = [yy(:)'*rfmask(:) tt(:)'*rfmask(:) xx(:)'*rfmask(:)];
    
    % peak lag in time and indices
    peaklagt = interp1(1:nlags, stat.timeax, ytx(2)); % peak lag using centroid
    peaklagf = floor(ytx(2));
    peaklagc = ceil(ytx(2));
    
    % "spatial" (orientation/frequency) RF
    if peaklagf==peaklagc
        srf = squeeze(rf3(peaklagf,:,:));
    else
        srf = squeeze(mean(rf3(peaklagf:peaklagc,:,:), 1)); % average over peak lags
    end
    
    stat.rffit(cc).srf = srf;
    
    [y0,x0]=find(srf==max(srf(:)));
    
    % initialize mean for fit based on centroid
    x0 = interp1(1:numel(stat.xax), stat.xax, x0);
    y0 = interp1(1:numel(stat.yax), -stat.yax, y0);
    
    stat.peaklag(cc) = round(ytx(2));
    stat.peaklagt(cc) = peaklagt;

    % fit parametric receptive field model
    I = srf; % change of variable
    
    % --- fit parametric RF
    mxI = max(srf(:));
    mnI = min(srf(:));
    
    % We have two stimulus sets for the grating condition 
    if forceHartley % stimulus was hartley basis
        
        [ori0, cpd0] = cart2pol(y0, x0); %#ok<*ASGLU>
        [xx,yy] = meshgrid(grating.oris, -grating.cpds);
        [X0(:,1), X0(:,2)] = cart2pol(yy(:), xx(:));
        minSF = min(X0(X0(:,2)>0,2));
        if ip.Results.upsample > 1
            xx = imresize(xx, ip.Results.upsample, 'bilinear');
            yy = imresize(yy, ip.Results.upsample, 'bilinear');
            I = imresize(I, ip.Results.upsample, 'bilinear');
        end
        [X(:,1), X(:,2)] = cart2pol(yy(:), xx(:));
        
        [kx,ky] = meshgrid(grating.oris, -grating.cpds);
        sf = hypot(kx(:), ky(:));
        sfs = min(sf):max(sf);
        
        % fit orientation initial
        oris = 0:(pi/10):pi;
        [Yq, Xq] = pol2cart(oris, cpd0*ones(size(oris)));
    
        r = interp2(grating.oris, -grating.cpds, srf, Xq, Yq);
        oris = oris(~isnan(r))';
        r = r(~isnan(r))';
        x = [oris ones(size(oris(:)))*cpd0];
        
        %         try
        fun = @(params, x) prf.parametric_rf([params(1:2) cpd0 2 params(3:4)], x, log_gauss, vm01);
        evalc("phat = lsqcurvefit(fun, [ori0 1 mxI mnI], x, r);");
        ori0 = phat(2);
        orib0 = phat(1);
        minR = phat(4); %#ok<*NASGU>
        maxR = phat(3);
%         catch
%             ori0 = theta0;
%             orib0 = 0; %initialize with no orientation tuning
%         end
        
        if ip.Results.debug
            figure(99); clf
            plot(oris/pi*180, r, 'o'); hold on
            plot(linspace(0, 180, 100), fun(phat, [linspace(0, pi, 100)', ones(100,1)*cpd0]))
            xlabel('Orientation')
            ylabel('Firing Rate')
            
        end
        
        % fit SF initial
        [Yq, Xq] = pol2cart(ori0*ones(size(sfs)), sfs);
        r = interp2(grating.oris, -grating.cpds, srf, Xq, Yq);
        sfs = sfs(~isnan(r));
        r = r(~isnan(r))';
        x = [ones(size(sfs(:)))*ori0 sfs(:)];
        fun = @(params, x) prf.parametric_rf([orib0 ori0 params], x, log_gauss, vm01);
        
        evalc("phat = lsqcurvefit(fun, [cpd0 2 maxR minR], x, r);");
        if ip.Results.debug
            figure(101); clf
            plot(sfs, r, 'o'); hold on
            plot(linspace(0, 10, 100), fun(phat,[zeros(100,1) linspace(0, 10, 100)']))
            xlabel('Spatial Frequency')
            ylabel('Firing Rate')
        end
        
        cpd0 = max(real(phat(1)), cpd0);
        cpdb0 = real(phat(2));
        
        
    
    else % stimulus was orientation/frequency
        
        
        [xx,yy] = meshgrid(grating.oris/180*pi, grating.cpds);
        X0 = [xx(:), yy(:)];
        minSF = min(grating.cpds(grating.cpds>0));
        if ip.Results.upsample > 1
            xx = imresize(xx, ip.Results.upsample, 'bilinear');
            yy = imresize(yy, ip.Results.upsample, 'bilinear');
            I = imresize(I, ip.Results.upsample, 'bilinear');
        end
        X = [xx(:) yy(:)];
        
        
        
        [cpdIx,oriIx] = find(I==max(I(:)));
        ori0 = xx(1,oriIx)/180*pi;
        cpd0 = yy(cpdIx,1);
        
        % fit orientation initial
        oris = xx(1,:)'/pi*180;
        r = I(cpdIx,:)';
        x = [oris/180*pi ones(size(oris(:)))*cpd0];
        
        fun = @(params, x) prf.parametric_rf([params(1:2) cpd0 cpdb0 params(3:4)], x, log_gauss, vm01);
        evalc("phat = lsqcurvefit(fun, [ori0 1 mxI mnI], x, r);");
        ori0 = phat(2);
        orib0 = phat(1);
        maxR = phat(3);
        minR = phat(4);
        
        if ip.Results.debug
            figure(99); clf
            plot(oris, r, '-o'); hold on
            plot(linspace(0, 180, 100), fun(phat,[linspace(0, pi, 100)' cpd0*ones(100,1)]))
            xlabel('Orientation')
            ylabel('Firing Rate')
        end

        
        % fit SF initial (note: this will be really prone to over fitting
        % because there are so few spatial frequencies tested in many
        % datasets here. We use this as an initialization for the full
        % joint fit)
        sfs = yy(:,1);
        r = I(:,oriIx);
        
        x = [ones(size(sfs(:)))*ori0 sfs(:)];
        fun = @(params, x) prf.parametric_rf([orib0 ori0 params], x, log_gauss, vm01);
        
        evalc("phat = lsqcurvefit(fun, [cpd0 2 maxR minR], x, r);");
        if ip.Results.debug
            figure(101); clf
            plot(sfs, r, 'o'); hold on
            plot(linspace(0, 10, 100), fun(phat,[zeros(100,1) linspace(0, 10, 100)']))
            xlabel('Spatial Frequency')
            ylabel('Firing Rate')
        end
        
        cpd0 = max(real(phat(1)), cpd0);
        cpdb0 = real(phat(2));
        
        
    end
    
    
    % Do the fitting
    if log_gauss
        par0 = [orib0 ori0    cpd0  cpdb0  maxR     minR]; % initial parameters
        lb =   [0     -pi     0.1   -10    0  -inf];
        ub =   [500   2*pi    20    10     2*maxR   maxR];
    else
        par0 = [orib0 ori0 cpd0 2 maxR minR]; % initial parameters
        lb = [0 -pi 0.1 1.5 0 -inf];
        ub = [500 2*pi 20 10 2*maxR   maxR];
    end
    
    
    I = I(:);
    fitix = X(:,2) >= minSF;
    
    % refit min and max
    fun = @(params, x) prf.parametric_rf([par0(1:4) params], x, log_gauss, vm01);
    evalc("phat = lsqcurvefit(fun, [mxI mnI], X(fitix,:), I(fitix), lb(5:6), ub(5:6));");
    par0(5:6) = phat;
    
    % fit everythiing
    fun = @(params, x) prf.parametric_rf(params, x, log_gauss, vm01);
    evalc("phat = lsqcurvefit(fun, par0, X(fitix,:), I(fitix), lb, ub);");
    
    par = phat;
    
        
%         % least-squares with some scaling by firing rate
%         %         lossfun = @(r,lambda) sqrt(r)'*((r - lambda).^2);
%         lossfun = @(r,lambda) sum((r - lambda).^2);
%         
%         
%         fun = @(params) lossfun(I(fitix), prf.parametric_rf(params, X(fitix,:), log_gauss, vm01));
%         
%         opts = optimoptions('fmincon');
%         problem = createOptimProblem('fmincon','objective', ...
%             @(params) fun(params),'x0',par0,'lb',lb, ...
%             'ub',ub,'options',opts);
%         
%         gs = GlobalSearch('Display','none','XTolerance',1e-3,'StartPointsToRun','bounds');
%         ms = MultiStart(gs,'UseParallel',true);
%         [phat,~] = run(ms,problem, 100);
        
%         fun = @(params, x) prf.parametric_rf(params, x, log_gauss, vm01);
%         evalc("phat = lsqcurvefit(fun, par0, X(fitix,:), I(fitix));");
        
%         par = phat;
        
        % ori bandwidth
        try
            orientation = 0:.1:pi;
            if vm01
                orientationTuning = (exp(par(1)*cos(orientation).^2) - 1) / (exp(par(1))-1);
            else
                orientationTuning = (exp(par(1)*cos(orientation).^2)) / (exp(par(1)));
            end
            
            if vm01 % bounded von Mises, use numerical interpolation to find bandwidth
                id = find(orientationTuning == min(orientationTuning), 1);
                hbw = interp1(orientationTuning(1:id), orientation(1:id), .5);
            else
                hbw = acos(abs(sqrt( log(0.5)/par(1) +1)));
            end
            
            obw = hbw*2;
            
        catch
            obw = nan;
        end
        
        
        % Handle Edge Cases in orientation tuning
        if (vm01 && obw >=1.39) % bandiwdth too wide for vm01
            vm01 = false;
            
            fun = @(params, x) prf.parametric_rf(params, x, log_gauss, vm01);
            evalc("phat = lsqcurvefit(fun, par0, X(fitix,:), I(fitix), lb, ub);");
        
            par = phat;
        
            % ori bandwidth
            orientation = 0:.1:pi;
            orientationTuning = (exp(par(1)*cos(orientation).^2)) / (exp(par(1)));
            hbw = acos(abs(sqrt( log(0.5)/par(1) +1)));
            obw = hbw*2;
            if real(obw)==0 % undefined par(1) led to abs(cos(x)) > 1
                obw = nan;
            end
        end
        

        if isnan(obw) % no orientation tuning
            % refit without orientation tuning
%             fun = @(params) lossfun(I(fitix), prf.parametric_rf([0 0 params], X(fitix,:), log_gauss, vm01));
%             
%             opts = optimoptions('fmincon');
%             problem = createOptimProblem('fmincon','objective', ...
%                 @(params) fun(params),'x0',phat(3:end),'lb',lb(3:end), ...
%                 'ub',ub(3:end),'options',opts);
%             
%             gs = GlobalSearch('Display','none','XTolerance',1e-3,'StartPointsToRun','bounds');
%             ms = MultiStart(gs,'UseParallel',true);
%             [phat,~] = run(ms,problem, 100);
            
            
            fun = @(params, x) prf.parametric_rf([0 0 params], x, log_gauss, vm01);
            evalc("phat = lsqcurvefit(fun, phat(3:end), X(fitix,:), I(fitix), lb(3:end), ub(3:end));");
            phat = [0 0 phat]; %#ok<*AGROW>
            
            obw = nan;
        end
        
        % -- GET METRICS
        par = phat;
        
        if log_gauss
            % sf Bandwidth (if log-gaussian)
            a = sqrt(-log(.5) * par(4)^2 * 2);
            b1 = (par(3) + 1) * exp(-a) - 1;
            b2 = (par(3) + 1) * exp(a) - 1;
            sfbw = b2 - b1;
        else
            % sf Bandwidth (if raised cosine)
            a = acos(.5);
            logbase = log(par(4));
            b1 = par(3)*exp(-a*logbase);
            b2 = par(3)*exp(a*logbase);
            sfbw = b2 - b1;
        end
        
        
%     catch
%         obw = nan;
%         sfbw = nan;
%         phat = par0;
%         
%     end
        
    fprintf('Success\n')
    %% evaluate
    Ihat = reshape(prf.parametric_rf(phat, X0, log_gauss, vm01), size(xx)/ip.Results.upsample);
    Ihat0 = reshape(prf.parametric_rf(par0, X0, log_gauss, vm01), size(xx)/ip.Results.upsample);
    r2 = rsquared(srf(:), Ihat(:));
    r20 = rsquared(srf(:), Ihat0(:));
    if r20 > r2
        phat = par0;
        Ihat = Ihat0;
        r2 = r20;
    end
    
    if ip.Results.debug
        figure(55); clf
        subplot(2,3,1)
        plot(stat.timeax, rfflat); hold on
        plot(stat.peaklagt(cc)*[1 1], ylim, 'k', 'Linewidth', 2)
        title("Temporal lags")
        subplot(2,3,2)
        imagesc(stat.xax, stat.yax, stat.rffit(cc).srf)
        axis xy
        title("Raw")
        
        subplot(2,3,3)
        imagesc(stat.xax, stat.yax, Ihat)
        title( phat(2)/pi*180)
        axis xy
        title("Fit")
        
        subplot(2,2,3)
        contourf(stat.xax, stat.yax, stat.rffit(cc).srf, 10, 'Linestyle', 'none'); hold on
        title("Raw")
        
        subplot(2,2,4)
        contourf(stat.xax, stat.yax, Ihat, 'Linestyle', 'none'); hold on
        title("Fit")
        
        drawnow
        keyboard
    end
    
    % save 
    stat.rffit(cc).pHat = phat;
    stat.rffit(cc).oriPref = phat(2)/pi*180;
    stat.rffit(cc).oriBandwidth = obw/pi*180;
    stat.rffit(cc).sfPref = phat(3);
    stat.rffit(cc).sfBandwidth = sfbw;
    stat.rffit(cc).base = phat(6);
    stat.rffit(cc).amp = phat(5);
    stat.rffit(cc).r2 = r2;
    stat.rffit(cc).srfHat = Ihat;
    stat.rffit(cc).cid = cc;
    stat.rffit(cc).vm01 = vm01;
    stat.ishartley = forceHartley;
    stat.sftuning = ip.Results.sftuning;
    
end

% check significance
stat.sig = false(NC,1);

for cc = 1:NC
    if isempty(stat.rffit(cc).pHat) || stat.peaklag(cc)==0
        stat.sig(cc) = false;
    else
        zrf = stat.rf(:,:,cc)*stat.fs_stim / stat.sdbase(cc);
        z = reshape(zrf(stat.timeax>=0,:), [], 1);
        zthresh = 6;
        stat.sig(cc) = sum(z > zthresh) > (1-normcdf(zthresh));
    end
end