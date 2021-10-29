function stat = grat_rf_helper_mle(Exp, varargin)
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
%   

ip = inputParser();
ip.addParameter('win', [-5 25])
ip.addParameter('plot', true)
ip.addParameter('fitminmax', false)
% ip.addParameter('sdthresh', 8)
ip.addParameter('stat', [])
ip.addParameter('debug', false)
ip.addParameter('boxfilt', 1, @(x) mod(x,2)~=0);
ip.addParameter('sftuning', 'loggauss', @(x) ismember(x, {'loggauss', 'raisedcosine'}))
ip.addParameter('upsample', 1)
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
% 1) Threshold RF to find centroid
% 2) Fit parametric model
for cc = 1:NC
    fprintf('Fitting unit %d/%d...\t', cc, NC)
    
    % time x space (flattened) receptive field in units of spikes/sec
    rfflat = stat.rf(:,:,cc)*grating.fs_stim;
    
%     %%
%     figure(1); clf
%     plot(rfflat)
%     
%     
%     %%
    
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
    thresh = max(max(adiff(stat.timeax<=0,:)));
    
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
    
    stat.maxV(cc) = maxV;
    if maxV < 3
        fprintf('No RF\n')
        continue
    end
    
    rfmask = rf3 .* (bw == bigg);
    rfmask = rfmask / sum(rfmask(:));
    sz = size(rfmask);
    [tt,yy,xx] = ndgrid(1:sz(1), 1:sz(2), 1:sz(3));
    
    ytx = [yy(:)'*rfmask(:) tt(:)'*rfmask(:) xx(:)'*rfmask(:)];
    
%     ytx = s.Centroid(bigg,:); % location (in space / time)
    peaklagt = interp1(1:nlags, stat.timeax, ytx(2)); % peak lag using centroid
    
    peaklagf = floor(ytx(2));
    peaklagc = ceil(ytx(2));
    
    % "spatial" (orientation/frequency) RF
    srf = squeeze(mean(rf3(peaklagf:peaklagc,:,:), 1)); % average over peak lags
    stat.rffit(cc).srf = srf;
    
    [y0,x0]=find(srf==max(srf(:)));
    % initialize mean for fit based on centroid
    x0 = interp1(1:numel(stat.xax), stat.xax, x0);
    y0 = interp1(1:numel(stat.yax), stat.yax, y0);
    
    stat.peaklag(cc) = round(ytx(2));
    stat.peaklagt(cc) = peaklagt;

    keyboard
    %%
    figure(1); clf
%     cc = cc + 1;
%     if cc > NC
%         cc = 1;
%     end
    
    cc = 1;
    peaklag = 6;
    ori = grating.ori(1:end-peaklag+1);
    cpd = grating.cpd(1:end-peaklag+1);
    n = size(Robs,1);
    r = Robs(peaklag:n,cc) + Robs((peaklag:n)-1,cc) + Robs(min((peaklag:n)+1,n),cc);
    ix = r > 0;
    h = scatter3(ori(ix),cpd(ix),r(ix)+randn(size(r(ix)))*.02, 100*r(ix), '.');
%     h.MarkerFaceColor = 'b';
    
    %%
    
    
    
    % fit parametric receptive field model
    I = srf; % change of variable
    
    % --- fit parametric RF
    mxI = max(srf(:));
    mnI = min(srf(:));
    
    if forceHartley % stimulus was hartley basis
        [theta0, cpd0] = cart2pol(y0, x0);
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
        
        ori0 = cart2pol(y0,x0);
        
        [Yq, Xq] = pol2cart(ori0*ones(size(sfs)), sfs);
        
        % fit SF initial
        r = interp2(grating.oris, -grating.cpds, srf, Xq, Yq);
        sfs = sfs(~isnan(r));
        r = r(~isnan(r))';
        x = [ones(size(sfs(:)))*ori0 sfs(:)];
        fun = @(params, x) prf.parametric_rf([0 0 params max(r) min(r)], x);
        
        evalc("phat = lsqcurvefit(fun, [cpd0 2], x, r);");
%         figure
%         plot(sfs, r, 'o'); hold on
%         plot(linspace(0, 10, 100), fun(phat,[zeros(100,1) linspace(0, 10, 100)']))
        
        cpd0 = max(real(phat(1)), cpd0);
        cpdb0 = real(phat(2));
        
        % fit orientation initial
        oris = 0:(pi/10):pi;
        [Yq, Xq] = pol2cart(oris, cpd0*ones(size(oris)));
    
        r = interp2(grating.oris, -grating.cpds, srf, Xq, Yq);
        oris = oris(~isnan(r))';
        r = r(~isnan(r))';
        x = [oris ones(size(oris(:)))*cpd0];
        
        fun = @(params, x) prf.parametric_rf([params cpd0 cpdb0 max(r) min(r)], x);
        evalc("phat = lsqcurvefit(fun, [ori0 1], x, r);");
        
%         figure
%         plot(oris, r, 'o'); hold on
%         plot(linspace(0, pi, 100), fun(phat, [linspace(0, pi, 100)', ones(100,1)*cpd0]))
        
        ori0 = phat(2);
        orib0 = phat(1);
    
    else % stimulus was orientation/frequency
        theta0 = x0/180*pi;
        cpd0 = y0;
        [xx,yy] = meshgrid(grating.oris/180*pi, grating.cpds);
        X0 = [xx(:), yy(:)];
        minSF = min(grating.cpds(grating.cpds>0));
        if ip.Results.upsample > 1
            xx = imresize(xx, ip.Results.upsample, 'bilinear');
            yy = imresize(yy, ip.Results.upsample, 'bilinear');
            I = imresize(I, ip.Results.upsample, 'bilinear');
        end
        X = [xx(:) yy(:)];
    end
    
    
    % Do the fitting
    
    try
        % try global search
        if log_gauss
            par0 = [orib0 ori0 cpd0 cpdb0  mxI     mnI]; % initial parameters
            lb =   [0    -pi     0.1  -10      .5*mxI  -mxI];
            ub =   [500  2*pi    20   10     2*mxI   .5*mxI];
        else
            par0 = [orib0 ori0 cpd0 2 mxI mnI]; % initial parameters
            lb = [0 -pi 0.1 1.5 .5*mxI -mxI];
            ub = [500 2*pi 20 10 2*mxI .5*mxI];
        end
        
        I = I(:);
        fitix = X(:,2) >= minSF;
        fun = @(params, x) prf.parametric_rf([par0(1:4) params], x);
        evalc("phat = lsqcurvefit(fun, [mxI mnI], X(fitix,:), I(fitix));");
        par0(5:6) = phat;
        % least-squares with some scaling by firing rate
%         lossfun = @(r,lambda) sqrt(r)'*((r - lambda).^2);
        lossfun = @(r,lambda) sum((r - lambda).^2);
        
        
        fun = @(params) lossfun(I(fitix), prf.parametric_rf(params, X(fitix,:), log_gauss));
        
        opts = optimoptions('fmincon');
        problem = createOptimProblem('fmincon','objective', ...
            @(params) fun(params),'x0',par0,'lb',lb, ...
            'ub',ub,'options',opts);
        %     
        gs = GlobalSearch('Display','none','XTolerance',1e-3,'StartPointsToRun','bounds');
        ms = MultiStart(gs,'UseParallel',true);
        [phat,~] = run(ms,problem, 100);
        
        if phat(1) < 1e-5 % refit without orientation tuning
            fun = @(params) lossfun(I(fitix), prf.parametric_rf([0 0 params], X(fitix,:), log_gauss));
            
            opts = optimoptions('fmincon');
            problem = createOptimProblem('fmincon','objective', ...
                @(params) fun(params),'x0',phat(3:end),'lb',lb, ...
                'ub',ub,'options',opts);
            %
            gs = GlobalSearch('Display','none','XTolerance',1e-3,'StartPointsToRun','bounds');
            ms = MultiStart(gs,'UseParallel',true);
            [phat,~] = run(ms,problem, 100);
            phat = [0 0 phat];
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
        
        % ori bandwidth
        try
            orientation = 0:.1:pi;
            orientationTuning = (exp(par(1)*cos(orientation).^2) - 1) / (exp(par(1)) - 1);
            id = find(orientationTuning < .1, 1);
            hbw = interp1(orientationTuning(1:id), orientation(1:id), .5);
            obw = 2*hbw;
        catch
            obw = nan;
        end
        
    catch
%         
        obw = nan;
        sfbw = nan;
% %         %% fit with nlinfit
% %         try
% %             par0 = [2 theta0 1.5*cpd0 1.5 mxI mnI]; % initial parameters
% %             
% %             fun = @(params, x) prf.parametric_rf(params, x);
% %             options = statset('RobustWgtFun', 'bisquare', 'Tune', 10, 'MaxIter', 1000);
% %             cstr = evalc("[phat,R,~,COVB,~,minfo] = nlinfit(X, I(:), fun, par0, options);");
% %             if contains(cstr, 'Warning:')
% %                 warningFlag = true;
% %             else
% %                 warningFlag = false;
% %             end
% %             CI = nlparci(phat, R, 'covar', COVB);
% %         catch
% %             % Don't fit min and max
% %             if ip.Results.fitminmax
% %                 % Do fit min and max
% %                 par0 = [2 theta0 cpd0 1.5 mxI mnI]; % initial parameters
% %                 lb = [.01 -pi 0.1 1.01 .5*mxI mnI];
% %                 ub = [10 2*pi 20 10 mxI .5*mxI];
% %                 fun = @(params) (prf.parametric_rf([params], X) - I(:)).^2;
% %             else
% %                 fun = @(params) (prf.parametric_rf([params mxI mnI], X) - I(:)).^2;
% %                 par0 = [2 theta0 cpd0 1]; % initial parameters
% %                 lb = [.01 -pi 0.1 .2];
% %                 ub = [10 2*pi 20 1];
% %             end
% %             %% OLD SCHOOL LSQNONLIN
% %             % least-squares for parametric rf
% %             try
% %                 %  parameters are:
% %                 %   1. Orientation Kappa
% %                 %   2. Orientation Preference
% %                 %   3. Spatial Frequency Preference
% %                 %   4. Spatial Frequency Sigma
% %                 %   5. Gain
% %                 %   6. Offset
% %                 fprintf('Oldschool\n')
% %                 evalc('phat = lsqnonlin(fun, par0, lb, ub)');
% %                 %         evalc('phat = lsqcurvefit(fun, par0, X, I(:), lb, ub);');
% %             catch
                phat = par0;
% %                 
% %             end
% %             
% %             if ~ip.Results.fitminmax
% %                 phat = [phat mxI mnI];
% %             end
% %             
% %             CI = nan(numel(phat), 2);
% %             warningFlag = 1;
% %         end
%         phat = nan(size(par0));
    end
%         
    fprintf('Success\n')
    %% evaluate
    Ihat = reshape(prf.parametric_rf(phat, X0, log_gauss), size(xx)/ip.Results.upsample);
    Ihat0 = reshape(prf.parametric_rf(par0, X0, log_gauss), size(xx)/ip.Results.upsample);
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
        subplot(2,3,2)
        imagesc(stat.xax, stat.yax, stat.rffit(cc).srf)
        axis xy
        subplot(2,3,3)
        imagesc(stat.xax, stat.yax, Ihat)
        title( phat(2)/pi*180)
        axis xy
        subplot(2,2,3)
        contourf(stat.xax, stat.yax, stat.rffit(cc).srf, 10, 'Linestyle', 'none'); hold on
        plot(x0, y0, 'or')
        subplot(2,2,4)
        contourf(stat.xax, stat.yax, Ihat, 'Linestyle', 'none'); hold on
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