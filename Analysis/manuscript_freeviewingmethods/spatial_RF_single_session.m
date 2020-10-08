function S = spatial_RF_single_session(Exp, varargin)
% get spatial receptive fields for one session using dot stimuli
% Run in coarse to fine analysis
% S = spatial_RF_single_session(Exp, varargin)

ip = inputParser();
ip.addParameter('ROI', [-10 -10 10 10])
ip.addParameter('binSize', 1.5)
ip.addParameter('numlags', 10)
ip.addParameter('numspace', 20)
ip.addParameter('plot', true)
ip.parse(varargin{:})


%% Step 1: use large ROI and large bin size
ppd = Exp.S.pixPerDeg;
ROI = ip.Results.ROI; % initial ROI
binSize = ip.Results.binSize; % initial binSize

% step 1: get visually driven units, get  find ROI
spkS = io.get_visual_units(Exp, 'plotit', false, 'ROI', ROI*ppd, 'binSize', binSize*ppd);

figure(1); clf
NC = numel(spkS);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
for cc = 1:NC
    subplot(sx, sy, cc)
    imagesc(spkS(cc).xax/ppd, spkS(cc).yax/ppd, spkS(cc).srf)
    axis xy
end

S.coarse.ROI = ROI;
S.coarse.binSize = binSize;
S.coarse.details = spkS;

% find neurons that are significantly modulated by stimuli
vis = find([spkS.BackImage] | [spkS.Grating]);

%% step 2: run again but on a finer grid with a new ROI for each neuron

NC = numel(spkS);
rfdeets = repmat(struct('sta', nan), NC, 1); % initialize struct
for cc = 1:NC
        rfdeets(cc).sta = nan;
        rfdeets(cc).gfit.mu = nan(1,2);
        rfdeets(cc).gfit.cov = nan(2);
        rfdeets(cc).gfit.base = nan;
        rfdeets(cc).gfit.amp = nan;
        rfdeets(cc).peaklag = nan;
        rfdeets(cc).srf = nan;
        rfdeets(cc).xax = nan;
        rfdeets(cc).yax = nan;
        rfdeets(cc).cid = cc;
        rfdeets(cc).binSize = nan;
        rfdeets(cc).ROI = nan;
end


for cc = vis(:)'

  
    if ip.Results.plot
        figure(1); clf
    end
    
    sigpixels = abs(spkS(cc).sta) > spkS(cc).thresh;
    
    if sum(sigpixels(:)) < 1
        continue
    end
    
    umask = reshape(sum(sigpixels), size(spkS(cc).srf));
    
    % find new ROI
    [j, i] = find(umask > 0.01);

    nx = numel(spkS(1).xax);
    ny = numel(spkS(1).yax);
    pad = 1;
    xax = [min(spkS(1).xax(max(i-pad, 1))) max(spkS(1).xax(min(i+pad,nx)))]/ppd;
    yax = [min(spkS(1).yax(max(j-pad, 1))) max(spkS(1).yax(min(j+pad,ny)))]/ppd;
    
    if ip.Results.plot
        
        subplot(1,2,1)
        imagesc(spkS(cc).xax/ppd, spkS(cc).yax/ppd, spkS(cc).srf);
        hold on
        plot(xax, yax(1)*[1 1], 'r', 'Linewidth', 2)
        plot(xax, yax(2)*[1 1], 'r', 'Linewidth', 2)
        plot(xax(1)*[1 1], yax, 'r', 'Linewidth', 2)
        plot(xax(2)*[1 1], yax, 'r', 'Linewidth', 2)
    end
    % redefine ROI
    ROI = [xax(1) yax(1) xax(2) yax(2)];
    ppd = Exp.S.pixPerDeg;
    binSize = hypot(ROI(3)-ROI(1), ROI(4)-ROI(2))/ip.Results.numspace;

    % discretize stimulus
    [Stim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', ROI*ppd, 'binSize', ppd*binSize);

    RobsSpace = RobsSpace(:,cc);
    
    % embed time
    nlags = ip.Results.numlags;
    Xstim = makeStimRows(Stim, nlags);

    % find valid fixations
    eyeTimes = Exp.vpx2ephys(Exp.vpx.smo(:,1));

    cnt = find(histcounts(opts.frameTimes(opts.validFrames>0), eyeTimes)); % map frame times to eye samples

    % find if samples are fixations and map backwards to find valid frames
    valid = find(histcounts(eyeTimes(cnt(Exp.vpx.Labels(cnt)==1)), opts.frameTimes(opts.validFrames>0)));

    nValid = numel(valid);
    fprintf('%d valid (fixation) samples\n', nValid)

    
    %% estimate receptive field
    % assume smoothness in space and time. 
    % estimate the amount of smoothness by maximizing the model evidence
    % using a gridsearch
    
    % compute STA (in units of delta spike rate)
    Rvalid = RobsSpace - mean(RobsSpace);
    
%   train = valid(1:floor(4*nValid/5))';
%   test = valid(ceil(4*nValid/5):nValid)';
    % random sample train and test set
    test = randsample(valid, floor(nValid/5)); 
    train = setdiff(valid, test);
    
    XX = (Xstim(train,:)'*Xstim(train,:)); % covariance
    XY = (Xstim(train,:)'*Rvalid(train));
    
    Rtest = imgaussfilt(Rvalid(test), 2);
    
    CpriorInv = qfsmooth3D([nlags, fliplr(opts.dims)], .5);
    CpriorInv = CpriorInv + eye(size(CpriorInv,2));
    
    % if you want to check that you're in the right range   
    
    
    if ip.Results.plot
        % initialize
        lambda0 = 1e3;
        kmap0 = (XX+lambda0*CpriorInv)\XY;% ./sum(Xstim(valid,:))';
        
        rfspt = reshape(kmap0, nlags, []);
        
        figure(2);
        clf
        
        subplot(1,2,1)
        plot(rfspt)
        subplot(1,2,2)
        imagesc(reshape(rfspt(4,:), (opts.dims)))
    end
    
    
    % loop over hyperparameter
    lambdas = (2.^(-1:20));

    r2test = zeros(numel(lambdas),1);
    
    for il = 1:numel(lambdas)
        
        % get posterior weights
        lambda = lambdas(il);
        
        H = XX + lambda*CpriorInv;

        kmap = H\XY;  % Posterior Mean (MAP estimate)
        
        % predict on test set
        Rhat = Xstim(test,:)*kmap;
    
    
        r2test(il) = rsquared(Rtest, Rhat);
    
    end
    
    
%     figure(1); clf
%     plot(lambdas, r2test)
    [~, id] = max(r2test);
    lambda = lambdas(id);
    
    
    H = XX + lambda*CpriorInv;
    kmap = H\XY;
    
    sta = kmap;
    
    rfspt = reshape(sta, nlags, []);
    
    if ip.Results.plot
        figure(2); clf
        
        subplot(1,2,2)
        imagesc(reshape(rfspt(4,:), (opts.dims)))
        
        subplot(1,2,1)
        plot(rfspt, 'k')
        thresh = sqrt(robustcov(sta(:)))*4;
        hold on
        
        plot(xlim, thresh*[1 1])
    end
    
    rfdeets(cc).sta = rfspt;
    
    % get temporal kernel (assuming separability)
    [u,~,~] = svd(rfspt); 
    tk = u(:,1) ./ sum(u(:,1));
    
    % find peak lag
    [~, peaklag] = max(tk);
    
    % reshape dimensions
    I = reshape(rfspt(peaklag,:), opts.dims);
    I = I ./ max(I(:));
    
    if ip.Results.plot
        figure(3); clf
        imagesc(opts.xax/ppd, opts.yax/ppd, I); colormap gray
        hold on
        axis xy
        grid('on')
        ax.GridColor = 'y';
        ax.GridAlpha = .5;
    end
    
    % fit gaussian
    [x0,y0] = radialcenter(I); % initialize center with radial symmetric center
    
    x0 = interp1(1:size(I,2), opts.xax/ppd, x0);
    y0 = interp1(1:size(I,1), opts.yax/ppd, y0);
    
    par0 = [x0 y0 1 0 min(I(:)) max(I(:))];
    
    % gaussian function
    gfun = @(params, X) params(5) + params(6) * exp(-.5 * sum(((X-[params(1) params(2)])*pinv([params(3) params(4); params(4) params(3)]'*[params(3) params(4); params(4) params(3)])).*(X-[params(1) params(2)]),2));
    
    [xx,yy] = meshgrid(opts.xax/ppd,  opts.yax/ppd);
    X = [xx(:) yy(:)];
    
    % least-squares
    phat = lsqcurvefit(gfun, par0, X, I(:));    
    mu = phat(1:2);
    C = [phat(3) phat(4); phat(4) phat(3)]'*[phat(3) phat(4); phat(4) phat(3)];
    
    if ip.Results.plot
        plot.plotellipse(mu, C, 1, 'Linewidth', 2);
        drawnow
    end
    
    rfdeets(cc).gfit.mu = mu;
    rfdeets(cc).gfit.cov = C;
    rfdeets(cc).gfit.base = phat(5);
    rfdeets(cc).gfit.amp = phat(6);
    ghat = gfun(phat, X);
    rfdeets(cc).gfit.r2 = rsquared(I(:), ghat(:));
    rfdeets(cc).peaklag = peaklag;
    rfdeets(cc).srf = I;
    rfdeets(cc).xax = opts.xax/ppd;
    rfdeets(cc).yax = opts.yax/ppd;
    rfdeets(cc).cid = cc;
    rfdeets(cc).binSize = binSize;
    rfdeets(cc).ROI = ROI;
end

S.fine = rfdeets;





