function [rfsF, rfsC, groupid] = getSpatialRFLocations(Exp, splitShanks, plotit)
% [rfsF, rfsC, groupid] = getSpatialRFLocations(Exp, splitShanks, PlotIt)
% Get the locations of receptive fields using a coarse-to-fine search and a
% 2D Gaussian fit

% Two stage RF mapping
if nargin < 3
    plotit = false;
    if nargin < 2
        splitShanks = 1;
    end
end

%% Get unit locations, combine units on the same probe

% check if multiple shanks were present
[ux,~] = io.getUnitLocations(Exp.osp, 4);

xc = Exp.osp.xc;
    
% assign each unit a group
if splitShanks == 0
    groupid = ones(size(ux));
else
    [~, groupid] = min((ux - unique(xc)').^2,[],2);
end
    

nGroups = numel(unique(groupid));
rfsC = repmat(struct('separability', [], 'sta', [], 'trf', [], 'srf', [], 'xax', [], 'yax', [], 'x', [], 'y', []), nGroups, 1);
rfsF = repmat(struct('separability', [], 'sta', [], 'trf', [], 'srf', [], 'xax', [], 'yax', [], 'mu', [], 'cov', []), nGroups, 1);

for iGroup = 1:nGroups
    
    if plotit
        figure(iGroup); clf
    end
    
    unitix = groupid==iGroup;

    % Stage 1: full screen 1 d.v.a. bin size
    ROI =  [-200 -200 300 200]*2;
    [X, Robs, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI',ROI, 'binSize', ceil(Exp.S.pixPerDeg));

    nlags = 12;

    Robs = sum(Robs(:,unitix),2);
    Y = Robs - mean(Robs);

    sta = simpleRevcorr(X, Y, nlags);

    [u,s,v] = svd(sta);
    sd = sign(sum(u(:,1)));

    rfsC(iGroup).separability = s(1) / sum(diag(s));
    rfsC(iGroup).sta = sta;
    rfsC(iGroup).trf = u(:,1)*sd;
    rfsC(iGroup).srf = reshape(v(:,1)*sd, opts.dims);
    rfsC(iGroup).xax = opts.xax/Exp.S.pixPerDeg;
    rfsC(iGroup).yax = opts.yax/Exp.S.pixPerDeg;

    if plotit
        imagesc(rfsC(iGroup).xax, rfsC(iGroup).yax, rfsC(iGroup).srf)
        hold on
    end
    
    I = rfsC(iGroup).srf;
    I = imgaussfilt(I.^3, 1);
    I = I/max(I(:));
    
    
    [x,y] = radialcenter(I);
    x = interp1(1:numel(rfsC(iGroup).xax), rfsC(iGroup).xax, x);
    y = interp1(1:numel(rfsC(iGroup).yax), rfsC(iGroup).yax, y);

    rfsC(iGroup).x = x;
    rfsC(iGroup).y = y;
    
    if plotit
        plot(x, y, 'ro')
        drawnow
    end

    %% Stage 2: full screen 1/4 d.v.a. bin size

    ROI = ([-2 -2 2 2] + [x y x y])*Exp.S.pixPerDeg;
    if hypot(x,y) > 5
        ROI = ([-5 -5 5 5] + [x y x y])*Exp.S.pixPerDeg;
    end
    bs = max(0.25, .1*hypot(x,y));
    binSize = bs * Exp.S.pixPerDeg;

    [X, Robs, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI',ROI, 'binSize', binSize);

    Robs = sum(Robs(:,unitix),2);
    Y = Robs - mean(Robs);
    Y = mean(Y,2);

    sta = simpleRevcorr(X, Y, nlags);

    [u,s,v] = svd(sta);
    sd = sign(sum(u(:,1)));

    rfsF(iGroup).separability = s(1) / sum(diag(s));
    rfsF(iGroup).sta = sta;
    rfsF(iGroup).trf = u(:,1)*sd;
    rfsF(iGroup).srf = reshape(v(:,1)*sd, opts.dims);
    rfsF(iGroup).xax = opts.xax/Exp.S.pixPerDeg;
    rfsF(iGroup).yax = opts.yax/Exp.S.pixPerDeg;
    
    if plotit
        imagesc(rfsF(iGroup).xax, rfsF(iGroup).yax, rfsF(iGroup).srf)
        colormap gray
        hold on
    end
    
    I = rfsF(iGroup).srf;
    I = I/max(I(:));
    [x,y] = radialcenter(I);
    x = interp1(1:numel(rfsF(iGroup).xax), rfsF(iGroup).xax, x);
    y = interp1(1:numel(rfsF(iGroup).yax), rfsF(iGroup).yax, y);

    if plotit
        plot(x, y, 'ro')
    end

    %% fit gaussian to hi-res RF

    % initial guess
    mu = [x, y];
    sigma = .25;

    % build grid
    [xx,yy] = meshgrid(rfsF(iGroup).xax, rfsF(iGroup).yax);
    X = [xx(:) yy(:)];
    gfun = @(params, x) mvnpdf(x, params(1:2), [params(3:4); params(4:5)]) * params(6);

    pguess = [mu sigma 0 sigma 1];

    phat = lsqcurvefit(gfun, pguess, X, I(:));

    if plotit
        hold on
        plot.plotellipse(phat(1:2), [phat(3:4); phat(4:5)], 1, 'Color', 'g', 'Linewidth', 2)
        plot(phat(1), phat(2), '+g')
        drawnow
    end
    
    rfsF(iGroup).mu = phat(1:2);
    rfsF(iGroup).cov = [phat(3:4); phat(4:5)];
end
