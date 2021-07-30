function [eyePos2, eyePos] = getEyeCalibrationFromRaw(Exp, varargin)


ip = inputParser();
ip.addParameter('usebilinear', false)
ip.addParameter('usesmo', false)
ip.addParameter('plot', false)
ip.addParameter('cmat', [])
ip.parse(varargin{:})

%% organize the data
fprintf('Correcting eye pos by reanalyzing FaceCal\n')

validTrials = io.getValidTrials(Exp, 'FaceCal');

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

eyeTime = Exp.vpx2ephys(Exp.vpx.raw(:,1));
validIx = getTimeIdx(eyeTime, tstart, tstop);

if ip.Results.usesmo
    xy = Exp.vpx.smo(validIx,2:3);
else
    xy = Exp.vpx.raw(validIx,2:3);
end
spd = Exp.vpx.smo(validIx,7);

trialTargets = cellfun(@(x) x.PR.faceconfig(:,1:2), Exp.D(validTrials), 'uni', 0);
targets = unique(cell2mat(trialTargets),'rows');
ntargs = size(targets,1);

n = sum(validIx);
ix = true(n,1);
ix = all(abs(zscore(xy(ix,:)))<1,2); % remove outliers
ix = ix & ( spd / median(spd) < 2); % find fixations

%% check original
if ~isempty(ip.Results.cmat)
    
    phat = ip.Results.cmat;
    
else
    cmat = [1 1 0 0 0];
    
    
    [~, id] = calibration_loss(cmat, xy(ix,:), targets);
    
    figure(1); clf
    subplot(1,2,1)
    cmap = jet(ntargs);
    inds = find(ix);
    for j = 1:ntargs
        plot(xy(inds(id==j),1), xy(inds(id==j),2), '.', 'Color', cmap(j,:)); hold on
        plot(targets(j,1), targets(j,2), 'ok', 'MarkerFaceColor', cmap(j,:), 'Linewidth', 2); hold on
    end
    %% Calibrate from RAW
    
    % ctr = mean(xy(ix,:)); % get center
    
    [C, xax, yax] = histcounts2(xy(ix,1), xy(ix,2), 1000);
    [xx,yy] = meshgrid(xax(1:end-1), yax(1:end-1));
    
    I =  imgaussfilt(C', 10);
    
    % figure(1); clf
    % imagesc(xax, yax, I); hold on
    %
    
    wts = I(:).^9 / sum(I(:).^9);
    x = xx(:)'*wts;
    y = yy(:)'*wts;
    ctr = [x,y];
    
    % plot(ctr(1), ctr(2), 'or')
    % plot(x,y,'mo')
    %%
    
    lossfun = @(params) sum(calibration_loss([params ctr], xy(ix,:), targets));
    
    % fit model
    opts = optimset('Display', 'iter');
    phat = fminsearch(lossfun, [std(xy)/10 0], opts);
    phat = [phat ctr];
end

%% check whether we're in the right space
th = phat(3);
R = [cosd(th) -sind(th); sind(th) cosd(th)];
S = [phat(1) 0; 0 phat(2)];
A = (R*S)';
xxy = targets*A + phat(4:5);

[~, id] = calibration_loss(phat, xy(ix,:), targets);

figure(1); clf
subplot(1,2,1)
cmap = jet(ntargs);
inds = find(ix);
for j = 1:ntargs
    plot(xy(inds(id==j),1), xy(inds(id==j),2), '.', 'Color', cmap(j,:)); hold on
    plot(xxy(j,1), xxy(j,2), 'ok', 'MarkerFaceColor', cmap(j,:), 'Linewidth', 2); hold on    
end


%% Do calibration
R = [cosd(th) -sind(th); sind(th) cosd(th)];
S = [phat(1) 0; 0 phat(2)];
A = (R*S)';
Ainv = pinv(A);

if ip.Results.usesmo
    eyePos = (Exp.vpx.smo(:,2:3) - phat(4:5))*Ainv;
else
    eyePos = (Exp.vpx.raw(:,2:3) - phat(4:5))*Ainv;
end

xy2 = (xy - phat(4:5))*Ainv;
subplot(1,2,2)
for j = 1:ntargs
    plot(xy2(inds(id==j),1), xy2(inds(id==j),2), '.', 'Color', cmap(j,:)); hold on
    plot(targets(j,1), targets(j,2), 'ok', 'MarkerFaceColor', cmap(j,:), 'Linewidth', 2); hold on    
end
title('Degrees')


% %% test go forwards and backwards
% X = targets;
% A = (R*S)';
% Y = X*A + phat(4:5);
% Xhat = (Y - phat(4:5))*pinv(A);
% 
% 
% figure(1); clf
% plot(X(:,1), X(:,2), 'ob'); hold on
% plot(Xhat(:,1), Xhat(:,2), '+r')
% xlim([-1 1]*20)
% ylim([-1 1]*20)

%% 2nd Stage: fixation by fixation calibration

dxdy = diff(eyePos); % velocity
spd = hypot(dxdy(:,1), dxdy(:,2)); % speed


% figure(1); clf
% plot(spd)

fixatedLoss = [];

% check if flipped calibrations are better
lossFlipX = []; 
lossFlipY = [];
lossFlipXY = [];

fixatedTarget = [];
fixatedTime = [];
fixatedXY = [];

fixations = spd < .025; % identify fixations as low-speed moments

for iTrial = 1:numel(tstart)
    iix = eyeTime > tstart(iTrial) & eyeTime < tstop(iTrial);
    
    targetlist = trialTargets{iTrial};
    
    [lp, id] = calibration_loss([1 1 0 0 0], eyePos(iix,:), targetlist);
    % check if flipped calibrations are better
    lpFlipX = calibration_loss([1 1 0 0 0], eyePos(iix,:).*[-1 1], targetlist);
    lpFlipY = calibration_loss([1 1 0 0 0], eyePos(iix,:).*[1 -1], targetlist);
    lpFlipXY = calibration_loss([1 1 0 0 0], eyePos(iix,:).*[-1 -1], targetlist);
    
    
    tax = eyeTime(iix);
    exy = eyePos(iix,:);
    
    fixated = lp < 1 & fixations(iix);
    fixated = imboxfilt(double(fixated), 3) > 0;
            
    
    
    fixationList = bwlabel(fixated);
    nFix = max(fixationList);

    figure(2); clf
    subplot(2,2,1)
    plot(tax, exy(:,1), 'k'); hold on
    
    for iFix = 1:nFix
        fixix = fixationList == iFix;
    
        fixatedTargs = targetlist(id(fixix),:);
        [~, targId] = min(hypot(fixatedTargs(:,1) - targets(:,1)', fixatedTargs(:,2) - targets(:,2)'),[],2);
    
        fixatedLoss = [fixatedLoss; mean(lp(fixix))];
    
        % store flipped loss
        lossFlipX = [lossFlipX; mean(lpFlipX(fixix))]; 
        lossFlipY = [lossFlipY; mean(lpFlipY(fixix))];
        lossFlipXY = [lossFlipXY; mean(lpFlipXY(fixix))];

        fixatedTarget = [fixatedTarget; mean(targId)];
        fixatedTime = [fixatedTime; mean(tax(fixix))];
        fixatedXY = [fixatedXY; mean(exy(fixix,:))];
        
        plot(tax(fixix), exy(fixix,1), '.')
    
    end
    ylim([-10 10])
    
    fixatedTargs = targetlist(id(fixated),:);
    [~, targId] = min(hypot(fixatedTargs(:,1) - targets(:,1)', fixatedTargs(:,2) - targets(:,2)'),[],2);
    
    subplot(2,2,3)
    plot(tax, exy(:,2), 'k'); hold on
    plot(tax(fixated), exy(fixated,2), '.r')
    ylabel('horizontal (d.v.a.)')
    ylim([-10 10])
    
    subplot(2,2,[2 4])
    plot(exy(:,1), exy(:,2), 'k'); hold on
    fixinds = find(fixated);
    for t = targId(:)'
        iix = fixinds(t == targId);
        plot(exy(iix,1), exy(iix,2), '.', 'Color', cmap(t,:))
    end
    ylabel('vertical (d.v.a.)')
    
    plot(targetlist(:,1), targetlist(:,2), 'sk', 'MarkerSize', 20)
    xlim([-1 1]*12)
    ylim([-1 1]*12)
    title(sprintf('Trial %d', iTrial))
    
    drawnow
end

%% check for flips
L0 = sum(fixatedLoss);
    
% store flipped loss
lX = sum(lossFlipX);
lY = sum(lossFlipY);
lXY = sum(lossFlipXY);

flips = {[1 1], [-1 1], [1 -1], [-1 -1]};

try
    [~, bestflip] = min([L0 lX lY lXY]);
    
    switch bestflip
        case 1
            disp("No flipping necessary")
        case 2
            disp("Flipping X")
        case 3
            disp("Flipping Y")
        case 4
            disp("Flipping X and Y")
    end
    
    eyePos = eyePos.*flips{bestflip};
    fixatedXY = fixatedXY.*flips{bestflip};
end
%% redo calibration using 2nd-order polynomial

targXY = targets(round(fixatedTarget),:);
features = [fixatedXY fixatedXY.^2];

nFix = numel(fixatedTarget);

nboots = 100;
ntrain = ceil(nFix/2);
ntest = nFix - ntrain;
serror = zeros(ntest, nboots);
wtsXs = zeros(5, nboots);
wtsYs = zeros(5, nboots);

for iboot = 1:nboots
    trainindex = randsample(nFix, ntrain, false);
    testindex = setdiff(1:nFix,trainindex);

    wtsX = regress(targXY(trainindex,1), [ones(ntrain,1) features(trainindex,:)]);
    wtsY = regress(targXY(trainindex,2), [ones(ntrain,1) features(trainindex,:)]);
%     wtsX = robustfit(features(trainindex,:), targXY(trainindex,1));%, 'talwar', 1);
%     wtsY = robustfit(features(trainindex,:), targXY(trainindex,2));%, 'talwar', 1);

    exHat = wtsX(1) + features*wtsX(2:end);
    eyHat = wtsY(1) + features*wtsY(2:end);

    wtsXs(:,iboot) = wtsX;
    wtsYs(:,iboot) = wtsY;
    
    serror(:,iboot) = hypot( exHat(testindex) - targXY(testindex,1), eyHat(testindex) - targXY(testindex,2));
end

merror = serror ./ median(serror);
[~, id] = min(sum(merror > 2));

wtsX = wtsXs(:,id);
wtsY = wtsYs(:,id);


figure(1); clf
plot(exHat, eyHat,'.'); hold on
plot(fixatedXY(:,1), fixatedXY(:,2), '.')

[l2, id] = calibration_loss([1 1 0 0 0], [exHat eyHat], unique(targXY, 'rows'));
l1 = calibration_loss([1 1 0 0 0], fixatedXY, unique(targXY, 'rows'));

[xx,yy] = meshgrid(unique(targets(:)));
plot(xx, yy, 'k')
plot(yy, xx, 'k')
% disp(l2 - l1)

%%
clf
nFix = size(targXY,1);
plot(targXY(:,1)+.1*randn(nFix,1), exHat(:,1), '.'); hold on
plot(xlim, xlim, 'k')

% evaluate
% 
% iTarg = 13;
% iix = fixatedTarget == iTarg;
% 
% [C, ~, ~] = histcounts2(fixatedXY(iix,1)-targets(iTarg,1), fixatedXY(iix,2)-targets(iTarg,2), -2:.1:2, -2:.1:2);
% [C2, xax, yax] = histcounts2(exHat(iix)-targets(iTarg,1), eyHat(iix)-targets(iTarg,2), -2:.1:2, -2:.1:2);
% C2 = imgaussfilt(C2,.75);
% 
% [x,y] = radialcenter(C2');
% figure(1); clf
% imagesc(C2'); hold on
% plot(x,y, 'or')
% 
% %%
% figure(1); clf
% subplot(1,2,1)
% imagesc(xax, yax, C')
% hold on
% plot(xlim, [0 0], 'r')
% plot([0 0], ylim, 'r')
% 
% subplot(1,2,2)
% imagesc(xax, yax, C2')
% hold on
% plot(xlim, [0 0], 'r')
% plot([0 0], ylim, 'r')


%%
eyePosX = wtsX(1) + [eyePos eyePos.^2]*wtsX(2:end);
eyePosY = wtsY(1) + [eyePos eyePos.^2]*wtsY(2:end);

eyePos2 = [eyePosX eyePosY];