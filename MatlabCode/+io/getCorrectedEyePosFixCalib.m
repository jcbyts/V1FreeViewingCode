function eyePos = getCorrectedEyePosFixCalib(Exp, varargin)
% eyePos = getCorrectedEyePos(Exp, varargin)
% 'plot', false

ip = inputParser();
ip.addParameter('plot', false)
ip.addParameter('usebilinear', false)
ip.parse(varargin{:})

fprintf('Correcting eye pos by reanalyzing FaceCal\n')

validTrials = io.getValidTrials(Exp, 'FixCalib');
if isempty(validTrials)
    fprintf('No FixCalib trials\n')
    eyePos = Exp.vpx.smo(:,2:3);
    return
end
    
% get trial times
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));
eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
validIx = getTimeIdx(eyeTime, tstart, tstop);

xy = Exp.vpx.smo(validIx,2:3);
fixations = Exp.vpx.Labels==1;
fixations = fixations(validIx);


% target locations

targX = cell2mat(cellfun(@(x) x.PR.fixList(:,3), Exp.D(validTrials), 'uni', 0));
targY = cell2mat(cellfun(@(x) x.PR.fixList(:,4), Exp.D(validTrials), 'uni', 0));

% if ip.Results.plot
% eyeTime = eyeTime(validIx);
% targTime = Exp.ptb2Ephys(cell2mat(cellfun(@(x) x.PR.fixList(:,1), Exp.D(validTrials), 'uni', 0)));
%     figure(1); clf;
%     plot(eyeTime, xy(:,1)); hold on
%     plot(targTime, targX, 'o')
%     plot(eyeTime(fixations), xy(fixations,1), '.')
% end

% targets
[xx,yy] = meshgrid(unique(targX),unique(targY));

binSize = .05;
bins = {-20:binSize:20, -20:binSize:20};

% -- find fixations

% bin eye position during fixations
C = hist3(xy(fixations,:), 'Ctrs', bins ); colormap parula
% smooth
C = imgaussfilt(C, 2)';
% log
% C = log(C);
% threshold
C(C<10) = 0;

if ip.Results.plot
    figure(1); clf
    imagesc(bins{1}, bins{2}, C); hold on
    colorbar
    xlim([-18 18])
    ylim([-18 18])
    
    plot(xlim, [0 0], 'r')
    plot([0 0], ylim, 'r')
    
    plot(xx(:), yy(:), 's', 'MarkerFaceColor', 'r')
    xlim([-10 10])
    ylim([-10 10])
    title('Raw Eye position')
end

nTargs = numel(xx);
fixXY = nan(nTargs,2);
h = plot(xx(1),yy(1), '+g');
for iTarg = 1:nTargs
    
    iix = abs(xx(iTarg)-bins{1}) < 1;
    iiy = abs(yy(iTarg)-bins{2}) < 1;

    I = C(iiy, iix);
    sz = size(I)/2;
    [xc, yc] = radialcenter(I);
    
    if isnan(xc) || I(ceil(yc), ceil(xc)) == 0
        continue
    end
        
    if ip.Results.plot
        h.XData = xx(iTarg);
        h.YData = yy(iTarg);
        
        figure(2); clf
        imagesc(I); hold on
        plot(xc, yc, 'o')
        plot(sz(1), sz(2), '+r')
        drawnow
    end
    
    % store fixation center
    
    fixXY(iTarg,1) = (xc-sz(1))*binSize+xx(iTarg);
    fixXY(iTarg,2) = (yc-sz(2))*binSize+yy(iTarg);
    
end

if ip.Results.usebilinear
    % build design matrix for cubic fit
    X = [fixXY fixXY.^2 fixXY.^3];
else
    X = fixXY;
end

Y = [xx(:) yy(:)];

% remove non-fixated targets
bad = isnan(X(:,1));
X(bad,:) = [];
Y(bad,:) = [];

% fit model with robust regression
mdlx = fitlm(X, Y(:,1), 'RobustOpts','on');
mdly = fitlm(X, Y(:,2), 'RobustOpts','on');

xcoef = mdlx.Coefficients.Variables;
ycoef = mdly.Coefficients.Variables;

if ip.Results.usebilinear
    Xeye = [Exp.vpx.smo(:,2:3) Exp.vpx.smo(:,2:3).^2 Exp.vpx.smo(:,2:3).^3];
else
    Xeye = Exp.vpx.smo(:,2:3);
end

% correct eye position
x = xcoef(1) + Xeye*xcoef(2:end,1);
y = ycoef(1) + Xeye*ycoef(2:end,1);
eyePos = [x y];

% if ip.Results.plot
% figure(4); clf
% plot(eyePos(validIx,1), eyePos(validIx,2), '.'); hold on
% plot(xx(:), yy(:), '+')
% end

% plot new eye position
if ip.Results.plot
    figure(2); clf
    
    bins = {-20:binSize:20, -20:binSize:20};
    C = hist3(eyePos(validIx,:), 'Ctrs', bins ); colormap parula
    C = imgaussfilt(C,2);
    C = log(C');
    C(C<2.5) = 0;
    imagesc(bins{1}, bins{2},C ); hold on
    colorbar
    xlim([-18 18])
    ylim([-18 18])
    
    plot(xlim, [0 0], 'r')
    plot([0 0], ylim, 'r')
    
    
    plot(xx(:), yy(:), 's', 'MarkerFaceColor', 'r')
    xlim([-10 10])
    ylim([-10 10])
    title('Corrected')

end