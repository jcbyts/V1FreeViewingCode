function eyePos = getCorrectedEyePos(Exp, varargin)
% eyePos = getCorrectedEyePos(Exp, varargin)
% 'usebilinear', false

ip = inputParser();
ip.addParameter('usebilinear', false)
ip.addParameter('plot', false)
ip.parse(varargin{:})

fprintf('Correcting eye pos by reanalyzing FaceCal\n')

validTrials = io.getValidTrials(Exp, 'FaceCal');

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
validIx = getTimeIdx(eyeTime, tstart, tstop);

xy = Exp.vpx.smo(validIx,2:3);

% target locations
[xx,yy] = meshgrid(-10:5:10, -10:5:10);

xd = xy(:,1) - xx(:)';
yd = xy(:,2) - yy(:)';

% distance from targets
d = hypot(xd, yd);

% assign targets id
[d,id] = min(d, [], 2);

thresh = 1.5; % trheshold to count fixation
xy = xy(d<thresh,:);
id = id(d<thresh);

if ip.Results.plot
figure(1); clf
bins = {-20:.25:20, -20:.25:20};
C = hist3(xy, 'Ctrs', bins ); colormap parula
C = log(C');
C(C<1) = 0;
imagesc(bins{1}, bins{2},C ); hold on
colorbar
xlim([-18 18])
ylim([-18 18])

plot(xlim, [0 0], 'r')
plot([0 0], ylim, 'r')

plot(xx(:), yy(:), 's')


% Exp.D{validTrials(1)}.PR.NoiseHistory

figure(2); clf
plot(xx(id), xy(:,1), '.'); hold on
plot(yy(id)+.5, xy(:,2), '.')
plot(xlim, xlim)
end

% regress
if ip.Results.usebilinear
    X = [xy xy.^2 xy.^3];
else
    X = [xy];
end

Y = [xx(id) yy(id)];

mdlx = fitlm(X, Y(:,1), 'RobustOpts','on');
mdly = fitlm(X, Y(:,2), 'RobustOpts','on');

xcoef = mdlx.Coefficients.Variables;
ycoef = mdly.Coefficients.Variables;

if ip.Results.usebilinear
    Xeye = [Exp.vpx.smo(:,2:3) Exp.vpx.smo(:,2:3).^2 Exp.vpx.smo(:,2:3).^3];
else
    Xeye = [Exp.vpx.smo(:,2:3)];
end

x = xcoef(1) + Xeye*xcoef(2:end,1);
y = ycoef(1) + Xeye*ycoef(2:end,1);
eyePos = [x y];


if ip.Results.plot
figure(2); clf

bins = {-20:.25:20, -20:.25:20};
C = hist3(eyePos(validIx,:), 'Ctrs', bins ); colormap parula
C = log(C');
C(C<1) = 0;
imagesc(bins{1}, bins{2},C ); hold on
colorbar
% plot(xy(:,1), xy(:,2),'.'); hold on
xlim([-18 18])
ylim([-18 18])

plot(xlim, [0 0], 'r')
plot([0 0], ylim, 'r')

[xx,yy] = meshgrid(-10:5:10, -10:5:10);
plot(xx(:), yy(:), 's')

figure(3); clf
plot(eyePos(:,1)); hold on
plot(Exp.vpx.smo(:,2))
end