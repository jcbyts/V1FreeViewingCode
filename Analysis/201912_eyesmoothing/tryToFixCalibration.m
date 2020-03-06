
%% 

[Exp, S] = io.dataFactory(8);


%%

validTrials = io.getValidTrials(Exp, 'Forage');

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
validIx = getTimeIdx(eyeTime, tstart, tstop);

xy = Exp.vpx.smo(validIx,2:3);

figure(1); clf
% plot(xy(:,1), xy(:,2), '.')


bins = {-20:.25:20, -20:.25:20};
C = hist3(xy, 'Ctrs', bins ); colormap parula
imagesc(bins{1}, bins{2}, log(C')); hold on
xlim([-18 18])
ylim([-18 18])

r = Exp.D{validTrials(1)}.P.stimEcc;
th = linspace(0, 2*pi, 100);

plot(r*cos(th), r*sin(th), 'r')

%%


eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
figure(1); clf
counter = 0;

for iTrial = 1:numel(validTrials)
    thisTrial = validTrials(iTrial);
    
    tptb = Exp.D{thisTrial}.rewardtimes;
    t = Exp.ptb2Ephys(tptb);
    
    for ii = 1:numel(tptb)
        idx = find(Exp.D{thisTrial}.PR.ProbeHistory(:,4) < tptb(ii), 1, 'last');
        eyeidx = find(eyeTime < t(ii), 1, 'last');
        
        counter = counter + 1;
        probeX(counter) = Exp.D{thisTrial}.PR.ProbeHistory(idx-1,1);
        probeY(counter) = Exp.D{thisTrial}.PR.ProbeHistory(idx-1,2);
        
        iix = (-50:0)+eyeidx;
        xy = Exp.vpx.smo(iix, 2:3);
        bw = bwlabel(Exp.vpx.Labels(iix)==1);
        
        eyeX(counter) = mean(xy(bw == max(bw),1));
        eyeY(counter) = mean(xy(bw == max(bw),2));
        
        subplot(1,2,1)
        plot(probeX(counter), eyeX(counter), 'ob')
        hold on
        subplot(1,2,2)
        plot(probeY(counter), eyeY(counter), 'ob')
        hold on
        
    end
    
end

subplot(1,2,1)
plot(xlim, xlim, 'k')
subplot(1,2,2)
plot(xlim, xlim, 'k')

%%
X = [eyeX(:) eyeY(:) eyeX(:).^2 eyeY(:).^2 ones(counter,1)];
Y = [probeX(:) probeY(:)];

bad = isnan(sum(X,2)) | isnan(sum(Y,2));
X(bad,:) = [];
Y(bad,:) = [];

w = (X'*X) \ (X'*Y);
w(1,1) = 1;
w(2,2) = 1;
Xeye = [Exp.vpx.smo(:,2:3) Exp.vpx.smo(:,2:3).^2 ones(size(Exp.vpx.smo,1),1)];
xy = Xeye*w;

figure(1); clf
plot(xy(:,1)); hold on
plot(Exp.vpx.smo(:,2))

%%
validTrials = io.getValidTrials(Exp, 'Forage');

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
validIx = getTimeIdx(eyeTime, tstart, tstop);

% xy = Exp.vpx.smo(validIx,2:3);

figure(1); clf
% plot(xy(:,1), xy(:,2), '.')


bins = {-20:.25:20, -20:.25:20};
C = hist3(xy, 'Ctrs', bins ); colormap parula
imagesc(bins{1}, bins{2}, log(C')); hold on
xlim([-18 18])
ylim([-18 18])

r = Exp.D{validTrials(1)}.P.stimEcc;
th = linspace(0, 2*pi, 100);

plot(r*cos(th), r*sin(th), 'r')

%% check fixflash
validTrials = io.getValidTrials(Exp, 'FaceCal');

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
validIx = getTimeIdx(eyeTime, tstart, tstop);

xy = Exp.vpx.smo(validIx,2:3);
% xy = io.getCorrectedEyePos(Exp, 'usebilinear', true);
% xy = xy(validIx,:);

xd = xy(:,1) - xx(:)';
yd = xy(:,2) - yy(:)';
d = hypot(xd, yd);
[d,id] = min(d, [], 2);

xy = xy(d<1.5,:);
id = id(d<1.5);


figure(1); clf
% plot(xy(:,1), xy(:,2), '.')


bins = {-20:.25:20, -20:.25:20};
C = hist3(xy, 'Ctrs', bins ); colormap parula
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


% Exp.D{validTrials(1)}.PR.NoiseHistory

figure(2); clf
plot(xx(id), xy(:,1), '.'); hold on
plot(yy(id)+.5, xy(:,2), '.')
plot(xlim, xlim)

% X = [xx(id) yy(id) xx(id).^2 yy(id).^2 ones(numel(id), 1)];
% Y = xy;

X = [xy xy.^2 ones(numel(id), 1)];
Y = [xx(id) yy(id)];

w = (X'*X)\(X'*Y);

eyePos = Exp.vpx.smo(validIx,2:3);
Xeye = [eyePos eyePos.^2 ones(size(eyePos,1),1)];

xy = Xeye*w;

figure(2); clf
% plot(xy(:,1), xy(:,2), '.')


bins = {-20:.25:20, -20:.25:20};
C = hist3(xy, 'Ctrs', bins ); colormap parula
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


    