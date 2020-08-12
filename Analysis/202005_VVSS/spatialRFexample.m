%% load data
close all
sessId = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'kilo', 'cleanup_spikes', 1);
% eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);

lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);

%% basic unit stats
spkS = io.get_visual_units(Exp);
Wf = io.get_waveform_stats(Exp.osp);

%%
[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', [-250 -250 250 250], 'binSize', Exp.S.pixPerDeg/2, 'eyePos', eyePos);
%%

% 3, 25, 27, 31
cc = 31; %cc+ 1;

nlags = 10;
sta = simpleRevcorr(Xstim, RobsSpace(:,cc)-mean(RobsSpace(:,cc)), nlags);


% find spatial RF
[~, bestlag] = max(std(sta,[],2));
        
        
srf = reshape(sta(bestlag,:), opts.dims);
        


%% 
figure(1); clf
set(gcf, 'Color', 'w')
imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, srf); axis xy
hold on
colormap parula
title(cc)
plot.fixfigure(gcf, 18, [6 6], 'OffsetAxes', false)
saveas(gcf, 'Figures/exampleRF_noGrid.pdf')

c = 'y';
plot(xlim, [0 0], 'Color', c)
plot([0 0],ylim, 'Color', c)
th = linspace(0, 2*pi, 100);
r = 1.6;
plot(r*cos(th), r*sin(th), 'Color', c)
plot.fixfigure(gcf, 18, [6 6], 'OffsetAxes', false)
saveas(gcf, 'Figures/exampleRF_Grid.pdf')

%% Eye position

t = Exp.vpx.smo(:,1);
x = sgolayfilt(Exp.vpx.smo(:,2), 1, 7);
y = sgolayfilt(Exp.vpx.smo(:,3), 1, 7);
L = Exp.vpx.Labels;
t0 = 4018;


figure(1); clf
plot(t-t0, x, 'k', 'Linewidth', 2); hold on
plot(t-t0, y, 'Color', .5*[1 1 1], 'Linewidth', 2); hold on
% plot(t(L==2)-t0, x(L==2), '.')
% plot(t(L==2)-t0, y(L==2), '.')
xlim([0 8])
xlabel('Time (seconds)')
ylabel('Degrees')
plot.fixfigure(gcf, 18, [12 5], 'OffsetAxes', false)
saveas(gcf, 'Figures/exampleEyeTraces.pdf')

plot(t(L==2)-t0, x(L==2), '.')
plot(t(L==2)-t0, y(L==2), '.')
plot.fixfigure(gcf, 18, [12 5], 'OffsetAxes', false)
saveas(gcf, 'Figures/exampleEyeTracesSaccades.pdf')

ylim([-1 1])
saveas(gcf, 'Figures/exampleEyeTracesZoom.pdf')

%%
xax = linspace(-15, 15, 100);
C = histcounts2(x(L==1),y(L==1),xax, xax);
figure(1); clf

h = imagesc(xax, xax, sqrt(imgaussfilt((C'))));
% h.AlphaData = log(C')/max(log(C(:)));
colormap parula
xlim([-12 12])
ylim([-7 7])
axis xy
%%

nsac = size(Exp.slist,1);
fprintf('%d saccades\n', nsac)
sacon = Exp.slist(:,4);
sacoff = Exp.slist(:,5);

invalid = Exp.vpx.Labels==4;

invalid  = invalid(sacoff) | invalid(sacon);
ninv = sum(invalid);
fprintf('%d invalid saccades\n', ninv);

sacon(invalid) = [];
sacoff(invalid) = [];

dx = Exp.vpx.smo(sacoff,2) - Exp.vpx.smo(sacon,2);
dy = Exp.vpx.smo(sacoff,3) - Exp.vpx.smo(sacon,3);

sacAmp = hypot(dx, dy);
sacDuration = Exp.vpx.smo(sacoff,1)-Exp.vpx.smo(sacon,1);




nsac = numel(sacon);
pv = nan(nsac,1);
for i = 1:nsac
    pv(i) = max(Exp.vpx.smo(sacon(i):sacoff(i), 7));
end

goodDur = sacDuration > 0.005 & sacDuration < .1;
goodV = pv > 20 & pv < 1.2e3;
goodIx = find(goodDur & goodV);

dx = dx(goodIx);
dy = dy(goodIx);
sacAmp = sacAmp(goodIx);
sacDuration = sacDuration(goodIx);
pv = pv(goodIx);

xax = -6:.1:6;
xc = (xax(1:end-1) + xax(2:end)) / 2;
C = histcounts2(dx,dy,xax, xax);
figure(1); clf

imagesc(xc, xc, log(imgaussfilt((C'))))
xlabel('d.v.a')
title('Saccade end points')


figure(2); clf
hold on
histogram(sacAmp, 'binEdges', linspace(0, 20, 200), 'EdgeColor', 'none')
xlabel('Amplitude (d.v.a)')
ylabel('Count')
title('Saccade Amplitude')
plot.fixfigure(gcf, 18, [5 3], 'OffsetAxes', false)
saveas(gcf, 'Figures/sacAmplitude.pdf')

figure(3); clf
histogram(sacDuration*1e3, linspace(0, .06, 50)*1e3, 'EdgeColor', 'none')
xlabel('Duration (ms)')
plot.fixfigure(gcf, 18, [5 3], 'OffsetAxes', false)
saveas(gcf, 'Figures/sacDuration.pdf')

nsac = numel(sacDuration);
fprintf('%d saccades\n', nsac)

%%
figure(4); clf

[C, xax, yax] = histcounts2(sacAmp, pv, 100);
h = imagesc(xax, yax, log(C')); axis xy
colormap(flipud(parula))
h.AlphaData = log(C')/max(log(C(:)));
hold on
% plot(sacAmp(1:1e3), pv(1:1e3), '.')

good = ~isnan(pv);
fun = @(params, x) (params(1)*x).^params(2);

evalc('phat = robustlsqcurvefit(fun, [50 1], sacAmp(good), pv(good));');
plot(linspace(0, 10, 100), fun(phat, linspace(0, 10, 100)), 'k')
fprintf('Slope: %02.2f, Exponent: %02.2f\n', phat(1), phat(2))
xlabel('Amplitude (d.v.a.)')
ylabel('Peak Velocity (deg/sec)')
plot.fixfigure(gcf, 18, [8 8], 'OffsetAxes', false)
saveas(gcf, 'Figures/mainSequence.pdf')


% plot(sta(:,find(srf == max(srf(:)))))