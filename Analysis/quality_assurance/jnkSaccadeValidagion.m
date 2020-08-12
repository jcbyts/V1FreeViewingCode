
%%
Exp = io.dataFactoryGratingSubspace(1);

%% redetect saccades
% clean up and resample
Exp.vpx.raw = Exp.vpx.raw0;

Exp.vpx.raw(isnan(Exp.vpx.raw)) = 32000; % there shouldn't be any, but just in case

x = double(Exp.vpx.raw(:,2)==32000);
x(1) = 0;
x(end) = 0;

bon = find(diff(x)==1);
boff = find(diff(x)==-1);

bon = bon(2:end); % start of bad
boff = boff(1:end-1); % end of bad

gdur = Exp.vpx.raw(bon,1)-Exp.vpx.raw(boff,1);

remove = gdur < 1;

bremon = bon(remove);
bremoff = boff(remove);
gdur(remove) = [];

figure(1); clf,
histogram(gdur, 'binEdges', linspace(0, 5, 1e3))
goodSnips = sum(gdur/60);

totalDur = Exp.vpx.raw(end,1)-Exp.vpx.raw(1,1);
totalMin = totalDur/60;

fprintf('%02.2f/%02.2f minutes are usable\n', goodSnips, totalMin)

% go back through and eliminate snippets that are not analyzable
n = numel(bremon);
for i = 1:n
    Exp.vpx.raw(bremoff(i):bremon(i),2:end) = 32e3;
end


[~,ia] =  unique(Exp.vpx.raw(:,1));
Exp.vpx.raw = Exp.vpx.raw(ia,:);

% upsample eye traces to 1kHz
new_timestamps = Exp.vpx.raw(1,1):1e-3:Exp.vpx.raw(end,1);
new_EyeX = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,2), new_timestamps);
new_EyeY = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,3), new_timestamps);
new_Pupil = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,4), new_timestamps);
bad = interp1(Exp.vpx.raw(:,1), double(Exp.vpx.raw(:,2)>31e3), new_timestamps);
Exp.vpx.raw = [new_timestamps(:) new_EyeX(:) new_EyeY(:) new_Pupil(:)];

Exp.vpx.raw(bad>0,2:end) = nan;

figure(1); clf
plot(Exp.vpx.raw(:,2))
hold on

%%
% Saccade processing:
% Perform basic processing of eye movements and saccades
Exp = saccadeflag.run_saccade_detection_cloherty(Exp, ...
    'ShowTrials', false,...
    'accthresh', 2e4,...
    'velthresh', 8,...
    'velpeak', 10,...
    'isi', 0.02);


Exp.vpx.Labels(isnan(Exp.vpx.raw(:,2))) = 4;

%%

t = Exp.vpx.smo(:,1);
x = sgolayfilt(Exp.vpx.smo(:,2), 1, 7);
y = sgolayfilt(Exp.vpx.smo(:,3), 1, 7);
L = Exp.vpx.Labels;
t0 = 4018;


figure(1); clf
plot(t-t0, x, 'k'); hold on
plot(t-t0, y, 'Color', .5*[1 1 1]); hold on
% plot(t(L==2)-t0, x(L==2), '.')
% plot(t(L==2)-t0, y(L==2), '.')
xlim([0 8])
%%
nsac = size(Exp.slist,1);
fprintf('%d saccades\n', nsac)
sacon = Exp.slist(:,4);
sacoff = Exp.slist(:,5);

invalid = Exp.vpx.Labels==4;

% sacon = find([0; diff(Exp.vpx.Labels==2)==1]);
% sacoff = find([0; diff(Exp.vpx.Labels==2)==-1]);

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

contourf(xc, xc, log(imgaussfilt((C'), 1)))
xlabel('d.v.a')
title('Saccade end points')


figure(2); clf
hold on
histogram(sacAmp, 'binEdges', linspace(0, 20, 200), 'EdgeColor', 'none')
xlabel('Amplitude (d.v.a)')
ylabel('Count')
title('Saccade Amplitude')


figure(3); clf
histogram(sacDuration, linspace(0, .06, 50), 'EdgeColor', 'none')

nsac = numel(sacDuration);
fprintf('%d saccades\n', nsac)

%%
figure(4); clf

[C, xax, yax] = histcounts2(sacAmp, pv, 100);
imagesc(xax, yax, log(C')); axis xy
colormap gray
hold on
plot(sacAmp(1:1e3), pv(1:1e3), 'r.')

good = ~isnan(pv);
fun = @(params, x) (params(1)*x).^params(2);

evalc('phat = robustlsqcurvefit(fun, [50 1], sacAmp(good), pv(good));');
plot(linspace(0, 10, 100), fun(phat, linspace(0, 10, 100)), 'r')
fprintf('Slope: %02.2f, Exponent: %02.2f\n', phat(1), phat(2))

%%
clf
d = (pv(good) - fun(phat, sacAmp(good))).^2;
dx = (d./pv(good));
thresh = prctile(dx, 95);
bad = dx > thresh;
ind = find(good);
histogram(d(~bad)./pv(ind(~bad)))

clf
x = sacAmp(ind(~bad));
y = pv(ind(~bad));
phat = robustlsqcurvefit(fun, [50 1], x, y);
fprintf('Slope: %02.2f, Exponent: %02.2f\n', phat(1), phat(2))
plot(sacAmp(ind(~bad)), pv(ind(~bad)), '.')
hold on
plot(linspace(0, 10, 100), fun(phat, linspace(0, 10, 100)), 'r')

%%
micro = goodIx(sacDuration < .25);
figure(1); clf
pos = hypot(Exp.vpx.smo(:,2), Exp.vpx.smo(:,3));
for i = 1:numel(micro)
    id = sacon(micro(i));
    ix = id + (-100:100);
    plot(ix, pos(ix))
    hold on
    ix = sacon(micro(i)):sacoff(micro(i));
    plot(ix, pos(ix), '.')
    pause
    hold off
    
end
    





