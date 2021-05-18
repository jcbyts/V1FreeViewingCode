ls
%%
Exp = io.dataFactoryGratingSubspace(18);
%%
cc = 0;

%%
validTrials = io.getValidTrials(Exp, 'Grating');
starts = cellfun(@(x) x.START_EPHYS, Exp.D(validTrials));
ends = cellfun(@(x) x.END_EPHYS, Exp.D(validTrials));



%%

addpath ~/Dropbox/MatlabCode/Repos/pdstools/
figure(1); clf
cc = cc+ 1; %cc + 1;
if cc > 35
    cc = 1;
end
ev = Exp.vpx2ephys(Exp.slist(:,1));
vsacix = getTimeIdx(ev, starts, ends);

cids = Exp.osp.cids;

dx = Exp.vpx.smo(Exp.slist(vsacix,5),2)-Exp.vpx.smo(Exp.slist(vsacix,4),2);
dy = Exp.vpx.smo(Exp.slist(vsacix,5),3)-Exp.vpx.smo(Exp.slist(vsacix,4),3);
ev = ev(vsacix);

mu = [1.6, -1];
sigma = 2;

d = exp( - ( (dx - mu(1)).^2 + (dy - mu(2)).^2)/2/sigma^2 ); 

[~, ind] = sort(d);



[~,~,bc, ~,wfs] = pdsa.eventPsth(Exp.osp.st(Exp.osp.clu==cids(cc)), ev(ind), [-.2 .25], 1e-3);


[i,j] = find(wfs);

plot.raster(bc(j),i,10)
title(cc)
axis tight

figure(2); clf
[~,~,bc, ~,wfs] = pdsa.eventPsth(Exp.osp.st(Exp.osp.clu==cids(cc)), ev(ind), [-.2 .25], 10e-3);
imagesc(imgaussfilt(wfs, [20 1]))
axis xy

figure(3); clf
plot(bc, mean(wfs(d(ind)>.5,:))); hold on
plot(bc, mean(wfs(d(ind)<.01,:)))

%%


%% plotting
figure(1); clf
isi = Exp.slist(2:end,1)-Exp.slist(1:end-1,2); % inter-saccade interval
histogram(isi, 'binEdges', 0:.01:1); hold on
title(size(Exp.slist, 1))


%% saccade detection
Exp = saccadeflag.run_saccade_detection_cloherty(Exp, 'ShowTrials', false, ...
    'accthresh', 2e4, ...
    'velthresh', 8, ...
    'velpeak', 10, ...
    'isi', 0.04);

%%
% Exp.

spd = Exp.vpx.smo(:,7);
figure(1); clf
plot(Exp.vpx.Labels==1 | Exp.vpx.Labels==2)
xlim([3e6 3.1e6])
%%
cc = cc + 1
ev = Exp.vpx2ephys(Exp.slist(:,1:2));
st = Exp.osp.st(Exp.osp.clu==cids(cc));
sm = 7;
[spbn_pre, bcenters, Spre] = prf.sac_trig_var_dur(st, ev, 'alignment', 'pre', 'win', [-.3 .1], ...
    'smoothing', sm);

xd = [-0.2 .1];
subplot(4,1,1:3)
axis tight
xlim(xd)
subplot(4,1,4)
axis tight
xlim(xd)

%%
figure(1);
j = j+1;
nsac = size(Exp.slist, 1);
for i = [ind(j) ind(j) + 1]
    plot(Exp.slist(i,4):Exp.slist(i,5), pos(Exp.slist(i,4):Exp.slist(i,5)), 'r.')
end

xlim(Exp.slist(i,4) + [-200 250])
ylim([pos(Exp.slist(i,4)) pos(Exp.slist(i,5))] + [-2 2])



iix = (Exp.slist(i-1,4)-10):(Exp.slist(i,5)+10);
%%

% lsqcurvefit
tic
x = iix; %-mean(iix);
y = spd(iix)';
mny = min(y);
mxy = max(y);
y = (y - mny)/(mxy-mny);

use = y > .2;

abc = polyfit(x(use), log(y(use)), 2);

a = abc(3);
b = abc(2);
c = abc(1);

mu = -b/(2*c);
sigma = sqrt(-1/(2*c));
amp = exp(a - b^2/(4*c));
toc

% figure(2); clf
% plot(x(use),log(y(use)))
% hold on
% plot(x, a + b*x + c*x.^2)
% f = polyval(abc, x);
% plot(x, f)

%
figure(2); clf
plot(x, y); hold on
yhat = amp*exp(-(x - mu).^2/2/sigma^2);
plot(x, yhat)

% 
phat = lsqcurvefit(fun, [1, 0, 5], x,y,[],[],opts);
yhat = fun(phat, x);
plot(x, yhat)

rsquared(yhat,y)

%%
tic
phat = lsqcurvefit(fun, [1, 0, 5], x,y,[],[],opts);
yhat = fun(phat, x);
plot(x, yhat)
toc
%%
x = iix;
y = spd(iix)';

figure(2); clf
plot(x,y); hold on

[params, fun] = fit_gaussian_poly(x,y);
plot(x, fun(params, x))
legend({'Data', 'Fit'})

%%
figure(3); clf
% plot(pos); hold on
% plot(sgolayfilt(pos, 2, 51))
plot( (pos - sgolayfilt(pos, 2, 51)).^2)
% xlim(Exp.slist(i,4) + [-200 250])
ylim([0 10])

%%
figure(1); clf
% spd=Exp.vpx.smo(:,7);
spd=Exp.vpx.smo(:,7);
acc=[0; diff(spd)];

acc = zscore(acc);
spd = zscore(spd);

outliers = abs(acc) > 100 | abs(spd) > 100;

acc(outliers) = 0;
spd(outliers) = 0;

plot(acc, spd, '.')

n = numel(acc);

ix = hypot(acc, spd) > 15;
plot(acc, spd, '.'); hold on
plot(acc(ix), spd(ix), '.')

%%
figure(1); clf
plot(Exp.vpx.smo(:,2), 'k'); hold on
plot(find(ix), Exp.vpx.smo(ix, 2), 'r')


%%
figure(1); clf
smpos = imgaussfilt(pos, 10);

plot(pos); hold on
plot(smpos)

%%
clf
plot(sgolayfilt(pos, 1, 7)-smpos)


