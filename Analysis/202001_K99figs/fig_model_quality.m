
sessId = 12;
[Exp,S] = io.dataFactory(sessId);
%% 
tmp = load('Data/model1231_predictions2.mat');
stas = load('Data/model_stas.mat');

%%
model = load('Data/export1231_csn1.mat');

%%
cc = 36
model2 = model;
% figure 3 model
figure(3); clf,
% plot(model.ws01
lev2 = model2.ws10(:,cc);
model2.ws10([17:36 50:60],:) = -model2.ws10([17:36 50:60],:);
nsubs = size(model2.ws00,2);
subs  = find(abs(lev2) > 0.001);

subsFromLayer1 = subs(subs < nsubs);
subsFromLayer2 = subs(subs > nsubs) - nsubs;
[~, ind] = sort(sum(model2.ws01(:,subsFromLayer2)));
subsFromLayer2 = subsFromLayer2(ind);



model2.ws01((36/2)+1:36,:) = -model2.ws01((36/2)+1:36,:);
[ss, ind] = sort(sum(model2.ws01(:,subsFromLayer2),2), 'descend');

l2subsFromL1 = ind(ss > 0);

l2subsFromL1(ismember(l2subsFromL1, subsFromLayer1)) = [];

L1subs = [subsFromLayer1; l2subsFromL1];

nL1 = numel(L1subs);
nL2 = numel(subsFromLayer2);

NX = 24;
nlags = 10;

step = .9/nL1;
for i = 1:nL1
    axes('Position', [i*step .5 step .3])
    isub = L1subs(i);
    
    w = reshape(model2.ws00(:,isub), [nlags, NX*NX]);
    [~, peaklag] = max(sum(w.^2,2));
    
    I = reshape(w(peaklag,:), [NX NX]);
    w2 = sum(abs(model2.ws01(isub, subsFromLayer2)));
    w3 = (model2.ws10(isub,cc));
    I = I .* max(w2,w3);
    
        
    imagesc(I, [-1 1]*.2);
    axis off
    if ismember(isub, subsFromLayer1)
        axis on
        set(gca, 'Color', 'r', 'XTick', '', 'YTick', '')
    end
    title(isub)
end
colormap gray

step = .6/nL2;
for i = 1:nL2
    axes('Position', [.05+i*step .1 step .3])
    isub = subsFromLayer2(i);
    w = model2.ws00*model2.ws01(:,isub);
    w = reshape(w, [nlags, NX*NX]);
    w2 = model2.ws10(36+isub,cc);
    [~, peaklag] = max(sum(w.^2,2));
    I = reshape(w(peaklag,:), [NX NX]);
    I = I.*w2;
    imagesc(I, [-1 1]*.05);
    axis off
    
%     model2.ws01(:,isub)
    title(isub)
end

    

ev = find(tmp.sac_offBi);

plot.fixfigure(gcf, 8, [6 3], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', sprintf('subunitsModelExample%02.0f.pdf', cc)))

%%
figure(1); clf
w = model2.ws00*model2.ws01;
n2 = size(w,2);
step = .6/n2;
for i = 1:n2
    axes('Position', [.05+i*step .1 step .3])
    a = reshape(w(:,i), [nlags NX*NX]);
    [~, peaklag] = max(sum(a.^2,2));
    I = reshape(a(peaklag,:), [NX NX]);
    I = I.*w2;
    imagesc(I);
    axis off
end

colormap gray
plot.fixfigure(gcf, 8, [6 3], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', sprintf('subunitsModelL2Example%02.0f.pdf', cc)))
NC = size(tmp.RobsAll,2);
%%
cc = mod(cc + 1, NC); cc = max(cc, 1);

[~, ind] = sort(model2.ws10(:,44), 'descend');

w = [model2.ws00 model2.ws00*model2.ws01];
n = 10;
step = .6/n;

figure(1); clf
for i = 1:n
    isub = ind(i);
    axes('Position', [.05+i*step .1 step .3])
    a = reshape(w(:,isub), [nlags NX*NX]);
    [~, peaklag] = max(sum(a.^2,2));
    I = reshape(a(peaklag,:), [NX NX]);
    imagesc(I)
    axis off
end
    


%% predict with

load('Data/L20191231_Grating_Cstim.mat')
st = load('Data/L20191231_GratingL.mat')
%%
nlags = 10;
dims = sqrt(size(Cstim,2))*[1 1];

spar = NIM.create_stim_params([nlags dims]); %, 'tent_spacing', 1);
X = NIM.create_time_embedding(Cstim, spar);

%%
cc = 44;
w = squeeze(stas.stas(:,:,cc))';

Xfilt = zeros(size(Cstim));
for i = 1:prod(dims)
    Xfilt(:,i) = filter(w(:,i), 1, Cstim(:,i));
end

RobSTA = (sum(Xfilt,2));
%%
[stim, Robs, opts, params, kx, ky] = io.preprocess_grating_subspace_data(Exp);
NT = size(stim{1},1);


Xstim = NIM.create_time_embedding(stim{1}, params(1));

Xd = [Xstim ones(NT,1)];
C = Xd'*Xd;

%%
ix = tmp.frameTimesAll~=0;
R = interp1(tmp.frameTimesAll(ix), tmp.RobsAll(ix,:), opts.frameTime);
Rhat = interp1(tmp.frameTimesAll(ix), tmp.pred1All(ix,:), opts.frameTime);
Rsta = interp1(st.frameTimes, RobSTA, opts.frameTime);
Rsta(abs(Rsta)>200)=0;
Rsta(isnan(Rsta)) = 0;
Rsta = Rsta./std(Rsta)*std(R(:,44));
Rsta = max(Rsta,0);

figure(1); clf
plot(Rsta)
%%
NC = size(R,2);

tic
nBoot = 50;
    oriTun = zeros(opts.num_kx, NC, nBoot);
    oriTunH = zeros(opts.num_kx, NC, nBoot);
    oriTunS = zeros(opts.num_kx, nBoot);
for iBoot = 1:nBoot
    iix = randi(NT,NT,1);
    xy = Xd(iix,:)'*R(iix,:);
    xyHat = Xd(iix,:)'*Rhat(iix,:);
    xyS = Xd(iix,:)'*Rsta(iix);
    
    % wls = (C + 1e2*eye(size(C,2)))\xy;
    % sta = wls(1:end-1,:);
    staSubspace = xy(1:end-1,:);
    staSubspace = staSubspace./sum(Xstim)';
    staSubspaceH = xyHat(1:end-1,:);
    staSubspaceH = staSubspaceH./sum(Xstim)';
    
    staSubspaceS = xyS(1:end-1);
    staSubspaceS = staSubspaceS./sum(Xstim)';
    
    a = reshape(staSubspaceS, [opts.num_lags_stim, prod(opts.dim)]);
    [~, peakLag] = max(mean(a,2));
    a = (reshape(a(peakLag,:), opts.dim));
    [~, peakSF] = max(mean(a));
    oriTunS(:,iBoot) = a(:,peakSF)*120;
    
    for cc = 1:NC
        a = reshape(staSubspace(:,cc), [opts.num_lags_stim, prod(opts.dim)]);
        b = reshape(staSubspaceH(:,cc), [opts.num_lags_stim, prod(opts.dim)]);
        [~, peakLag] = max(mean(a,2));
        a = (reshape(a(peakLag,:), opts.dim));
        b = (reshape(b(peakLag,:), opts.dim));
        [~, peakSF] = max(mean(a));
        oriTun(:,cc,iBoot) = a(:,peakSF)*120;
        oriTunH(:,cc,iBoot) = b(:,peakSF)*120;
    end
end
toc
% 
% figure(1); clf
% sx = ceil(sqrt(NC));
% sy = round(sqrt(NC));
% for cc = 1:NC
%     subplot(sx, sy, cc)
%     a = reshape(staSubspace(:,cc), [opts.num_lags_stim, prod(opts.dim)]);
%     imagesc(a);
%     drawnow
% end
% 
% figure(2); clf
% sx = ceil(sqrt(NC));
% sy = round(sqrt(NC));
% for cc = 1:NC
%     subplot(sx, sy, cc)
%     a = reshape(staSubspaceH(:,cc), [opts.num_lags_stim, prod(opts.dim)]);
%     imagesc(a);
%     drawnow
% end

%%
cc = mod(cc + 1, NC); cc = max(cc, 1);
% cc = [24 36 44 47 48]
cc = 44;
figure(1); clf
nlags = size(stas.predstas,2);
NX = 40;
sta = squeeze(stas.stas(:,:,cc));
sta = (sta - min(sta(:)))/(max(sta(:)) - min(sta(:)));
staH = squeeze(stas.predstas(:,:,cc));
staH = (staH - min(staH(:)))/(max(staH(:)) - min(staH(:)));

for ilag = 1:nlags
    subplot(2,nlags,ilag, 'align')
    imagesc(reshape(sta(:,ilag), [NX, NX]), [0 1]); hold on
    
    subplot(2,nlags,nlags +ilag, 'align')
    imagesc(reshape(staH(:,ilag), [NX, NX]), [0 1]); hold on
end
colormap gray
NC = size(tmp.pred1Rsvp,2);

figure(2); clf

oriTuning = prctile(squeeze(oriTun(:,cc,:)),[2.5 50 97.5], 2);
oriTuningH = prctile(squeeze(oriTunH(:,cc,:)),[2.5 50 97.5], 2);

plot(opts.kxs, oriTuning, 'b'); hold on
plot(opts.kxs, oriTuningH, 'r')
plot(opts.kxs, oriTuning(:,2), 'b.'); hold on
plot(opts.kxs, oriTuningH(:,2), 'r.')
plot(opts.kxs, prctile(oriTunS, [2.5 50 97.5], 2), 'g')
title(cc)
xlabel('Orientation')
ylabel('Firing Rate')
plot.fixfigure(gcf, 8, [4 4])
saveas(gcf, fullfile('Figures', 'K99', sprintf('oriTuningModel%02.0f.pdf', cc)))

%%
model2 = model;
% figure 3 model
figure(3); clf,
% plot(model.ws01
lev2 = model2.ws10(:,cc);
model2.ws10([17:36 50:60],:) = -model2.ws10([17:36 50:60],:);
nsubs = size(model2.ws00,2);
subs  = find(abs(lev2) > 0.001);

subsFromLayer1 = subs(subs < nsubs);
subsFromLayer2 = subs(subs > nsubs) - nsubs;
[~, ind] = sort(sum(model2.ws01(:,subsFromLayer2)));
subsFromLayer2 = subsFromLayer2(ind);



model2.ws01((36/2)+1:36,:) = -model2.ws01((36/2)+1:36,:);
[ss, ind] = sort(sum(model2.ws01(:,subsFromLayer2),2), 'descend');

l2subsFromL1 = ind(ss > 0);

l2subsFromL1(ismember(l2subsFromL1, subsFromLayer1)) = [];

L1subs = [subsFromLayer1; l2subsFromL1];

nL1 = numel(L1subs);
nL2 = numel(subsFromLayer2);

NX = 24;
nlags = 10;

step = .9/nL1;
for i = 1:nL1
    axes('Position', [i*step .5 step .3])
    isub = L1subs(i);
    
    w = reshape(model2.ws00(:,isub), [nlags, NX*NX]);
    [~, peaklag] = max(sum(w.^2,2));
    I = reshape(w(peaklag,:), [NX NX]);
    imagesc(I, [-1 1]*.2);
    axis off
    if ismember(isub, subsFromLayer1)
        axis on
        set(gca, 'Color', 'r', 'XTick', '', 'YTick', '')
    end
    title(isub)
end
colormap gray

step = .6/nL2;
for i = 1:nL2
    axes('Position', [.05+i*step .1 step .3])
    isub = subsFromLayer2(i);
    w = model2.ws00*model2.ws01(:,isub);
    w = reshape(w, [nlags, NX*NX]);
    [~, peaklag] = max(sum(w.^2,2));
    I = reshape(w(peaklag,:), [NX NX]);
    imagesc(I, [-1 1]*.25);
    axis off
    
%     model2.ws01(:,isub)
    title(isub)
end

    

ev = find(tmp.sac_offBi);

plot.fixfigure(gcf, 8, [6 3], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', sprintf('subunitsModelExample%02.0f.pdf', cc)))

%%
figure(4); clf
win = [-20 20];
cc = mod(cc + 1, NC); cc = max(cc, 1);
% cc = 25;
% subplot(1,2,1)
[an, sd, lags] = eventTriggeredAverage((double(tmp.RobsBi(2:end,:))), ev, win);
plot.errorbarFill(lags, an(:,cc)*120, sd(:,cc)*120./sqrt(numel(ev)), 'b', 'FaceColor', 'b', 'FaceAlpha', .5); hold on
% subplot(1,2,2)
[an2, sd, lags] = eventTriggeredAverage((double(tmp.pred1Bi)), ev, win);
% an2 = an2 - mean(an2) + mean(an);
plot.errorbarFill(lags, an2(:,cc)*120, sd(:,cc)*120./sqrt(numel(ev)), 'r', 'FaceColor', 'r', 'FaceAlpha', .5); hold on

% plot(model.ws01(:,subs)

%%
figure(10); clf
[tt, sublist] = sort(sum(model.ws01,2));
sublist(tt==0) = [];

sublist = find(tt~=0);
n = numel(sublist);
for i = 1:n
    subplot(2,ceil(n/2), i, 'align')
    isub = sublist(i);
    
    w = reshape(model.ws00(:,isub), [nlags, NX*NX]);
    [~, peaklag] = max(sum(w.^2,2));
    I = reshape(w(peaklag,:), [NX NX]);
    imagesc(I, [-1 1]*.2);
    axis off
    if ismember(isub, subsFromLayer1)
        axis on
        set(gca, 'Color', 'r', 'XTick', '', 'YTick', '')
    end
    title(isub)
end
colormap gray


plot.fixfigure(gcf, 8, [14 3], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', sprintf('sharedsubunitsL1%02.0f.pdf', cc)))

%%
figure(10); clf
[tt, sublist] = sort(sum(model.ws01,2));
sublist(tt==0) = [];

sublist = find(tt~=0);
n = numel(sublist);
for i = 1:n
    subplot(2,ceil(n/2), i, 'align')
    isub = sublist(i);
    
    w = reshape(model.ws00(:,isub), [nlags, NX*NX]);
    [~, peaklag] = max(sum(w.^2,2));
    I = reshape(w(peaklag,:), [NX NX]);
    imagesc(I, [-1 1]*.2);
    axis off
    if ismember(isub, subsFromLayer1)
        axis on
        set(gca, 'Color', 'r', 'XTick', '', 'YTick', '')
    end
    title(isub)
end
colormap gray


plot.fixfigure(gcf, 8, [14 3], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', sprintf('sharedsubunitsL1%02.0f.pdf', cc)))

%% fancy sta plot
dims = [40 40];
xax = ((-dims(2)/2 + 1):dims(2)/2)*1.5;
yax = -(1:dims(1))*1.5;
figure(1); clf
% NC = size(tmp.stas,3);
% 
%    imagesc(reshape(sta(:,ilag), [NX, NX]), [0 1]); hold on
%     
%     subplot(2,nlags,nlags +ilag, 'align')
%     imagesc(reshape(staH(:,ilag), [NX, NX]), [0 1]); hold on
%     
    
clim = max(abs(sta(:)))*[-1 1]*1;    
for ilag = 1:nlags
    subplot(1,nlags,ilag, 'align')
    imagesc(xax, yax, reshape(sta(:,ilag), dims), clim)
    axis xy
%     contourf(reshape(a(:,ilag), [40 40]),[-4:.1:-2  0 2:6], 'Linestyle', 'none')
end
title(cc)
colormap(gray)


%%

cmap = gray;
[xx,tt,yy] = meshgrid(xax, (1:nlags)*8, yax);
% [xx,yy] = meshgrid(xax, yax);
I = reshape(sta, [dims nlags]);
I = permute(I, [3 2 1]);

figure(2); clf
set(gcf, 'Color', 'w')
h = slice(xx,tt, yy, I, [], (1:10)*8,[]);
set(gca, 'CLim', clim)
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
%     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end
view(79,11)
colormap(cmap)
axis off

hold on
% plot3(10+[10 16], [1 1]*39, -[50 50], 'r', 'Linewidth', 2)
% plot3(10+[10 16], [1 1]*30.5, -[50 50], 'r', 'Linewidth', 2)
% text(40, 32, -50, '0.1 d.v.a', 'Color', 'r')

plot.fixfigure(gcf, 8, [14 3])
saveas(gcf, fullfile('Figures', 'K99', sprintf('staTrue%02.0f.png', cc)))

I = reshape(staH, [dims nlags]);
I = permute(I, [3 2 1]);

figure(2); clf
set(gcf, 'Color', 'w')
h = slice(xx,tt, yy, I, [], (1:10)*8,[]);
set(gca, 'CLim', clim)
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
%     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end
view(79,11)
colormap(cmap)
axis off

hold on
% plot3(10+[10 16], [1 1]*39, -[50 50], 'r', 'Linewidth', 2)
% plot3(10+[10 16], [1 1]*30.5, -[50 50], 'r', 'Linewidth', 2)
% text(40, 32, -50, '0.1 d.v.a', 'Color', 'r')

plot.fixfigure(gcf, 8, [14 3])
saveas(gcf, fullfile('Figures', 'K99', sprintf('staModel%02.0f.png', cc)))

%%


ev = find(tmp.sac_offBi);


figure(4); clf
win = [-20 500];
% cc = 25;
% subplot(1,2,1)
[an, sd, lags, wfs] = eventTriggeredAverage((double(tmp.RobsBi(2:end,cc))), ev, win);
subplot(1,2,1)
imagesc(wfs)
% subplot(1,2,2)
[an2, sd, lags, wfs2] = eventTriggeredAverage((double(tmp.pred1Bi(:,cc))), ev, win);

subplot(1,2,2)
imagesc(wfs2)

%%
clf
iTrial = iTrial + 1;
plot(wfs(iTrial,:)); hold on
plot(wfs2(iTrial,:))
title(iTrial)

%%
figure(1); clf
rho = corr(double(tmp.RobsBi(2:end,:)), double(tmp.pred1Bi(1:end-1,:)));
rho = diag(rho);
plot(rho); hold on

rho = corr(double(tmp.RobsBi(:,:)), double(tmp.pred1Bi(:,:)));
rho = diag(rho);
plot(rho); hold on
% plot(mo


%%

eyeTime = Exp.vpx2ephys(Exp.vpx.raw(:,1));
remove = find(diff(eyeTime)==0);

eyeX = sgolayfilt(Exp.vpx.smo(:,2), 3, 9);
eyeX(isnan(eyeX)) = 0;
eyeY = sgolayfilt(Exp.vpx.smo(:,3), 3, 9);
eyeY(isnan(eyeY)) = 0;

eyeTime(remove) = [];
eyeX(remove) = [];
eyeY(remove) = [];

% resample at higher temporal resolution with fixed sample times
nuEyeX = interp1(eyeTime, eyeX, tmp.frameTimesBi, 'pchip');
nuEyeY = interp1(eyeTime, eyeY, tmp.frameTimesBi, 'pchip');


%%
figure(2); clf
cc = mod(cc + 1, NC); cc = max(cc, 1);

win = [-20 50];
% cc = 25;
% subplot(1,2,1)
[an, sd, lags, wfs] = eventTriggeredAverage((double(tmp.RobsBi(:,cc))), ev, win);
[an2, sd, lags, wfs2] = eventTriggeredAverage((double(tmp.pred1Bi(:,cc))), ev, win);


nt = size(wfs,1);
rtmodel = zeros(nt,1);
for t = 1:nt
    ix = ~isnan(wfs(t,:));
    if sum(ix)<5
        continue
    end
%     rtmodel(t) = rsquared(wfs(t,ix), wfs2(t,ix));
    
    rtmodel(t) = bitsPerSpike(wfs2(t,ix)',wfs(t,ix)', nanmean(wfs(t,ix)));
    
end
max(rtmodel)

[m, ind] = sort(rtmodel, 'descend');


bad = isnan(m)|m==0;
m(bad) = [];
ind(bad) = [];


subplot(1,2,1)
imagesc(wfs(ind(1:20),:))
% subplot(1,2,2)


[~, ~, ~, wfsEyeX] = eventTriggeredAverage(nuEyeX, ev, win);

subplot(1,2,2)
imagesc(wfs2(ind(1:50),:))
title(cc)

%
figure(3); clf
plot(nanmean(wfs(ind(m>median(m)),:))); hold on
plot(nanmean(wfs2(ind(m>median(m)),:))); hold on

%%
% [m, id] = max(rtmodel);


for i = 1:25
    id = ind(i);
figure(1); clf
subplot(2,1,1)
plot(wfs(id,:)); hold on
plot(wfs2(id,:))
title(cc)
subplot(2,1,2)
plot(wfsEyeX(id,:))
title(m(i))
pause
end


%%
plot.fixfigure(gcf, 8, [6 3], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', sprintf('samplefix%02.0f.pdf', cc)))

%%
figure(1); clf
iTrial = iTrial + 1;
plot(wfs(iTrial,:)); hold on
plot(wfs2(iTrial,:))
title(iTrial)
% tmp
%%
figure(1); clf
[a, trials] = sort(sum(wfs.*(wfs2),2), 'descend');
trials(isnan(a)) = [];
figure(1); clf
t = t+1;
iTrial = trials(t);
plot(wfs(iTrial,:)); hold on
plot(wfs2(iTrial,:))