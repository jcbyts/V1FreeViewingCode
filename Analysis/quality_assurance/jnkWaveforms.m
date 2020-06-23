
Exp = io.dataFactoryGratingSubspace(1);
%%

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(:)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(:)));

epochs = [tstop(1:end-1) tstart(2:end)];

%%
W1 = io.get_waveform_stats(Exp.osp);
W = io.get_waveform_stats(Exp.osp, 'validEpochs', epochs);


%%
figure(1); clf
NC = numel(W);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    ix = W(cc).lags>=0;
    plot(W(cc).lags(ix), W(cc).isi(ix)); hold on
    plot(W1(cc).lags(ix), W1(cc).isi(ix))
    set(gca, 'xscale', 'log')
end

legend({'ITI', 'ALL'})

figure, histogram(arrayfun(@(x) x.BRI, W), 100)
%%
[stim, robs, grating] = io.preprocess_grating_subspace_data(Exp);
[Xspat, Rspat, spopts] = io.preprocess_spatialmapping_data(Exp, 'ROI', [-200 -200 300 300]);

%%
X = stim{1};
X = X(:,sum(X)>0);

y = robs(:,Exp.osp.cids(2));
sta = simpleRevcorr(X, y-mean(y), 15);

figure(1); clf
imagesc(sta)

%%

cc = cc + 1;
sta = simpleRevcorr(Xspat, Rspat(:,cc)-mean(Rspat(:,cc)), 20);

figure(1); clf
imagesc(reshape(sum(sta), spopts.dims))
%% get waveform stats across all experimental sessions
W = [];
exid = [];
for iEx = 1:55
    tic; 
    Exp = io.dataFactoryGratingSubspace(iEx);
    
    % trial starts and stops
    tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(:)));
    tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(:)));

    % inter-trial intervals
    epochs = [tstop(1:end-1) tstart(2:end)];
    
    % get waveform statistics over valid epochs
    W_ = io.get_waveform_stats(Exp.osp, 'validEpochs', epochs);
    
    % append session to struct-array of units
    exid = [exid; iEx*ones(numel(W_),1)];
    W = [W; W_];
    toc
end
%% get all waveforms and useful metrics

fprintf('%d total units\n', numel(W))

% waveforms shifted to peak or trough
wf = cell2mat(arrayfun(@(x) x.shiftwaveform(:,3)', W, 'uni', 0));

% ISI autocorrelation
isi = cell2mat(arrayfun(@(x) x.isi(:)', W, 'uni', 0));

p2t = arrayfun(@(x) x.peaktime - x.troughtime, W, 'uni', 1);

% unit quality measures
isiL = arrayfun(@(x) x.isiL, W); % variability around detrended isi
isiR = arrayfun(@(x) x.isiRate, W); % refractory rate / expected rate

localIdx = arrayfun(@(x) x.localityIdx, W); % wf amplitude ch-2 / ch center
BRI = arrayfun(@(x) x.BRI, W); % burstiness-refractoriness index

figure(1); clf
ix = localIdx < .5 & isiR<1 & p2t > 0;
fprintf('%d/%d units are putative SU\n', sum(ix), numel(W))

histogram(p2t(ix)*1e3, linspace(-1, 1, 80) );
xlabel('Peak - Trough (ms)')

figure(2); clf
wx = (wf(ix,:)); %./max(wf(ix,:),[],2));
imagesc( wx )

figure(3); clf
% normalized waveforms
wxn = (wx - min(wx,[],2))./(max(wx,[],2)- min(wx,[],2));

peaksorters = wxn(:,21) > 0;
troughsorters = wxn(:,21)==0;

p  = p2t(ix)*1e3; % peak - trough (miliseconds)
bri = BRI(ix);

a = wxn(peaksorters,:);

[~, ind] = sort(p(troughsorters));
w = wxn(troughsorters,:);
b = w(ind,:);
imagesc([a; b])
plot.fixfigure(gcf, 12, [4 6], 'offsetAxes', false)

figure(4); clf
% histogram(p2t(:)*1e3, 100, 'EdgeColor', 'none', 'FaceColor', .5*[1 1 1]); hold on
h = histogram(p(:), 50, 'EdgeColor', 'none', 'FaceColor', lines(1), 'Normalization', 'pdf'); hold on
xlabel('Peak - Trough (ms)')
plot.fixfigure(gcf, 12, [4 4])

n = 5;
AIC = zeros(n,1);
BIC = zeros(n,1);
for c = 1:n
    gmdist = fitgmdist(p,c);
    xx = linspace(min(xlim), max(xlim), 100);
    plot(xx, gmdist.pdf(xx'))
    AIC(c) = gmdist.AIC;
    BIC(c) = gmdist.BIC;
end

%%
figure(1); clf
plot(BIC); hold on
xlabel('# of mixing distributions')
ylabel('BIC')
[~, id] = min(BIC);
plot([1 1]*id, ylim, 'k--')
plot.fixfigure(gcf, 12, [4 4])

figure(4); clf
% histogram(p2t(:)*1e3, 100, 'EdgeColor', 'none', 'FaceColor', .5*[1 1 1]); hold on
h = histogram(p(:), 50, 'EdgeColor', 'none', 'FaceColor', lines(1), 'Normalization', 'pdf'); hold on
xlabel('Peak - Trough (ms)')
plot.fixfigure(gcf, 12, [4 4])

gmdist = fitgmdist(p,id);
xx = linspace(min(xlim), max(xlim), 100);
plot(xx, gmdist.pdf(xx'))

%%
clusts = cluster(gmdist, p);


figure(6); clf
clu = unique(clusts);
nclust = numel(clu);
cmap = lines(nclust);
cmap = [cmap .25*ones(nclust,1)];

for cc = 1:nclust
    ii = clusts==clu(cc);
    nts = size(wxn,2);
    plot(W(1).shiftlags*1e3,   wxn(ii, :)', 'Color', cmap(cc,:)); hold on
    text(nts, .5 + .3*cc/nclust, sprintf('n=%d', sum(ii)), 'Color', cmap(cc,:))
end
axis tight
xlabel('Milliseconds from trough')
ylabel('Amplitude (normalized)')
plot.fixfigure(gcf, 12, [4 4])


%%
figure(7); clf
ax = subplot(3,3,[1 4]);
histogram(bri, 100)
xlim(ax, [0 20])
view(-90,90)

ax = subplot(3,3,[2:3 5:6]);
for cc = 1:nclust
    ii = clusts==clu(cc);
    plot(p(ii), bri(ii), 'o', 'Color', cmap(cc,:), 'MarkerFaceColor', cmap(cc,1:3), 'MarkerSize', 5); hold on
end
% set(gca, 'yscale', 'log')
xlabel('Peak - Trough (ms)')
ylabel('BRI')
plot.fixfigure(gcf, 12, [4 4])
ylim(ax,[0 20])
xlim(ax,[0 .85])

ax = subplot(3,3,8:9);
histogram(p, 100)
xlim(ax, [0 .85])
% figure(2); clf
% histogram(

%%
ac = isi(ix,:);
ac = (ac - min(ac,[],2)) ./ (max(ac,[],2) - min(ac,[],2));

wxn(isnan(wxn)) = 0;

figure(1); clf
for cc = 1:nclust
    subplot(nclust, 2, (cc-1)*2 + 1, 'align')
    ii = clusts==clu(cc);
    wtmp = wxn(ii,:);
    
    ptmp = p(ii);
    [~, ind] = sort(ptmp);
    imagesc(wtmp(ind,:))
    title(sprintf('Waveforms Group %d', cc))
    
    subplot(nclust, 2, (cc-1)*2 + 2, 'align')
    atmp = ac(ii,:);
    lags = W(1).lags;
    atmp = atmp(:,lags>0.001);
    [~, id] = max(atmp, [], 2);
    [~, ind] = sort(id);
    imagesc(atmp(ind,:))
    title(sprintf('AutoCorrelation Group %d', cc))
end

%%
    
    

figure(5); clf

lags = W(1).lags;
[~, id] = max(ac(:,lags>0.001), [], 2);
[~, ind] = sort(id);
imagesc(ac(ind,lags>0.001))

figure(6); clf
[X, binsx, binsy] = histcounts2(p, bri, 0:.05:1.5, 0:.5:15);
h = imagesc(binsx,binsy, X'); colormap(flipud(viridis))
h.AlphaData = h.CData/max(h.CData(:));
axis xy
hold on
% f = plot(p, lx(id)', '.', 'Color', cmap(1,:));

xlabel('Peak - Trough')
ylabel('AC peak')
%%

figure(1); clf
C = corr(wxn') + corr(ac');


C = 1 - C;
imagesc(C)

%%


Z = linkage(C, 'weighted');
d = pdist([wxn isi(ix,:)]);
order = optimalleaforder(Z, d);

figure
imagesc(C(order,order))

figure
dendrogram(Z)

c = cluster(Z, 'maxclust', 10);


%%
figure(1); clf
cs = unique(c);
cmap = lines(numel(cs));
cmap = [cmap .25*ones(numel(cs),1)];
for i = 1:numel(cs)
    ix = c==cs(i);

    f = plot(wxn(ix, :)', 'Color', cmap(i,:)); hold on

end
%%

%%
% ac = imgaussfilt(ac, [.001 3]);
a = ac(peaksorters,:);
w = ac(troughsorters,:);
b = w(ind,:);
imagesc(([a; b]), [0 20])
plot.fixfigure(gcf, 12, [6 4])

%%
figure
plot(ac')
%%
figure(7); clf


ac = (ac - min(ac,[],2)) ./ (max(ac, [], 2) - min(ac, [], 2));
ii = p < 0 | p > 1;
n(1) = sum(ii);
nts = size(ac,2);
f = plot(ac(ii, :)', 'g'); hold on
for i = 1:numel(f)
    f(i).Color(4) = .5;
end
text(nts, .7, sprintf('n=%d', n(1)), 'Color', 'g')

ii = p > 0 & p < .35;
n(2) = sum(ii);
f = plot(ac(ii, :)', 'b'); hold on
for i = 1:numel(f)
    f(i).Color(4) = .5;
end
text(nts, .82, sprintf('n=%d', n(2)), 'Color', 'b')

ii = p > 0.35 & p < 1;
n(3) = sum(ii);
f = plot(ac(ii, :)', 'r');
for i = 1:numel(f)
    f(i).Color(4) = .5;
end
text(nts, .88, sprintf('n=%d', n(3)), 'Color', 'r')


plot.fixfigure(gcf, 12, [6 4])

%%
x = w;

Y = tsne(x, 'NumPCAComponents', 10, 'Algorithm','barneshut',...
    'Distance', 'minkowski', 'Standardize',true,'Perplexity',50);

X = Y;
d = pdist(X);
z = linkage(d, 'ward');
c = cluster(z, 'maxclust', 5);
figure(1); clf
gscatter(X(:,1), X(:,2), c)

%% clustering
wtmp = wxn;
itmp = isi(ix,:);
cinds = unique(c);
nc = numel(cinds);
cmap = lines(nc);
figure(2); clf
for ci = 1:nc
    subplot(1, 2, 1)
    x = wtmp(c==ci,:);
    m = nanmean(x);
    s = 2*nanstd(x)/sqrt(sum(c==ci));
    plot.errorbarFill(1:numel(m), m,s, 'k', 'EdgeColor', 'none', 'FaceAlpha', .25, 'FaceColor', cmap(ci,:)); hold on
    plot(m, 'Color', cmap(ci,:))
    subplot(1, 2, 2)
    plot(mean(itmp(c==ci,:))); hold on
end



%%
wi = reshape(wx, sum(ix), [], 5);
I = wi(:,:,3);
imagesc(I)
% 
% plot(std(I,[],2), '.')
% w = I(std(I,[],2)>4,:);
% imagesc(w)
% 
% figure(3); clf
% plot(w(:,20:80)')

%%
i = i + 1;
if i > numel(W)
    i = 1;
end
figure(1); clf
subplot(121)
plot(W(i).waveform)
title(exid(i))
subplot(122)
plot(isi(i,:))
title(i)
%%

cids = Exp.osp.cids;
pairs = nchoosek(cids, 2);

%%
ipair = ipair + 1;
if ipair > size(pairs,1)
    ipair = 1;
end

st1 = Exp.osp.st(Exp.osp.clu==cids(pairs(ipair,1)));
st2 = Exp.osp.st(Exp.osp.clu==cids(pairs(ipair,2)));

[xc, lags, err] = crossCorrelation(st1, st2);

figure(1); clf
plot.errorbarFill(lags, xc, err); hold on
plot(lags, xc, 'k')

%%

%%
figure(1); clf
isi = isi;
x = isi(:,1:100); x = x - mean(x,2);
s = std(x,[],2);
% ix = sum(isnan(wf),2) == 0 & uQ > 1 & isi(:,202)<0;% & isiV < .5 & s < 0.0015;

subplot(121)
imagesc(wf(ix,:))
subplot(122)
imagesc(isi(ix,:))

%%
% -mean(wf(ix,:),2)
x = wf(ix,:)-mean(wf(ix,:),2);
x = x ./ max(x,[],2);

% x = [isi(ix,:)];
Y = tsne(x, 'NumPCAComponents', 4, 'Algorithm','barneshut',...
    'Distance', 'minkowski', 'Standardize',true,'Perplexity',50);
% Z = tsne(isi(ix,:)-mean(isi(ix,:),2), 'NumPCAComponents',4, 'Algorithm','exact','Standardize',true,'Perplexity',50);


X = Y;
d = pdist(X);
z = linkage(d, 'ward');
c = cluster(z, 'maxclust', 5);
figure(1); clf
gscatter(X(:,1), X(:,2), c)

%%
wtmp = wf(ix,:);
itmp = isi(ix,:);
cinds = unique(c);
nc = numel(cinds);
cmap = lines(nc);
figure(2); clf
for ci = 1:nc
    subplot(1, 2, 1)
    x = wtmp(c==ci,:);
    m = mean(x);
    s = 2*std(x)/sqrt(sum(c==ci));
    plot.errorbarFill(1:numel(m), m,s, 'k', 'EdgeColor', 'none', 'FaceAlpha', .25, 'FaceColor', cmap(ci,:)); hold on
    plot(m, 'Color', cmap(ci,:))
    subplot(1, 2, 2)
    plot(mean(itmp(c==ci,:))); hold on
end

%%
figure(1); clf

for ci = 1:nc
    x = itmp(c==ci,:);
    subplot(1,nc,ci)
    imagesc(x)
    
end

%%
figure(1); clf
[~, ind] = sort(p2t(ix));
% [~, ind] = sort(itmp(:,199));
subplot(121)
imagesc(wtmp(ind,:))
subplot(122)
imagesc(itmp(ind,:))

%%
clf
plot(p2t(ix), mean(itmp(:,198:199),2), '.')