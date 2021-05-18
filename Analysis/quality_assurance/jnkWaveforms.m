
%% get waveform stats across all experimental sessions
W = [];
exid = [];
cgs = [];
sessions = {};
csdStim = {};
csdSac = {};

csdExclusionList = {'ellie_20190107', ...
    'ellie_20190111', ...
    'logan_20191119', ...
    'logan_20191121', ...
    'logan_20191209', ...
    'logan_20200228', ...
    'logan_20200229', ...
    'logan_20200302', ...
    'milo_20190607', ...
    'milo_20190621'};
%%
for iEx = 56
    tic; 
   
    [Exp, ~, lfp] = io.dataFactoryGratingSubspace(iEx, 'spike_sorting', 'jrclustwf');
    
    session = strrep(Exp.FileTag, '.mat', '');
    sessions{iEx} = session;

    % saccade-triggered CSD
    excludeCSD = any(strcmp(csdExclusionList, Exp.FileTag(1:end-4)));
    
    if excludeCSD
        et = [];
    else
        et = Exp.vpx2ephys(Exp.slist(:,1));
    end
    cstruct = csd.getCSD(lfp, et, 'spatsmooth', 2.5, 'method', 'standard', 'plot', false, 'debug', false);
    
    cstruct.session = session;
    csdSac{iEx} = cstruct;
    
    % flash-triggered CSD
    if excludeCSD
        et = [];
    else
        et = csd.getCSDEventTimes(Exp);
    end
    cstruct = csd.getCSD(lfp, et, 'spatsmooth', 2.5, 'method', 'standard', 'plot', false);
    cstruct.session = session;
    csdStim{iEx} = cstruct;

    % trial starts and stops
    tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(:)));
    tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(:)));

    % inter-trial intervals
    epochs = [tstop(1:end-1) tstart(2:end)];
    
    % get waveform statistics over valid epochs
    W_ = io.get_waveform_stats(Exp.osp, 'validEpochs', epochs, 'binSize', .25e-3, 'debug', false);
    for cc = 1:numel(W_)
        W_(cc).session = session;
%         W_(cc).depth = W_(cc).depth - csdReversal;
    end
    
    % append session to struct-array of units
    exid = [exid; iEx*ones(numel(W_),1)];
    cgs = [cgs; Exp.osp.cgs(:)];
    W = [W; W_];
    toc
    drawnow
end

% BRI = [1-3/3-6]
% BRI2 = [ ((1-3) - (3-6)) / ((1-3) + (3-6)]
%%  plot all CSDs

C = csdStim;
figure; clf
field = 'csd';
ix = ~cellfun(@isempty, csdStim);
Nsess = numel(csdSac);
sx = ceil(sqrt(Nsess));
sy = round(sqrt(Nsess));
ax = plot.tight_subplot(sx,sy,0.01, 0.01);
for iEx = 1:Nsess
    try
    set(gcf, 'currentaxes', ax(iEx));
    if strcmp(field, 'csd')
        if numel(size(C{iEx}.CSD))==3
            imagesc(C{iEx}.time, C{iEx}.chDepths(2:31), C{iEx}.CSD(:,:,1))
        else
            imagesc(C{iEx}.time, C{iEx}.chDepths(2:31), C{iEx}.CSD)
        end
    else
        imagesc(C{iEx}.time, C{iEx}.chDepths, C{iEx}.STA)
    end
    hold on
    plot([0 0], ylim, 'k')
    t = C{iEx}.latency;
    plot(t*[1 1], ylim, 'r')
    plot(xlim, C{iEx}.reversalPointDepth{1}*[1 1], '-r')
    xlim([20 80])
    end
end

%%
ix = ~cellfun(@isempty, csdStim);
figure(1); clf
rsac = cellfun(@(x) x.reversalPointDepth{1}, csdSac(ix));
rst = cellfun(@(x) x.reversalPointDepth{1}, csdStim(ix));

plot(rsac, rst, '.'); hold on
plot(xlim, xlim, 'k')

%%
monkix = arrayfun(@(x) strcmp(x.session(1), 'e'), W);
p2t = arrayfun(@(x) x.peaktime - x.troughtime, W)*1e3;
prt = arrayfun(@(x) abs(x.peakval./x.troughval), W);
ps = arrayfun(@(x) x.PeakSlope, W);
figure(1); clf
subplot(121)
% plot(p2t, prt, '.'); hold on
plot(p2t(monkix), prt(monkix), '.')
subplot(122)
plot(p2t, ps, '.')

figure(2); clf
monkix = arrayfun(@(x) strcmp(x.session(1), 'e'), W);
histogram(p2t(monkix), 200); hold on
monkix = arrayfun(@(x) strcmp(x.session(1), 'l'), W);
histogram(p2t(monkix), 200); hold on



w1 = arrayfun(@(x) x.ExtremityCiRatio(1), W);
w2 = arrayfun(@(x) x.ExtremityCiRatio(2), W);

figure(3); clf
plot(w1, w2, '.'); hold on
ix = w1 > 1 & w2 > 2;
% ix = ~monkix;
plot(w1(ix), w2(ix), '.')
xlim([0 20])
ylim([0 20])
% set(gca, 'xscale', 'log', 'yscale', 'log')
%% get all waveforms and useful metrics

fprintf('%d total units\n', numel(W))

% waveforms shifted to peak or trough
wf = cell2mat(arrayfun(@(x) x.shiftwaveform(:,3)', W, 'uni', 0));
% wf = cell2mat(arrayfun(@(x) x.shiftwaveform(:)', W, 'uni', 0));
% wf = cell2mat(arrayfun(@(x) x.ctrChWaveform(:)', W, 'uni', 0));

% ISI autocorrelation
isi = cell2mat(arrayfun(@(x) x.isi(:)', W, 'uni', 0));

p2t = arrayfun(@(x) x.peaktime - x.troughtime, W, 'uni', 1);
p2v = arrayfun(@(x) x.peakval./x.troughval, W, 'uni', 1);

% unit quality measures
isiL = arrayfun(@(x) x.isiL, W); % variability around detrended isi
isiR = arrayfun(@(x) x.isiRate, W); % refractory rate / expected rate

d = arrayfun(@(x) x.depth, W);

localIdx = arrayfun(@(x) x.localityIdx, W); % wf amplitude ch-2 / ch center
BRI = arrayfun(@(x) x.BRI, W); % burstiness-refractoriness index

figure(1); clf
monkix = arrayfun(@(x) strcmp(x.session(1), 'e'), W);
% monkix = true(numel(W),1);
w1 = arrayfun(@(x) x.ExtremityCiRatio(1), W);
w2 = arrayfun(@(x) x.ExtremityCiRatio(2), W);
ix = w1 > 1 & w2 > 2 & monkix;
ix = cgs == 2;


% ix = localIdx < .5 & isiR<=1 & p2t > 0 & p2v > 50;% & p2v > 10 & p2t < .5/1e3;
fprintf('%d/%d units are putative SU\n', sum(ix), numel(W))

histogram(p2t(ix)*1e3, linspace(-1, 1, 80) );
xlabel('Peak - Trough (ms)')

wx = (wf(ix,:)); %./max(wf(ix,:),[],2));


figure(3); clf
% normalized waveforms
wxn = (wx - min(wx,[],2))./(max(wx,[],2)- min(wx,[],2));
% wxn = wx ./ abs(min(wx, [], 2));

p  = p2t(ix)*1e3; % peak - trough (miliseconds)
bri = BRI(ix);
depth = d(ix);

[~, pind] = sort(p);

a = wxn(pind,:);

imagesc(a)
plot.fixfigure(gcf, 12, [4 6], 'offsetAxes', false)

figure(2); clf

imagesc( wx(pind,:) )

figure(4); clf
% histogram(p2t(:)*1e3, 100, 'EdgeColor', 'none', 'FaceColor', .5*[1 1 1]); hold on
h = histogram(p(:), 150, 'EdgeColor', 'none', 'FaceColor', lines(1), 'Normalization', 'pdf'); hold on
xlabel('Peak - Trough (ms)')
plot.fixfigure(gcf, 12, [4 4])

n = 3;
AIC = zeros(n,1);
BIC = zeros(n,1);
for c = 1:n
    gmdist = fitgmdist(p,c);
    xx = linspace(min(xlim), max(xlim), 1000);
    plot(xx, gmdist.pdf(xx'))
    AIC(c) = gmdist.AIC;
    BIC(c) = gmdist.BIC;
end

[~, id] = min(BIC);

xax = linspace(min(xlim), max(xlim), 100);
yax = linspace(min(ylim), max(ylim), 100);
[xx,yy] = meshgrid(xax, yax);
contourf



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
h = histogram(p(:), 150, 'EdgeColor', 'none', 'FaceColor', lines(1), 'Normalization', 'pdf'); hold on
xlabel('Peak - Trough (ms)')
plot.fixfigure(gcf, 12, [4 4])

gmdist = fitgmdist(p,3);
xx = linspace(min(xlim), max(xlim), 1000);
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
figure(1); clf
plot(p, -depth, '.')
%%
figure(7); clf
ax = subplot(3,3,[1 4]);
histogram(ax, bri, 1000)
set(ax, 'xscale', 'log')
xlim(ax, [0 20])
view(-90,90)

ax = subplot(3,3,[2:3 5:6]);
for cc = 1:nclust
    ii = clusts==clu(cc);
    plot(p(ii), bri(ii), 'o', 'Color', cmap(cc,:), 'MarkerFaceColor', cmap(cc,1:3), 'MarkerSize', 5); hold on
end
set(gca, 'yscale', 'log')
xlabel('Peak - Trough (ms)')
ylabel('BRI')
plot.fixfigure(gcf, 12, [4 4])
% ylim(ax,[0 20])
% xlim(ax,[0 .85])

ax = subplot(3,3,8:9);
histogram(p, 100)
xlim(ax, [0 .85])
% figure(2); clf
% histogram(

figure(2); clf
for cc = 1:nclust
    subplot(nclust, 1, cc, 'align')
    ii = clusts==clu(cc);
    histogram(log(bri(ii)), 'binEdges', -3:.1:3, 'FaceColor', cmap(cc,1:3)); hold on
end
xlabel('log BRI')
plot.fixfigure(gcf, 12, [4 4])
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
[X, binsx, binsy] = histcounts2(p, bri, 0:.025:.5, 0:.15:15);
h = imagesc(binsx,binsy, X'); colormap(flipud(viridis))
h.AlphaData = h.CData/max(h.CData(:));
axis xy
hold on
% f = plot(p, lx(id)', '.', 'Color', cmap(1,:));

xlabel('Peak - Trough')
ylabel('AC peak')
%%

figure(1); clf
C = nancov(wxn') + nancov(ac');


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