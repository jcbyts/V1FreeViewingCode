
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

addpath Analysis/202001_K99figs_01  
addpath Analysis/manuscript_freeviewingmethods/
%% load data saccade variance data

if exist('D.mat', 'file')==2
    disp('Loading')
    load('D.mat')
else
    D = {};
    for sessId = 1:57
        
        try
            [Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'jrclustwf');
        catch
            disp('loading kilow file instead')
            [Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'kilowf');
        end
        try
            D{sessId} = get_saccade_triggered_variance(Exp, 'stimulusSets', {'BackImage', 'Grating', 'ITI'}, 'plotit', false);
        end
    end
    
    save('D.mat', 'D')
end
%% load grating data
sesslist = io.dataFactoryGratingSubspace;
Sgt = [];
if exist('Sgt.mat', 'file')==2
    disp('Loading')
    load Sgt.mat
else
    
    for iEx = 1:numel(sesslist)
        try
            if isfield(D, sesslist{iEx})
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', D.(sesslist{iEx}).spike_sorting);
            else
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', 'kilowf');
            end
            
            evalc('rfdeets = grating_RF_single_session(Exp, ''plot'', false);');
            Sgt(iEx).rfs = rfdeets;
            
        catch
            disp('ERROR ERROR')
        end
    end
    
    save('Sgt.mat', '-v7.3', 'Sgt')
end

%%

figure(1); clf
imagesc(Sgt(1).rfs(1).sta)


%%


subplot(1,2,1)
imagesc( (d > 5*mad))



imagesc(d/mad); colorbar





%% loop over sessions, extract unit values
thresh = 5; % in MADs

mbi = [];
vbi = [];

miti = [];
viti = [];

mgr = [];
vgr = [];

cid = [];
cg = [];

rfMad =[];
sfPref = [];
oriPref = [];
r2fit = [];

for sessId = 1:57
    if isempty(D{sessId})
        continue
    end
    
    NC = numel(D{sessId}.BackImage);
    if NC ~= numel(Sgt(sessId).rfs)
        continue
    end
    
    for cc = 1:NC
        mbi = [mbi; D{sessId}.BackImage(cc).mean];
        vbi = [vbi; D{sessId}.BackImage(cc).var];
        miti = [miti; D{sessId}.ITI(cc).mean];
        viti = [viti; D{sessId}.ITI(cc).var];
        mgr = [mgr; D{sessId}.Grating(cc).mean];
        vgr = [vgr; D{sessId}.Grating(cc).var];
        cg = [cg; D{sessId}.Grating(cc).cg];
        
        % grating
        if isempty(Sgt(sessId).rfs)
            sfPref = [sfPref; nan];
            oriPref = [oriPref; nan];
            r2fit = [r2fit; nan];
            rfMad = [rfMad; nan];
        else
            
            sfPref = [sfPref; Sgt(sessId).rfs(cc).rffit.sfPref];
            oriPref = [oriPref; Sgt(sessId).rfs(cc).rffit.oriPref];
            r2fit = [r2fit; Sgt(sessId).rfs(cc).rffit.r2];
            
            I = Sgt(sessId).rfs(cc).sta;
            d = abs(I - median(I(:))); % deviations
            mad = median(d(:)); % median absolute deviation
            rfMad = [rfMad; mean(d(:) > thresh*mad)];
        end
        
    end
end


lags = D{1}.BackImage(1).lags;
ffbi = vbi./mbi;
ffiti = viti./miti;
ffgr = vgr./mgr;



%% estimate fano factor with line fits

flin = @(b, x) b(1) + b(2)*x;
fquad = @(b, x) b(1) + b(2)*x + b(3)*x.^2;

NC = size(mgr,1);
blin = zeros(NC,2);
bquad = zeros(NC,3);
blin0 = zeros(NC,2);
m0 = zeros(NC,1);
peakloc = zeros(NC,1);
peak = zeros(NC,1);

for cc = 1:NC
    if mod(cc, 10)==0
        fprintf('%d/%d\n', cc, NC)
    end
    m = [mbi(cc,:)];% mgr(cc,:)]; 
    v = [vbi(cc,:)];% vgr(cc,:)];
    
    m0(cc) = norm(m);
    
    evalc('bl0 = lsqcurvefit(flin, [0 1], miti(cc,:), viti(cc,:),[0 0], [0 10]);');
    evalc('bl = lsqcurvefit(flin, [0 bl0(2)], m, v, [0 bl0(2)], [10 bl0(2)]);');
    evalc('bq = lsqcurvefit(fquad, [bl(1:2) 0], m, v, [bl(1) bl(2) -10], [bl(1) bl(2) 10]);');
    
    blin(cc,:) = bl;
    bquad(cc,:) = bq;
    blin0(cc,:) = bl0; 
    
    [peakloc(cc), peak(cc)] = get_peak_and_trough(lags, mbi(cc,:), [-0.01 .25]);
    
end
 
%%
ix = rfMad > 0.02; %cg==2;

b1 = blin(:,1); %./mean(mbi,2);
b1(b1 < 0.001) = 0;

figure(1), clf
subplot(1,3,1)
plot(blin0(:,2), blin(:,2)+b1, '.'); hold on
plot(blin0(ix,2), blin(ix,2)+b1(ix), '.'); hold on
plot(xlim, xlim, 'k')

subplot(1,3,2)
plot(b1, '.'); hold on
plot(blin0(:,1), '.')

subplot(1,3,3)
histogram(bquad(:,3), 100); hold on

figure(3); clf
subplot(1,2,1)
histogram(rfMad)
hold on
histogram(rfMad(ix))
sum(rfMad(ix) > 0.01) 
% plot(blin(:,2)./blin0(:,2),'.')

% histogram(blin(:,1) - blin0(:,1))

subplot(1,2,2)
histogram((sfPref(ix)))

%%



%%
pval = zeros(NC,1);
zval = zeros(NC,1);
for cc = 1:NC
   d = vbi(cc,:) - flin(blin0(cc), mbi(cc,:));
   [pval(cc), h, stats] = signrank(d);
   zval(cc) = stats.zval;
end

%%
idx = (cg == 2);
clf
histogram(zval(idx), 100)

%%
figure(1); clf
plot(blin0(idx), blin(idx), '.');

hold on
plot(xlim, xlim, 'k')
figure(2); clf
% plot(bquad(:,2), 1./max(mbi, [], 2), '.')
% histogram(bquad(:,2))
plot(blin(idx)./blin0(idx), bquad(idx), '.')


%%
figure(1); clf
idx = find(cg == 2);
c = c + 1;
if c > numel(idx)
    c = 1;
end
cc = idx(c);

subplot(211,'align')
plot(lags, mbi(cc,:)); hold on
plot(lags, miti(cc,:));
plot(lags, mgr(cc,:));
xlim([-.05 .25])
xlabel('Lag from fix onset')
ylabel('Firing Rate')
legend({'NI','ITI', 'GR'})
subplot(2,1,2, 'align')
plot(lags, ffbi(cc,:)); hold on
plot(lags, ffiti(cc,:))
plot(lags, ffgr(cc,:))
xlim([-.05 .25])
xlabel('Lag from fix onset')
ylabel('Fano Factor')

figure(2); clf
set(gcf, 'color', 'w')
m = [mbi(cc,:) mgr(cc,:)];
v = [vbi(cc,:) vgr(cc,:)];

plot(mbi(cc,:), vbi(cc,:), '.'); hold on
plot(miti(cc,:), viti(cc,:), '.')
plot(mgr(cc,:), vgr(cc,:), '.')
plot(xlim, xlim, 'k')

xax = linspace(min(xlim), max(xlim), 100);
cmap = lines;
plot(xax, flin(blin(cc,:), xax), 'Color', cmap(1,:))
plot(xax, fquad(bquad(cc,:), xax), 'Color', cmap(1,:))
plot(xax, flin(blin0(cc,:), xax), 'Color', cmap(2,:))
title([blin(cc,:) blin0(cc,2)])
legend({'NI', 'ITI', 'GR'})
xlabel('mean')
ylabel('variance')

% figure(3); clf
% histogram(vbi(cc,:) - flin(blin0(cc), mbi(cc,:)))

%%
ix = (cg == 2) & zval < 0;

figure; clf
subplot(211,'align')
errorbar(lags, mean(mbi(ix,:)),nanstd(mbi(ix,:))/sqrt(sum(ix))); hold on
errorbar(lags, mean(miti(ix,:)),nanstd(miti(ix,:))/sqrt(sum(ix)));
errorbar(lags, mean(mgr(ix,:)),nanstd(mgr(ix,:))/sqrt(sum(ix)));
xlim([-.05 .25])
xlabel('Lag from fix onset')
ylabel('Firing Rate')
legend({'NI','ITI', 'GR'})

subplot(2,1,2, 'align')
errorbar(lags, nanmean(ffbi(ix,:)), nanstd(ffbi(ix,:))/sqrt(sum(ix))); hold on
errorbar(lags, nanmean(ffiti(ix,:)), nanstd(ffiti(ix,:))/sqrt(sum(ix)));
errorbar(lags, nanmean(ffgr(ix,:)), nanstd(ffgr(ix,:))/sqrt(sum(ix)));
xlim([-.05 .25])
xlabel('Lag from fix onset')
ylabel('Fano Factor')


%%
ix = cg==2;
[mx, mxid] = max(mbi(ix,:),[],2);
I = mbi(ix,:)./mx;

[mxsort, ind] = sort(mxid, 'ascend');
iix = mxsort > 120 & mxsort < 200;
clf
imagesc(I(ind(iix),:))
set(gcf, 'DefaultAxesColorOrder', hsv(sum(iix)))
plot(I(ind(iix),:)')

%% get response variance aligned to peak
peakloc2 = median(peakloc)*ones(numel(peakloc), 1);

lagsnew = (-100:100)/1e3;
wfnew = zeros(NC, numel(lagsnew));
for cc = 1:NC
    if peakloc(cc) < .1 && peakloc(cc) > 0.02
        peakloc2(cc) = peakloc(cc);
    end
    wfnew(cc,:) = interp1(lags, vbi(cc,:),  lagsnew + peakloc2(cc));
end
%%

figure(1); clf
imagesc(wfnew./m0)

%%

ix = cg>=0;
sum(ix)
% peakloc

% X = wfnew;
% ix = true(1, numel(lagsnew));

X = vbi(ix,:)./mbi(ix,:);
ix = lags > -.05 & lags < .25;


X(any(isnan(X),2),:) = [];


n = zeros(size(X,1), 1);
for i = 1:size(X,1)
    n(i) = norm(X(i,ix));
end

figure(1); clf
xtmp = X(:,ix)./n;


% xtmp = xtmp - mean(xtmp,2);


C = cov(xtmp);

[u,s,v] = svd(C);
figure(3); clf
imagesc(C)


figure(2); clf
subplot(1,2,1)
plot(lags(ix), v(:,1:2))
subplot(1,2,2)
plot(diag(s), '.')

figure(1); clf
plot(xtmp(ix,:)*v(:,1), xtmp(ix,:)*v(:,2), 'o')
%%
figure(1); clf
ix = cg==2;
lagix = lags > -.05 & lags < .25;

xtmp = ffbi(ix,:);
xtmp = xtmp(:,lagix);
plot(xtmp*v(:,1), xtmp*v(:,2), 'o'); hold on

xtmp = ffiti(ix,:);
xtmp = xtmp(:,lagix);
plot(xtmp*v(:,1), xtmp*v(:,2), 'o');

xtmp = ffgr(ix,:);
xtmp = xtmp(:,lagix);
plot(xtmp*v(:,1), xtmp*v(:,2), 'o');

%%
figure(1); clf
histogram(xtmp*v(:,1) - xtmp*v(:,2))

%%
% cc=1;
% cc = 18;
% cc = 42;
cc = mod(cc + 1, NC); cc = max(cc, 1);
figure(1); clf
% subplot(4,1,1:3)
rast = squeeze(spks(ind,cc,:));
[i, j] = find(rast);
plot.raster(lagsrast(j), i, 20); hold on
axis off
set(gcf, 'PaperSize', [4 5], 'PaperPosition', [0 0 4 5])
saveas(gcf, fullfile('Figures', 'K99', sprintf('raster%d.png', cc)))
axis on
plot([0 0], ylim, 'r')
plot(fixdur(ind), 1:numel(ind), 'g')
xlim(win)
ylim([500 numel(ind)])
title(sprintf('Unit: %d', cc))
ylabel('Fixation #')
plot.fixfigure(gcf, 10, [4 5])
% saveas(gcf, fullfile('Figures', 'K99', sprintf('raster%d.pdf', cc)))

% get saccade size
sstart = Exp.slist( validSaccadesNI,4);
sstop = Exp.slist( validSaccadesNI,5);
dx = Exp.vpx.smo(sstop,2)-Exp.vpx.smo(sstart,2);
dy = Exp.vpx.smo(sstop,3)-Exp.vpx.smo(sstart,3);
sacSize = hypot(dx, dy);

% sort by saccade Size
sacSizes = [1 2 4 6 8 10];
ns = numel(sacSizes)-1;
figure(2); clf
cmap = jet(ns);
for i = 1:ns
    ix = sacSize > sacSizes(i) & sacSize < sacSizes(i+1) & fixdurNI > .1;
    disp(sum(ix))
    m = mean(squeeze(spks(ix,cc,:)));
    m = imgaussfilt(m, 2.5);
    h(i) = plot(lagsrast, m/binsize, 'Color', cmap(i,:)); hold on
end
xlim([-.05 .15])
xlabel('Time from saccade offset')
ylabel('Firing Rate')
title(cc)
legend(h, arrayfun(@(x) sprintf('%02.2f', x), sacSizes(1:ns), 'uni', 0), 'Location', 'Best')

% sort by saccade direction
sacTheta = wrapTo360(cart2pol(dx,dy)/pi*180);
sacThetas = [-45:45:(360+45)];
ns = numel(sacThetas)-1;
figure(3); clf
cmap = jet(ns);
for i = 1:ns
    ix = sacTheta > sacThetas(i) & sacTheta < sacThetas(i+1) & fixdurNI > .1;
%     disp(sum(ix))
    m = mean(squeeze(spks(ix,cc,:)));
    m = imgaussfilt(m, 2.5);
    h(i) = plot(lagsrast, m/binsize, 'Color', cmap(i,:)); hold on
end
xlim([-.05 .15])
xlabel('Time from saccade offset')
ylabel('Firing Rate')
title(cc)
legend(h, arrayfun(@(x) sprintf('%02.2f', x), sacThetas(1:ns), 'uni', 0), 'Location', 'Best')
%%

figure(2); clf
thresh = 0.2;
binsize = 1e-3;

iix = fixdurNI > thresh;
sp = squeeze(spksNI(iix,cc,:))/binsize;
sp = filter(ones(10,1)/10, 1, sp')';
mFR = mean(sp);
sFR = std(sp)/sqrt(size(sp,1));

cmap = lines;
plot.errorbarFill(lagsrast, mFR, sFR, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .5); hold on

iix = fixdurG > thresh;
sp = squeeze(spksG(iix,cc,:))/binsize;
sp = filter(ones(10,1)/10, 1, sp')';
mFR = mean(sp);
sFR = std(sp)/sqrt(size(sp,1));

cmap = lines;
plot.errorbarFill(lagsrast, mFR, sFR, 'k', 'FaceColor', cmap(4,:), 'EdgeColor', cmap(4,:), 'FaceAlpha', .5)

xlabel('Time from fixation onset')
ylabel('Firing rate')
legend('BackImage', 'Gabor Noise')
plot.fixfigure(gcf, 10, [4 2])
saveas(gcf, fullfile('Figures', 'K99', sprintf('sacpsth%d.pdf', cc)))

%%
figure(2); clf
fd = fixdurNI(fixdurNI>0.05);
mean(fd>0.1 & fd < 0.2)
histogram(fd, 0:.01:.5, 'FaceColor', .5*[1 1 1], 'EdgeColor', .5*[1 1 1])
xlim(win)
ylim([0 200])
plot.fixfigure(gcf, 10, [4 2])
saveas(gcf, fullfile('Figures', 'K99', 'fixduration.pdf'))
%% plot response variability early, late
iix = fixdurNI > thresh;

lagix = lagsrast > 0.04 & lagsrast < 0.1;
w = hanning(sum(lagix));
% w = w./sum(w);
spE = squeeze(spksNI(iix,cc,lagix))*w;

lagix = lagsrast > 0.1 & lagsrast < 0.2;
w = hanning(sum(lagix));
% w = w./sum(w);
spL = squeeze(spksNI(iix,cc,lagix))*w;

figure(1); clf
histogram(spE-spL)
% plot(spE, spL, '.'); hold on
% plot(xlim, xlim, 'k')

%%
clf
C = hist3([spE(:,cc), spL(:,cc)]/.1, {0:1:50, 0:1:50});
imagesc(C)
axis xy
% plot(xlim, xlim, 'k')




%%
cc = mod(cc + 1, NC); cc = max(cc, 1);


figure(2); clf
% backimage
iix = fixdurNI > thresh;
mFR = squeeze(mean(spksNI(iix,:,:)))'/binsize;
mFR = filter(ones(10,1)/10, 1, mFR);
tot = mean(mFR);
% plot(lagsrast, mean(mFR(:,cc)./tot,2)); hold on
plot(lagsrast, mFR(:,cc)); hold on

% gabor
iix = fixdurG > thresh;
mFR = squeeze(mean(spksG(iix,:,:)))'/binsize;
mFR = filter(ones(10,1)/10, 1, mFR);
% plot(lagsrast, mean(mFR./tot,2)); hold on
plot(lagsrast, mFR(:,cc)); hold on
% 
% grating
iix = fixdurGr > thresh;
mFR = squeeze(mean(spksGr(iix,:,:)))'/binsize;
mFR = filter(ones(10,1)/10, 1, mFR);
% plot(lagsrast, mean(mFR./tot,2)); hold on
plot(lagsrast, mFR(:,cc)); hold on
% % grating
% iix = fixdurF > thresh;
% mFR = squeeze(mean(spksF(iix,:,:)))'/binsize;
% mFR = filter(ones(10,1)/10, 1, mFR);
% plot(lagsrast, mFR(:,cc)); hold on
title(cc)
%%
k = mod(k + 1, NC); k = max(k, 1);
figure(2); clf
set(gcf, 'Color', 'w')
% plot(lags, squeeze(var(spks(fixdur > .25,k,:)))'/binsize); hold on
thresh = .25;
mFR = squeeze(mean(spks(fixdur > thresh,:,:)))'/binsize;
mFR = filter(ones(5,1)/5, 1, mFR);

plot(lags, mFR); hold on
plot(thresh*[1 1], ylim, 'k--')
xlabel('Time from fixation onset')
ylabel('Spike Rate')

% plot(squeeze(mean(spks(fixdur < .2,:,:),1))');
figure(3); clf
set(gcf, 'Color', 'w')
% plot(lags, squeeze(var(spks(fixdur > .25,k,:)))'/binsize); hold on
thresh = .25;
mFR = squeeze(mean(spks(fixdur > thresh,:,:)))'/binsize;
mFR = filter(ones(5,1)/5, 1, mFR);
% mTot = full(mean(Y(:,S.cids)))/binsize;
mTot = mean(mFR(lags < .250 & lags > .1,:));
mTot(mTot < 1) = nan;
plot(lags, mFR./mTot); hold on
plot(thresh*[1 1], ylim, 'k--')
plot(xlim, [1 1], 'k--')
xlabel('Time from fixation onset')
ylabel('Relative Rate')

figure(1); clf
plot(mean(mFR(lags > .15,:)), mean(mFR(lags < .1,:)), '.'); hold on
plot(xlim, xlim, 'k')

%%

labels = Exp.vpx.Labels;
good = labels == 1 | labels ==2;
[bw, num] = bwlabel(good);
ar = nan(num, 1);
for i = 1:num
    ar(i) = sum(bw==i);
end

%%
   
[~, ind] = sort(ar, 'descend');

longest = bw == ind(1);


t = Exp.vpx.smo(:,1);
x = sgolayfilt(Exp.vpx.smo(:,2), 2, 7);
y = Exp.vpx.smo(:,3);

figure(1); clf
plot(t(longest), x(longest), 'k')

