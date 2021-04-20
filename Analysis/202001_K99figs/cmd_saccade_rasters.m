
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

%% load data
sessId = 45;
[Exp, S] = io.dataFactory(sessId, 'spike_sorting', 'jrclustwf');

stat = fixrate_by_stim(Exp, 'stimSets', {'Grating', 'BackImage'}, 'plot', true);
%% get spike rate in every fixation
alignto = 'fixon';
binsize = 1e-3;
win = [-.1 .5];
lags = win(1):binsize:win(2);
numlags = numel(lags);

cids = Exp.osp.cids;
NC = numel(cids);

% fixation times
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));

bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];

numFixations = numel(fixon);

disp('Binning Spikes')
st = Exp.osp.st;
clu = Exp.osp.clu;
keep = ismember(clu, cids);
st = st(keep);
clu = double(clu(keep));

bs = (st==0) + ceil(st/binsize);
spbn = sparse(bs, clu, ones(numel(bs), 1));
spbn = spbn(:,cids);
blags = ceil(lags/binsize);
switch alignto
    case 'fixon'
        balign = ceil(fixon/binsize);
    case 'sacon'
        balign = ceil(sacon/binsize);
end

% Do the binning here
spks = zeros(numFixations, NC, numlags);
for i = 1:numlags
    spks(:,:,i) = spbn(balign + blags(i),:);
end

disp('Done')

% Get trial number for each fixation
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(:)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(:)));

[valid, trialNum] = getTimeIdx(fixon, tstart, tstop);


%%

%%
stimSets = {'Grating', 'BackImage'};
S = struct();
figure(1); clf
fixdur = sacon - fixon;
for iStim = 1:2
    
    stimSet = stimSets{iStim};
    validTrials = io.getValidTrials(Exp, stimSet);
    
    
    
    ix = ismember(trialNum, validTrials);
    ix = ix & fixdur > .2;
    
    S.(stimSet).numFix = sum(ix);
    
    sp = spks(ix,:,:);
    
    mu = squeeze(mean(sp,1));
    S.(stimSet).peakloc = nan(NC,1);
    S.(stimSet).peak = nan(NC,1);
    S.(stimSet).troughloc = nan(NC,1);
    S.(stimSet).trough = nan(NC,1);
    
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    figure(1);
    for cc = 1:NC
        subplot(sx,sy,cc)
        mu(cc,:) = imgaussfilt(mu(cc,:), 5)/binsize;
        
        
        m = mu(cc,:);
        m = m ./ median(m) - 1;
        
        h = plot(lags, m); hold on
        
        [S.(stimSet).peakloc(cc), S.(stimSet).peak(cc), S.(stimSet).troughloc(cc), S.(stimSet).trough(cc)] = get_peak_and_trough(lags, m, [-inf inf]);
        
        plot(S.(stimSet).peakloc(cc), S.(stimSet).peak(cc), 'o', 'Color', h.Color, 'MarkerFaceColor', h.Color);
        plot(S.(stimSet).troughloc(cc), S.(stimSet).trough(cc), 'v', 'Color', h.Color, 'MarkerFaceColor', h.Color);
        
    end
end

%%
figure(2); clf
fields = fieldnames(S);
cmap = lines;
for iStim = 1:numel(fields)
    field = fields{iStim};
    plot(S.(field).peakloc(:), S.(field).peak(:), 'o', 'Color', cmap(iStim,:), 'MarkerFaceColor', cmap(iStim,:)); hold on
    plot(S.(field).troughloc(:), S.(field).trough(:), 'v', 'Color', cmap(iStim,:), 'MarkerFaceColor', cmap(iStim,:));
end
    


%% FORAGE STATIC LINES DOWN HERE
% Try to linearly decode the noise pattern
stimSet = 'ForageStaticLines';
validTrials = io.getValidTrials(Exp, stimSet);
fixdur = sacon - fixon;

noisePat = cellfun(@(x) unique(x.PR.NoiseHistory(:,2)), Exp.D(validTrials));

R = [];
X = [];
inds = zeros(numFixations,1);
for i = 1:2
    ix = ismember(trialNum, validTrials(noisePat==i));
    ix = ix & fixdur > .2;
    inds(ix)=i;
    sp = spks(ix,:,:);
%     R_ = var(sp(:,:,lags>0 & lags<0.24),[],3);
    R_ = sum(sp(:,:,lags>0 & lags<0.24),3);
    R = [R; R_];
    X = [X; i*ones(sum(ix),1)];
end

figure(1); clf
B = glmfit(R,X==1, 'binomial');

figure(1); clf
histogram(R(X==1,:)*B(2:end), 50); hold on
histogram(R(X==2,:)*B(2:end), 50);


nfix = sum(inds~=0);



%%

xproj = reshape(permute(spks, [1 3 2]), [], NC)*B(2:end);
xproj = reshape(xproj, numFixations, numlags);
xproj = imboxfilt(xproj, [1 51], 'Padding', 'symmetric');
figure(1); clf
subplot(2,1,1)
plot(lags, (xproj(inds==1,:))', 'r'); hold on
plot(lags, (xproj(inds==2,:))', 'b')
subplot(2,1,2)

errorbar(lags, mean(xproj(inds==1,:)), std(xproj(inds==1,:)), 'r'); hold on
errorbar(lags, mean(xproj(inds==2,:)), std(xproj(inds==2,:)), 'b')

%%

%%
figure(2); clf
for cc = 1:NC
    subplot(sx,sy,cc)
    
    m = squeeze(var(spks(inds==1,cc,:)));
    m = imgaussfilt(m, 5)/binsize;
    plot(lags, m); hold on
    m = squeeze(var(spks(inds==2,cc,:)));
    m = imgaussfilt(m, 5)/binsize;
    plot(lags, m); hold on
end
% rproj = size(
%%
plot(mean(R(X==1,:))); hold on
plot(mean(R(X==2,:))); hold on
%% plot raster
cc = cc + 1;
if cc > NC
    cc = 1;
end
% cc = 17;
figure(1); clf
sp = squeeze(spks(inds==1,cc,:));
[i,j] = find(sp);
plot.raster(lags(j),i, 5, 'b'); hold on
off = max(i);
sp = squeeze(spks(inds==2,cc,:));
[i,j] = find(sp);
plot.raster(lags(j),i+off, 5, 'r');

title(cc)
%%
%%

% iTrial = iTrial + 1;
% if iTrial > numel(validTrials)
%     iTrial = 1;
% end

for iTrial = 1:numel(validTrials)
    thisTrial = validTrials(iTrial);

tstart = Exp.ptb2Ephys(Exp.D{thisTrial}.STARTCLOCKTIME);
tend = Exp.ptb2Ephys(Exp.D{thisTrial}.ENDCLOCKTIME);

tt = Exp.vpx2ephys(Exp.vpx.smo(:,1));
xpos = Exp.vpx.smo(:,2);
ypos = Exp.vpx.smo(:,3);

ixstart = find(tstart < tt,1);
ixend = find(tend < tt, 1);

ix = ixstart:ixend;
xpos = xpos(ix);
ypos = ypos(ix);
tt = tt(ix);
xpos(Exp.vpx.Labels(ix)==4) = nan;
ypos(Exp.vpx.Labels(ix)==4) = nan;
figure(1); clf
plot(tt, xpos); hold on
plot(tt, ypos)
ii = Exp.vpx.Labels(ix)==1;
plot(tt(ii), xpos(ii), '.r')
title(thisTrial)

RobsTrial = zeros(numel(tt), NC);
for cc = 1:NC
    RobsTrial(:,cc) = histcounts(Exp.osp.st(Exp.osp.clu==cids(cc)), [tt; tt(end) + 1e-3]);
end

spix = Exp.osp.st > tstart & Exp.osp.st < tend;
plot.raster(Exp.osp.st(spix), Exp.osp.clu(spix));

if strcmp(stimSet, 'ForageStaticLines')
    noisePat = Exp.D{thisTrial}.PR.NoiseHistory(:,2);
    noisePats = unique(noisePat);
    if numel(noisePats) > 1
        error('more than one noise pattern. what is up')
    end

    I = buildLineTexture(Exp.D{thisTrial}.PR.hNoise(noisePats(1)));

    ppd = Exp.S.pixPerDeg;
    ctr = Exp.S.centerPix;
    figure(2); clf
    imagesc(I); colormap gray
    title(noisePats); 
    hold on
    plot(ctr(2) + ppd*xpos(ii), ctr(1) + ppd*ypos(ii), '.r')
else
    noisePats = 1;
end


fixations = bwlabel(ii);
fixes = unique(fixations);
numFixations = numel(fixes);
for iFix = 1:numFixations
    ix = fixations==fixes(iFix);
    fix_ = struct();
    fix_.tt = tt(ix);
    fix_.xpos = xpos(ix);
    fix_.ypos = ypos(ix);
    fix_.RobsTrial = RobsTrial(ix,:);
    fix_.noisePat = noisePats;
    
    fixDat = [fixDat; fix_];
end

end

%%
noisePat = arrayfun(@(x) x.noisePat, fixDat);
fixDur = arrayfun(@(x) numel(x.tt), fixDat);

ix = fixDur > 200;
cc = cc + 1;
x1 = cell2mat(arrayfun(@(x) x.RobsTrial(1:200,cc)', fixDat(ix & noisePat==1), 'uni', 0));
x2 = cell2mat(arrayfun(@(x) x.RobsTrial(1:200,cc)', fixDat(ix & noisePat==2), 'uni', 0));

figure(1); clf
plot(mean(x1)); hold on
plot(mean(x2))
% fixDat(1).RobsTrial

%%
fixDur = arrayfun(@(x) numel(x.tt), fixDat);
goodFix = fixDur > 100;

xpos = cell2mat(arrayfun(@(x) detrend(x.xpos(100:400)'), fixDat(goodFix), 'uni', 0));
ypos = cell2mat(arrayfun(@(x) detrend(x.ypos(100:400)'), fixDat(goodFix), 'uni', 0));
xpos2 = cell2mat(arrayfun(@(x) stdfilt(x.xpos(100:400)'), fixDat(goodFix), 'uni', 0));
ypos2 = cell2mat(arrayfun(@(x) stdfilt(x.ypos(100:400)'), fixDat(goodFix), 'uni', 0));

noisePat = arrayfun(@(x) x.noisePat, fixDat(goodFix));

nwin = size(xpos,2);
win = hanning(nwin);

% xpos = xpos.*win';
% ypos = ypos.*win';

bad = ~all(hypot(xpos, ypos) < .2,2);

xpos = xpos2;
ypos = ypos2;
figure(1); clf
imagesc(xpos(~bad,:), [-.1 .1])

%%

figure(1); clf
x = xpos(~bad,:);
y = ypos(~bad,:);

vx = var(hypot(x,y), [],2);

cc = cc + 1;
if cc > NC
    cc = 1;
end
robs = cell2mat(arrayfun(@(x) x.RobsTrial(100:400,cc)', fixDat(goodFix), 'uni', 0));

rs = sum(robs(~bad,:),2);

[~, ind] = sort(vx);
plot(imgaussfilt(rs(ind), 2), '.')
title(cc)

%%
i = i+1
if i > size(x,1)
    i = 1;
end

for j=1:300
    iix = (j-40):j;
    iix = max(iix, 1);
    
    subplot(121)
    plot(x(i,iix), y(i,iix), '-ok', 'MarkerSize', 2); hold on
    plot(x(i,j), y(i,j), 'or'); hold off
    subplot(122)
    
    
    plot(y(i,iix), 'r'); hold on
    plot(x(i,iix), '-k'); hold off
    drawnow
end

%%

var(xpos,[],2)


% plot(x(i,:)); hold on


%%
padding = zeros(size(xpos));

trix = noisePat == 1;
ix = ~bad & trix;
x = reshape([xpos(ix,:) padding(ix,:)]', [], 1);
y = reshape([ypos(ix,:) padding(ix,:)]', [], 1);



%%
cc = cc + 1;
if cc > NC
    cc = 1;
end
robs = cell2mat(arrayfun(@(x) x.RobsTrial(100:400,cc)', fixDat(goodFix), 'uni', 0));

figure(2); clf
[iii,jjj] = find(robs);
plot.raster(jjj,iii, 2);
title(cc)
% %%
% figure(1); clf
% i = i + 1
% if i > size(xpos, 1)
%     i = 1;
% end
% plot(xpos(i,:)); hold on
% plot(robs(i,:))
% title(i)
% 
% %%


r = reshape([robs(ix,:) padding(ix,:)]', [], 1);
sum(r)
r0 = rand(size(r))<mean(r);
r = r - mean(r);

nlags = 50;
x(isnan(x)) = 0;
y(isnan(y)) = 0;

x = [(hypot(x,y))];
% x = stdfilt(x, ones(3)) + stdfilt(y, ones(3));

[xcr, lags] = xcorr(x, r, nlags, 'unbiased');
% [xcl, ~] = xcorr(y, r, nlags, 'unbiased');

[xcr0] = xcorr(x, r0, nlags, 'unbiased');
% [xcl0] = xcorr(y, r0, nlags, 'unbiased');

figure(1); clf
plot(lags, xcr); hold on
% plot(lags, xcl);

plot(lags, xcr0, 'k')
% plot(lags, xcl0, 'k')

% ylim([-.1 .1])

%%

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));


fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));


win = [-.1 .5];
bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];

valid = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);

fixdur = sacon - fixon;
[~, ind] = sort(fixdur);


binsize = 1e-3;

lags = win(1):binsize:win(2);
nlags = numel(lags);
n = numel(fixon);
NC = numel(S.cids);
spks = zeros(n,NC,nlags);
for i = 1:nlags
%     .st(Exp.osp.clu==S.cids(2))
    y = binNeuronSpikeTimesFast(Exp.osp,fixon+lags(i), binsize);
    spks(:,:,i) = full(y(:,S.cids));
end
k = 1;

%% 
tic
binTimes =  min(Exp.osp.st):binsize:max(Exp.osp.st);
valid = getTimeIdx(fixon, tstart, tstop);
binTimes = binTimes(valid);
Y = binNeuronSpikeTimesFast(Exp.osp, binTimes, binsize);
toc
%%

% k = mod(k + 1, NC); k = max(k, 1);


figure(1); clf
set(gcf, 'Color', 'w')
for cc = 1:NC
    subplot(6, ceil(NC/6), cc, 'align')
    [i, j] = find(spks(ind,cc,:));
    plot.raster(lags(j), i, 10); hold on
    plot([0 0], ylim, 'r')
    plot(fixdur(ind), 1:numel(ind), 'g')
    xlim(win)
    ylim([1 numel(ind)])
    title(sprintf('Unit: %d', cc))
    axis off
%     ylim(ind(end)-[400 0])
end


%%
k = mod(k + 1, NC); k = max(k, 1);
figure(2); clf
% set(gcf, 'Color', 'w')
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
% set(gcf, 'Color', 'w')
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

