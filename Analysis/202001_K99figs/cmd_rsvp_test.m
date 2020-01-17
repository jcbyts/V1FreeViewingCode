

[Exp, S] = io.dataFactory(12);

%%
validTrials = io.getValidTrials(Exp, 'FixRsvpStim');
%%
numTrials = numel(validTrials);
iTrial = 1;
thisTrial = validTrials(iTrial);


%%
n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials));
tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1,1), Exp.D(validTrials)));
bad = n < 100;
tstart(bad) = [];
n(bad) = [];

win = [-.1 2];

figure(1); clf

[~, ind] = sort(n, 'descend');


binsize = 2e-3;

lags = win(1):binsize:win(2);
nlags = numel(lags);
nt = numel(tstart);
NC = numel(S.cids);
spks = zeros(nt,NC,nlags);
for i = 1:nlags
    y = binNeuronSpikeTimesFast(Exp.osp,tstart+lags(i), binsize);
    spks(:,:,i) = full(y(:,S.cids));
end
k = 1;



%% 
tic
binTimes =  min(Exp.osp.st):binsize:max(Exp.osp.st);
% valid = getTimeIdx(fixon, tstart, tstop);
% binTimes = binTimes(valid);
Y = binNeuronSpikeTimesFast(Exp.osp, binTimes, binsize);
toc
cc = 0;
%%

cc = mod(cc + 1, NC); cc = max(cc, 1);
cc = 20;

figure(1); clf
set(gcf, 'Color', 'w')

subplot(4,1,1:3)
[i, j] = find(spks(ind,cc,:));
plot.raster(lags(j), i, 1); hold on
plot([0 0], ylim, 'r')
plot(n(ind)/Exp.S.frameRate, 1:numel(ind), 'g')
xlim(lags([1 end]))
ylabel('Trial')

title(cc)
subplot(4,1,4)
m = squeeze(mean(spks(:,cc,:),1))/binsize;
plot(lags, m, 'k')
xlim(lags([1 end]))
xlabel('Time from fixation onset')
ylabel('Firing Rate')




%%


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




ori = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,2), Exp.D(validTrials), 'uni', 0));
cpd = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,3), Exp.D(validTrials), 'uni', 0));
frameTime = Exp.ptb2Ephys(cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,1), Exp.D(validTrials), 'uni', 0)));


ix = ~isnan(cpd);
cpds = unique(cpd(ix));
oris = unique(ori(ix));
ncpd = numel(cpds);
nori = numel(oris);

[~, oriid] = max(ori==oris', [], 2);
[~, cpdid] = max(cpd==cpds', [], 2);
blank = cpdid==1;
ncpd = ncpd - 1;
cpdid = cpdid - 1;
cpds(1) = [];

ind = sub2ind([nori ncpd], oriid(~blank), cpdid(~blank));

binsize = 1./Exp.S.frameRate;

NT = numel(frameTime);
ft = (1:NT)';
stim = sparse(ft(~blank), ind, ones(numel(ind),1), NT, nori*ncpd);
Robs = binNeuronSpikeTimesFast(Exp.osp, frameTime, binsize);

%%
nlags = 30;
Xd = makeStimRows(stim, nlags);

sta = [Xd ones(NT,1)]'*Robs;
sta = sta(1:end-1,:);

figure
plot(sta)
% sta = sta./sum(X)';
%%
NC = size(Robs,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf

[~, cid] = sort(Exp.osp.clusterDepths);
cids = Exp.osp.cids(cid);
for ci = 1:NC
    subplot(sx, sy, ci, 'align')
    cc = cids(ci);
    
%     cc = mod(cc+1, NC); cc(cc==0)=1;
    a = reshape(sta(:,cc), [nlags, nori*ncpd]);

    plot(oris, reshape(mean(a), [nori, ncpd]))
    ylim([0 40])
    title(cc)
end
%%
% 


