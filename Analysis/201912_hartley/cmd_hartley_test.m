

[Exp, S] = io.dataFactory(12);

validTrials = io.getValidTrials(Exp, 'Grating');
%%
numTrials = numel(validTrials);
iTrial = 1;
thisTrial = validTrials(iTrial);


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


