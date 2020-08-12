function [mFR, mRR, mRRsd, lags, spks, fixdur, valid_saccades] = get_saccade_relative_rate(Exp, S,validTrials, varargin)
% [RR, mRR, mRRsd] = get_saccade_relative_rate(Exp, validTrials)

ip = inputParser();
ip.addOptional('binsize', 5e-3)
ip.addOptional('smoothing', 5)
ip.addOptional('win', [-.1 .5])
ip.addOptional('sacexclusion', 0.25)
ip.parse(varargin{:})

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

thresh = ip.Results.sacexclusion;

fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));

valid_saccades = 1:size(Exp.slist,1)-1;

win = ip.Results.win;
bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];
valid_saccades(bad) = [];

valid = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);
valid_saccades = valid_saccades(valid);

fixdur = sacon - fixon;

binsize = ip.Results.binsize;

lags = win(1):binsize:win(2);
nlags = numel(lags);
n = numel(fixon);
NC = numel(S.cids);
spks = zeros(n,NC,nlags);
for i = 1:nlags
    y = binNeuronSpikeTimesFast(Exp.osp,fixon+lags(i), binsize);
    spks(:,:,i) = full(y(:,S.cids));
end

mFR = squeeze(mean(spks(fixdur > thresh,:,:)))'/binsize;
mFR = filter(ones(5,1)/5, 1, mFR);
% mTot = full(mean(Y(:,S.cids)))/binsize;
mTot = mean(mFR(lags < .250 & lags > .1,:));
mTot(mTot < 1) = nan;
% plot(lags, mFR./mTot); hold on
% plot(thresh*[1 1], ylim, 'k--')
% plot(xlim, [1 1], 'k--')
% xlabel('Time from fixation onset')
% ylabel('Relative Rate')
% clf
RR = mFR./mTot;
mRR = nanmean(RR,2);
mRRsd = nanstd(mFR./mTot,[],2)/sqrt(NC);
% plot.errorbarFill(lags, mRR, mRRsd)
