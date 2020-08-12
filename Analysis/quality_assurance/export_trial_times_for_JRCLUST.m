
%%

iEx = 56;
[Exp, S, lfp] = io.dataFactoryGratingSubspace(iEx);
%%

validTrials = io.getValidTrials(Exp, 'FixRsvpStim');
n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials));

bad = n < 100;

validTrials(bad) = [];

tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1), Exp.D(validTrials)));

numel(tstart)

times = tstart - lfp.timestamps(1);

save(sprintf('~/Dropbox/%s.mat', strrep(Exp.FileTag, '.mat', '_rsvptrial')), 'times')

%%
figure, 
plot(diff(lfp.timestamps), '.')