

for iEx = 1:57

[Exp, S, lfp, mua] = io.dataFactoryGratingSubspace(iEx);

end


%% test single session extraction from server

iEx = 10;
[Exp, S, lfp] = io.dataFactoryGratingSubspace(iEx);

ops = io.loadOps(S.rawFilePath);


if numel(ops) > 1
    warning('more than one ops')
    ops = ops(1);
end

ops = io.convertOpsToNewDirectory(ops, S.rawFilePath);
% get MUA

%%
[MUA, timestamps] = io.getMUA(ops, true, true);

%%

[data, ts] = io.loadRaw(ops, 100e3+[0 30e3], true);

Nchan = size(data,1);

%%

figure(1); clf
plot(bsxfun(@plus, data' - mean(data',2), (1:Nchan)*500), 'k')

%%
% data = preprocess.artifRemovAcrossChannel(MUA, 150, 12, 50);
data = MUA;
et = csd.getCSDEventTimes(Exp);
% et = Exp.vpx2ephys(Exp.slist(:,1));
[~, ~, ev] = histcounts(et, timestamps);

[an, sd, xax] = eventTriggeredAverage((data), ev, [-100, 300]);

an =  an - mean(an(xax < 0 ,:));

an = an ./ max(an(xax < 100,:));

for i = 1:size(an,2)
   an(:,i) =  imgaussfilt(an(:,i), 5);
end

figure(1); clf
plot(xax, an)


figure(2); clf
imagesc(an')
% imagesc( xax, lfp.ycoords, an')
% xlim([100 200])

cstruct = csd.getCSD(lfp, et, 'spatsmooth', 2.5, 'method', 'standard');

csd.plotCSD(cstruct)
xlim([0 100])
%%
data = preprocess.artifRemovAcrossChannel(MUA, 150, 12, 50);
[an, sd, xax] = eventTriggeredAverage(data, ev, [-100, 300]);

figure(1); clf
plot(xax, bsxfun(@plus, zscore(an), (1:32)*4), 'k')


figure(2); clf
imagesc(zscore(an)')
axis xy

%%

data = preprocess.artifRemovAcrossChannel(MUA, 150, 12, 50);
figure(1); clf
plot(MUA(:,1)); hold on
plot(data(:,1))
