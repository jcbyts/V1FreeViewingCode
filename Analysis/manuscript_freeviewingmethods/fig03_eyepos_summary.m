figDir = 'Figures/manuscript_freeviewing';

fid = 1; % print to command window

clear S;
addpath Analysis/manuscript_freeviewingmethods/

fprintf(fid, '******************************************************************\n');
fprintf(fid, '******************************************************************\n');
fprintf(fid, 'Statistics for figure 03 (Eye position / saccades)\n\n\n')
%% Run main analysis on each session
fprintf(fid, '******************************************************************\n');
fprintf(fid, '******************************************************************\n');
fprintf(fid, '\t\tRUNNING EYE POSITION ANALYSIS ON All SESSIONS\n');
fprintf(fid, '******************************************************************\n');
fprintf(fid, '******************************************************************\n');

for sessId = 1:57
    fprintf(fid, '******************************************************************\n');
    fprintf(fid, 'Session %d\n', sessId)
    Exp = io.dataFactoryGratingSubspace(sessId);
      
%     ds = io.detect_doublestep_saccades(Exp);
%     % flag ds sacs
%     ix = ds.dotprod > .9;
%     
%     % new end
%     Exp.slist(ds.suspectSaccadePairs(ix,1), 2) = Exp.slist(ds.suspectSaccadePairs(ix,2), 2);
%     Exp.slist(ds.suspectSaccadePairs(ix,1), 5) = Exp.slist(ds.suspectSaccadePairs(ix,2), 5);
%     % remove second
%     Exp.slist(ds.suspectSaccadePairs(ix,2), :) = [];
    
    S(sessId) = sess_eyepos_summary(Exp);
end


%%
% clear ds
% for sessId = 1:57
%     fprintf(fid, '******************************************************************\n');
%     fprintf(fid, 'Session %d\n', sessId)
%     Exp = io.dataFactoryGratingSubspace(sessId);
% 
%     ds(sessId) = io.detect_doublestep_saccades(Exp);
% end
% 
% %%
% 
% dp = cell2mat(arrayfun(@(x) real(acosd(x.dotprod(:))), ds(:), 'uni', 0));
% figure(1); clf
% histogram(dp, 100);
% xlabel('Angle between saccade vectors')

%%
stim = 'All'; % loop over stimuli
monkey = 'All'; % loop over monkeys

nTrials = arrayfun(@(x) x.(stim).nTrials, S);

switch monkey
    case {'Ellie', 'E'}
        monkIx = arrayfun(@(x) strcmp(x.exname(1), 'e'), S);
        Exp = io.dataFactoryGratingSubspace(5);
    case {'Logan', 'L'}
        monkIx = arrayfun(@(x) strcmp(x.exname(1), 'l'), S);
        Exp = io.dataFactoryGratingSubspace(56);
    case {'Milo', 'M'}
        monkIx = arrayfun(@(x) strcmp(x.exname(1), 'm'), S);
    case {'All'}
        monkIx = true(size(nTrials));
end
        
good = (nTrials > 0) & monkIx;
nGood = sum(good);

bins = S(1).(stim).positionBins;
posCnt = arrayfun(@(x) x.(stim).positionCount, S(good), 'uni', 0);

dims = size(posCnt{1});
X=reshape(cell2mat(posCnt), [dims, nGood]);

figure(2); clf
C = log10(imgaussfilt(sum(X,3), .5));
h = imagesc(bins, bins, C); axis xy
colormap(plot.viridis(50))
colorbar



% plot screen width
w = Exp.S.screenRect(3)/Exp.S.pixPerDeg;
h = Exp.S.screenRect(4)/Exp.S.pixPerDeg;

hold on
plot([-w/2 w/2], [h/2 h/2], 'r', 'Linewidth', 2)
plot([-w/2 w/2], -[h/2 h/2], 'r', 'Linewidth', 2)
plot([-w/2 -w/2], [-h/2 h/2], 'r', 'Linewidth', 2)
plot([w/2 w/2], [-h/2 h/2], 'r', 'Linewidth', 2)
xlabel('Horizontal Eye Position (d.v.a.)')
ylabel('Horizontal Eye Position (d.v.a.)')
title(sprintf('Monkey: %s', monkey))

plot.fixfigure(gcf, 10, [5 4])
saveas(gcf, fullfile(figDir, sprintf('Fig02a_eyeposition_monk%s.pdf', monkey)))

% stim = 'All';
totalDuration = arrayfun(@(x) x.(stim).totalDuration, S(good))/60;

% --- plot fixation duration
figure(4); clf
bins = S(1).(stim).fixationDurationBins * 1e3;
cnt = cell2mat(arrayfun(@(x) x.(stim).fixationDurationCnt, S(good)', 'uni', 0));
cnt = cnt ./ sum(cnt,2);
x = mean(cnt);
s = std(cnt) / sqrt(nGood);

ix = find(bins <= 100);
fill([bins(ix) bins(ix(end))], [x(ix) x(1)], 'r', 'FaceColor', .8*[1 1 1], 'EdgeColor', 'none'); hold on
ix = find(bins >= 100);
fill([bins(ix(1)) bins(ix) bins(ix(end))], [0 x(ix) x(1)], 'r', 'FaceColor', .5*[1 1 1], 'EdgeColor', 'none');
cmap=lines;
plot.errorbarFill(bins, x, 2*s, 'b', 'FaceColor', cmap(1,:), 'EdgeColor', 'none'); hold on
plot(bins, x, 'Color', cmap(1,:))

m = nan(nGood,1);
for i = 1:nGood
    [ux, ia] = unique(cumsum(cnt(i,:))./sum(cnt(i,:)));
    m(i) = interp1(ux,bins(ia), .5);
end

s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
plot(mean(m)*[1 1], ylim, 'r', 'Linewidth', 2)
% xx = prctile(m, [25 75]);
% yy = ylim;
% fill(xx([1 1 2 2]), yy([1 2 2 1]), 'r', 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', .5)


ylabel('Probability')
xlabel('Fixation Duration (ms)')
ylim([0 0.05])
set(gca, 'XTick', 0:250:1000, 'YTick', 0:.025:.05, 'TickDir', 'out', 'Box', 'off')
plot.fixfigure(gcf, 10, [3 3])
saveas(gcf, fullfile(figDir, sprintf('Fig02d_fixationdur_monk%s.pdf', monkey)))

% --- Plot saccade amplitude
figure(3); clf
bins = S(1).(stim).sacAmpBinsBig;
cnt = cell2mat(arrayfun(@(x) x.(stim).sacAmpCntBig, S(good)', 'uni', 0));
% cnt = cnt ./ sum(cnt,2);
cnt = cnt ./ totalDuration(:);

x = mean(cnt);

m = nan(nGood,1);
for i = 1:nGood
    [ux, ia] = unique(cumsum(cnt(i,:))./sum(cnt(i,:)));
    try
    m(i) = interp1(ux,bins(ia), .5);
    end
end

s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
xx = prctile(m, [25 75]);
yy = ylim;
fill(xx([1 1 2 2]), yy([1 2 2 1]), 'r', 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', .5)
ylabel('Saccade Rate (Count / Minute)')
xlabel('Amplitude (d.v.a.)')
set(gca, 'XTick', 0:5:15, 'TickDir', 'out', 'Box', 'off')

% get microsaccade plot
bins = S(1).(stim).sacAmpBinsMicro;
cnt = cell2mat(arrayfun(@(x) x.(stim).sacAmpCntMicro, S(good)', 'uni', 0));
cnt = cnt ./ totalDuration(:);

axinset = axes('Position', [.5 .5 .4 .4]);
set(gcf, 'currentaxes', axinset)

x = mean(cnt);
s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
xlim([0 1])
ylabel('Count / Minute')
xlabel('Amplitude (d.v.a.)')
set(axinset, 'Box', 'off', 'TickDir', 'out')

% ylim([0 0.05])

plot.fixfigure(gcf, 10, [3 3])
saveas(gcf, fullfile(figDir, sprintf('Fig02d_saccadeamp_monk%s.pdf', monkey)))



bins = S(1).(stim).DistanceBins;
cnt = cell2mat(arrayfun(@(x) x.(stim).DistanceCountCtrScreen, S(good)', 'uni', 0));
cnt = cnt ./ sum(cnt,2);

% percent within 5 d.v.a of center of screen
pc5 = sum(cnt(:,bins < 5),2) ./ sum(cnt,2);
fprintf(fid, "Monkey %s spent %02.2f %% +- %02.2f of the time < 5 d.v.a\n", monkey, mean(pc5)*100, std(pc5)*100/sqrt(nGood))
pc10 = sum(cnt(:,bins < 10),2) ./ sum(cnt,2);
fprintf(fid, "Monkey %s spent %02.2f %% +- %02.2f of the time < 10 d.v.a\n", monkey, mean(pc10)*100, std(pc10)*100/sqrt(nGood))

figure(1); clf
x = mean(cnt);
s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
ylabel('Probability')
xlabel('Distance (d.v.a.)')
xlim([0 20])
ylim([0 .1])
set(gca, 'XTick', 0:5:20, 'YTick', 0:.05:.1, 'TickDir', 'out', 'Box', 'off')
plot.fixfigure(gcf, 10, [3 3])
saveas(gcf, fullfile(figDir, sprintf('Fig02c_distance_monk%s.pdf', monkey)))
% imagesc(cnt)

%% Time fixating
stim = 'All';
good = true(numel(S),1);

totalDuration = arrayfun(@(x) x.(stim).totalDuration, S(good))/60;
nValidSamples = arrayfun(@(x) x.(stim).nValidSamples, S(good));
nInvalidSamples = arrayfun(@(x) x.(stim).nInvalidSamples, S(good));
nTotalSamples = arrayfun(@(x) x.(stim).nTotalSamples, S(good));
nFixationSamples = arrayfun(@(x) x.(stim).nFixationSamples, S(good));

figure(1); clf
cmap = lines;
monkeys = {'E', 'L', 'M'};
rng(555)
for m = 1:3
    ix = arrayfun(@(x) strcmpi(x.exname(1), monkeys{m}), S(good));
    n = sum(ix);
    x = nFixationSamples(ix) ./ nTotalSamples(ix) * 60;
    fprintf(fid, 'Monkey %s median fixation: %02.2f, [%02.2f, %02.2f]\n', monkeys{m}, median(x), prctile(x, 2.5), prctile(x, 97.5))
    jitter = randn(n,1);
    plot(m + .1*jitter, x, 'o', 'Color', cmap(m,:), 'MarkerFaceColor', cmap(m,:)); hold on
    plot(m + [-.2 .2], median(x)*[1 1], 'k', 'Linewidth', 2)
    
    x = nValidSamples(ix) ./ nTotalSamples(ix) * 60;
    plot(m + .1*jitter, x, 'o', 'Color', cmap(m,:)); hold on
    plot(m + [-.2 .2], median(x)*[1 1], 'k--', 'Linewidth', 2)
end

ylim([30 60])
set(gca, 'XTick', 1:3, 'XTickLabel', monkeys, 'YTick', 30:5:60, 'TickDir', 'Out', 'Box', 'off')
xlabel('Subject')
ylabel('Time (s) per minute of recording')
plot.fixfigure(gcf, 10, [4 4])
saveas(gcf, fullfile(figDir, 'Fig02d_usableFixationTime.pdf'))





%%
plot(nValidSamples ./ nTotalSamples, '-o')

figure(2); clf
plot(nFixationSamples ./ nValidSamples, 'o')
ylabel('Percent Time Fixating')
xlabel('Session')
