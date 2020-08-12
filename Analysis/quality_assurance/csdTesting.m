
iEx = 49;
[Exp, ~, lfp, mua] = io.dataFactoryGratingSubspace(iEx);

%% CSD test cell
et = csd.getCSDEventTimes(Exp);

% --- get CSD
cstruct1 = csd.getCSD(lfp, et, 'spatsmooth', 2.5, 'method', 'standard', 'plot', false, 'debug', true);

csd.plotCSD(cstruct1)
%%
% loop over sessions and make plots of MUA and CSD
for iEx = 1:57
    
    try
        %%
%         iEx = iEx + 1;

        % load data
        [Exp, ~, lfp, mua] = io.dataFactoryGratingSubspace(iEx);
        
        et = csd.getCSDEventTimes(Exp);
        
        % --- get CSD
        cstruct1 = csd.getCSD(lfp, et, 'spatsmooth', 2.5, 'method', 'standard', 'plot', false);
        
        % --- get MUA triggered average
        % convert to sample indexes
        [~, ~, ev] = histcounts(et, mua.timestamps);
        
        % trigger average
        % [fav, ~, xax] = eventTriggeredAverage((mua.data - nanmean(mua.data))./nanstd(mua.data), ev, [-150 300]);
        [fav, ~, xax] = eventTriggeredAverage(mua.data, ev, [-150 300]);
        
        fav0 = fav;
        
        % smooth
        fav = fav0;
        for ch = 1:size(fav, 2)
            fav(:,ch) = sgolayfilt(fav(:,ch), 3, 15);
        end
        
        
        figure(1); clf
        subplot(2,2,1, 'align')
        plot(xax, fav);
        
        xlabel('Time from Flash')
        ylabel('MUA');
        title('RAW MUA')
        axis tight
        xlim([-100 200])
        
        subplot(2,2,2, 'align')
        
        % subtrack baseline
        fav = fav - mean(fav(xax > 0 & xax < 10,:));
        inds = setdiff(1:32, mua.deadChan);
        fav(:,mua.deadChan) = nan;
        % normalize
        fav = fav ./ max(fav(xax>0 * xax < 50,:));
        
        depths = (32:-1:1)*abs(diff(mua.ycoords(1:2)));
        ix = xax > 15 & xax < 80;
        x = xax(ix);
        y = fav(:,inds)';
        imagesc(x, depths(inds), y(:,ix))
        xlim([0 100])
        hold on
        plot(xlim, cstruct1.reversalPointDepth{1}*[1 1], 'r--', 'Linewidth', 2)
        title('Normalized / Baseline Subtract MUA')
        
        subplot(2,2,3, 'align')
        n = numel(size(cstruct1.STA));
        if n ==3
            imagesc(cstruct1.time, cstruct1.depth, cstruct1.CSD(:,:,1))
        else
            imagesc(cstruct1.time, cstruct1.depth, cstruct1.CSD)
        end
        hold on
        plot(xlim, cstruct1.reversalPointDepth{1}*[1 1], 'r--', 'Linewidth', 2)
        title('CSD')
        
        subplot(2,2,4,'align')
        
        cids = Exp.osp.cids;
        NC = numel(cids);
        win = [-150 300];
        nw = diff(win)+1;
        spks = nan(NC, nw);
        for cc = 1:NC
            spcnt = histcounts(Exp.osp.st(Exp.osp.clu==cids(cc)), mua.timestamps);
            [av, ~, xax] = eventTriggeredAverage(spcnt(:), ev, win);
            av = av - mean(av(xax<0));
            av = av / max(av);
            spks(cc,:) = av;
        end
        
        [depths, inds]= sort(Exp.osp.clusterDepths(:));
        p = pcolor(xax, depths, spks(inds,:));
        axis ij
        p.EdgeColor = 'none';
        xlabel('Time from Flash')
        ylabel('Cluster Depth')
        title('Kilo Clusters')
        xlim([0 100])

        exname = strrep(Exp.FileTag(1:end-4), '_', '-');
        plot.suplabel(exname, 't')
        
        
        plot.fixfigure(gcf, 12, [8 8], 'OffsetAxes', false);
        saveas(gcf, sprintf('Figures/MUACSD/%s.pdf', Exp.FileTag(1:end-4)))
    end
end
%%

%


% plot
Nchan = size(mua.data,2);

%
figure(iEx); clf

subplot(121,'align')
imagesc(xax, (Nchan:-1:1)*50, fav');
hold on
plot(xlim, cstruct1.reversalPointDepth{1}*[1 1], 'r--', 'Linewidth', 2)
title('MUA')
xlim([0 80])

subplot(122,'align')
imagesc(cstruct1.time, cstruct1.depth, cstruct1.CSD); hold on
plot(xlim, cstruct1.reversalPointDepth{1}*[1 1], 'r--', 'Linewidth', 2)
xlim([0 80])
title('CSD')

exname = strrep(Exp.FileTag(1:end-4), '_', '-');
plot.suplabel(exname, 't')

%%
fun = @(par,t) exp(-.5 * (t - par(1)).^2 / par(2).^2) + par(3) * exp(-.5 * (t - par(4)).^2 / par(5).^2);
par0 = [50, 10, .1, 100, 100];

% size(fun(xax, par0))
fit = zeros(numel(xax), Nchan);
for ch = 1:Nchan
    xv = xax(xax > 0 & xax < 100);
    [~, id] = max(fav(xax > 0 & xax < 100,ch));
    par0(1) = xv(id);
    parHat = lsqcurvefit(fun, par0, xax, fav(:,ch)');
    fit(:,ch) = fun(parHat, xax);
end

figure(1); clf
subplot(121,'align')
imagesc(xax, 1:Nchan, fav')
xlim([0 100])
subplot(122, 'align')
imagesc(xax, 1:Nchan, fit')
xlim([0 100])
%%

% lfp.data = abs(lfp.data);
% mua.data = zscore(mua.data);

% lfp.data =


%%
plot(-cstruct1.STA(:,13))
%%

% et = Exp.vpx2ephys(Exp.slist(:,1));
cstruct0 = csd.getCSD(lfp, et, 'spatsmooth', 0, 'method', 'spline', 'plot', false);

csd.plotCSD(cstruct0)
title('Flash (spline)')

%%

% et = csd.getCSDEventTimes(Exp);
et = Exp.vpx2ephys(Exp.slist(:,1));
cstruct2 = csd.getCSD(lfp, et, 'spatsmooth', 2.5, 'method', 'standard', 'plot', false);
csd.plotCSD(cstruct2);

%%
% trial starts and stops
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(:)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(:)));

% inter-trial intervals
epochs = [tstop(1:end-1) tstart(2:end)];

% get waveform statistics over valid epochs
% W = io.get_waveform_stats(Exp.osp, 'validEpochs', epochs);


%%
figure;
imagesc(cstruct1.time, cstruct1.chDepths, cstruct1.STA)

figure
imagesc(cstruct2.time, cstruct2.chDepths, cstruct2.STA)
%%
figure(1); clf
imagesc(cstruct2.STA);

%%

spower = cstruct2.STA(:,find(cstruct1.time==38));
figure(2); clf
plot(1:32, spower);
hold on
plot(2:32, diff(spower))
hold on
plot(2:31, diff(diff(spower)))

% cstruct2 = csd.getCSD(lfp, et)