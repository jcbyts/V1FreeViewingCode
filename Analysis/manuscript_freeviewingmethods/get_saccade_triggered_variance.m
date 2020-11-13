function S = get_saccade_triggered_variance(Exp, varargin)
% run saccade-triggered analysis of variance

ip = inputParser();
ip.addParameter('binsize', 1e-3);
ip.addParameter('win', [-.1 .5])
ip.addParameter('timebins', -.1:0.02:.28)
ip.addParameter('stimulusSets', {'BackImage'})
ip.addParameter('validrange', [.25 .5])
ip.addParameter('slidingwin', 20)
ip.addParameter('plotit', false)
ip.parse(varargin{:})

% parameters
binsize = ip.Results.binsize;
win = ip.Results.win;
timebins = ip.Results.timebins;
stimulusSets = ip.Results.stimulusSets;
validrange = ip.Results.validrange; % fixation durations within this range are used
sm = ip.Results.slidingwin; % sliding window
plotit = ip.Results.plotit;

S = struct();

for iStim = 1:numel(stimulusSets)
    
    stimulusSet = stimulusSets{iStim};
    
    tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D));
    tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D));

    if strcmp(stimulusSet, 'ITI')
        timings = [tstop(1:end-1) tstart(2:end)];
    else
        validTrials = io.getValidTrials(Exp, stimulusSet);
        timings = [tstart(validTrials) tstop(validTrials)];
    end
    
    [~, ~, ~, lags, spks, fixdur, valid_saccades] = get_saccade_relative_rate(Exp, Exp.osp, timings, 'binsize', binsize, 'win', win);
    [~, ind] = sort(fixdur, 'ascend');
    
    % saccade vectors
    dx = Exp.vpx.smo(Exp.slist(:,5),2) - Exp.vpx.smo(Exp.slist(:,4),2);
    dy = Exp.vpx.smo(Exp.slist(:,5),3) - Exp.vpx.smo(Exp.slist(:,4),3);
    dx = dx(valid_saccades);
    dy = dy(valid_saccades);
    sacamp = hypot(dx, dy);
    
    
    NC = size(spks, 2);
    
    S.(stimulusSet) = repmat(struct('mean', [], ...
        'var', [], 'n', [], 'blin', [], 'bquad', [], 'lags', [], 'cg', [], 'cid', []), NC, 1);
    
    %% plot raster
    if plotit
        lagsrast = lags;
        figure(4); clf
        win = [-.1 .5];
        figure(1); clf
        set(gcf, 'Color', 'w')
        
        
        for cc = 1:NC
            subplot(6, ceil(NC/6), cc, 'align')
            [i, j] = find(spks(ind,cc,:));
            plot.raster(lagsrast(j), i, 10); hold on
            plot([0 0], ylim, 'r')
            plot(fixdur(ind), 1:numel(ind), 'g')
            xlim(win)
            ylim([1 numel(ind)])
            title(sprintf('Unit: %d', cc))
            axis off
        end
    end
    
    
    for cc = 1:NC
        
        S.(stimulusSet)(cc).cid = Exp.osp.cids(cc);
        S.(stimulusSet)(cc).cg = Exp.osp.cgs(cc);
        
        ix = fixdur > validrange(1) & fixdur < validrange(2);
        
        fprintf('%d valid fixations\n', sum(ix))
        
        % smooth spikes with a boxcar
        bc = ones(sm,1);
        spsm = filter(bc, 1, squeeze(spks(ix,cc,:))')';
        
        mu = mean(spsm)/binsize/sm;
        v = var(spsm)/binsize/sm;
        
        % subselect a single saccade vector
        iisac = dx > 1.5 & dx < 4 & abs(dy) < 1;
        figure(2); clf
        plot(dx, dy, '.'); hold on
        plot(dx(iisac), dy(iisac), '.')
        
        n2 = sum(ix & iisac);
        m2 = mean(spsm(iisac(ix),:))/binsize/sm;
        v2 = var(spsm(iisac(ix),:))/binsize/sm;
        
        if plotit
            figure(1); clf
            subplot(1,3,1)
            [i,j] = find(squeeze(spks(ind,cc,:)));
            plot.raster(lags(j),i,10); hold on
            
            cmap = hot(numel(timebins)+5);
            fixix = fixdur(ind) > .25 & fixdur(ind) < .5;
            for ibin = 2:numel(timebins)
                iix = lags(j)' >= timebins(ibin-1) & lags(j)' < timebins(ibin);
                iix = iix & fixix(i);
                plot.raster(lags(j(iix)),i(iix),10, 'Color', cmap(ibin,:));
            end
            axis tight
            title(cc)
            
            figure(2); clf
            [~, ind2] = sort(sacamp(ix), 'ascend');
            stmp = squeeze(spks(ix,cc,:));
            [i,j] = find(stmp(ind2,:));
            plot.raster(lags(j), i, 10)
            axis tight
            xlim([-.1 .26])
            
            
            figure(1);
            
            subplot(1,3,2)
            for ibin = 2:numel(timebins)
                iix = lags >= timebins(ibin-1) & lags < timebins(ibin);
                plot(mu(iix), v(iix), '.', 'Color', cmap(ibin,:)); hold on
            end
            plot(xlim, xlim, 'k')
            xlabel('Mean (sp/s)')
            ylabel('Variance')
            
            subplot(2,3,3)
            for ibin = 2:numel(timebins)
                iix = lags >= timebins(ibin-1) & lags < timebins(ibin);
                fill([lags(iix) fliplr(lags(iix))],[mu(iix) fliplr(v(iix))], 'w', 'FaceColor', cmap(ibin,:)); hold on
            end
            xlim([-.05 .25]);
            plot(lags, v, 'Color', .5*[1 1 1], 'Linewidth', 1.5); hold on
            plot(lags, v, 'k', 'Linewidth', 1.5);
            xlabel('Time from fixation onset (s)')
            ylabel('Firing Rate')
            plot(lags, m2, 'b')
            
            subplot(2,3,6)
            ff = v./mu;
            plot(lags, ff, 'k'); hold on
            for ibin = 2:numel(timebins)
                iix = lags >= timebins(ibin-1) & lags < timebins(ibin);
                plot(lags(iix), ff(iix),'.', 'Color', cmap(ibin,:)); hold on
            end
            xlabel('Time from fixation onset (s)')
            ylabel('Fano Factor')
            plot(xlim, [1 1], 'k--')
            xlim([-.05 .25]);
            ylim([.5 2])
            
            plot(lags, v2./m2, 'b')
            input('Check')
        end
        
        
        fun = @(b, x) b(1)*x + b(2)*x.^2;
        b0 = [1 0];
        bquad = lsqcurvefit(fun, b0, mu, v);
        
        fun = @(b, x) b(1)*x;
        b0 = [1];
        blin = lsqcurvefit(fun, b0, mu, v);
        
        S.(stimulusSet)(cc).lags = lags;
        S.(stimulusSet)(cc).mean = mu;
        S.(stimulusSet)(cc).var = v;
        S.(stimulusSet)(cc).meanfixsac = m2;
        S.(stimulusSet)(cc).varfixsac = v2;
        S.(stimulusSet)(cc).nfixsac = n2;
        
        S.(stimulusSet)(cc).betaQuad = bquad;
        S.(stimulusSet)(cc).betaLin = blin;
    end
end