function [spkS,W] = get_visual_units(Exp, varargin)
% spkS = get_visual_units(Exp, varargin)

ip = inputParser();
ip.addParameter('plotit', false)
ip.addParameter('visStimField', 'BackImage')
ip.addParameter('ROI', [-100 -100 100 100])
ip.addParameter('binSize', 10)
ip.addParameter('waveforms', [])
ip.addParameter('numTemporalBasis', 5)
ip.parse(varargin{:});

% get waveforms
if isempty(ip.Results.waveforms)
    W = io.get_waveform_stats(Exp.osp);
else
    W = ip.Results.waveforms;
end

figure(66); clf
plotWaveforms(W)
title(strrep(Exp.FileTag(1:end-4), '_', ' '))

%% measure visual drive
plotit = ip.Results.plotit;

cids = Exp.osp.cids;
NC = numel(cids);

% trial starts
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D));

% start / stop / start / stop
bEdges = reshape([tstart tstop]', [], 1);

% calculate firing rate during trials and during ITIs
NT = numel(bEdges)-1;
Robs = zeros(NT, NC);
for cc = 1:NC
    Robs(:,cc) = histcounts(Exp.osp.st(Exp.osp.clu==cids(cc)), bEdges, 'normalization', 'countdensity');
end

% eyePos = io.getCorrectedEyePos(Exp, 'usebilinear', false);
eyePos = Exp.vpx.smo(:,2:3);

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', ip.Results.ROI, 'binSize', ip.Results.binSize, 'eyePos', eyePos);

nlags = 15;
% find(opts.eyeLabel==1)
    
stas = forwardCorrelation(Xstim, RobsSpace-mean(RobsSpace), [0 nlags-1], [], ip.Results.numTemporalBasis);

stimSets = {'Grating', 'Dots', 'BackImage', 'FixRsvpStim'};

spkS = [];

for cc = 1:NC
    
    stimFr = Robs(1:2:end-1,cc); % firing rate during trial
    isiFr = Robs(2:2:end,cc); % firing rate during ITI
    % compare images to gray screens as measure of visual drive
    
    
    Stmp = struct();
    Stmp.cid = cids(cc);
    for iStim = 1:numel(stimSets)
        validTrials = io.getValidTrials(Exp, stimSets{iStim});
        if numel(validTrials) < 10
            Stmp.(stimSets{iStim}) = nan;
        else
            Stmp.(stimSets{iStim}) = ismodulated(stimFr, isiFr, validTrials);
        end
    end
    
    validTrials = io.getValidTrials(Exp, ip.Results.visStimField); % we know we show this every time
    
    sfr = stimFr(validTrials(1:end-1));
    ifr = isiFr(validTrials(1:end-1));
    isviz = Stmp.BackImage;
    if isnan(isviz)
        isviz = false;
    end
    
    gtrials = validTrials;
    isiX = isiFr(validTrials(1:end-1));
    
    % stability analysis
    [ipoints, ~] = findchangepts(isiX, 'Statistic', 'mean', 'MinDistance', 1, 'MaxNumChanges', 2);
    
    n=numel(isiX);
    ipoints = [0; ipoints; n]; %#ok<AGROW>
    stableFr = [];
    for ichng = 2:numel(ipoints)
        i0 = ipoints(ichng-1)+1;
        i1 = ipoints(ichng);
        iix = i0:i1;
        stableFr = [stableFr mean(isiX(iix))*ones(size(iix))]; %#ok<AGROW>
    end
    
    if rsquared(isiX, stableFr) > .2 % changepoints explain 10% more variance
        
        len = diff(ipoints);
        [~, bigi] = max(len);
        
        goodix = gtrials(ipoints(bigi)+1):gtrials(ipoints(bigi+1));
    else
        goodix = gtrials(1):gtrials(n);
    end
    
    Stmp.stableIx = goodix;
    
    if plotit
        figure(1); clf
        
        subplot(2,4,1, 'align') % waveform template
        plotWaveforms(W(cc))
        
        
        subplot(2,4,2, 'align') % autocorrelation
        sptimes = Exp.osp.st(Exp.osp.clu==cids(cc));
        nbins = 100;
        K = ccg(sptimes, sptimes, nbins, 1e-3);
        K(nbins+1) = 0;
        plot(K, '-o', 'MarkerSize', 2)
        
        subplot(2,4,3, 'align') % natural images vs. ITI
        plot(ifr, sfr, '.'); hold on
        xlabel('Firing Rate (ITI)')
        ylabel('Firing Rate (Image)')
        plot(xlim, xlim, 'k')
        
        subplot(2,3,4:6) % firing rate over time (for stability)
        cmap = lines;
        plot(stimFr, 'Color', [cmap(1,:) .5]); hold on
        plot(isiFr, 'Color', [cmap(2,:) .5])
        clear h
        h(1) = plot(imboxfilt(stimFr, 11), 'Color', cmap(1,:));
        h(2) = plot(imboxfilt(isiFr, 11), 'Color', cmap(2,:));
        
        title(cc)
        h(3) = plot(goodix, mean(isiFr(goodix))*ones(size(goodix)), 'g');
        plot(validTrials(1)*[1 1], ylim, 'r--')
        legend(h, {'Trial', 'ITI', 'Good Trials'}, 'Location', 'Best', 'Box', 'off')
        
        
        fprintf('%d good trials\n', numel(goodix))
        
    end
    
    if ~isempty(Xstim)
        
        sta = stas(:,:,cc);
%         sta = simpleRevcorr(Xstim, RobsSpace(:,cc)-mean(RobsSpace(:,cc)), nlags);
        thresh = sqrt(robustcov(sta(:)))*4;
        
        [sm,la] = bounds(sta(:));
        
        
        if plotit
            subplot(2,4,4, 'align')
            imagesc(sta)
            
            figure(3); clf
            
            plot(sta, 'Color', .5*[1 1 1 1])
            hold on
            plot(xlim, thresh*[1 1], 'r--')
            plot(xlim, -thresh*[1 1], 'r--')
        end
        
        sta_ = nan(size(sta));
        
        sta_(abs(sta)>thresh) = sta(abs(sta)>thresh);
        if plotit
            plot(sta_, 'r')%sta > thresh
        end
        
        if abs(sm) > la
            sta_ = sta_ ./ sm;
        else
            sta_ = sta ./ la;
        end
        
        sta_(isnan(sta_)) = 0;
        sta_ = max(sta_, 0);
        
        
        % find spatial RF
        [~, bestlag] = max(std(sta_,[],2));
        
        I = imgaussfilt(reshape(sta_(bestlag,:), opts.dims),.5);
        srf = reshape(sta(bestlag,:), opts.dims);
        
        [x0,y0] = radialcenter(I);
        
        unit_mask = exp(-hypot((1:opts.dims(2)) - x0, (1:opts.dims(1))' - y0)) .* srf;
        
        if plotit
            figure(5); clf
            imagesc(unit_mask); hold on
            
            plot(x0, y0, '.r')
        end
        
        Stmp.sta = sta;
        Stmp.thresh = thresh;
        Stmp.best_lag = bestlag;
        Stmp.unit_mask = unit_mask;
        Stmp.srf = srf;
        Stmp.xax = opts.xax;
        Stmp.yax = opts.yax;
        Stmp.x0 = interp1(1:opts.dims(2), opts.xax, x0);
        Stmp.y0 = interp1(1:opts.dims(1), opts.yax, y0);
    else
        Stmp.best_lag = nan;
        try
            Stmp.unit_mask = zeros(opts.dims);
            Stmp.srf = zeros(opts.dims);
            Stmp.xax = opts.xax;
            Stmp.yax = opts.yax;
        catch
            Stmp.unit_mask = nan;
            Stmp.srf = nan;
            Stmp.xax = nan;
            Stmp.yax = nan;
        end
        Stmp.x0 = nan;
        Stmp.y0 = nan;
        
    end
    
    spkS = [spkS; Stmp]; %#ok<AGROW>
end



function plotWaveforms(W)
    
    if ~isfield(W, 'waveform') && isfield(W, 'osp')
        W = io.get_waveform_stats(W.osp);
    end


%     figure; clf
    NC = numel(W);
    cmap = lines(NC);
    
    SU = false(NC,1);
    
    for cc = 1:NC
        nw = norm(W(cc).waveform(:,3));
        SU(cc) = nw > 20 & W(cc).isiL > .1 & W(cc).isiL < 1 & W(cc).uQ > 5 & W(cc).isi(200) < 0;
        
        nts = size(W(1).waveform,1);
        xax = linspace(0, 1, nts) + cc;
        if SU(cc)
            clr = cmap(cc,:);
        else
            clr = .5*[1 1 1];
        end
        plot(xax, W(cc).waveform + W(cc).spacing + W(cc).depth, 'Color', clr, 'Linewidth', 2); hold on
        text(mean(xax),  W(cc).spacing(end)+20 + W(cc).depth, sprintf('%d', cc))
        
    end
    
    xlabel('Unit #')
    ylabel('Depth')
    plot.fixfigure(gcf, 12, [8 4], 'OffsetAxes', false)




function [isviz,pval,stats] = ismodulated(stimFr, isiFr, validTrials)
sfr = stimFr(validTrials(1:end-1));
ifr = isiFr(validTrials(1:end-1));
% paired wilcoxon sign-rank test for significant difference in median firing rate
[pval, isviz, stats] = signrank(sfr, ifr, 'alpha', 0.05/numel(sfr)); % scale alpha by the number of trials (more trials, more conservative)
