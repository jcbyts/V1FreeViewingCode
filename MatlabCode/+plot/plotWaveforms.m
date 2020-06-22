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