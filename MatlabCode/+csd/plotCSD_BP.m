function plotCSD_BP(stats, gamma, onlyPlotOneShank)
% plots gamam power landmarks (such as input layer prediction) on top  of
% CSD for verification
% Inputs:
%   stats              [struct] - stats struct from csd.getCSD
%   gamma              [struct] - gamma from csd.getGamma
%   onlyPlotOneShank   true/false - only plots one shank if more than one
%   (default = false)
% 
% ghs wrote it 2020

    if nargin<3
        onlyPlotOneShank = false;
    end
    
    if onlyPlotOneShank 
        numShanks = 1;
    else
        numShanks = size(stats.CSD, 3);
    end
    

    if numShanks == 1
        %figure; clf;
        curCSD = stats.CSD(:,:,1);
        imagesc(stats.time, stats.depth, curCSD-mean(curCSD(:))); axis ij
        %colormap(colormap_redblackblue);
        hold on
        plot(stats.time, bsxfun(@plus, stats.STA(:,:,1), stats.chDepths), 'Color', repmat(.1, 1, 3))
        xlim(stats.time([1 end]))
        %plot(stats.time([1 end]), stats.sinkDepth(1)*[1 1], 'w--', 'Linewidth', 2)
        if ~isempty(stats.reversalPointDepth{1})
            h3 = plot(stats.time([1 end]), [1; 1]*stats.reversalPointDepth{1}(1), 'r--', 'Linewidth', 2);
        end
        %tmp = abs(stats.reversalPointDepth{1} - stats.sinkDepth(shankInd));
        %tmp = tmp + stats.sinkDepth(shankInd);
        %plot(stats.time([1 end]), [1; 1]*tmp, 'r--', 'Linewidth', 2)
        
        xLimits = get(gca,'XLim');
        scale = xLimits(2);
        h1 = plot(gamma.lgPower(:,1).*scale, stats.chDepths, 'Color', 'black', 'Linewidth', 1);
        %h(2) = plot(gamma.hgPower(:,shankInd).*scale, stats.chDepths, 'Color', 'green', 'Linewidth', 1);
        
        h2 = plot(stats.time([1 end]), [1; 1]*gamma.lgInputLayerDepths(:,:,1), 'black--', 'Linewidth', 2);
        %plot(stats.time([1 end]), [1; 1]*gamma.lgMinDepth(shankInd), 'w--', 'Linewidth', 2)
        
        %                 axis ij
        xlabel('Time (ms)')
        ylabel('Depth (nm)')
        title(['Shank #' num2str(1)])
        hold off
    elseif numShanks > 1
        %figure; clf;
        for shankInd = 1:numShanks
        %curShankInds = shankInd*lenShanks-lenShanks+1:shankInd*lenShanks;
        curCSD = stats.CSD(:,:,shankInd);
        subplot(1,numShanks,shankInd)
        imagesc(stats.time, stats.depth, curCSD-mean(curCSD(:))); axis ij
        %colormap(colormap_redblackblue);
        hold on
        plot(stats.time, bsxfun(@plus, stats.STA(:,:,shankInd), stats.chDepths), 'Color', repmat(.1, 1, 3))
        xlim(stats.time([1 end]))
        %plot(stats.time([1 end]), stats.sinkDepth(shankInd)*[1 1], 'w--', 'Linewidth', 2)
        if ~isempty(stats.reversalPointDepth{shankInd})
            h3 = plot(stats.time([1 end]), [1; 1]*stats.reversalPointDepth{shankInd}(1), 'r--', 'Linewidth', 2);
        end
        %tmp = abs(stats.reversalPointDepth{1} - stats.sinkDepth(shankInd));
        %tmp = tmp + stats.sinkDepth(shankInd);
        %plot(stats.time([1 end]), [1; 1]*tmp, 'r--', 'Linewidth', 2)
        
        xLimits = get(gca,'XLim');
        scale = xLimits(2);
        h1 = plot(gamma.lgPower(:,shankInd).*scale, stats.chDepths, 'Color', 'black', 'Linewidth', 1);
        %h(2) = plot(gamma.hgPower(:,shankInd).*scale, stats.chDepths, 'Color', 'green', 'Linewidth', 1);
        
        h2 = plot(stats.time([1 end]), [1; 1]*gamma.lgInputLayerDepths(:,:,shankInd), 'black--', 'Linewidth', 2);
        %plot(stats.time([1 end]), [1; 1]*gamma.lgTroughDepth(shankInd), 'w--', 'Linewidth', 2)
        
        %                 axis ij
        xlabel('Time (ms)')
        ylabel('Depth (nm)')
        title(['Shank #' num2str(shankInd)])
        hold off

        end
        
    end
    colorbar
        legend([h3; h1; h2], 'CSD reversal point', 'low gamma power', 'low G input layer depths')
end

