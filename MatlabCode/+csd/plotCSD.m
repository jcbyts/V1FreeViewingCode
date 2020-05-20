function plotCSD(stats)
% gets the event times of the current source density trials
% Inputs:
%   stats              [struct] - stats struct from csd.getCSD
% 
% ghs wrote it 2020

    numShanks = size(stats.CSD, 3);
    

    if numShanks == 1
        figure; clf;
        imagesc(stats.time, stats.chDepths, stats.CSD-mean(stats.CSD(:))); axis xy
        colormap jet
        hold on
        plot(stats.time, bsxfun(@plus, stats.STA, stats.chDepths), 'Color', repmat(.1, 1, 3))
        xlim(stats.time([1 end]))
        plot(stats.time([1 end]), stats.sinkDepth(1)*[1 1], 'w--', 'Linewidth', 2)
        plot(stats.time([1 end]), [1; 1]*stats.reversalPointDepth{1}, 'r--', 'Linewidth', 2)
        tmp = abs(stats.reversalPointDepth{1} - stats.sinkDepth(1));
        tmp = tmp + stats.sinkDepth(1);
        plot(stats.time([1 end]), [1; 1]*tmp, 'r--', 'Linewidth', 2)
        %                 axis ij
        hold off
        colorbar
    elseif numShanks > 1
        figure; clf;
        for shankInd = 1:numShanks
        %curShankInds = shankInd*lenShanks-lenShanks+1:shankInd*lenShanks;
        curCSD = stats.CSD(:,:,shankInd);
        subplot(1,numShanks,shankInd)
        imagesc(stats.time, stats.chDepths, curCSD-mean(curCSD(:))); axis xy
        colormap jet
        hold on
        plot(stats.time, bsxfun(@plus, stats.STA(:,:,shankInd), stats.chDepths), 'Color', repmat(.1, 1, 3))
        xlim(stats.time([1 end]))
        plot(stats.time([1 end]), stats.sinkDepth(shankInd)*[1 1], 'w--', 'Linewidth', 2)
        plot(stats.time([1 end]), [1; 1]*stats.reversalPointDepth{shankInd}, 'r--', 'Linewidth', 2)
        tmp = abs(stats.reversalPointDepth{1} - stats.sinkDepth(shankInd));
        tmp = tmp + stats.sinkDepth(shankInd);
        plot(stats.time([1 end]), [1; 1]*tmp, 'r--', 'Linewidth', 2)
        %                 axis ij
        hold off
        colorbar
        end
    end
end
