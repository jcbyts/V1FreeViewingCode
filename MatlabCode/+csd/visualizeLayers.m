function visualizeLayers(sessNums)
% Plotting function that allows you to visualize layer analyses of multiple
% sessions (plots CSD and BP with landmarks)
%   input:
%        sessNums          [1xnumSessions] vector of session numbers

dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');

meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');

data = readtable(meta_file);

shankLen = 32; % number of channels on each shank
numPlots = sum(table2array(data(sessNums, 16)))/shankLen;

di = 1:numPlots;
di = di(~(rem(numPlots, di)));
if mod(length(di),2) == 0
    num_vert = di(floor(length(di)/2));
    num_hor = di(floor(length(di)/2)+1);
else
    num_vert = median(di);
    num_hor = num_vert;
end

figure;
h = gcf; clf;
gap = 0.005;
ha = csd.tight_subplot(num_vert,num_hor, [gap gap],[gap gap],[gap gap]);

plotInd = 1;
for indSess = 1:length(sessNums)
    [Exp, ~, lfp] = io.dataFactoryGratingSubspace(sessNums(indSess));
    
    stats = csd.getCSD(lfp, Exp, 'window', [-100 200], 'plot', false, 'method',...
        'spline', 'sampleRate', 1000, 'exclude', true); % compute CSD
    
    gamma = csd.getGamma(lfp);
    
    numShanks = size(stats.CSD, 3);
    for shankInd = 1:numShanks
        
        axes(ha(plotInd))
        
        curCSD = stats.CSD(:,:,shankInd);
        imagesc(stats.time, stats.depth, curCSD-mean(curCSD(:))); axis ij
        colormap(colormap_redblackblue);
        hold on
        plot(stats.time, bsxfun(@plus, stats.STA(:,:,shankInd), stats.chDepths), 'Color', repmat(.1, 1, 3))
        xlim(stats.time([1 end]))
        plot(stats.time([1 end]), stats.sinkDepth(shankInd)*[1 1], 'w--', 'Linewidth', 2)
        if ~isempty(stats.reversalPointDepth{shankInd})
            plot(stats.time([1 end]), [1; 1]*stats.reversalPointDepth{shankInd}, 'r--', 'Linewidth', 2)
        end
        tmp = abs(stats.reversalPointDepth{1} - stats.sinkDepth(shankInd));
        tmp = tmp + stats.sinkDepth(shankInd);
        if ~isempty(tmp)
            plot(stats.time([1 end]), [1; 1]*tmp, 'r--', 'Linewidth', 2)
        end
        xLimits = get(gca,'XLim');
        scale = xLimits(2);
        shankInds = (shankLen*(shankInd-1)+1):(shankLen*shankInd);
        
        plot(gamma.lgPower(:,shankInd).*scale, stats.chDepths, 'Color', 'white', 'Linewidth', 3)
        plot(gamma.hgPower(:,shankInd).*scale, stats.chDepths, 'Color', 'green', 'Linewidth', 3)
        plot(stats.time([1 end]), [1; 1]*gamma.lgInputLayerDepths(:,:,shankInd), 'white', 'Linewidth', 2)
        plot(stats.time([1 end]), [1; 1]*gamma.hgInputLayerDepths(:,:,shankInd), 'green', 'Linewidth', 2)

        
        hold off
        
        yticks(35:35:1120)
        yticklabels({'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'})
        title([num2str(sessNums(indSess)) ',' num2str(shankInd)])
        
        plotInd = plotInd + 1;
        
    end
    
end

colorbar

