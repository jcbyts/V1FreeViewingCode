function visualizeLayers(sessNums, varargin)
% Plotting function that allows you to visualize layer analyses of multiple
% sessions (plots CSD and BP with landmarks)
%   input:
%        sessNums          [1xnumSessions] vector of session numbers

ip = inputParser();
ip.addParameter('gamma_method', 'weightedMin')
ip.addParameter('CSD_method', 'standard')
ip.parse(varargin{:});

gamma_method = ip.Results.gamma_method;
CSD_method = ip.Results.CSD_method;

dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');

meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');

data = readtable(meta_file);

numPlots = length(sessNums);

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
ha = csd.tight_subplot(num_vert,num_hor, [0.04 gap],[gap 0.04],[gap gap]);

for indSess = 1:length(sessNums)
    [Exp, ~, lfp] = io.dataFactoryGratingSubspace(sessNums(indSess));
    
    axes(ha(indSess))
    %csd.plotGammaLayerPrediction(lfp, Exp, 'method', method, 'onlyPlotOneShank', true)
    
    gamma = csd.getGamma(lfp, 'method', gamma_method);

    stats = csd.getCSD(lfp, Exp, 'window', [-100 200], 'plot', false, 'method',...
        CSD_method, 'sampleRate', 1000, 'exclude', true);

    csd.plotCSD_BP(stats, gamma, ip.Results.onlyPlotOneShank)
    
    colorbar off
    
    if ~(indSess==length(sessNums))
        set(gca,'xticklabel',{[]})
        set(gca,'yticklabel',{[]})
    end
    title(strcat(string(data{sessNums(indSess), 2}), '-', string(data{sessNums(indSess), 1})))
    
    
end

colorbar

