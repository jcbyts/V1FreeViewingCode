function plotGammaLayerPrediction(lfp, Exp, varargin)
% Plots gamma layer prediction on top of CSD for verification
% Inputs: lfp & Expstructs from io.dataFactoryGratingSubspace

ip = inputParser();
ip.addParameter('onlyPlotOneShank', false)
ip.addParameter('method', 'weightedMin')
ip.parse(varargin{:});

gamma = csd.getGamma(lfp, 'method', ip.Results.method);

stats = csd.getCSD(lfp, Exp, 'window', [-100 200], 'plot', false, 'method',...
        'spline', 'sampleRate', 1000, 'exclude', true);

csd.plotCSD_BP(stats, gamma, ip.Results.onlyPlotOneShank)




