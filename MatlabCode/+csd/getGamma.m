function gamma = getGamma(lfp, varargin)
% Gets gamma
%
%   Input:
%   lfp              [struct] - lfp struct from io.dataFactoryGratingSubspace
%                                   -- OR --
%                    [double] - lfp data that you want to get power from                                  
%
%   Parser inputs:
%       - sampleRate: sampling rate Hz (default: 1000)
%       - plotIt: true/false - plot power dens (default: true)
%       - inds: indices of lfp to compute power (default: [1 100000])
%                   - note: leave inds empty ([]) if want entire signal
%       - exclude: true/false - exclude and interpolate bad channels
%   Output:
%       - power: average power density - 1 x nChan
%
% ghs wrote it 2019
% ghs edit it 2020

lgDist = 58; %nm distance below to reversal point (low gamma)
hgDist = 222; %nm distance above reversal point (high gamma)
inputLayerDist = 500; %nm length of input layer

ip = inputParser();
ip.addParameter('sampleRate', 1e3)
ip.addParameter('plotIt', false)
%ip.addParameter('norm', true)
ip.addParameter('inds', [100000 200000])
ip.addParameter('exclude', true)
ip.parse(varargin{:});

numShanks = size(lfp.ycoords, 2);
lenShanks = size(lfp.ycoords, 1);

lgPower = csd.getLFPPower(lfp, 'low gamma', 'sampleRate', ip.Results.sampleRate, 'plotIt', ip.Results.plotIt,...
    'norm', true, 'inds', ip.Results.inds, 'exclude', ip.Results.exclude); % compute
lgPower = reshape(lgPower, [lenShanks numShanks]);

hgPower = csd.getLFPPower(lfp, 'high gamma', 'sampleRate', ip.Results.sampleRate, 'plotIt', ip.Results.plotIt,...
    'norm', true, 'inds', ip.Results.inds, 'exclude', ip.Results.exclude);
hgPower = reshape(hgPower, [lenShanks numShanks]);


ycoords = lfp.ycoords(:,1); % asssuming ycoords are same for all shanks
lgMinDepth = nan(1, numShanks);
hgMaxDepth = nan(1, numShanks);
lgInputLayerDepths = nan(1, 2, numShanks);
hgInputLayerDepths = nan(1, 2, numShanks);

for shankInd = 1:numShanks
    
    curShankInds = shankInd*lenShanks-lenShanks+1:shankInd*lenShanks;
    
    currLGPower = lgPower(curShankInds);
    
    % Take weighted average of channels with power one std above mean power
    % Find corresponding depth of this weighted average
    % search for minimum only below this depth
    meanlgpower = mean(lgPower(curShankInds));
    stdlgpower = std(lgPower(curShankInds));
    threshLGPower = currLGPower(lgPower(curShankInds)>meanlgpower+stdlgpower);
    threshLGCoords = ycoords(lgPower(curShankInds)>meanlgpower+stdlgpower);
    tmp = threshLGPower*threshLGCoords;
    tmp = tmp / sum(threshLGPower);
    
    [~,lgMinDepth(shankInd)] = min(currLGPower(ycoords>tmp));
    lgMinDepth(shankInd) = ycoords(lgMinDepth(shankInd));
    
    [~,hgMaxDepth(shankInd)] = max(hgPower(curShankInds));
    hgMaxDepth(shankInd) = ycoords(hgMaxDepth(shankInd));
    
    lgInputLayerDepths(:,:,shankInd) = [lgMinDepth(shankInd)-lgDist lgMinDepth(shankInd)-lgDist-inputLayerDist];
    hgInputLayerDepths(:,:,shankInd) = [hgMaxDepth(shankInd)+hgDist hgMaxDepth(shankInd)+hgDist-inputLayerDist];
    
end

gamma.lgPower = lgPower;
gamma.hgPower = hgPower;
gamma.lgMinDepth = lgMinDepth;
gamma.hgMaxDepth = hgMaxDepth;
gamma.lgInputLayerDepths = lgInputLayerDepths;
gamma.hgInputLayerDepths = hgInputLayerDepths;




