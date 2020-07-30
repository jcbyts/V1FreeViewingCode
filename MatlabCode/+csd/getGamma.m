function gamma = getGamma(lfp, varargin)
% Gets low gamma (20-60Hz) and high gamma power (80-200Hz) used for 
% identifying input layer on probe
%
%   Input:
%   lfp              [struct] - lfp struct from io.dataFactoryGratingSubspace
%                                   -- OR --
%                    [double] - lfp data that you want to get power from                                  
%
%   Parser inputs:
%       - sampleRate: sampling rate Hz (default: 1000)
%       - plotIt: true/false - plots minimum verificiation (default: true)
%       - inds: indices of lfp to compute power (default: [100000 200000])
%                   - note: leave inds empty ([]) if want entire signal
%       - exclude: true/false - exclude and interpolate bad channels
%       - method: either 'softmax' or 'weightedMin' - method for finding
%       low gamma trough
%               - softmax takes power softmax (x.^p/sum(x.^p)) and takes weighted average of this
%                   function to get trough value 
%               - weightMin finds minimum of interpolated power below the
%                   max value (low gamma trough is always below a peak in
%                   input layer - similar to high gamma peak)
%
%   Output:
%       - gamma strcut containing:
%                  - lgPower: low gamma power at each channel (nChans x
%                  nShanks)
%                  - hgPower: high gamma power at each channel (nChans x
%                  nShanks)
%                  - lgMinDepth: depth on probe of low gamma trough 
%                  - hgMinDepth: depth on probe of high gamma peak
%                  - lgInputLayerDepths: input layer depth based on low
%                  gamma trough (usually more accurate)
%                  - hgInputLayerDepths: input layer depth based on high
%                  gamma peak 
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
ip.addParameter('method', 'weightedMin')
ip.parse(varargin{:});

method = ip.Results.method; % method for fidning low gamma trough - NOTE: hard coded

numShanks = size(lfp.ycoords, 2);
lenShanks = size(lfp.ycoords, 1);

lgPower = csd.getLFPPower(lfp, 'low gamma', 'sampleRate', ip.Results.sampleRate, 'plotIt', false,...
    'norm', true, 'inds', ip.Results.inds, 'exclude', ip.Results.exclude); % compute
lgPower = reshape(lgPower, [lenShanks numShanks]);

hgPower = csd.getLFPPower(lfp, 'high gamma', 'sampleRate', ip.Results.sampleRate, 'plotIt', false,...
    'norm', true, 'inds', ip.Results.inds, 'exclude', ip.Results.exclude);
hgPower = reshape(hgPower, [lenShanks numShanks]);


ycoords = lfp.ycoords(:,1); % asssuming ycoords are same for all shanks
lgTroughDepth = nan(1, numShanks);
hgMaxDepth = nan(1, numShanks);
lgInputLayerDepths = nan(1, 2, numShanks);
hgInputLayerDepths = nan(1, 2, numShanks);
reversal = nan(1, numShanks);

for shankInd = 1:numShanks
    
    curShankInds = shankInd*lenShanks-lenShanks+1:shankInd*lenShanks;
    
    currLGPower = lgPower(curShankInds);
    

    % Use softmax function (instead of exponential use power p) to find
    % trough
    if strcmp(method, 'softmax')
        disp('Method: softmax')
        p = 5; % power taken for softmax equation (not using exp)
        x = currLGPower;
        x = 1-x;
        x = (x.^p/sum(x.^p)); % apply power softmax
        
        % get minimum of softmax power
        [~,lgTroughDepth(shankInd)] = max(x);
        lgTroughDepth(shankInd) = ycoords(lgTroughDepth(shankInd));
        
        if ip.Results.plotIt
            figure(30); clf
            plot(ycoords, x); hold on; 
            plot(lgTroughDepth(shankInd)*[1 1], ylim, 'r');
            plot(ycoords, currLGPower); hold off
            legend('softmax', 'softmax trough', 'original lg power')
            title(['lg softmax, depth=' num2str(lgTroughDepth(shankInd))])
        end
        
    % Take weighted average of channels with power one std above mean power
    % Find corresponding depth of this weighted average
    % search for minimum only below this depth by using softmax  
    elseif strcmp(method, 'weightedMin')
        disp('Method: weighted min')
        meanlgpower = mean(lgPower(curShankInds));
        stdlgpower = std(lgPower(curShankInds));
        threshLGPower = currLGPower(lgPower(curShankInds)>meanlgpower+stdlgpower);
        threshLGCoords = ycoords(lgPower(curShankInds)>meanlgpower+stdlgpower);
        if size(currLGPower, 2) == 1 % Needed for logan sessions only
            threshLGPower = threshLGPower';
        end
        weightedmax = threshLGPower*threshLGCoords;
        weightedmax = weightedmax / sum(threshLGPower); % find weighted max depth of power
        lgplot = currLGPower; % for plotting purposes
        
%         p = 5; % power taken for softmax equation (not using exp)
%         x = currLGPower;
%         %x = 1-x;
%         x = (x.^p/sum(x.^p)); % apply power softmax
%         weightedmax = x*ycoords;
        
        currLGPower(ycoords<weightedmax) = 1; % only search for min below weighted max depth
        
        % Interpolate for better estimate
        ycoords_new = linspace(min(ycoords), max(ycoords), 400);
        interpLgPower = interp1(ycoords,currLGPower,ycoords_new, 'spline');
        
        % get minimum of interpolated power
        [~,lgTroughDepth(shankInd)] = min(interpLgPower);
        lgTroughDepth(shankInd) = ycoords_new(lgTroughDepth(shankInd));
        
        % plot it
        if ip.Results.plotIt
            figure(29); clf
            plot(ycoords_new, interpLgPower); hold on; 
            plot(lgTroughDepth(shankInd)*[1 1], ylim, 'r');
            plot(ycoords, lgplot); 
            plot(weightedmax*[1 1], ylim, 'r'); hold off
            legend('interp power', 'min trough', 'original lg power', 'weightedmax')
            title(['lg weighted min, depth=' num2str(lgTroughDepth(shankInd))])
        end
    else
        error('Wrong method input')
    end
    
    [~,hgMaxDepth(shankInd)] = max(hgPower(curShankInds));
    hgMaxDepth(shankInd) = ycoords(hgMaxDepth(shankInd));
    
    lgInputLayerDepths(:,:,shankInd) = [lgTroughDepth(shankInd)-lgDist lgTroughDepth(shankInd)-lgDist-inputLayerDist];
    hgInputLayerDepths(:,:,shankInd) = [hgMaxDepth(shankInd)+hgDist hgMaxDepth(shankInd)+hgDist-inputLayerDist];
    reversal(shankInd) = lgTroughDepth(shankInd)-lgDist;
end

% See documentation for explanations of outputs
gamma.lgPower = lgPower;
gamma.hgPower = hgPower;
gamma.lgTroughDepth = lgTroughDepth;
gamma.hgMaxDepth = hgMaxDepth;
gamma.lgInputLayerDepths = lgInputLayerDepths; 
gamma.hgInputLayerDepths = hgInputLayerDepths; 
gamma.reversal = reversal;




