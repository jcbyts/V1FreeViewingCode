function stats = getCSD(lfp, eventTimes, varargin)
% CSDBASIC computes the current source density
% Inputs:
%   lfp              [struct] - lfp struct from io.dataFactoryGratingSubspace
%   eventTimes       [nEvents x 1] - timestamps of the events
%                                --- OR ---
%                    [struct] - Exp struct from io.dataFactoryGratingSubspace
%
% optional arguments (as argument pairs):
%   'channelDepths'  [nChannels x 1] - array of channel depths
%   'window'         [1 x 2]         - start and stop of analysis window
%                                      (aligned to event time)
%   'plot'           logical         -  plot if (default: true)
%   'method'         string          - csd method (default: 'spline')
%
%   'sampleRate'     int             - sampleRate of LFP
%
%   'exclude'        logical         - exclude lfp.deadChan channels and
%                                      interpolate to get them
%
% valid csd methods:
%       'standard' - second spatial derivative
%       'step'     - stepwise inverse method (not really sure)
%       'spline'   - interpolated inverse CSD
%
% 2017 jly wrote it
% 2020 ghs edit it

ip = inputParser();
ip.addParameter('window', [-100 200])
ip.addParameter('channelDepths', [])
ip.addParameter('plot', true)
ip.addParameter('method', 'spline')
ip.addParameter('sampleRate', 1000)
ip.addParameter('exclude', true)
ip.parse(varargin{:});
exclude = ip.Results.exclude;

if exclude % excluded channels
    excChan = str2num(lfp.deadChan);
end


if isa(eventTimes, 'double')
    eventTimes = eventTimes(:);
elseif isa(eventTimes, 'struct') % Get CSD event times (if not already input)
    eventTimes = csd.getCSDEventTimes(eventTimes);
    eventTimes = eventTimes(:);
else
    error('eventTimes input must be double or struct')
end

if isempty(eventTimes)
    error('eventTimes empty')
end

[~,~,ev] = histcounts(eventTimes, lfp.timestamps); % convert time to samples

if isempty(lfp.ycoords)
    ch0_all = ((-32:-1)*50)';
else
    ch0_all = lfp.ycoords; % row vector
end

numShanks = size(ch0_all, 2); % number of shanks
lenShanks = size(ch0_all, 1); % number of channels on each shank
for shankInd = 1:numShanks
    ch0 = ch0_all(:,shankInd);
    
    curShankInds = shankInd*lenShanks-lenShanks+1:shankInd*lenShanks;
    
    % event-triggered LFP
    [sta,~, time] = eventTriggeredAverage(lfp.data(:,curShankInds), ev(:), ip.Results.window);
    
    if exclude
        curDeadChan = excChan(excChan>=curShankInds(1)&excChan<=curShankInds(end));
        curDeadChan = curDeadChan - (shankInd-1)*lenShanks; 
        
        for indDChan = 1:length(curDeadChan)% interpolate dead channels
            DChan = curDeadChan(indDChan);
            abovebelow = [sta(:,DChan-1) sta(:,DChan+1)];
            vq = interp1(abovebelow',1.5, 'linear');
            sta(:,DChan) = vq;
        end
        
    end
    
    switch ip.Results.method
        case 'spline'
            % compute the CSD using the spline inverse method
            CSD = csd.splineCSD(sta', 'el_pos', ch0);
        case 'standard'
            CSD = csd.standardCSD(sta', 'el_pos', ch0);
        case 'step'
            CSD = csd.stepCSD(sta', 'el_pos', ch0);
        otherwise
            error('valid methods are {spline, standard, step}')
    end
    
    % find the sink and reversal point
    ix = time > 0 & time < 100; % look over time window after flash
    
    % sink should be the minimum value
    [~,id] = min(reshape(CSD(:,ix), [], 1));
    % convert to indices
    [depthIndex,timeIndex] = ind2sub(size(CSD(:,ix)), id);
    % upsample channels to index into them
    chUp   = linspace(1, numel(ch0), size(CSD,1));
    depthUp= linspace(ch0(1), ch0(end), size(CSD,1));
    
    % find reversal point
    CSD_ = CSD(:,ix);
    reversalPoints = findZeroCrossings(CSD_(:,timeIndex));
    
    % output structure
    stats.STA(:,:,shankInd) = sta';
    stats.CSD(:,:,shankInd) = CSD;
    stats.reversalPointDepth{shankInd} = depthUp(reversalPoints);
    stats.sinkDepth(shankInd) = depthUp(depthIndex);
    stats.sinkChannel(shankInd) = chUp(depthIndex);
end

% Output
stats.time  = time;
stats.depth = depthUp;
stats.chDepths = ch0;
stats.chUp  = chUp;
stats.numShanks = numShanks;

if ip.Results.plot % afterall, it is a plot function
    
    csd.plotCSD(stats) % Plot CSD
    
end
end

function i = findZeroCrossings(data, mode)
%FINDZEROCROSSINGS Find zero crossing points.
%   I = FINDZEROCROSSINGS(DATA,MODE) returns the indicies into the supplied
%   DATA vector, corresponding to the zero crossings.
%
%   MODE specifies the type of crossing required:
%     MODE < 0 - results in indicies for the -ve going zero crossings,
%     MODE = 0 - results in indicies for ALL zero crossings (default), and
%     MODE > 0 - results in indicies for the +ve going zero crossings.

% $Id: findZeroCrossings.m,v 1.1 2008-07-21 23:31:50 shaunc Exp $

if nargin < 2
    mode = 0;
end

[i,~,p] = find(data); % ignore zeros in the data vector

switch sign(mode)
    case -1
        % find -ve going crossings
        ii = find(diff(sign(p))==-2);
    case 0
        % find all zero crossings
        ii = find(abs(diff(sign(p)))==2);
    case 1
        % find +ve going crossings
        ii = find(diff(sign(p))==2);
end;

i = round((i(ii)+i(ii+1))/2);
end
