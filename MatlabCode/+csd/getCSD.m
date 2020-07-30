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
%   'spatsmooth'     double          - apply spatial smoothing to sta
%                                      before computing CSD
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
ip.addParameter('plotIt', true)
ip.addParameter('method', 'standard')
ip.addParameter('sampleRate', 1000)
ip.addParameter('exclude', true)
ip.addParameter('spatsmooth', 1.5)
ip.parse(varargin{:});

exclude = ip.Results.exclude;
excChan = lfp.deadChan;

if isempty(eventTimes)
    stats = struct();
    stats.time = ip.Results.window(1):ip.Results.window(2);
    stats.STA = nan(32, numel(stats.time));
    stats.CSD = nan(30, numel(stats.time));
    stats.latency = nan;
    stats.reversalPointDepth{1} = nan;
    stats.sinkDepth = nan;
    stats.sourceDepth = nan;
    stats.sinkChannel=nan;
    stats.sourceChannel = nan;
    stats.depth = nan(32,1);
    stats.depth = nan(32,1);
    stats.depth = nan(32,1);
    stats.numShanks = 0;
    return
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
    ch0_all = ((32:-1:1)*50)';
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
    
    %     if ip.Results.spatsmooth > 0
    %         for t = 1:size(sta,1)
    %             sta(t,:) = imgaussfilt(sta(t,:), ip.Results.spatsmooth);
    %         end
    %     end
    
    switch ip.Results.method
        case 'spline'
            % compute the CSD using the spline inverse method
            CSD = csd.splineCSD(sta', 'el_pos', ch0);
        case 'standard'
            
            sta = -1.*sta;
            
                        %spatial smoothing
                        if ip.Results.spatsmooth > 0
                            for t = 1:size(sta,1)
                                sta(t,:) = imgaussfilt(sta(t,:), ip.Results.spatsmooth);
                            end
                        end
            
%             % spatial smoothing
%             if ip.Results.spatsmooth > 0
%                 for t = 1:size(sta,1)
%                     sta(t,:) = smooth(sta(t,:),11,'lowess');
%                 end
%             end
%             
%             % temporal smoothing
%             if ip.Results.tempsmooth > 0
%                 for ch = 1:size(sta,2)
%                     sta(:,ch) = smooth(sta(:,ch),11,'sgolay');
%                 end
%             end
            
            
            
            % compute CSD
            CSD = diff(sta, 2, 2)';
            %             CSD = csd.standardCSD(sta', 'el_pos', ch0);
            
            %             % remove first and last depth since CSD takes derivative twice
            %             chSTA = ch0;
            %             ch0(1) = [];
            %             ch0(end) = [];
        case 'step'
            CSD = csd.stepCSD(sta', 'el_pos', ch0);
        otherwise
            error('valid methods are {spline, standard, step}')
    end
    
    
    % find the sink and reversal point
    
    %     figure(1); clf
    %     imagesc(CSD');
    
    
    tpower = std(CSD).^4.*-1;
    ix = time < 0;
    
    tpower = imgaussfilt(tpower, 3);
    
    tpower = fix(tpower/ (10*max(tpower(ix))));
    %     plot(tpower)
    
    dpdt = diff(tpower);
    
    inds = find(sign(dpdt)~=0); % remove indices that don't have a sign
    
    zc = find(diff(sign(dpdt))==-2);
    [~, ind] = sort(tpower(zc), 'descend');
    
    if length(zc) >= 3
        numP = 3;
    else
        numP = length(zc);
    end
    
    %numP = 3;
    
    %zc = sort(zc(ind(1:numP))); % three biggest peaks in order
    
    zc = sort(zc(ind(1:min(numel(ind), 3))));
    
    ch00 = ch0;
    if strcmp(ip.Results.method, 'spline')
        ch0 = imresize(ch0, size(CSD,1)/numel(ch0));
        ch0 = ch0(:,1);
    end
    
    if isempty(zc)
        stats.STA(:,:,shankInd) = sta';
        stats.CSD(:,:,shankInd) = CSD;
        stats.time  = time;
        stats.chDepths = ch00;
        stats.depth = ch0;
        stats.chUp  = ch0;
        stats.numShanks = numShanks;
        stats.latency = nan;
        stats.sinkDepth(shankInd) = nan;
        stats.sourceDepth(shankInd) = nan;
        stats.sinkChannel(shankInd) = nan;
        stats.sourceChannel(shankInd) = nan;
        
        % try to locate using other method
        ix = time > 0 & time < 100;
        % sink should be the minimum value
        [~,id] = min(reshape(CSD(:,ix), [], 1));
        % convert to indices
        [depthIndex,timeIndex] = ind2sub(size(CSD(:,ix)), id);
        % find reversal point
        CSD_ = CSD(:,ix);
        reversalPoints = findZeroCrossings(CSD_(:,timeIndex));
        stats.reversalPointDepth{shankInd} = reversalPoints(1);
        
        continue
        %return
    end
    
    spower = CSD(:,zc(1));
    
    % get peak
    [~, peaks] = findpeaks(spower, 'MinPeakWidth', 2, 'MinPeakHeight', .5);
    [~, vals] = findpeaks(-spower, 'MinPeakWidth', 2, 'MinPeakHeight', .5);
    
    mx = min(peaks);
    mn = min(vals);
    
    %     [~, mx] = max(spower);
    %     [~, mn] = min(spower);
    
    if mx > mn
        ind = mn:mx;
        rvrsl = ind(find(diff(sign(spower(ind)))==2, 1));
    else
        ind = mx:mn;
        rvrsl = ind(find(diff(sign(spower(ind)))==-2, 1));
    end
    
    t = time(zc(1));
    
    source = ch0(mx+1); % plus 1 because CSD cuts channel (due to derivative)
    sink = ch0(mn+1); % same here
    reversal = ch0(rvrsl+2); % plus 2 because we took the derivative again to

    if isempty(source)
        source = nan;
        mx = nan;
    end
    if isempty(sink)
        sink = nan;
        mn = nan;
    end
    
    if isempty(reversal)
        reversal = nan;
    end
    % output structure
    stats.STA(:,:,shankInd) = sta';
    stats.CSD(:,:,shankInd) = CSD;
    stats.latency = t;
    stats.reversalPointDepth{shankInd} = reversal;
    stats.sinkDepth(shankInd) = sink;
    stats.sourceDepth(shankInd) = source;
    stats.sinkChannel(shankInd) = mn+1;
    stats.sourceChannel(shankInd) = mx+1;
end

% Output
stats.time  = time;
stats.chDepths = ch00;
stats.chUp  = ch0;
stats.depth = ch0;
stats.numShanks = numShanks;

if ip.Results.plotIt
    
    figure(3); clf
    subplot(121)
    plot(time, tpower); hold on
    cmap = lines;
    for i = 1:numP
        plot(time(zc(i)), tpower(zc(i)), 'o', 'Color', cmap(i,:), 'Linewidth', 2)
    end
    xlim([0 150])
    
    figure(2); clf
    imagesc(time, ch0, CSD)
    hold on
    for i = 1:numP
        plot(time(zc(i))*[1 1], ylim, 'Linewidth', 2)
    end
    
    if exist('sink','var')
        plot(t, sink, 'or')
        plot(t, source, 'sr')
        plot(t, reversal, 'dr')
    end
    
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
