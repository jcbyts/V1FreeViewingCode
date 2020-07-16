function W = get_waveform_stats(osp, varargin)
% get waveform details
% Inputs:
%   osp <struct> the spike struct from getSpikesFromKilo
% Outputs:
%   w <struct array> array of structs with info about each unit
%       cid             Unit Id
%       depth           Depth along probe
%       uQ              Unit Quality Metric (from getSpikesFromKilo)
%       isiV            ISI violation rate
%       x               X position along probe(s) 
%       lags            Time lags for ISI distribution
%       isi             ISI distribution (in excess spikes/sec)
%       isiE            2 x SE on ISI distribution
%       isiL            A crude measure of line noise
%       peaktime        Time of the peak 
%       peakval         Value at peaktime
%       troughtime      Time of the trough
%       troughval       Value at troughtime
%       waveform        Waveform at "spacing" centered on Unit
%       spacing         spacing to calculate waveform

ip = inputParser();
ip.addParameter('binSize', 1e-3)
ip.addParameter('numLags', 200)
ip.addParameter('spacing', [-100 -50 0 50 100])
ip.addParameter('validEpochs', [])
ip.parse(varargin{:});

S = io.getUnitLocations(osp);

binSize = ip.Results.binSize;
numLags = ip.Results.numLags;
spacing = ip.Results.spacing;

NC = numel(osp.cids);

W = repmat(struct('cid', [], ...
    'depth', [], ...
    'uQ', [], ...
    'isiV', [], ...
    'x', [], ...
    'lags', [], ...
    'isi', [], ...
    'isiE', [], ...
    'isiL', [], ...
    'isiRate', [], ...
    'localityIdx', [], ...
    'peaktime', [], ...
    'peakval', [], ...
    'troughtime', [], ...
    'troughval', [], ...
    'waveform', [], ...
    'shiftlags', [], ...
    'shiftwaveform', [], ...
    'BRI', [], ...
    'normrate', [], ...
    'spacing', []), NC, 1);

% bin spike times
binEdges = min(osp.st):binSize:max(osp.st);
if ~isempty(ip.Results.validEpochs)
    nepochs = size(ip.Results.validEpochs,1);
    vix = false(1, numel(binEdges)-1);
    for epoch = 1:nepochs
        vix(binEdges(1:end-1) >= ip.Results.validEpochs(epoch,1) & binEdges(2:end) < ip.Results.validEpochs(epoch,2)) = true;
    end
else
    vix = true(1, numel(binEdges)-1);
end
        
    
for cc = 1:NC

    cid = osp.cids(cc);

    sptimes = osp.st(osp.clu==cid);
    
    if ~isempty(ip.Results.validEpochs)
        vixsp = getTimeIdx(sptimes, ip.Results.validEpochs(:,1), ip.Results.validEpochs(:,2));
        sptimes = sptimes(vixsp);        
    end

    
    [spcnt, ~, id] = histcounts(sptimes, binEdges);
    
    
    % get autocorrelation in units of excess firing rate
    lags = -numLags:numLags;
    ix = id + lags;
    ix(any(ix<1 | ix > numel(spcnt),2),:) = [];
    I = spcnt(ix);
    I(:,lags==0)= 0; % remove zero-lag spike
    
    mu = mean(I);
    mu0 = mean(spcnt(vix));
        
    % binomial confidence intervals
    n = size(I,1);
    binoerr = 2*sqrt( (mu - mu.^2)/n);

    xc = (mu - mu0) / binSize; % in excess spikes/sec
    
    % push error through same nonlinearity ( baseline subtract / binsize)
    err = ((mu + binoerr) - mu0)/binSize - xc;
    
    % rate in the refractory period (1ms after spike)
    refrate = xc(lags==1);
    
    % expected rate using shoulders of the autocorrelation as a baseline
    expectedminrate = mean(xc([1 end])) - mean(err([1 end]));
    
    normrate = mu/mu0;
    
    % burst-refractoriness index
    BRI = mean(normrate(lags>=1 & lags<=4));
        
    % interpolate waveform around the center of mass (handles different
    % electrode spacing / centers)
    nsp = numel(spacing);
    nts = size(S.templates, 1);
    wf = zeros(nts, nsp);
    nshanks = numel(unique(S.xcoords));
    if S.useWF
        offset = 1;
    else
        offset = 20;
    end
    
    for i = 1:nts
        
        if nshanks > 1
            mask = abs(S.x(cc) - S.xcoords) < mean(diff(unique(S.xcoords)))/2;
        else
            mask = true(numel(S.ycoords),1);
        end
        
        if sum(mask) >= numel(spacing)
            wf(i,:) = interp1(S.ycoords(mask), squeeze(S.templates(i, mask, cc)), S.y(cc)+spacing);
        else
            wf_ = squeeze(S.templates(i, mask, cc));
            wf(i,1:numel(wf_)) = wf_(:);
        end
    end

    % center waveform on trough
    wf = wf(offset:end,:);
    nts = size(wf,1);
    if S.useWF
        ts = osp.WFtax/1e3;
        centeredTimestamps = (-15:30)/30e3;
    else
        ts = (1:nts)/30e3; % NOTE: sampling rate is hard coded here    
        % shift waveform aligned to maximum excursion (peak or trough)
        centeredTimestamps = (-20:40)/30e3;
    end
    
    %--- find peak and trough
    dwdt = diff(wf(:,3));
    
    % get trough
    troughloc = findZeroCrossings(dwdt, 1);
    troughloc = troughloc(ts(troughloc) < .35/1e3); % only troughs before .35 ms after clipping
    tvals = wf(troughloc,3);
    [~, mxid] = min(tvals);
    troughloc = ts(troughloc(mxid));

    % get peak
    peakloc = findZeroCrossings(dwdt, -1);
    peakloc = peakloc(ts(peakloc) < .65/1e3 & ts(peakloc) > -.25/1e3); % only troughs before .35 ms after clipping
    pvals = wf(peakloc,3);
    [~, mxid] = max(pvals);
    peakloc = ts(peakloc(mxid));
    
    if isempty(peakloc)
        peakloc = nan;
    end
    
    if isempty(troughloc)
        troughloc = 0;
    end
    
    % do the centering
    wnew = zeros(numel(centeredTimestamps), size(wf,2));
   
    for isp = 1:size(wf,2)
        wnew(:,isp) = interp1(ts, wf(:,isp), centeredTimestamps+troughloc);
    end
    
    wamp = sqrt(sum(wf.^2));
    locality = wamp(1) / wamp(3);
    
    % interpolate trough / peak with softamx
    softmax = @(x,y,p) x(:)'*(y(:).^p./sum(y(:).^p));
    
    
    winix = abs(ts - troughloc) < .25/1e3;
    
    troughloc = softmax(ts(winix), max(-wf(winix,3),0), 10);
    
    winix = abs(ts - peakloc) < .25/1e3;
    peakloc = softmax(ts(winix), max(wf(winix,3),0), 10);
    
    trough = interp1(ts, wf(:,3), troughloc);
    peak = interp1(ts, wf(:,3), peakloc);

    
    if isempty(peakloc)
        peakloc = nan;
        peak = nan;
    end
    
    if isempty(troughloc)
        troughloc = nan;
        trough = nan;
    end
    
    W(cc).cid = cid;
    W(cc).isiV = osp.isiV(cc);
    W(cc).uQ = osp.uQ(cc);
    W(cc).depth = S.y(cc);
    W(cc).x = S.x(cc);
    W(cc).lags = lags;
    W(cc).isi = xc;
    W(cc).isiRate = refrate/expectedminrate;
    W(cc).localityIdx = locality;
    W(cc).isiE = err;
    W(cc).isiL = std(detrend(xc(1:floor(numLags/2)))); % look for line noise in ISI:
    W(cc).peaktime = peakloc;
    W(cc).peakval = peak;
    W(cc).troughtime = troughloc;
    W(cc).troughval = trough;
    W(cc).waveform = wf;
    W(cc).wavelags = ts;
    W(cc).spacing = spacing;
    W(cc).shiftlags = centeredTimestamps;
    W(cc).shiftwaveform = wnew;
    W(cc).BRI = BRI;
    W(cc).normrate = normrate;
end
