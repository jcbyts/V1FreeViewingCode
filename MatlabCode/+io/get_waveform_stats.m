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
    'peaktime', [], ...
    'peakval', [], ...
    'troughtime', [], ...
    'troughval', [], ...
    'waveform', [], ...
    'shiftlags', [], ...
    'shiftwaveform', [], ...
    'spacing', []), NC, 1);

for cc = 1:NC

    cid = osp.cids(cc);

    sptimes = osp.st(osp.clu==cid);

    % bin spike times
    [spcnt, ~, id] = histcounts(sptimes, min(sptimes):binSize:max(sptimes));
    
    % get autocorrelation in units of excess firing rate
    lags = -numLags:numLags;
    ix = id + lags;
    ix(any(ix<1 | ix > numel(spcnt),2),:) = [];
    I = spcnt(ix);
    I(:,lags==0)= 0; % remove zero-lag spike
    
    mu = mean(I);
    mu0 = mean(spcnt);
    
    % binomial confidence intervals
    n = size(I,1);
    binoerr = 2*sqrt( (mu - mu.^2)/n);

    xc = (mu - mu0) / binSize; % in excess spikes/sec
    
    % push error through same nonlinearity ( baseline subtract / binsize)
    err = ((mu + binoerr) - mu0)/binSize - xc;
    
%     % get autocorrelation the matlab way
%     [xc, lags] = xcorr(spcnt, numLags, 'coef');
%     xc(lags==0) = 0;
    
%     figure(1); clf
%     subplot(121)
%     plot(S.templates(:,:,cc) + (1:size(S.templates,2))*10, 'k')
%     title('Waveform')
%     subplot(122)
%     plot(lags, xc, '-')
%     title('ISI')
    
    % interpolate waveform around the center of mass (handles different
    % electrode spacing / centers)
    nsp = numel(spacing);
    nts = size(S.templates, 1);
    wf = zeros(nts, nsp);
    nshanks = numel(unique(osp.xcoords));
    offset = 20;
    
    for i = 1:nts
        
        if nshanks > 1
            mask = abs(S.x(cc) - osp.xcoords) < mean(diff(unique(osp.xcoords)))/2;
        else
            mask = true(numel(osp.ycoords),1);
        end
        
        if sum(mask) >= numel(spacing)
            wf(i,:) = interp1(osp.ycoords(mask), squeeze(S.templates(i, mask, cc)), S.y(cc)+spacing);
        else
            wf_ = squeeze(S.templates(i, mask, cc));
            wf(i,1:numel(wf_)) = wf_(:);
        end
    end

    wf = wf(offset:end,:);
    nts = size(wf,1);
    ts = (1:nts)/30e3; % NOTE: sampling rate is hard coded here
    
    % shift waveform aligned to maximum excursion (peak or trough)
    centeredTimestamps = (-20:40)/30e3;
    wnew = zeros(numel(centeredTimestamps), size(wf,2));
    
    if max(-wf(:,3))> max(wf(:,3)) % align to trough
        [pks, locs] = findpeaks(-wf(:,3),ts);
        [~,id] = max(pks);
        tloc = locs(id);
        
        if isempty(tloc)
            tloc = nan;
        end
        
        for isp = 1:size(wf,2)
            wnew(:,isp) = interp1(ts, wf(:,isp), centeredTimestamps+tloc);
        end
    else % align to peak
        [pks, locs] = findpeaks(wf(:,3),ts);
        [~,id] = max(pks);
        tloc = locs(id);
        
        if isempty(tloc)
            tloc = nan;
        end
        
        for isp = 1:size(wf,2)
            wnew(:,isp) = interp1(ts, wf(:,isp), centeredTimestamps+tloc);
        end
    end
    
    
    % trough
    [pks, locs] = findpeaks(-wf(:,3),ts);
    [trough,id] = max(pks);
    trough = -trough;
    troughloc = locs(id);
    if isempty(trough)
        trough = nan;
        troughloc = nan;
    end
    
    % peak
    [pks, locs] = findpeaks(wf(:,3),ts);
    [peak,id] = max(pks);
    peakloc = locs(id);
    if isempty(peak)
        peak = nan;
        peakloc = nan;
    end
    
    W(cc).cid = cid;
    W(cc).isiV = osp.isiV(cc);
    W(cc).uQ = osp.uQ(cc);
    W(cc).depth = S.y(cc);
    W(cc).x = S.x(cc);
    W(cc).lags = lags;
    W(cc).isi = xc;
    W(cc).isiE = err;
    W(cc).isiL = std(detrend(xc(1:floor(numLags/2)))); % look for line noise in ISI:
    W(cc).peaktime = peakloc;
    W(cc).peakval = peak;
    W(cc).troughtime = troughloc;
    W(cc).troughval = trough;
    W(cc).waveform = wf;
    W(cc).spacing = spacing;
    W(cc).shiftlags = centeredTimestamps;
    W(cc).shiftwaveform = wnew;
end
