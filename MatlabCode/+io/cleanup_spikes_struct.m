function sp = cleanup_spikes_struct(sp, varargin)
% sp = cleanup_spikes_struct(sp, varargin)

ip = inputParser();
ip.addParameter('firingratethresh', 1)
ip.addParameter('ampthresh', 1)
ip.addParameter('verbose', 0)
ip.parse(varargin{:});

% remove units with firing rates below 
firingratethresh = ip.Results.firingratethresh; % sp/s

% if one of the violating spikes is more than a standard deviation away from the cluster mean, remove it
ampthresh = ip.Results.ampthresh; 

nbins = 200;
dt = 1e-3;

flagtoremove = [];

DT = max(sp.st)-min(sp.st);

cids = sp.cids;
NC = numel(cids);

for cc = 1:NC
    
    spix = find(sp.clu==cids(cc));
    
    if (numel(spix)/DT) < firingratethresh
        flagtoremove = [flagtoremove cc];
        continue
    end
    
    sptimes = sp.st(spix); % unit spike times
    spamps = sp.spikeAmps(spix); % unit amplitudes (for each spike)
    sdev = sqrt(robustcov(spamps)); % robust standard deviation of 

    K = ccg(sptimes, sptimes, nbins, dt);
    K(nbins+1) = 0;
    
    if ip.Results.verbose
        figure(1); clf
        plot(K)
    end
    
    % find all refractory violations
    rfviols = find(diff(sptimes)<dt);
    rempots = rfviols(:)+[1 0];
    nviol = numel(rfviols);
    
    if ~isfield(sp, 'clusterAmps')
        mamp = mean(sp.spikeAmps(spix));
    else
        mamp = sp.clusterAmps(cc);
    end
    
    ampdev = abs((spamps(rempots) - mamp) / sdev);
    if nviol==1
        ampdev = ampdev(:)';
    end
    [~, id] = max(ampdev, [], 2);
    
    maxmask = false(nviol, 2);
    ampmask = ampdev > ampthresh;
    
    maxmask(sub2ind([nviol, 2], 1:nviol, id(:)')) = true;
    
    toremove = rempots(ampmask & maxmask);
    
    sp.clu(spix(toremove)) = 0;
    
    % recompute spike times
    spix = sp.clu==cids(cc);
    sptimes = sp.st(spix);
    
    K = ccg(sptimes, sptimes, nbins, dt);
    K(nbins+1) = 0;
    if ip.Results.verbose
        hold on
        plot(K)
    end
end

% remove units that didn't meat the firing rate threshold
fields = fieldnames(sp);
fields = fields(cellfun(@(x) numel(sp.(x)), fields)==NC);
numFields = numel(fields);
for iField = 1:numFields
    sp.(fields{iField})(flagtoremove) = [];
end


