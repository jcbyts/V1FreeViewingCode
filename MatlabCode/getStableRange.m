function goodix = getStableRange(R, varargin)
% get stable range using change point analysis

ip = inputParser();
ip.addParameter('plot', true)
ip.parse(varargin{:})


% stability analysis
% [ipoints, ~] = findchangepts(R, 'Statistic', 'mean', 'MinDistance', 1, 'MaxNumChanges', 10);
[ipoints, ~] = findchangepts(R, 'Statistic', 'linear', 'MinThreshold', 50*var(R));
   
if ip.Results.plot
    figure(111); clf
end

n=numel(R);
ipoints = [0; ipoints; n];
stableFr = [];
for ichng = 2:numel(ipoints)
    i0 = ipoints(ichng-1)+1;
    i1 = ipoints(ichng);
    iix = i0:i1;
    stableFr = [stableFr mean(R(iix))*ones(size(iix))]; %#ok<AGROW>
end

if ip.Results.plot
    plot(R); hold on
    plot(stableFr);
end

if rsquared(R, stableFr) > .2 % changepoints explain 10% more variance
    
    len = diff(ipoints);
    [~, bigi] = max(len);
    
    goodix = (ipoints(bigi)+1):(ipoints(bigi+1));
else
    goodix = 1:n;
end

if ip.Results.plot
    plot(goodix, R(goodix), '.')
end

