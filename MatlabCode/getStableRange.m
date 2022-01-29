function goodix = getStableRange(R, varargin)
% get stable range using change point analysis
% goodix = getStableRange(R, varargin)

ip = inputParser();
ip.addParameter('plot', true)
ip.addParameter('changeptargs', {'Statistic', 'linear', 'MinThreshold', 50*var(R)}) %{'Statistic', 'mean', 'MinDistance', 1, 'MaxNumChanges', 3})
ip.parse(varargin{:})


% stability analysis
[ipoints, ~] = findchangepts(R, ip.Results.changeptargs{:});
% [ipoints, ~] = findchangepts(R, );
   
if ip.Results.plot
    figure(111); clf
end

n=numel(R);
ipoints = [0; ipoints; n];
stableFr = [];
Tspikes = [];
for ichng = 2:numel(ipoints)
    i0 = ipoints(ichng-1)+1;
    i1 = ipoints(ichng);
    iix = i0:i1;
    Tspikes = [Tspikes; sum(R(iix))];
    stableFr = [stableFr mean(R(iix))*ones(size(iix))]; %#ok<AGROW>
end

if ip.Results.plot
    plot(R); hold on
    plot(stableFr);
end

if rsquared(R, stableFr) > .2 % changepoints explain 10% more variance
    
%     len = diff(ipoints);
    [~, bigi] = max(Tspikes);
%     [~, bigi] = max(len);
    
    goodix = (ipoints(bigi)+1):(ipoints(bigi+1));
else
    goodix = 1:n;
end

if ip.Results.plot
    plot(goodix, R(goodix), '.')
end

