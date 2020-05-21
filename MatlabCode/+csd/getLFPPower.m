function power = getLFPPower(lfp, frange, varargin)
% Computes average spectral power density for each channel in LFP
%
%   Input:
%   lfp              [struct] - lfp struct from io.dataFactoryGratingSubspace
%                                   -- OR --
%                    [double] - lfp data that you want to get power from
%
%   frange           [string] - frequency range to compute bandpower Hz (default: 'low gamma')
%           *frange options: 'low gamma' (20-80Hz), 'high gamma' (80-200Hz)
%                                   --OR--
%                    [double] - two element vector containing frequencies                                   
%
%   Parser inputs:
%       - sampleRate: sampling rate Hz (default: 1000)
%       - plotIt: true/false - plot power dens (default: true)
%       - norm: true/false - normalize power (b/t 0-1)
%       - inds: indices of lfp to compute power (default: [1 100000])
%                   - note: leave inds empty ([]) if want entire signal
%       - exclude: true/false - exclude and interpolate bad channels
%   Output:
%       - power: average power density - 1 x nChan
%
% ghs wrote it 2019
% ghs edit it 2020

ip = inputParser();
ip.addParameter('sampleRate', 1e3)
ip.addParameter('plotIt', true)
ip.addParameter('norm', true)
ip.addParameter('inds', [1 100000])
ip.addParameter('exclude', true)
ip.parse(varargin{:});
inds = ip.Results.inds;

if nargin < 2
    frange = [20 80];
elseif isa(frange, 'double')
    %frange = frange;
elseif  strcmp(frange,'low gamma')
    frange = [20 80];
elseif strcmp(frange, 'high gamma')
    frange = [80 200];
else
    error('Wrong frange input: see documentation')
end

if isa(lfp, 'double')
    data = lfp;
elseif isfield(lfp,'data')
    if isempty(inds)
        data = lfp.data;
    else
        data = lfp.data(inds(1):inds(2), :);
    end
else
    error('Wrong LFP input: see documentation')
end

p = bandpower(data,ip.Results.sampleRate,frange);

if ip.Results.exclude
    p(lfp.deadChan) = NaN;
    p(lfp.deadChan) = interp1(p, lfp.deadChan, 'spline');
end

if ip.Results.norm
    normP = p - min(p(:));
    power = normP ./ max(normP(:));
else
    power = p;
end

if ip.Results.plotIt == true
    figure(30); clf;
    plot(power)
    xlabel('Channel')
    ylabel('Power')
end

