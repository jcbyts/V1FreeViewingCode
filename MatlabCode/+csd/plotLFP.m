function plotLFP(lfp, inds, excluded)
% gets the event times of the current source density trials
% Inputs:
%   lfp              [struct] - lfp struct from io.dataFactoryGratingSubspace
%                                   -- OR --
%                    [double] - lfp data that you want to plot
%                                   -- OR --
%                    [struct] - csd struct from csd.getCSD (plots STA)
%   inds             [1x2double] - indices of plotting LFP (default [1 1000])
%   excluded          [1xdeadChan] - visualize dead channels (optional)
% 
% ghs wrote it 2020

if nargin < 3
    excluded = [];
end

if nargin < 2
    inds = [1 1000];
end

if isfield(lfp,'STA')
    data = lfp.STA;
    data = reshape(data,[size(data,1)*size(data,3), size(data,2)])';
    %channelOffsets = lfp.chDepths;
elseif isa(lfp, 'double')
    data = lfp;
elseif isfield(lfp,'data')
    data = lfp.data(inds(1):inds(2), :);
else
    error('Bad LFP input: see documentation')
end

Nchan = size(data,2);

adjNchan = 1:Nchan;
adjNchan(excluded) = [];
dataExcl = data(:, excluded);
data = data(:,adjNchan);

figure(1); clf
channelOffsets =(adjNchan)*100;
%dtmp = data(channelMap,:)';
%dtmp = dtmp - mean(dtmp,2);
plot(bsxfun(@plus, data, channelOffsets), 'Color', 'black'); hold on
if ~isempty(excluded)
    channelOffsets =(excluded)*100;
    plot(bsxfun(@plus, dataExcl, channelOffsets), 'Color', 'red');
end
legend('Not excl', 'Excluded data')
yticks(channelOffsets)
yticklabels(num2cell(adjNchan))



end

