function [TCs, TCerr, ptun, stimid] = computeTuningCurves(spks, stim, nBasis)
% [TCs, TCerr] = computeTuningCurves(spks, stim)

rr = spks;
nneur = size(spks,2);

if nargin < 3 || nBasis == 0
    stimid = unique(stim)';
    ss = stim(:)==stimid(:)';
else
    [g, stimid] = vm_basis_deg(nBasis);
    % stimulus evaluated on basis
    ss = g(stim);
end


npres = sum(ss);  % number of times each stimulus presented
% disp(npres)
TCs = bsxfun(@rdivide,(ss'*rr),npres');  % tuning curve (avg response per neuron)
% TCs(npres < 2,:) = nan;
TCsd = sqrt(bsxfun(@rdivide, ss'*(rr - (ss*TCs)).^2, npres'-1)); % standard deviation
TCerr = bsxfun(@rdivide, TCsd, sqrt(npres)');

if nargout > 2
    % remove untuned neurons
    ptun = zeros(1,nneur);
    for k = 1:nneur
        ptun(k) = anova1(rr(:,k), stim, 'off');
    end
end