function xidxs = xvalidationLeaveOneOut(nTotalSamples)
% xidxs = xvalidationLeaveOneOut(nTotalSamples)
% Get k-fold cross-validation indices.
%
% Input
%   nTotalSamples: integer number of samples or indices
% Output
%   xidxs: {kxValidation x 2} training, test indices

xidxs = cell(nTotalSamples, 2);
ix = 1:nTotalSamples;
for k = 1:nTotalSamples
    xidxs{k,1} = setdiff(ix, k);
    xidxs{k,2} = k;
end
