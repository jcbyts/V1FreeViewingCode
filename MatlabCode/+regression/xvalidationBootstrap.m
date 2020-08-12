function xidxs = xvalidationBootstrap(nTotalSamples, Boots)
% xidxs = xvalidationBootstrap(nTotalSamples, Boots)
% Get Bootstrap cross-validation indices.
%
% Input
%   nTotalSamples: integer number of samples or indices
%   Boots:         integer number of bootstrap iterations
% Output
%   xidxs: {Boots x 2} training, test indices

xidxs = cell(Boots, 2);
ix = 1:nTotalSamples;

for k = 1:Boots
    B = randi(nTotalSamples, [nTotalSamples 1]);
    xidxs{k,1} = B;
    xidxs{k,2} = setdiff(ix, B);
end
