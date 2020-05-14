function S = getUnitLocations(osp, pow)
% S = getUnitLocations(osp)
% Input:
%   osp [struct] - getSpikesFromKilo output
%   pow [double] - soft max power    
% Output:
%   S [struct]
%       x 
%       y 
%       templates
if nargin < 2
    pow = 2;
end

NC = numel(osp.cids);

x = nan(NC,1);
y = nan(NC,1);

nLags = size(osp.temps,2);
nChan = size(osp.temps,3);

templates = zeros(nLags, nChan, NC);
for cc = 1:NC
    tempix = unique(osp.spikeTemplates(osp.clu==osp.cids(cc)))+1;

    unitTemp = squeeze(mean(osp.tempsUnW(tempix, :, :),1));
    templates(:,:,cc) = unitTemp;
    
    u = sum(unitTemp.^2,1);
    w = u.^pow /sum(u(:).^pow);
    x(cc) = w*osp.xcoords;
    y(cc) = w*osp.ycoords;
end

S.x = x;
S.y = y;
S.templates = templates;