function [x, y] = getUnitLocations(osp, pow)
% [x, y] = getUnitLocations(Exp)

if nargin < 2
    pow = 2;
end

NC = numel(osp.cids);

x = nan(NC,1);
y = nan(NC,1);


for cc = 1:NC
    tempix = unique(osp.spikeTemplates(osp.clu==osp.cids(cc)))+1;

    unitTemp = squeeze(mean(osp.tempsUnW(tempix, :, :),1));
    u = sum(unitTemp.^2,1);
    w = u.^pow /sum(u(:).^pow);
    x(cc) = w*osp.xcoords;
    y(cc) = w*osp.ycoords;
end