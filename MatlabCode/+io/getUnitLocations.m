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

if isfield(osp, 'WFmed')
    useWF = true;
    temps = osp.WFmed;
else
    useWF = false;
    temps = osp.temps;
end

nLags = size(temps,2);
nChan = size(temps,3);

if nChan == 64
    warning('hack channel map for 2-shank probes')
    osp.xcoords = [zeros(32, 1); 200*ones(32, 1)];
    osp.ycoords = [(1:32)'*35; (1:32)'*35];
end

xcoords = osp.xcoords;
ycoords = osp.ycoords;
    
templates = zeros(nLags, nChan, NC);
for cc = 1:NC
    if useWF
        unitTemp = squeeze(temps(cc,:,:));
    else    
        tempix = unique(osp.spikeTemplates(osp.clu==osp.cids(cc)))+1;
        unitTemp = squeeze(mean(osp.tempsUnW(tempix, :, :),1));
    end
    templates(:,:,cc) = unitTemp;
    
    u = sum(unitTemp.^2,1);
    w = u.^pow /sum(u(:).^pow);
    
    x(cc) = w*xcoords;
    y(cc) = w*ycoords;
end

S.x = x;
S.y = y;
S.templates = templates;
S.xcoords = xcoords;
S.ycoords = ycoords;
S.useWF = useWF;