function [I, xax, yax] = kdeplot(x,y, varargin)
% kernel density estimate for bivariate data
% kdeplot(x,y)

ip = inputParser();
ip.addParameter('binSize', 1)
ip.addParameter('sig', [5 5])
ip.addParameter('dim', [])
ip.parse(varargin{:});

bs  = ip.Results.binSize;
sig = ip.Results.sig;
if numel(sig)==1
    sig = repmat(sig,1,2);
end

binfun = @(x) (x==0) + ceil(x/bs);

if isempty(ip.Results.dim)
    mnx = min(x);
    mxx = max(x);
else
    mnx = ip.Results.dim(1);
    mxx = ip.Results.dim(2);
end

d = mxx-mnx;

[xx,yy] = meshgrid(linspace(-d/2,d/2,binfun(d)));


smkernel = exp( - .5 * (xx.^2./sig(1).^2 + yy.^2./sig(2).^2)); 
smkernel = smkernel./sum(smkernel(:));


X = sparse(binfun(y), binfun(x), ones(numel(x), 1)/numel(x), binfun(d), binfun(d));
I = conv2(full(X), smkernel, 'same');
    
xax = linspace(mnx, mxx, binfun(d));
yax = linspace(mnx, mxx, binfun(d));